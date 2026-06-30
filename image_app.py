"""
Modal app — ComfyUI Qwen virtual try-on.

Architecture:
    POST /predict (proxy-auth-protected)
      → handler downloads input images from R2 (user-images or catalog bucket)
      → ComfyUI runs the workflow
      → handler uploads outputs to R2 (panneau-user-images)
      → returns the R2 keys

Models live in a modal.Volume seeded from R2; see scripts/sync_r2_to_volume.py.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field

# ─── App config ──────────────────────────────────────────────────────────────

app = modal.App("panneau-comfy")

models_vol = modal.Volume.from_name("panneau-models", create_if_missing=True)

# ─── Image ───────────────────────────────────────────────────────────────────

COMFY_DIR = "/comfyui"
COMFY_PORT = 8188

# GPU type — single source of truth for the @app.cls decorator AND the
# cost-attribution field returned to panneau. Change here when swapping GPUs;
# panneau's rate table keys on this exact string (must match Modal's GPU
# pricing label), so a swap needs no panneau code change — just a rate row.
GPU = "RTX-PRO-6000"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0", "libglib2.0-dev")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}",
        f"cd {COMFY_DIR} && pip install --no-cache-dir -r requirements.txt",
    )
    .run_commands(
        # Only nodes referenced by the active image try-on workflows are installed.
        # Removed (no class_type from these appears in any active workflow):
        #   ComfyUI-GGUF (workflows use stock UNETLoader, not GGUF loaders),
        #   ComfyUI-FASHN-VTON, rgthree-comfy (API-only deploy; no rgthree nodes used),
        #   ComfyUI-SeedVR2_VideoUpscaler, ComfyUI-WanVideoWrapper,
        #   ComfyUI-VideoHelperSuite, ComfyUI-Frame-Interpolation (video — disabled).
        # Removing the 4 video/upscale packs also drops the import-time CUDA init
        # (SeedVR2 bfloat16 probe, Wan get_torch_device) that slowed every cold start.
        f"git clone https://github.com/kijai/ComfyUI-KJNodes.git {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        f"git clone https://github.com/Acly/comfyui-inpaint-nodes.git {COMFY_DIR}/custom_nodes/comfyui-inpaint-nodes"
        f" && cd {COMFY_DIR}/custom_nodes/comfyui-inpaint-nodes && git checkout b9039c2",
        # ComfyMath — int/float arithmetic for in-workflow padding/dimension calculations
        f"git clone https://github.com/evanspearman/ComfyMath.git {COMFY_DIR}/custom_nodes/ComfyMath",
    )
    .pip_install("sageattention==1.0.6", "kornia==0.8.2")
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    .run_commands("pip install --no-cache-dir flash-attn --no-build-isolation")
    .pip_install(
        "boto3==1.35.0",
        "fastapi[standard]==0.115.0",
        "requests==2.32.3",
        "pydantic==2.9.2",
    )
    # Segformer B2 clothes (ATR human parsing) — garment isolation for try-on
    # reference images. Appended LATE on purpose: keeps the expensive flash-attn
    # build layer above cached.
    #   • NOT installing the node's requirements.txt — it pins transformers==4.33.2,
    #     which would downgrade the image's transformers and break the Qwen 2.5-VL
    #     CLIP loader. The existing transformers runs Segformer inference fine.
    #   • The pack's __init__.py imports BOTH segformer_b2_clothes AND
    #     segformer_b3_fashion, and EACH module loads its model at IMPORT time
    #     (module level). We only use b2 (clothes/ATR parsing). The b3 module's
    #     module-level load of the (unused) b3 model was failing the WHOLE pack
    #     import → neither node registered ("node not found"). Fix: strip every
    #     b3 reference from __init__.py so only b2 is imported — no b3 model needed.
    #   • b2 model baked in via the image's EXISTING huggingface_hub. Do NOT
    #     pin/upgrade huggingface_hub here: hf_hub>=0.25 dropped the top-level
    #     `is_offline_mode` symbol the image's transformers imports, which
    #     crash-loops ComfyUI at boot. The bundled hf_hub already ships
    #     snapshot_download.
    .run_commands(
        f"git clone https://github.com/StartHua/Comfyui_segformer_b2_clothes.git"
        f" {COMFY_DIR}/custom_nodes/Comfyui_segformer_b2_clothes",
        f"sed -i '/segformer_b3_fashion/d'"
        f" {COMFY_DIR}/custom_nodes/Comfyui_segformer_b2_clothes/__init__.py",
        "python -c \"from huggingface_hub import snapshot_download;"
        " snapshot_download(repo_id='mattmdjaga/segformer_b2_clothes',"
        f" local_dir='{COMFY_DIR}/models/segformer_b2_clothes')\"",
    )
    # SAM2 (segment-anything-2) — clean garment isolation, especially swimwear/
    # bikini where Segformer mislabels skin as garment (tan blob, ragged edges).
    # The node bundles its own sam2 code (pyproject dependencies=[]; imports the
    # relative .sam2 package and comfy.utils loader) so NO extra pip deps are
    # needed — just the clone. Checkpoint baked into models/sam2 to avoid a
    # cold-start HuggingFace download. Uses the image's existing huggingface_hub
    # (same as the Segformer block) — do NOT pin/upgrade it.
    .run_commands(
        f"git clone https://github.com/kijai/ComfyUI-segment-anything-2.git"
        f" {COMFY_DIR}/custom_nodes/ComfyUI-segment-anything-2",
        "python -c \"from huggingface_hub import hf_hub_download;"
        " hf_hub_download(repo_id='Kijai/sam2-safetensors',"
        " filename='sam2.1_hiera_base_plus.safetensors',"
        f" local_dir='{COMFY_DIR}/models/sam2')\"",
    )
    # Florence-2 (open-vocabulary grounding) — AUTOMATIC garment detection by text
    # ("clothing", "swimsuit") so SAM2 needs NO manual point coordinates and
    # generalizes to any garment, including swimwear that Segformer cannot parse.
    # Pipeline: Florence2Run(caption_to_phrase_grounding, text) → bbox →
    # Florence2toCoordinates (in the SAM2 node) → Sam2Segmentation → mask.
    #   • Transformers-based, NO CUDA-op build (unlike GroundingDINO).
    #   • The node's requirements pin nothing that touches the image's
    #     transformers (Qwen 2.5-VL CLIP loader) — we install only the light
    #     extras the Florence-2 remote code needs (timm, einops, matplotlib).
    #   • Florence-2-base baked into models/LLM to avoid a cold-start download.
    .run_commands(
        f"git clone https://github.com/kijai/ComfyUI-Florence2.git"
        f" {COMFY_DIR}/custom_nodes/ComfyUI-Florence2",
        "pip install --no-cache-dir timm einops matplotlib",
        "python -c \"from huggingface_hub import snapshot_download;"
        " snapshot_download(repo_id='microsoft/Florence-2-base',"
        f" local_dir='{COMFY_DIR}/models/LLM/Florence-2-base')\"",
    )
    # KeepLargeMaskComponents — drops small disconnected mask islands (e.g. skin
    # mislabelled as garment on swimwear) so the garment mask is clean before
    # feathering. Single-file node, isolated dir; does not touch the cloned packs.
    .add_local_dir("comfy_mask_cc", f"{COMFY_DIR}/custom_nodes/comfy_mask_cc")
    .add_local_dir("comfy", "/app/comfy")
)


# ─── Request / response models ───────────────────────────────────────────────


class ImageRef(BaseModel):
    bucket: Literal["user-images", "catalog"] = Field(
        ..., description="Which R2 bucket the image lives in."
    )
    key: str = Field(..., description="R2 object key within that bucket.")


class PredictRequest(BaseModel):
    tenant_id: str = Field(..., description="UUID of the tenant making the request")
    workflow: dict = Field(..., description="ComfyUI API-format workflow JSON")
    images: dict[str, ImageRef] = Field(
        default_factory=dict,
        description=(
            "Map of local filename (as referenced inside the workflow) → "
            "{bucket, key} pointer to an image in R2."
        ),
    )
    timeout: int = Field(default=600, ge=10, le=1800)


class PredictResponse(BaseModel):
    outputs: list[str]
    prompt_id: str
    # Cost attribution — read by panneau into usage_events. gpu_seconds is the
    # wall-clock this request held the (dedicated, max_inputs=1) GPU container,
    # i.e. the attributable active time. cold_start_seconds is the container
    # boot/load time and is non-zero ONLY when this request triggered the cold
    # start (the user waited for the wake) — attributable to this generation.
    # System/anticipatory warm-ups are separate calls that write no usage_event.
    gpu_type: str = GPU
    gpu_seconds: float = 0.0
    cold_start: bool = False
    cold_start_seconds: float = 0.0


# ─── The class ───────────────────────────────────────────────────────────────

TENANT_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

USER_IMAGES_BUCKET = "panneau-user-images"
CATALOG_BUCKET = "panneau-catalog"

BUCKETS = {
    "user-images": {
        "name": USER_IMAGES_BUCKET,
        "key_id_env": "AWS_ACCESS_KEY_ID",
        "secret_env": "AWS_SECRET_ACCESS_KEY",
    },
    "catalog": {
        "name": CATALOG_BUCKET,
        "key_id_env": "CATALOG_AWS_ACCESS_KEY_ID",
        "secret_env": "CATALOG_AWS_SECRET_ACCESS_KEY",
    },
}


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": models_vol},
    secrets=[
        modal.Secret.from_name("panneau-r2-user-images"),
        modal.Secret.from_name("panneau-r2-catalog"),
    ],
    min_containers=0,
    max_containers=5,
    scaledown_window=300,
    timeout=60 * 30,
)
@modal.concurrent(max_inputs=1)
class ComfyUI:
    @modal.enter()
    def boot(self):
        """Start ComfyUI subprocess and wait for it to be ready."""
        boot_t0 = time.perf_counter()
        # _cold flips to False after the first request this container serves, so
        # the boot/load time is attributed to exactly one generation — the one
        # that waited for the wake.
        self._cold = True
        self._boot_seconds = 0.0
        # Render extra_model_paths.yaml into the ComfyUI dir.
        src = Path("/app/comfy/extra_model_paths.yaml")
        dst = Path(f"{COMFY_DIR}/extra_model_paths.yaml")
        dst.write_text(src.read_text())

        # Launch the ComfyUI server as a subprocess.
        self._proc = subprocess.Popen(
            [
                "python",
                "main.py",
                "--listen",
                "0.0.0.0",
                "--port",
                str(COMFY_PORT),
                "--disable-auto-launch",
                "--fast",
                "--extra-model-paths-config",
                "extra_model_paths.yaml",
            ],
            cwd=COMFY_DIR,
        )

        # Wait for /system_stats to respond.
        import requests

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2
                )
                if r.ok:
                    self._boot_seconds = time.perf_counter() - boot_t0
                    print(f"[boot] ComfyUI ready in {self._boot_seconds:.1f}s.")
                    return
            except requests.RequestException:
                pass
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited prematurely with code {self._proc.returncode}"
                )
            time.sleep(2)
        raise RuntimeError("ComfyUI did not become ready within 5 minutes.")

    # ── helpers ────────────────────────────────────────────────────────────

    def _s3_for(self, bucket_alias: str):
        import boto3

        cfg = BUCKETS[bucket_alias]
        return boto3.client(
            "s3",
            endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            aws_access_key_id=os.environ[cfg["key_id_env"]],
            aws_secret_access_key=os.environ[cfg["secret_env"]],
        )

    def _validate(self, req: PredictRequest) -> None:
        from fastapi import HTTPException

        # tenant_id still validated because outputs are written under it.
        if not TENANT_UUID_RE.match(req.tenant_id):
            raise HTTPException(400, "invalid tenant_id format")
        try:
            uuid.UUID(req.tenant_id)
        except ValueError:
            raise HTTPException(400, "invalid tenant_id")

        for local_name, ref in req.images.items():
            if "/" in local_name or ".." in local_name or local_name.startswith("."):
                raise HTTPException(400, f"invalid local filename: {local_name}")
            if ref.bucket not in BUCKETS:
                raise HTTPException(400, f"unknown bucket: {ref.bucket}")
            if not ref.key or ".." in ref.key or ref.key.startswith("/"):
                raise HTTPException(400, f"invalid key: {ref.key}")

    def _download_inputs(self, req: PredictRequest) -> None:
        input_dir = Path(f"{COMFY_DIR}/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        # Cache one s3 client per bucket to avoid recreating each loop.
        clients: dict[str, object] = {}
        for local_name, ref in req.images.items():
            if ref.bucket not in clients:
                clients[ref.bucket] = self._s3_for(ref.bucket)
            dst = input_dir / local_name
            clients[ref.bucket].download_file(
                BUCKETS[ref.bucket]["name"], ref.key, str(dst)
            )

    def _queue(self, workflow: dict) -> str:
        import requests
        from fastapi import HTTPException

        client_id = str(uuid.uuid4())
        r = requests.post(
            f"http://127.0.0.1:{COMFY_PORT}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        if not r.ok:
            try:
                body = r.json()
            except ValueError:
                body = r.text[:2000]
            raise HTTPException(
                status_code=400 if r.status_code in (400, 422) else 500,
                detail={
                    "error": "comfyui rejected workflow",
                    "comfyui_status": r.status_code,
                    "comfyui_response": body,
                },
            )
        return r.json()["prompt_id"]

    def _wait(self, prompt_id: str, timeout: int) -> dict:
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(
                f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}", timeout=10
            )
            history = r.json()
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout}s")

    def _upload_outputs(self, tenant_id: str, history: dict) -> list[str]:
        import requests

        s3 = self._s3_for("user-images")
        uploaded: list[str] = []
        # ComfyUI groups outputs by type (images, gifs, videos). Handle all.
        for node_output in history.get("outputs", {}).values():
            for output_type in ("images", "gifs", "videos", "animated"):
                for item in node_output.get(output_type, []):
                    view = (
                        f"http://127.0.0.1:{COMFY_PORT}/view"
                        f"?filename={item['filename']}"
                        f"&subfolder={item.get('subfolder', '')}"
                        f"&type={item.get('type', 'output')}"
                    )
                    r = requests.get(view, timeout=120)
                    r.raise_for_status()
                    ext = Path(item["filename"]).suffix or ".png"
                    key = f"{tenant_id}/outputs/{uuid.uuid4()}{ext}"
                    # Pick content type based on extension
                    ext_lower = ext.lstrip(".").lower()
                    if ext_lower in ("mp4", "mov", "webm"):
                        content_type = f"video/{ext_lower}"
                    elif ext_lower == "gif":
                        content_type = "image/gif"
                    else:
                        content_type = f"image/{ext_lower}"
                    s3.put_object(
                        Bucket=USER_IMAGES_BUCKET,
                        Key=key,
                        Body=r.content,
                        ContentType=content_type,
                    )
                    uploaded.append(key)
        return uploaded

    # ── endpoint ───────────────────────────────────────────────────────────

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def predict(self, req: PredictRequest) -> PredictResponse:
        from fastapi import HTTPException

        t0 = time.perf_counter()
        # Attribute the container boot to the first request only (the user who
        # triggered the cold start). Subsequent requests on this warm container
        # report cold_start=False / 0s.
        cold = getattr(self, "_cold", False)
        cold_start_seconds = getattr(self, "_boot_seconds", 0.0) if cold else 0.0
        self._cold = False

        self._validate(req)
        try:
            self._download_inputs(req)
        except Exception as exc:
            raise HTTPException(400, f"failed to fetch inputs: {exc}")

        prompt_id = self._queue(req.workflow)
        history = self._wait(prompt_id, timeout=req.timeout)

        status = history.get("status", {})
        if status.get("status_str") == "error":
            raise HTTPException(
                500,
                detail={
                    "error": "workflow execution failed",
                    "messages": status.get("messages", []),
                },
            )

        outputs = self._upload_outputs(req.tenant_id, history)
        return PredictResponse(
            outputs=outputs,
            prompt_id=prompt_id,
            gpu_type=GPU,
            gpu_seconds=round(time.perf_counter() - t0, 3),
            cold_start=cold,
            cold_start_seconds=round(cold_start_seconds, 3),
        )
