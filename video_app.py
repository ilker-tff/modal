"""
Modal app — ComfyUI WAN 2.2 image-to-video.

Separate app from the image try-on (panneau-comfy) on purpose: a lean,
video-only custom-node set keeps cold starts fast (no segformer / ComfyMath /
inpaint / flash-attn build that the image app carries).

Architecture (identical contract to the image app):
    POST /predict (proxy-auth-protected)
      → handler downloads input images from R2 (user-images or catalog bucket)
      → ComfyUI runs the WAN workflow sent by the caller
      → handler uploads outputs (mp4/gif) to R2 (panneau-user-images)
      → returns the R2 keys

Models live in the shared "panneau-models" modal.Volume (read-only here);
the WAN unets / umt5 text encoder / wan vae / lightx2v loras must already be
present under it (seeded via scripts/sync_r2_to_volume.py).
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

app = modal.App("panneau-comfy-video")

models_vol = modal.Volume.from_name("panneau-models", create_if_missing=True)

# ─── Image ───────────────────────────────────────────────────────────────────

COMFY_DIR = "/comfyui"
COMFY_PORT = 8188

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    # ffmpeg is required by ComfyUI-VideoHelperSuite (VHS_VideoCombine -> mp4).
    .apt_install(
        "git", "wget", "curl", "ffmpeg", "libgl1", "libglib2.0-0", "libglib2.0-dev"
    )
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
        # Only the custom nodes the WAN 2.2 i2v workflow actually uses:
        #   • ComfyUI-KJNodes        — WanVideoNAG (quality)
        #   • rgthree-comfy          — Power Lora Loader (lightx2v high/low)
        #   • ComfyUI-VideoHelperSuite — VHS_VideoCombine (mp4 output)
        #   • ComfyUI-Image-Saver    — Seed Generator
        # WAN i2v / first-last-frame nodes themselves are native comfy_extras
        # (nodes_wan) — no WanVideoWrapper needed (it was the heaviest import).
        # Frame-Interpolation (FILM VFI) and RES4LYF (Film Grain) are intentionally
        # NOT installed: those post-processing nodes are bypassed in the workflow.
        f"git clone https://github.com/kijai/ComfyUI-KJNodes.git {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        f"git clone https://github.com/rgthree/rgthree-comfy.git {COMFY_DIR}/custom_nodes/rgthree-comfy"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/rgthree-comfy/requirements.txt",
        f"git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git {COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        f"git clone https://github.com/alexopus/ComfyUI-Image-Saver.git {COMFY_DIR}/custom_nodes/ComfyUI-Image-Saver"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-Image-Saver/requirements.txt",
    )
    .pip_install(
        "boto3==1.35.0",
        "fastapi[standard]==0.115.0",
        "requests==2.32.3",
        "pydantic==2.9.2",
    )
    .add_local_dir("comfy", "/app/comfy")
)


# ─── Request / response models (same contract as the image app) ──────────────


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
    timeout: int = Field(default=900, ge=10, le=1800)


class PredictResponse(BaseModel):
    outputs: list[str]
    prompt_id: str


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
    # A100-80GB is the proven WAN config. RTX-PRO-6000 (Blackwell sm_120) was
    # unreliable for WAN on older torch builds; revisit on cu128 for cost later.
    gpu="A100-80GB",
    volumes={"/models": models_vol},
    secrets=[
        modal.Secret.from_name("panneau-r2-user-images"),
        modal.Secret.from_name("panneau-r2-catalog"),
    ],
    min_containers=0,
    max_containers=2,
    # Video is sparse + long-running; a short grace avoids reloads within a
    # burst but we accept cold starts otherwise.
    scaledown_window=120,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=1)
class ComfyUI:
    @modal.enter()
    def boot(self):
        """Start ComfyUI subprocess and wait for it to be ready."""
        src = Path("/app/comfy/extra_model_paths.yaml")
        dst = Path(f"{COMFY_DIR}/extra_model_paths.yaml")
        dst.write_text(src.read_text())

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

        import requests

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2
                )
                if r.ok:
                    print("[boot] ComfyUI ready.")
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
        # ComfyUI groups outputs by type. VHS_VideoCombine reports mp4 under
        # "gifs"; handle every container type to be safe.
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
        return PredictResponse(outputs=outputs, prompt_id=prompt_id)
