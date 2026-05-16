"""
Modal app — ComfyUI Qwen virtual try-on.

Architecture:
    POST /predict (proxy-auth-protected)
      → handler downloads input images from R2 (panneau-user-images)
      → ComfyUI runs the workflow
      → handler uploads outputs back to R2
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

import modal
from pydantic import BaseModel, Field

# ─── App config ──────────────────────────────────────────────────────────────

app = modal.App("panneau-comfy")

models_vol = modal.Volume.from_name("panneau-models", create_if_missing=True)

# ─── Image ───────────────────────────────────────────────────────────────────

COMFY_DIR = "/comfyui"
COMFY_PORT = 8188

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
        f"git clone https://github.com/city96/ComfyUI-GGUF.git {COMFY_DIR}/custom_nodes/ComfyUI-GGUF"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-GGUF/requirements.txt",
        f"git clone https://github.com/drphero/ComfyUI-FASHN-VTON.git {COMFY_DIR}/custom_nodes/ComfyUI-FASHN-VTON"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-FASHN-VTON/requirements.txt",
        f"git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git {COMFY_DIR}/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt",
        f"git clone https://github.com/kijai/ComfyUI-KJNodes.git {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes/requirements.txt",
    )
    .pip_install("sageattention==1.0.6")
    .pip_install("wheel", "packaging", "ninja", "setuptools")
    .run_commands("pip install --no-cache-dir flash-attn --no-build-isolation")
    .pip_install(
        "boto3==1.35.0",
        "fastapi[standard]==0.115.0",
        "requests==2.32.3",
        "pydantic==2.9.2",
    )
    .add_local_dir("comfy", "/app/comfy")
)


# ─── Request / response models ───────────────────────────────────────────────


class PredictRequest(BaseModel):
    tenant_id: str = Field(..., description="UUID of the tenant making the request")
    workflow: dict = Field(..., description="ComfyUI API-format workflow JSON")
    images: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of local filename (as referenced inside the workflow) → "
            "R2 key in panneau-user-images. Keys must start with "
            "'<tenant_id>/inputs/'."
        ),
    )
    timeout: int = Field(default=600, ge=10, le=1800)


class PredictResponse(BaseModel):
    outputs: list[str]
    prompt_id: str


# ─── The class ───────────────────────────────────────────────────────────────

TENANT_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

USER_IMAGES_BUCKET = "panneau-user-images"


@app.cls(
    image=image,
    gpu=["A100-80GB", "H100"],
    volumes={"/models": models_vol},
    secrets=[
        modal.Secret.from_name("panneau-r2-user-images"),
    ],
    min_containers=0,
    max_containers=5,
    scaledown_window=60,
    timeout=60 * 30,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)
class ComfyUI:
    @modal.enter(snap=True)
    def boot(self):
        """Start ComfyUI subprocess and wait for it to be ready."""
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

    def _s3(self):
        import boto3

        return boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])

    def _validate(self, req: PredictRequest) -> None:
        if not TENANT_UUID_RE.match(req.tenant_id):
            from fastapi import HTTPException

            raise HTTPException(400, "invalid tenant_id format")
        try:
            uuid.UUID(req.tenant_id)
        except ValueError:
            from fastapi import HTTPException

            raise HTTPException(400, "invalid tenant_id")

        expected_prefix = f"{req.tenant_id}/inputs/"
        for local_name, key in req.images.items():
            if "/" in local_name or ".." in local_name or local_name.startswith("."):
                from fastapi import HTTPException

                raise HTTPException(400, f"invalid local filename: {local_name}")
            if not key.startswith(expected_prefix):
                from fastapi import HTTPException

                raise HTTPException(403, f"key out of tenant scope: {key}")
            basename = key[len(expected_prefix) :]
            if "/" in basename or ".." in basename or basename.startswith("."):
                from fastapi import HTTPException

                raise HTTPException(400, f"invalid key: {key}")

    def _download_inputs(self, req: PredictRequest) -> None:
        input_dir = Path(f"{COMFY_DIR}/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        s3 = self._s3()
        for local_name, key in req.images.items():
            dst = input_dir / local_name
            s3.download_file(USER_IMAGES_BUCKET, key, str(dst))

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

        s3 = self._s3()
        uploaded: list[str] = []
        for node_output in history.get("outputs", {}).values():
            for img in node_output.get("images", []):
                view = (
                    f"http://127.0.0.1:{COMFY_PORT}/view"
                    f"?filename={img['filename']}"
                    f"&subfolder={img.get('subfolder', '')}"
                    f"&type={img.get('type', 'output')}"
                )
                r = requests.get(view, timeout=60)
                r.raise_for_status()
                ext = Path(img["filename"]).suffix or ".png"
                key = f"{tenant_id}/outputs/{uuid.uuid4()}{ext}"
                s3.put_object(
                    Bucket=USER_IMAGES_BUCKET,
                    Key=key,
                    Body=r.content,
                    ContentType=f"image/{ext.lstrip('.').lower()}",
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
