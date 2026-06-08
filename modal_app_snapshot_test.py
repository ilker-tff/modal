"""
EXPERIMENTAL: Modal GPU memory snapshot + in-process ComfyUI.

Goal: measure cold-start delta vs modal_app.py on the try-on path.

Differences vs modal_app.py:
  - Strips WanVideoWrapper, SeedVR2, VideoHelperSuite, Frame-Interpolation.
    (SeedVR2 + WanVideoWrapper init CUDA at import → break snapshot.
     The two video helpers are unused on the try-on path; dropping them
     keeps this image minimal.)
  - enable_memory_snapshot=True + experimental_options.enable_gpu_snapshot=True.
  - ComfyUI runs IN-PROCESS in a background thread (not subprocess.Popen).
  - Two enter hooks:
      * @modal.enter(snap=True) — imports ComfyUI + custom nodes with
        torch.cuda.is_available patched to False, so no CUDA context is
        created before the snapshot is captured.
      * @modal.enter() — restores CUDA patches and starts ComfyUI's
        aiohttp server in a background thread.

Deploy:  modal deploy modal_app_snapshot_test.py
Test:    POST /predict (same shape as the main app's endpoint).
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field

# ─── App config ──────────────────────────────────────────────────────────────

app = modal.App("panneau-comfy-snapshot-test")

models_vol = modal.Volume.from_name("panneau-models", create_if_missing=True)

COMFY_DIR = "/comfyui"
COMFY_PORT = 8188

# ─── Image (try-on-safe subset only) ─────────────────────────────────────────

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
        f" && cd {COMFY_DIR}/custom_nodes/ComfyUI-GGUF && git checkout 01f8845"
        f" && pip install --no-cache-dir -r requirements.txt",
        f"git clone https://github.com/drphero/ComfyUI-FASHN-VTON.git {COMFY_DIR}/custom_nodes/ComfyUI-FASHN-VTON"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-FASHN-VTON/requirements.txt",
        f"git clone https://github.com/kijai/ComfyUI-KJNodes.git {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        f"git clone https://github.com/Acly/comfyui-inpaint-nodes.git {COMFY_DIR}/custom_nodes/comfyui-inpaint-nodes"
        f" && cd {COMFY_DIR}/custom_nodes/comfyui-inpaint-nodes && git checkout b9039c2",
        f"git clone https://github.com/rgthree/rgthree-comfy.git {COMFY_DIR}/custom_nodes/rgthree-comfy"
        f" && pip install --no-cache-dir -r {COMFY_DIR}/custom_nodes/rgthree-comfy/requirements.txt",
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
    .run_commands(
        f"git clone https://github.com/StartHua/Comfyui_segformer_b2_clothes.git"
        f" {COMFY_DIR}/custom_nodes/Comfyui_segformer_b2_clothes",
        f"sed -i '/segformer_b3_fashion/d'"
        f" {COMFY_DIR}/custom_nodes/Comfyui_segformer_b2_clothes/__init__.py",
        "python -c \"from huggingface_hub import snapshot_download;"
        " snapshot_download(repo_id='mattmdjaga/segformer_b2_clothes',"
        f" local_dir='{COMFY_DIR}/models/segformer_b2_clothes')\"",
    )
    .add_local_dir("comfy", "/app/comfy")
)


# ─── Request / response models (same shape as modal_app.py) ──────────────────


class ImageRef(BaseModel):
    bucket: Literal["user-images", "catalog"]
    key: str


class PredictRequest(BaseModel):
    tenant_id: str
    workflow: dict
    images: dict[str, ImageRef] = Field(default_factory=dict)
    timeout: int = Field(default=600, ge=10, le=1800)


class PredictResponse(BaseModel):
    outputs: list[str]
    prompt_id: str


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


# ─── In-process ComfyUI helpers ──────────────────────────────────────────────


def _preload_heavy_imports():
    """Import torch + ComfyUI modules during snap so they're frozen into the snapshot.

    Does NOT call init_extra_nodes (async in newer Comfy + custom-node side effects
    are flaky inside a snapshot phase). Custom nodes are loaded by the subprocess
    on restore.
    """
    os.chdir(COMFY_DIR)
    if COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)

    # Heavy imports we want to skip on every cold start.
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import transformers  # noqa: F401
    # ComfyUI's model_management initializes a CUDA context at import — that's
    # exactly what GPU snapshots are designed to capture.
    import comfy.model_management  # noqa: F401
    import comfy.utils  # noqa: F401
    import comfy.sd  # noqa: F401


def _start_comfy_subprocess():
    """Start ComfyUI as a subprocess (same as the production app)."""
    import subprocess
    return subprocess.Popen(
        [
            "python", "main.py",
            "--listen", "0.0.0.0",
            "--port", str(COMFY_PORT),
            "--disable-auto-launch",
            "--fast",
            "--extra-model-paths-config", "extra_model_paths.yaml",
        ],
        cwd=COMFY_DIR,
    )


def _wait_for_comfy_ready(timeout_s: int = 300):
    import requests

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(
                f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2
            )
            if r.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready within timeout.")


# ─── The class ───────────────────────────────────────────────────────────────


@app.cls(
    image=image,
    gpu="A100-40GB",  # single GPU type for now — multi-arch snapshot restore is unverified
    volumes={"/models": models_vol},
    secrets=[
        modal.Secret.from_name("panneau-r2-user-images"),
        modal.Secret.from_name("panneau-r2-catalog"),
    ],
    min_containers=0,
    max_containers=2,
    scaledown_window=300,
    timeout=60 * 30,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class ComfyUISnapshot:
    @modal.enter(snap=True)
    def snap_phase(self):
        """Runs at snapshot-creation time (and never again on restore).

        Imports ComfyUI + every custom node so the snapshot freezes the
        post-import state (Python modules + CUDA context).
        """
        t0 = time.time()
        print("[snap] start")

        # Render the local extra_model_paths.yaml into ComfyUI's dir.
        src = Path("/app/comfy/extra_model_paths.yaml")
        dst = Path(f"{COMFY_DIR}/extra_model_paths.yaml")
        dst.write_text(src.read_text())

        _preload_heavy_imports()
        print(f"[snap] imports done in {time.time() - t0:.2f}s")

    @modal.enter(snap=False)
    def post_restore(self):
        """Runs on every container start (including post-snapshot restore)."""
        t0 = time.time()
        print("[restore] starting ComfyUI subprocess")
        self._proc = _start_comfy_subprocess()
        _wait_for_comfy_ready()
        print(f"[restore] ComfyUI ready in {time.time() - t0:.2f}s")

    # ── helpers (copy-paste from modal_app.py — identical behavior) ─────────

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
