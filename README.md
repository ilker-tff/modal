# panneau-modal

Modal-hosted ComfyUI inference backends. Two separate apps so each carries only
the custom nodes it needs (faster cold starts):

| App | File | Modal app name | Workflow family |
|---|---|---|---|
| Image | `image_app.py` | `panneau-comfy` | Qwen image-edit try-on / composite (RTX-PRO-6000) |
| Video | `video_app.py` | `panneau-comfy-video` | WAN 2.2 image-to-video (A100-80GB) |

Both expose the **same** proxy-auth `/predict` contract and read/write the same
R2 buckets. Only the GPU and the installed custom nodes differ.

## Architecture

```
Browser → Cloud Run → Modal app (image | video) → R2 (panneau-user-images)
                          ↑
                   panneau-models (modal.Volume, shared, read-only)
```

- **Inputs**: Cloud Run uploads user images to `panneau-user-images/<tenant>/inputs/<uuid>.png`
  (or references catalog images in `panneau-catalog`), then calls Modal with `{bucket, key}` pointers.
- **Inference**: Modal downloads the referenced inputs, runs the caller-supplied
  ComfyUI workflow, uploads outputs to `panneau-user-images/<tenant>/outputs/<uuid>.<ext>`.
- **Output**: response returns the R2 keys. Cloud Run resolves them to URLs.

No base64 in flight, no bucket mounts, no per-request model writes.

## Custom node sets

- **image** (`image_app.py`): ComfyUI-KJNodes, ComfyMath, Comfyui_segformer_b2_clothes
  (+ comfyui-inpaint-nodes — pending confirmation it's used by any workflow).
- **video** (`video_app.py`): ComfyUI-KJNodes, rgthree-comfy, ComfyUI-VideoHelperSuite,
  ComfyUI-Image-Saver. WAN i2v / first-last-frame nodes are native comfy_extras.

## Prerequisites

- Modal account in the `panneau` workspace.
- Modal secrets (each exposes `AWS_ENDPOINT_URL` + access key/secret):
  - `panneau-r2-models` — read-only on `panneau-models`
  - `panneau-r2-user-images` — read+write on `panneau-user-images`
    (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`)
  - `panneau-r2-catalog` — read on `panneau-catalog`
    (`CATALOG_AWS_ACCESS_KEY_ID` / `CATALOG_AWS_SECRET_ACCESS_KEY`)

## One-time setup

Seed the shared `panneau-models` Volume from R2:

```bash
modal run scripts/sync_r2_to_volume.py
```

Re-run whenever model files change in R2.

## Deploy

```bash
modal deploy image_app.py      # panneau-comfy
modal deploy video_app.py      # panneau-comfy-video
```

Each prints its endpoint URL. Endpoints require proxy auth — generate a token in
the Modal dashboard (Settings → Proxy Auth Tokens) and store the
`Modal-Key` / `Modal-Secret` pair in GCP Secret Manager for Cloud Run.

## API

`POST /predict` (proxy-auth-protected) — same shape for both apps.

```json
{
  "tenant_id": "5ffa0e74-c6c6-4929-a65c-52572595677c",
  "workflow": { "...": "ComfyUI API-format JSON" },
  "images": {
    "person.png": { "bucket": "user-images", "key": "<tenant>/inputs/<uuid>.png" },
    "garment.png": { "bucket": "catalog", "key": "category/Apparel/AP0001/aaliyah.png" }
  },
  "timeout": 600
}
```

Response:

```json
{
  "outputs": ["5ffa0e74-c6c6-4929-a65c-52572595677c/outputs/<new-uuid>.png"],
  "prompt_id": "..."
}
```

Rules:
- `tenant_id` must be a well-formed UUID (outputs are written under it).
- `images` maps a workflow-local filename → `{bucket, key}`. `bucket` is
  `"user-images"` or `"catalog"`. `key` is any object key in that bucket
  (no `..`, no leading `/`). Local filenames must not contain `/` or `..`.
- `timeout`: seconds, clamped `[10, 1800]`.

## Local testing

```bash
modal serve image_app.py    # or video_app.py
```

Ephemeral copy of the endpoint, billed at the same rate. Proxy auth still applies.

## Concurrency / cost knobs

On each app's `@app.cls(...)` decorator: `gpu`, `min_containers`,
`max_containers`, `scaledown_window`. Image runs hot/cheap (RTX-PRO-6000,
scaledown 10s); video runs sparse/heavy (A100-80GB, scaledown 120s,
max_containers 2).

## Repo layout

```
image_app.py                    # image try-on / composite app (panneau-comfy)
video_app.py                    # WAN 2.2 i2v app (panneau-comfy-video)
comfy/
  extra_model_paths.yaml        # paths into /models (volume mount) — shared
  install_nodes.sh              # reference
scripts/
  sync_r2_to_volume.py          # R2 → modal.Volume seed/refresh
  bench/                        # cold-start benchmarking
pyproject.toml
```
