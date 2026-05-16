# panneau-modal

Modal-hosted ComfyUI for the Qwen virtual try-on workflow. Designed as the primary inference backend with Fal as a peer and RunPod as fallback.

## Architecture

```
Browser → Cloud Run → Modal (this app) → R2 (panneau-user-images)
                          ↑
                   panneau-models (modal.Volume, synced from R2)
```

- **Inputs**: Cloud Run uploads user images to `panneau-user-images/<tenant>/inputs/<uuid>.png`, then calls Modal with the R2 keys.
- **Inference**: Modal downloads the referenced inputs, runs the ComfyUI workflow, uploads the outputs to `panneau-user-images/<tenant>/outputs/<uuid>.png`.
- **Output**: response contains the R2 keys of the generated images. Cloud Run resolves them to URLs for the browser.

No base64 in flight, no per-request volume writes for model state, no shared bucket mount.

## Prerequisites

- Modal account in the `panneau` workspace.
- Modal secrets:
  - `panneau-r2-models` — read-only on `panneau-models`
  - `panneau-r2-user-images` — read+write on `panneau-user-images`

  Both expose `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION`.

## One-time setup

Seed the `panneau-models` Modal Volume from the R2 bucket:

```bash
modal run scripts/sync_r2_to_volume.py
```

Re-run whenever model files change in R2.

## Deploy

```bash
modal deploy modal_app.py
```

Modal prints the endpoint URL. Endpoint requires proxy auth — generate a token in the Modal dashboard (App → Web endpoint → Auth) and store the `Modal-Key` / `Modal-Secret` pair in GCP Secret Manager for Cloud Run to read.

## API

`POST /predict` (proxy-auth-protected)

```json
{
  "tenant_id": "5ffa0e74-c6c6-4929-a65c-52572595677c",
  "workflow": { ... ComfyUI API JSON ... },
  "images": {
    "woman-stairs.png": "5ffa0e74-c6c6-4929-a65c-52572595677c/inputs/<file-uuid>.png",
    "beige_outfit.jpg": "5ffa0e74-c6c6-4929-a65c-52572595677c/inputs/<file-uuid>.jpg"
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

Validation:
- `tenant_id` must be a well-formed UUID.
- Every `images[*]` value must start with `<tenant_id>/inputs/` (no `..`, no nested paths).
- Local filenames (`images` keys) must not contain `/` or `..`.

## Local testing

```bash
modal serve modal_app.py
```

Hits an ephemeral copy of the endpoint, charged at the same rates as deployed. Headers for proxy auth still apply.

## Concurrency / cost knobs

In `modal_app.py`, on the `@app.cls(...)` decorator:

- `gpu=["A100-80GB", "H100"]` — prefer A100, fall back to H100.
- `min_containers=0` — scale to zero when idle.
- `max_containers=5` — cost cap; raise once traffic justifies it.
- `scaledown_window=60` — short for dev; bump to 300 when there are real users.

## Repo layout

```
modal_app.py                    # the Modal app
comfy/
  extra_model_paths.yaml        # paths into /models (volume mount)
  install_nodes.sh              # reference; image installs via run_commands
  workflows/qwen_tryon.json     # canonical workflow JSON
scripts/
  sync_r2_to_volume.py          # R2 → modal.Volume seed/refresh
pyproject.toml
```
