"""
Sync model files from R2 (panneau-models bucket) to Modal Volume.

One-time seed, and re-run whenever model files change.

Usage:
    modal run scripts/sync_r2_to_volume.py
"""

import modal

app = modal.App("panneau-sync-models")

vol = modal.Volume.from_name("panneau-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3==1.35.0", "tqdm==4.66.5")
)


@app.function(
    image=image,
    volumes={"/models": vol},
    secrets=[modal.Secret.from_name("panneau-r2-models")],
    timeout=60 * 60,
    cpu=4,
)
def sync():
    import os
    from pathlib import Path

    import boto3
    from botocore.config import Config

    bucket = "panneau-models"
    dest_root = Path("/models")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    paginator = s3.get_paginator("list_objects_v2")
    total_new = 0
    total_skip = 0
    total_bytes = 0

    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            dest = dest_root / key

            if dest.exists() and dest.stat().st_size == size:
                total_skip += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"↓ {key} ({size / 1e6:.1f} MB)")
            s3.download_file(bucket, key, str(dest))
            total_new += 1
            total_bytes += size

    vol.commit()
    print(
        f"Done. Downloaded {total_new} new objects "
        f"({total_bytes / 1e9:.2f} GB), skipped {total_skip} already-current."
    )
