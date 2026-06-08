#!/usr/bin/env python3
"""
Bench Modal ComfyUI cold-start latency.

For each trial:
  - mutate the workflow seed (defeats Comfy's prompt cache)
  - stop all running containers of the target app (forces cold start)
  - POST /predict, time end-to-end

Reads secrets from gcloud secret manager. Never prints secret values.

Usage:
    python3 scripts/bench/bench_cold_start.py \
        --app panneau-comfy-snapshot-test \
        --endpoint https://.../predict \
        --trials 2
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import boto3
import requests


def gcloud_secret(name: str) -> str:
    """Fetch a secret value from GCP Secret Manager. Never logged."""
    out = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={name}"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.rstrip("\n")


def get_app_id(app_name: str) -> str | None:
    """Look up Modal App ID for a deployed app by name. Returns None if not deployed."""
    out = subprocess.run(
        ["modal", "app", "list", "--json"],
        capture_output=True, text=True, check=True,
    )
    for app in json.loads(out.stdout):
        if app.get("Description") == app_name and app.get("State") == "deployed":
            return app["App ID"]
    return None


def stop_all_containers(app_name: str, wait_after: int = 15) -> int:
    """Stop every running container for the app. Returns count stopped."""
    app_id = get_app_id(app_name)
    if not app_id:
        print(f"  app {app_name!r} not deployed — nothing to stop")
        return 0
    out = subprocess.run(
        ["modal", "container", "list", "--app-id", app_id, "--json"],
        capture_output=True, text=True, check=True,
    )
    containers = json.loads(out.stdout)
    if not containers:
        print(f"  no warm containers for {app_name}")
        return 0
    count = 0
    for c in containers:
        cid = c.get("Container ID") or c.get("id")
        if not cid:
            continue
        print(f"  stopping container {cid}")
        r = subprocess.run(
            ["modal", "container", "stop", "-y", cid],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"    warning: stop returned {r.returncode}: {r.stderr.strip()}")
        count += 1
    print(f"  waiting {wait_after}s for containers to drain...")
    time.sleep(wait_after)
    return count


def upload(s3, bucket: str, local: Path, key: str) -> None:
    s3.upload_file(str(local), bucket, key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, help="Modal app name (e.g. panneau-comfy)")
    parser.add_argument("--endpoint", required=True, help="Full /predict URL")
    parser.add_argument(
        "--workflow",
        default=str(Path(__file__).parent / "tryon_workflow.json"),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Space-separated person:garment paths (e.g. person.png:garment.png). One per trial; cycles if fewer than trials.",
    )
    parser.add_argument("--person", default=None, help="(legacy) single person path; use --pairs instead")
    parser.add_argument("--garment", default=None, help="(legacy) single garment path; use --pairs instead")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--skip-stop",
        action="store_true",
        help="Skip container-stop step (use only for the first trial after fresh deploy)",
    )
    args = parser.parse_args()

    # Build the list of (person, garment) pairs for trials.
    if args.pairs:
        pairs = []
        for spec in args.pairs:
            p, g = spec.split(":")
            pairs.append((Path(p), Path(g)))
    elif args.person and args.garment:
        pairs = [(Path(args.person), Path(args.garment))]
    else:
        raise SystemExit("Provide --pairs person.png:garment.png ... or --person/--garment")

    for p, g in pairs:
        assert p.exists(), f"missing {p}"
        assert g.exists(), f"missing {g}"

    print(f"=== Bench: {args.app} ===")
    print("Fetching secrets from gcloud...")
    modal_key = gcloud_secret("panneau-modal-key")
    modal_secret = gcloud_secret("panneau-modal-secret")
    r2_endpoint = gcloud_secret("panneau-r2-user-images-endpoint")
    r2_bucket = gcloud_secret("panneau-r2-user-images-bucket")
    r2_key_id = gcloud_secret("panneau-r2-user-images-access-key-id")
    r2_sak = gcloud_secret("panneau-r2-user-images-secret-access-key")

    s3 = boto3.client(
        "s3",
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_key_id,
        aws_secret_access_key=r2_sak,
    )

    # Upload every pair once under a unique prefix.
    bench_id = uuid.uuid4().hex[:8]
    uploaded_pairs: list[tuple[str, str]] = []
    for i, (p, g) in enumerate(pairs):
        person_key = f"bench-{bench_id}/person_{i}{p.suffix}"
        garment_key = f"bench-{bench_id}/garment_{i}{g.suffix}"
        print(f"Uploading {p.name} → s3://{r2_bucket}/{person_key}")
        upload(s3, r2_bucket, p, person_key)
        print(f"Uploading {g.name} → s3://{r2_bucket}/{garment_key}")
        upload(s3, r2_bucket, g, garment_key)
        uploaded_pairs.append((person_key, garment_key))

    workflow_template = json.loads(Path(args.workflow).read_text())
    tenant_id = "00000000-0000-0000-0000-000000000001"
    headers = {"Modal-Key": modal_key, "Modal-Secret": modal_secret}

    timings: list[tuple[float, bool]] = []
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1}/{args.trials} ---")
        if args.skip_stop:
            print("  (skip-stop) leaving containers warm (no stop)")
        else:
            print(f"  forcing cold start by stopping containers of {args.app}")
            stop_all_containers(args.app)

        # Rotate through the uploaded pairs.
        person_key, garment_key = uploaded_pairs[trial % len(uploaded_pairs)]
        print(f"  using person={person_key}, garment={garment_key}")

        wf = json.loads(json.dumps(workflow_template))
        new_seed = random.randint(1, 2**31 - 1)
        wf["900"]["inputs"]["value"] = new_seed
        print(f"  seed = {new_seed}")

        payload = {
            "tenant_id": tenant_id,
            "workflow": wf,
            "images": {
                "image1.png": {"bucket": "user-images", "key": person_key},
                "image2.png": {"bucket": "user-images", "key": garment_key},
            },
            "timeout": args.timeout,
        }

        print(f"  POST {args.endpoint}")
        t0 = time.time()
        try:
            r = requests.post(
                args.endpoint, json=payload, headers=headers,
                timeout=args.timeout + 30,
            )
        except requests.RequestException as exc:
            elapsed = time.time() - t0
            print(f"  REQUEST ERROR after {elapsed:.2f}s: {exc}")
            timings.append((elapsed, False))
            continue

        elapsed = time.time() - t0
        ok = r.ok
        print(f"  status={r.status_code} elapsed={elapsed:.2f}s ok={ok}")
        if not ok:
            print(f"  body: {r.text[:1000]}")
        else:
            try:
                outputs = r.json().get("outputs", [])
                print(f"  outputs ({len(outputs)}): {outputs[:3]}{'...' if len(outputs) > 3 else ''}")
            except Exception:
                pass
        timings.append((elapsed, ok))

    print(f"\n=== Summary: {args.app} ===")
    for i, (t, ok) in enumerate(timings):
        flag = "OK" if ok else "FAIL"
        print(f"  trial {i+1}: {t:.2f}s [{flag}]")
    ok_times = sorted(t for t, ok in timings if ok)
    if ok_times:
        median = ok_times[len(ok_times) // 2]
        print(f"  median (ok only): {median:.2f}s  min={ok_times[0]:.2f}s  max={ok_times[-1]:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
