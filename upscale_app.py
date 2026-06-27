# upscale_app.py — Panneau SeedVR2 (UPSCALE) — Modal ComfyUI app
# ------------------------------------------------------------------
# Aynı "panneau-models" volume'unu mount eder (SEEDVR2 modelleri oradan, SADECE okuma).
# panneau-comfy'ye DOKUNMAZ — tamamen ayri, yeni bir app (video_app.py ile ayni desen).
#
# TEST  (interaktif UI):  MODAL_PROFILE=panneau modal serve upscale_app.py
#                         -> verilen URL'de ComfyUI acilir, SeedVR2 upscale workflow'unu calistir
# HEADLESS (tek gorsel):  MODAL_PROFILE=panneau modal run upscale_app.py --src ~/Desktop/foto.png
#                         -> 1440px (canonical "2k") upscale -> ~/Desktop/foto_upscaled.png
# DEPLOY (kalici):        MODAL_PROFILE=panneau modal deploy upscale_app.py
#
# Asama 3 (sonra): R2-bagli predict endpoint -> Edit sayfasi "Upscale" butonu.

import subprocess
import modal

VOLUME = "panneau-models"
MODELS_DIR = "/root/comfy/ComfyUI/models"      # volume buraya mount edilir (SEEDVR2/ alt dizini icinde)
CUSTOM = "/root/comfy/ComfyUI/custom_nodes"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --nvidia")       # ComfyUI -> /root/comfy/ComfyUI
    # --- SeedVR2 upscale node'lari ---
    .run_commands(
        f"git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler {CUSTOM}/ComfyUI-SeedVR2_VideoUpscaler",
        f"git clone https://github.com/kijai/ComfyUI-KJNodes {CUSTOM}/ComfyUI-KJNodes",
    )
    # --- node bagimliliklari ---
    .run_commands(
        f"pip install -r {CUSTOM}/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt || true",
        f"pip install -r {CUSTOM}/ComfyUI-KJNodes/requirements.txt || true",
    )
    # comfy install models/ klasorunu olusturuyor; Modal volume'u SADECE bos path'e
    # mount eder -> mount noktasini bosalt ki panneau-models buraya mount olabilsin
    .run_commands(f"rm -rf {MODELS_DIR}")
    # predict endpoint icin: boto3 (R2) + fastapi. EN SONA koyuldu ki yukaridaki
    # agir layer'lar (comfy install + node clone'lar) cache'den dusmesin.
    .pip_install("boto3", "fastapi[standard]")
    # FaceDetailer (yuz repaint) icin: Impact Pack + Subpack (UltralyticsDetectorProvider).
    # EN SONA -> yukaridaki agir layer'lar cache'de kalir.
    .run_commands(
        f"git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack {CUSTOM}/ComfyUI-Impact-Pack",
        f"git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack {CUSTOM}/ComfyUI-Impact-Subpack",
    )
    .run_commands(
        f"pip install -r {CUSTOM}/ComfyUI-Impact-Pack/requirements.txt || true",
        f"pip install -r {CUSTOM}/ComfyUI-Impact-Subpack/requirements.txt || true",
    )
)

app = modal.App("panneau-comfy-upscale", image=image)
vol = modal.Volume.from_name(VOLUME)

@app.function(
    gpu="A100-40GB",                 # SeedVR2 3B; OOM olursa A100-80GB yap
    volumes={MODELS_DIR: vol},
    timeout=1800,
    scaledown_window=120,            # seyrek -> warm tutmuyoruz
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(8000, startup_timeout=300)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)


# ==================================================================
# HEADLESS UPSCALE (Asama 2) — proxy YOK, localhost'tan surulur
#   MODAL_PROFILE=panneau modal run upscale_app.py --src ~/Desktop/foto.png
#   -> SeedVR2 ile 1440px (uzun kenar, canonical "2k"; 4k icin 2160) upscale -> bytes geri doner
# video_app.py'deki gen_single ile BIREBIR ayni iskelet.
# Kaynak: workflow 3/seedvr2_upscale (canonical) + workflows 5/final-for-ilker/tryon_upscale_2k|4k
#   2k -> resolution 1440 (default) | 4k -> resolution 2160
# ==================================================================
PORT = 8188

def _graph(img, resolution, seed):
    # SeedVR2 graph — canonical workflow 3/seedvr2_upscale + workflows 5/final-for-ilker
    # ile BIREBIR ayni node yapisi. device="default" (canonical); Modal CUDA'yi kendi secer.
    # Modeller volume'de SEEDVR2/ altinda; node bare filename ile bulur.
    return {
      "1": {"class_type": "LoadImage", "inputs": {"image": img}},
      "2": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
                "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                "device": "cuda:0", "attention_mode": "sdpa"}},
      "3": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
                "model": "ema_vae_fp16.safetensors",
                "device": "cuda:0", "encode_tiled": False, "decode_tiled": False}},
      "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
                "image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0],
                "resolution": resolution, "batch_size": 1,
                "color_correction": "lab", "seed": seed,
                "max_resolution": 0, "uniform_batch_size": False}},
      "5": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": "panneau_upscale", "images": ["4", 0]}},
    }

@app.function(gpu="A100-40GB", volumes={MODELS_DIR: vol}, timeout=1800)
def gen_upscale(image_bytes: bytes, resolution: int = 1440, seed: int = 42):
    import subprocess, os, json, time, urllib.request
    def post(path, obj=None):
        data = json.dumps(obj).encode() if obj is not None else None
        h = {"Content-Type": "application/json"} if data else {}
        return json.loads(urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers=h), timeout=120).read())
    # 1) comfy'yi localhost'ta baslat + hazir bekle
    subprocess.Popen(f"comfy launch -- --listen 127.0.0.1 --port {PORT}", shell=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/object_info", timeout=10).read(); break
        except Exception:
            time.sleep(3)
    else:
        raise RuntimeError("local comfy ayaga kalkmadi")
    print(f"[+] comfy hazir ({int(time.time()-t0)}s)")
    # 2) kaynak gorseli input'a yaz
    inp = "/root/comfy/ComfyUI/input"; os.makedirs(inp, exist_ok=True)
    open(os.path.join(inp, "src.png"), "wb").write(image_bytes)
    outdir = "/root/comfy/ComfyUI/output"
    # 3) upscale
    pid = post("/prompt", {"prompt": _graph("src.png", resolution, seed), "client_id": "hl"})["prompt_id"]
    print(f"[>] kuyrukta {pid} (resolution={resolution})")
    s0 = time.time()
    while time.time() - s0 < 1500:
        hi = post(f"/history/{pid}")
        if pid in hi and hi[pid].get("outputs"):
            imgs = hi[pid]["outputs"].get("5", {}).get("images") or []
            if imgs:
                p = os.path.join(outdir, imgs[0]["subfolder"], imgs[0]["filename"])
                print(f"[+] bitti ({int(time.time()-s0)}s)")
                return open(p, "rb").read()
        # hata olduysa erken cik
        if pid in hi and hi[pid].get("status", {}).get("status_str") == "error":
            raise RuntimeError("workflow error: " + json.dumps(hi[pid]["status"].get("messages", []))[:1500])
        time.sleep(4)
    raise RuntimeError("timeout")

# ==================================================================
# PREDICT ENDPOINT (Asama 3) — ana panneau-comfy app ile AYNI proxy contract
#   POST  Headers: Modal-Key / Modal-Secret  (requires_proxy_auth)
#   Body: { tenant_id, workflow, images:{fn:{bucket,key}}, timeout }
#   ->    { outputs:["<tenant>/outputs/<uuid>.png"], prompt_id }
# R2'den indir -> ComfyUI'da calistir -> R2'ye yaz -> key dondur.
# Urun (comfy_generate modal path) bu endpoint'e gelir; SeedVR2 workflow'u
# admin'deki 'upscale' artifact'inden gelir (generic proxy, ana app gibi).
# ==================================================================
# Logical bucket adi -> gercek R2 bucket adi. Upscale kaynagi her zaman
# user-images (kaynak inputs/, cikti outputs/ ikisi de orada).
R2_BUCKETS = {"user-images": "panneau-user-images"}

@app.cls(
    gpu="A100-80GB",                  # SeedVR2 + Qwen FaceDetailer ayni container'da -> 80GB
    volumes={MODELS_DIR: vol},
    secrets=[modal.Secret.from_name("panneau-r2-user-images")],   # AWS_* (R2 creds + endpoint)
    timeout=1800,
    scaledown_window=120,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
class Worker:
    @modal.enter()
    def boot(self):
        # Container basina BIR kez ComfyUI'yi localhost'ta ayaga kaldir (warm).
        import subprocess, time, urllib.request
        subprocess.Popen(f"comfy launch -- --listen 127.0.0.1 --port {PORT}", shell=True)
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/object_info", timeout=10).read()
                print(f"[+] comfy hazir ({int(time.time()-t0)}s)")
                return
            except Exception:
                time.sleep(3)
        raise RuntimeError("comfy boot timeout")

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def predict(self, item: dict):
        import os, json, time, uuid, urllib.request, boto3
        from fastapi import HTTPException
        tenant = item.get("tenant_id") or "anon"
        workflow = item.get("workflow")
        images = item.get("images", {}) or {}
        timeout = int(item.get("timeout", 600))
        if not isinstance(workflow, dict):
            raise HTTPException(status_code=400, detail="workflow (object) required")

        s3 = boto3.client("s3")  # AWS_* env'lerden creds + endpoint
        inp = "/root/comfy/ComfyUI/input"; os.makedirs(inp, exist_ok=True)
        outdir = "/root/comfy/ComfyUI/output"

        # 1) referans gorselleri R2'den input'a indir
        for fn, ref in images.items():
            logical = (ref or {}).get("bucket", "user-images")
            bucket = R2_BUCKETS.get(logical)
            if not bucket:
                raise HTTPException(status_code=400, detail=f"unsupported bucket '{logical}'")
            try:
                s3.download_file(bucket, ref["key"], os.path.join(inp, fn))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"R2 download failed {fn}: {str(e)[:200]}")

        def post(path, obj=None):
            data = json.dumps(obj).encode() if obj is not None else None
            h = {"Content-Type": "application/json"} if data else {}
            try:
                return json.loads(urllib.request.urlopen(
                    urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers=h), timeout=120).read())
            except urllib.error.HTTPError as he:
                # ComfyUI /prompt 400 -> node_errors gövdede; logla + ileri tasi.
                body = he.read().decode("utf-8", "replace")[:1500]
                print(f"[!] comfy {path} {he.code}: {body}")
                raise HTTPException(status_code=502, detail=f"comfy {path} {he.code}: {body}")

        # 2) workflow'u kuyruga al
        pid = post("/prompt", {"prompt": workflow, "client_id": "proxy"})["prompt_id"]

        # 3) bitene kadar bekle, ciktiyi R2'ye yukle
        t0 = time.time()
        while time.time() - t0 < timeout:
            hi = post(f"/history/{pid}")
            if pid in hi and hi[pid].get("outputs"):
                outputs = []
                for node in hi[pid]["outputs"].values():
                    for im in node.get("images", []):
                        p = os.path.join(outdir, im.get("subfolder", ""), im["filename"])
                        ext = im["filename"].rsplit(".", 1)[-1].lower() if "." in im["filename"] else "png"
                        okey = f"{tenant}/outputs/{uuid.uuid4()}.{ext}"
                        # ContentType set et — yoksa R2 octet-stream kaydeder, run-outputs
                        # onu aynen serve eder, indirilen dosya .octet-stream olur.
                        ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                                 "webp": "image/webp"}.get(ext, "image/png")
                        s3.upload_file(p, R2_BUCKETS["user-images"], okey,
                                       ExtraArgs={"ContentType": ctype})
                        outputs.append(okey)
                return {"outputs": outputs, "prompt_id": pid}
            if pid in hi and hi[pid].get("status", {}).get("status_str") == "error":
                raise HTTPException(status_code=500,
                                    detail="workflow error: " + json.dumps(hi[pid]["status"].get("messages", []))[:1200])
            time.sleep(2)
        raise HTTPException(status_code=504, detail="timeout")


@app.local_entrypoint()
def main(src: str = "/Users/zehraceviker/Desktop/workflows 5/untitled folder/AP0024_blake_expose.png",
         resolution: int = 1440):
    import os
    data = open(os.path.expanduser(src), "rb").read()
    out = gen_upscale.remote(data, resolution)
    base, ext = os.path.splitext(os.path.expanduser(src))
    dst = f"{base}_upscaled.png"
    open(dst, "wb").write(out)
    print(f"KAYDEDILDI: {dst}  ({len(out)//1024} KB)")

