"""Quick connection test for Clicky's Gemini + Ollama models.
Run from the Clicky folder:  python tools\\check_models.py
"""
import base64, io, json, os, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip()

# tiny test image so vision paths are exercised
try:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (240, 60), "white"); ImageDraw.Draw(im).text((10, 20), "2 + 3 = ?", fill="black")
    buf = io.BytesIO(); im.save(buf, "JPEG"); IMG = base64.b64encode(buf.getvalue()).decode()
except Exception:
    IMG = None

print("=== Gemini ===")
key = env.get("GOOGLE_API_KEY", "")
if not key:
    print("  no GOOGLE_API_KEY in .env")
else:
    try:
        cache = json.loads((Path(os.environ.get("LOCALAPPDATA", "")) / "Clicky" / "models_gemini.json").read_text())
        cached = [m["id"] for m in cache.get("models", []) if m["id"].startswith("gemini")]
    except Exception:
        cached = []
    for m in dict.fromkeys(["gemini-3.8-flash", "gemini-flash-latest", "gemini-flash-lite-latest", *[c for c in sys.argv[1:] if not c.startswith("--")]]):
        parts = [{"text": "What is the answer in the image? One word."}]
        if IMG: parts.insert(0, {"inline_data": {"mime_type": "image/jpeg", "data": IMG}})
        t = time.time()
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                           headers={"x-goog-api-key": key}, json={"contents": [{"parts": parts}]}, timeout=90)
            if r.status_code == 200:
                txt = r.json()["candidates"][0]["content"]["parts"][0].get("text", "").strip()
                print(f"  OK   {m:32s} {time.time()-t:4.1f}s  -> {txt[:40]!r}")
            else:
                try: msg = r.json()["error"]["message"]
                except Exception: msg = r.text[:150]
                print(f"  FAIL {m:32s} HTTP {r.status_code}: {msg[:150]}")
        except Exception as e:
            print(f"  FAIL {m:32s} {type(e).__name__}: {e}")

print("=== Ollama ===")
host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
try:
    tags = httpx.get(f"{host}/api/tags", timeout=5).json()["models"]
except Exception as e:
    print(f"  server not reachable at {host}: {e}"); tags = []
def ask(name):
    body = {"model": name, "prompt": "What is the answer in the image? One word.",
            "images": [IMG] if IMG else [], "stream": False, "keep_alive": "10m"}
    if "qwen3" in name.lower():
        body["think"] = False
    return httpx.post(f"{host}/api/generate", timeout=300, json=body).json()

for t in tags:
    name, caps = t["name"], t.get("capabilities", [])
    if "vision" not in caps:
        print(f"  skip {name} (no vision)"); continue
    try:
        t0 = time.time(); ask(name); load = time.time() - t0          # first call loads model
        t0 = time.time(); r = ask(name); warm = time.time() - t0       # real speed
        gpu = ""
        for m in httpx.get(f"{host}/api/ps", timeout=5).json().get("models", []):
            if m["name"] == name and m.get("size"):
                gpu = f"{100*m.get('size_vram',0)/m['size']:.0f}% GPU"
        print(f"  OK   {name:20s} load {load:5.1f}s | warm {warm:5.1f}s | {gpu:8s} -> {r.get('response','').strip()[:30]!r}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
try:
    ps = httpx.get(f"{host}/api/ps", timeout=5).json().get("models", [])
    for m in ps:
        tot, vram = m.get("size", 0), m.get("size_vram", 0)
        pct = 100 * vram / tot if tot else 0
        print(f"  loaded {m['name']}: {pct:.0f}% on GPU ({vram/1e9:.1f} of {tot/1e9:.1f} GB)")
except Exception:
    pass

print("=== Antigravity (Gemini on Pro plan) ===")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
try:
    import asyncio
    from ai.agy_provider import AgyProvider, find_agy
    print("  agy:", find_agy())
    prov = AgyProvider()
    async def _agy():
        t0 = time.time(); out = ""
        async for c in prov.stream_response(
                "What is the answer in the screenshot? One word.", [IMG] if IMG else [], [],
                "You answer briefly.", "default"):
            out += c
        return out, time.time() - t0
    out, dt = asyncio.run(_agy())
    print(f"  {'OK  ' if out.strip() else 'EMPTY'} agy {dt:5.1f}s -> {out.strip()[:80]!r}")
    if not out.strip() or "--raw" in sys.argv:
        print("  raw agy events (paste these to Claude):")
        for ln in prov.last_events:
            print("   ", ln.strip()[:300])
except Exception as e:
    print(f"  FAIL agy: {e}")
