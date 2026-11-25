import csv
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from PIL import Image, ImageOps

MODEL = "timbrooks/instruct-pix2pix"
REVISION = "a8ebe0606a67d40e4ad5c26a71a25bdcdc6368ac"
PROMPT = "Change the car's paint to bright purple. Keep the wheels and background unchanged."


def prepare():
    original = ROOT / "original"
    edited = ROOT / "editted"
    original.mkdir(exist_ok=True)
    edited.mkdir(exist_ok=True)
    manifest = ROOT / "pilot.csv"
    if manifest.exists():
        with manifest.open(newline="") as handle:
            return list(csv.DictReader(handle))

    candidates = sorted((ROOT / "archive").glob("*.jpg"))
    random.Random(42).shuffle(candidates)
    rows = []
    makes = set()
    for path in candidates:
        make = path.name.split("_", 1)[0]
        if make in makes:
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if min(image.size) < 160:
                    continue
        except OSError:
            continue
        makes.add(make)
        shutil.copyfile(path, original / path.name)
        rows.append({
            "original": f"original/{path.name}",
            "edited": f"editted/{path.stem}.png",
            "prompt": PROMPT,
            "seed": 42 + len(rows),
            "model": MODEL,
            "revision": REVISION,
        })
        if len(rows) == 20:
            break
    if len(rows) != 20:
        raise RuntimeError("Could not select 20 decodable photos from different makes.")
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render(rows):
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"Loading {MODEL} on {device}", flush=True)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing("max") if device == "mps" else None
    for index, row in enumerate(rows, 1):
        target = ROOT / row["edited"]
        if target.exists():
            with Image.open(target) as existing:
                existing.load()
            continue
        with Image.open(ROOT / row["original"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((256, 256) if device == "mps" else (512, 512), Image.Resampling.LANCZOS)
        width, height = image.size
        image = image.crop((0, 0, width // 8 * 8, height // 8 * 8))
        with torch.inference_mode():
            result = pipe(
                prompt=row["prompt"], image=image, num_inference_steps=12 if device == "mps" else 20,
                guidance_scale=7.5, image_guidance_scale=1.5,
                generator=torch.Generator("cpu").manual_seed(int(row["seed"])),
            ).images[0]
        temporary = target.with_suffix(".tmp")
        result.save(temporary, format="PNG")
        temporary.replace(target)
        print(f"Saved {index}/{len(rows)}: {target.name}", flush=True)


if __name__ == "__main__":
    import sys

    selected = prepare()
    print(f"Selected {len(selected)} original photos.", flush=True)
    if "--prepare-only" not in sys.argv:
        render(selected)
