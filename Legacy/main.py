import importlib.util
import subprocess
import sys
from io import BytesIO
from pathlib import Path

# Install missing dependencies using this Python environment.
packages = {
    "diffusers": "diffusers",
    "transformers": "transformers<5",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "PIL": "pillow",
    "requests": "requests",
}

missing = [
    package
    for module, package in packages.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing]
    )

import requests
import torch
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. Use your GPU-enabled PyTorch environment."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path(__file__).resolve().parent / "pix2pix-results"
    output_dir.mkdir(exist_ok=True)

    # Download a sample image.
    url = (
        "https://raw.githubusercontent.com/"
        "timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(
        image, (512, 512), method=Image.Resampling.LANCZOS
    )

    # To use your own image instead:
    image = ImageOps.exif_transpose(
        Image.open("car.png")
    ).convert("RGB")
    image = ImageOps.fit(image, (512, 512))

    image.save(output_dir / "input.png")

    print("Loading model; the first run downloads several GB...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    # Reduce GPU memory usage by moving components between RAM and GPU.
    pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()
    
    pipe.enable_model_cpu_offload()

    prompt = "Add falling snow and snow on the road. Keep the car’s color, shape, and position unchanged."  # Change this instruction.

    with torch.inference_mode():
        edited = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=30,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            generator=torch.Generator(device="cpu").manual_seed(42),
        ).images[0]

    output_path = output_dir / "edited.png"
    edited.save(output_path)
    print(f"Saved edited image: {output_path}")


if __name__ == "__main__":
    main()