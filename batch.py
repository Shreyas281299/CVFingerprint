"""Prepare, render, and verify a portable, source-grouped main-study batch."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

MODEL = "timbrooks/instruct-pix2pix"
REVISION = "a8ebe0606a67d40e4ad5c26a71a25bdcdc6368ac"
CONFIG = {
    "id": "instruct-pix2pix-v1", "model": MODEL, "revision": REVISION,
    "max_side": 512, "steps": 20, "guidance": 7.5, "image_guidance": 1.5,
    "precision": "float16", "scheduler": "EulerAncestralDiscreteScheduler",
}
PROMPTS = {
    "paint": "Change the car's paint to {color}. Keep the wheels and background unchanged.",
    "background": "Replace the background with a quiet country road. Keep the car unchanged.",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inventory, batch):
    sources = read_csv(inventory)
    required = ("source_id", "group_id", "path", "creator", "source_url", "license", "target_color")
    if not sources or any(not row.get(key, "").strip() for row in sources for key in required):
        raise ValueError("Inventory requires nonempty columns " + ", ".join(required))
    ids = [row["source_id"] for row in sources]
    if len(set(ids)) != len(ids) or any(
        not name.isascii() or not name.replace("-", "").replace("_", "").isalnum()
        for name in ids
    ):
        raise ValueError("Source IDs must be unique ASCII letters, numbers, hyphens or underscores.")
    groups = sorted({row["group_id"] for row in sources})
    if len(groups) < 5:
        raise ValueError("Supply at least five independent groups.")
    np.random.default_rng(42).shuffle(groups)
    held_out = max(1, round(len(groups) * 0.2))
    splits = {group: "test" if i < held_out else "validation" if i < 2 * held_out
              else "train" for i, group in enumerate(groups)}

    batch.mkdir(parents=True)
    (batch / "inputs").mkdir()
    inventory_rows = []
    jobs = []
    used_hashes = set()
    for i, row in enumerate(sorted(sources, key=lambda row: row["source_id"])):
        path = inventory.parent / row["path"]
        source_hash = digest(path)
        if source_hash in used_hashes:
            raise ValueError(f"Duplicate source bytes at {path}.")
        used_hashes.add(source_hash)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((CONFIG["max_side"], CONFIG["max_side"]), Image.Resampling.LANCZOS)
        width, height = image.size
        if min(width, height) < 32:
            raise ValueError(f"Source is too small: {path}.")
        image = image.crop((0, 0, width // 8 * 8, height // 8 * 8))
        source_id = row["source_id"]
        prepared = f"inputs/{source_id}.png"
        image.save(batch / prepared)
        inventory_rows.append({
            **row, "split": splits[row["group_id"]], "sha256": source_hash,
            "prepared": prepared, "prepared_sha256": digest(batch / prepared),
        })
        for j, (category, prompt) in enumerate(PROMPTS.items()):
            jobs.append({
                "job_id": f"{source_id}-{category}", "source_id": source_id, "input": prepared,
                "category": category, "prompt": prompt.format(color=row["target_color"]),
                "seed": 42 + 2 * i + j, "config_id": CONFIG["id"],
            })
    write_csv(batch / "sources.csv", inventory_rows)
    write_csv(batch / "jobs.csv", jobs)
    (batch / "generation.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(inventory_rows)} sources in {len(groups)} groups and {len(jobs)} jobs.")


def read_batch(batch):
    sources = read_csv(batch / "sources.csv")
    jobs = read_csv(batch / "jobs.csv")
    config = json.loads((batch / "generation.json").read_text(encoding="utf-8"))
    if config != CONFIG:
        raise ValueError("Unexpected generation configuration.")
    source_map = {row["source_id"]: row for row in sources}
    expected = {(row["source_id"], category) for row in sources for category in PROMPTS}
    actual = {(job["source_id"], job["category"]) for job in jobs}
    if len(source_map) != len(sources) or actual != expected or len(actual) != len(jobs):
        raise ValueError("Missing or duplicate source jobs.")
    for row in sources:
        if digest(batch / row["prepared"]) != row["prepared_sha256"]:
            raise ValueError(f"Input checksum mismatch for {row['source_id']}.")
    for job in jobs:
        row = source_map[job["source_id"]]
        if (
            job["input"] != row["prepared"]
            or job["job_id"] != f"{job['source_id']}-{job['category']}"
            or job["config_id"] != CONFIG["id"]
        ):
            raise ValueError(f"Job manifest mismatch for {job['job_id']}.")
    return jobs


def read_attempts(batch, jobs):
    log = batch / "attempts.jsonl"
    attempts = {}
    valid_jobs = {job["job_id"]: job for job in jobs}
    if not log.exists():
        return attempts
    for line in log.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        job = valid_jobs.get(item["job_id"])
        if job is None or item["config_id"] != CONFIG["id"]:
            raise ValueError("Attempt references an unknown job or configuration.")
        history = attempts.setdefault(item["job_id"], [])
        attempt = len(history) + 1
        if attempt > 2 or item["attempt"] != attempt or item["seed"] != int(job["seed"]) + 100000 * (attempt - 1):
            raise ValueError(f"Unexpected attempt history for {item['job_id']}.")
        if history and history[-1]["status"] == "ok":
            raise ValueError(f"Duplicate successful job {item['job_id']}.")
        if item["status"] == "ok":
            expected = f"outputs/{job['job_id']}-{attempt}.png"
            if item["output"] != expected or digest(batch / expected) != item["sha256"]:
                raise ValueError(f"Output checksum or path mismatch for {item['job_id']}.")
            with Image.open(batch / job["input"]) as source, Image.open(batch / expected) as output:
                if output.format != "PNG" or output.size != source.size or not np.ptp(np.asarray(output)):
                    raise ValueError(f"Invalid successful output for {item['job_id']}.")
        elif item["status"] != "failed":
            raise ValueError(f"Unknown attempt status for {item['job_id']}.")
        history.append(item)
    return attempts


def render(batch):
    render_jobs(batch, read_batch(batch))


def render_jobs(batch, jobs):
    from importlib.metadata import version
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf-cache"))
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    attempts = read_attempts(batch, jobs)
    pending = [job for job in jobs if not attempts.get(job["job_id"])
               or (len(attempts[job["job_id"]]) < 2
                   and attempts[job["job_id"]][-1]["status"] == "failed")]
    if not pending:
        print("No jobs pending.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run the main-study render on the RTX laptop with CUDA.")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    (batch / "outputs").mkdir(exist_ok=True)
    versions = {name: version(name) for name in ("torch", "diffusers", "transformers", "Pillow")}
    with (batch / "attempts.jsonl").open("a", encoding="utf-8") as log:
        for job in pending:
            first = len(attempts.get(job["job_id"], [])) + 1
            for number in range(first, 3):
                seed = int(job["seed"]) + 100000 * (number - 1)
                output = f"outputs/{job['job_id']}-{number}.png"
                record = {"job_id": job["job_id"], "config_id": CONFIG["id"], "attempt": number,
                          "seed": seed, "versions": versions}
                start = time.monotonic()
                with Image.open(batch / job["input"]) as prepared:
                    image = prepared.convert("RGB")
                try:
                    with torch.inference_mode():
                        result = pipe(
                            prompt=job["prompt"], image=image, num_inference_steps=CONFIG["steps"],
                            guidance_scale=CONFIG["guidance"], image_guidance_scale=CONFIG["image_guidance"],
                            generator=torch.Generator("cuda").manual_seed(seed),
                        ).images[0]
                    if result.size != image.size or not np.ptp(np.asarray(result)):
                        raise ValueError("Blank or wrong-sized output.")
                except Exception as error:
                    record.update(status="failed", error=str(error))
                else:
                    target = batch / output
                    temp = target.with_suffix(".tmp")
                    result.save(temp, format="PNG")
                    temp.replace(target)
                    record.update(status="ok", output=output, sha256=digest(target),
                                  width=result.width, height=result.height)
                record["seconds"] = round(time.monotonic() - start, 3)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(job["job_id"], number, record["status"], flush=True)
                if record["status"] == "ok":
                    break


def verify(batch):
    jobs = read_batch(batch)
    attempts = read_attempts(batch, jobs)
    completed = {job_id for job_id, history in attempts.items() if history[-1]["status"] == "ok"}
    failed = set(attempts) - completed
    missing = {job["job_id"] for job in jobs} - set(attempts)
    expected = {item["output"] for history in attempts.values() for item in history
                if item["status"] == "ok"}
    actual = {path.relative_to(batch).as_posix() for path in (batch / "outputs").glob("*.png")}
    if actual != expected:
        raise ValueError("Unexpected or missing files in outputs.")
    print(f"Verified {len(completed)} successful jobs, {len(failed)} failed, {len(missing)} unattempted.")
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "render", "verify"))
    parser.add_argument("batch", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.inventory is None:
            parser.error("prepare requires --inventory")
        prepare(args.inventory, args.batch)
    elif args.action == "render":
        render(args.batch)
    else:
        verify(args.batch)
