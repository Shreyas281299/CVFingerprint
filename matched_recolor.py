"""Interactive matched-recolor development experiment. Run on a Windows RTX laptop."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from batch import CONFIG, PROMPTS, digest, read_attempts, render_jobs
from detector import score_image
from extract_features import prepare_original, write_csv

ROOT = Path(__file__).resolve().parent
COLORS = ("purple", "gold", "blue", "red", "green", "orange")
TARGET = 20


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def recolor(image, mask, color):
    hsv = np.array(image.convert("HSV"))
    hue = Image.new("RGB", (1, 1), ImageColor.getrgb(color)).convert("HSV").getpixel((0, 0))[0]
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 190
    # Preserve shading while making color visible on dark body panels.
    hsv[:, :, 2] = 30 + hsv[:, :, 2].astype(float) * 0.88
    tint = Image.frombytes("HSV", image.size, hsv.tobytes()).convert("RGB")
    return Image.composite(tint, image, mask)


def annotate(image, name, color):
    import cv2

    pixels = np.array(image)[:, :, ::-1].copy()
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    points = []
    title = f"{name[:28]} to {color} | click polygon, P add, R reset, Enter accept, S skip, Q pause"
    cv2.namedWindow(title)
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    cv2.setMouseCallback(title, click)
    try:
        while True:
            view = pixels.copy()
            view[mask > 0] = (view[mask > 0] * 0.5 + np.array([0, 100, 0])).astype(np.uint8)
            if points:
                cv2.polylines(view, [np.array(points)], False, (0, 255, 255), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(30) & 255
            if key == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("Paused. Rerun to resume.")
            if key == ord("s"):
                return None
            if key == ord("r"):
                mask.fill(0)
                points.clear()
            if key in (ord("p"), 10, 13):
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.array(points)], 255)
                    points.clear()
                if key in (10, 13) and mask.any() and not points:
                    return Image.fromarray(mask)
    finally:
        cv2.destroyAllWindows()


def archive_group(name):
    parts = re.split(r"_\d{4}_", Path(name).stem.casefold(), maxsplit=1)
    return parts[0] if len(parts) == 2 else Path(name).stem.casefold()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "masks", "conventional"):
        (folder / name).mkdir(exist_ok=True)
    state = folder / "selection.json"
    selected = json.loads(state.read_text()) if state.exists() else []
    processed = {r["file"] for r in selected}
    candidates = ROOT / "candidates"
    files = sorted(p for p in candidates.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    used_groups = {archive_group(r["file"]) for r in selected if r["accepted"]}
    used_hashes = {r["archive_sha256"] for r in selected}
    if not files and sum(r["accepted"] for r in selected) < TARGET:
        raise SystemExit(f"No candidate images found under {candidates}. Extract the candidate update beside this script.")
    for p in files:
        count = sum(r["accepted"] for r in selected)
        if count == TARGET:
            break
        if p.name in processed or archive_group(p.name) in used_groups or digest(p) in used_hashes:
            continue
        color = COLORS[count % len(COLORS)]
        image = prepare_original(p, CONFIG["max_side"])
        mask = annotate(image, p.stem, color)
        row = {"file": p.name, "accepted": mask is not None, "archive_sha256": digest(p)}
        if mask is not None:
            key = f"car{count+1:02d}"
            outputs = {"inputs": image, "masks": mask, "conventional": recolor(image, mask, color)}
            for directory, output in outputs.items():
                output.save(folder / directory / f"{key}.png")
            row.update(key=key, color=color, seed=7600+count, hashes={
                name: digest(folder / name / f"{key}.png") for name in outputs})
        used_hashes.add(row["archive_sha256"])
        if mask is not None:
            used_groups.add(archive_group(p.name))
        selected.append(row)
        save_json(state, selected)
    return [r for r in selected if r["accepted"]]


def run(folder):
    rows = prepare(folder)
    if len(rows) != TARGET:
        raise SystemExit(f"Selected {len(rows)}/20. Reviewed all eligible images under {ROOT / 'candidates'}. Add another batch there and rerun with the same --out folder.")
    jobs = []
    for r in rows:
        key = r["key"]
        for name, expected in r["hashes"].items():
            if digest(folder / name / f"{key}.png") != expected:
                raise ValueError(f"Changed {name} for {key}")
        jobs.append(dict(job_id=key, input=f"inputs/{key}.png", seed=r["seed"],
                         prompt=PROMPTS["paint"].format(color=r["color"]), config_id=CONFIG["id"]))
    save_json(folder / "jobs.json", jobs)
    save_json(folder / "generation.json", CONFIG)
    render_jobs(folder, jobs)
    return rows, jobs


def review_and_score(folder, rows, jobs):
    import cv2

    attempts = read_attempts(folder, jobs)
    review_path = folder / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    accepted = []
    for row, job in zip(rows, jobs, strict=True):
        key = row["key"]
        attempt = attempts[key][-1]
        if attempt["status"] != "ok":
            continue
        paths = [folder / job["input"], folder / "conventional" / f"{key}.png", folder / attempt["output"]]
        if key not in review:
            panes = [cv2.imread(str(path)) for path in paths]
            title = f"{key} {row['color']} | original - conventional - AI | A accept, X reject, Q pause"
            cv2.imshow(title, np.hstack(panes))
            while True:
                decision = cv2.waitKey(30) & 255
                if decision in (ord("a"), ord("x"), ord("q")):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    decision = ord("q")
                    break
            cv2.destroyAllWindows()
            if decision == ord("q"):
                raise SystemExit("Paused before scoring.")
            review[key] = "accepted" if decision == ord("a") else "rejected"
            save_json(review_path, review)
        if review[key] == "accepted":
            accepted.append((key, row["color"], paths))
    model_path = ROOT / "results/pilot/detector.json"
    model = json.loads(model_path.read_text())
    scored = [(key, color, *[score_image(p, model)["edit_score"] for p in paths])
              for key, color, paths in accepted]
    data = folder / "run-data"
    data.mkdir(exist_ok=True)
    write_csv(data / "scores.csv", ("source", "color", "original", "conventional", "ai"), scored)
    summary = {"attempted_sources": len(rows), "accepted_triplets": len(scored),
               "render_failures": len(rows)-len(review), "threshold": model["threshold"],
               "model_sha256": digest(model_path)}
    if scored:
        values = np.array([r[2:] for r in scored])
        rates = (values >= model["threshold"]).mean(axis=0)
        summary.update(original_fpr=rates[0], conventional_fpr=rates[1], ai_recall=rates[2],
                       ai_above_conventional=int(sum(values[:, 2] > values[:, 1])))
    save_json(data / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "matched-run")
    folder = parser.parse_args().out
    rows, jobs = run(folder)
    review_and_score(folder, rows, jobs)
