"""Score a labeled image manifest using a frozen detector."""
import argparse
import csv
import json
from pathlib import Path

from detector import score_image
from extract_features import ROOT, RUN_DATA, image_hash, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=RUN_DATA / "archive_test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "results/pilot/detector.json")
    parser.add_argument("--output", type=Path, default=RUN_DATA / "archive_test_predictions.csv")
    args = parser.parse_args()
    if args.output.resolve() in {args.manifest.resolve(), args.model.resolve()}:
        parser.error("--output must differ from the input manifest and model")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        entries = list(csv.DictReader(stream))
    if not entries:
        raise ValueError("Manifest must include at least one labeled image.")
    if len({row["image_id"] for row in entries}) != len(entries):
        raise ValueError("Each image ID must be unique.")

    seen_hashes = set()
    results = []
    tn = fp = fn = tp = 0
    training_groups = set(model.get("splits", {}).get("train", []))
    training_groups.update(model.get("splits", {}).get("validation", []))
    for row in entries:
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Label must be 0 or 1 for {row['image_id']}.")
        if row.get("source_id") in training_groups or row.get("group_id") in training_groups:
            raise ValueError(f"Development group in test manifest: {row['image_id']}.")
        path = args.manifest.parent / row["path"]
        digest = image_hash(path)
        if row.get("sha256") and digest != row["sha256"]:
            raise ValueError(f"Checksum mismatch for {row['image_id']}.")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate image bytes for {row['image_id']}.")
        seen_hashes.add(digest)
        result = score_image(path, model)
        label = int(row["label"])
        prediction = model["labels"].index(result["label"])
        tn += label == 0 and prediction == 0
        fp += label == 0 and prediction == 1
        fn += label == 1 and prediction == 0
        tp += label == 1 and prediction == 1
        results.append((row["image_id"], row["path"], label, result["edit_score"], prediction, result["label"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, ("image_id", "path", "label", "edit_score", "prediction", "predicted_label"), results)
    print(f"Scored {len(results)} images at threshold {model['threshold']:.3f}.")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    if tn + fp:
        print(f"False-positive rate: {fp / (tn + fp):.1%} ({fp}/{tn + fp})")
    if tp + fn:
        print(f"Edit recall: {tp / (tp + fn):.1%} ({tp}/{tp + fn})")
    if (tn + fp) and (tp + fn):
        print(f"Balanced accuracy: {0.5 * (tn / (tn + fp) + tp / (tp + fn)):.1%}")
    else:
        print("Balanced accuracy requires both classes.")
    print(f"Saved predictions to {args.output}.")


if __name__ == "__main__":
    main()
