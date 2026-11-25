"""Run source-grouped pilot ablations and matched export challenges."""

import argparse
import csv
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import (
    FEATURE_NAMES, FEATURE_VERSION, ROOT, RUN_DATA, extract_features, image_hash, load_rgb,
    prepare_original, write_csv,
)

SEED = 42
C_VALUES = [0.01, 0.1, 1, 10]
CHALLENGES = ("jpeg85", "jpeg65", "resize75", "brightness", "sharpen")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_dataset(include_rejected):
    index = read_csv(RUN_DATA / "pilot_feature_index.csv")
    cache = read_csv(RUN_DATA / "pilot_features.csv")
    features = {row["image_id"]: row for row in cache}
    ids = {row["image_id"] for row in index}
    if len(ids) != len(index) or len(features) != len(cache) or ids != features.keys():
        raise ValueError("Feature and index tables must have matching unique image IDs.")
    if {row["feature_version"] for row in index} != {FEATURE_VERSION}:
        raise ValueError("Feature version mismatch. Run extract_features.py again.")
    for row in index:
        feature = features[row["image_id"]]
        if (feature["feature_version"] != FEATURE_VERSION
            or feature["sha256"] != row["sha256"]
            or image_hash(ROOT / row["path"]) != row["sha256"]):
            raise ValueError(f"Stale feature cache for {row['image_id']}.")
    if not include_rejected:
        review = read_csv(RUN_DATA / "pilot_review.csv")
        statuses = {row["source_id"]: row["status"] for row in review}
        if set(statuses) != {row["source_id"] for row in index} or len(statuses) != len(review):
            raise ValueError("Expected one visual review per source.")
        if set(statuses.values()) - {"accepted", "rejected"}:
            raise ValueError("Each visual review must be accepted or rejected.")
        index = [row for row in index if statuses[row["source_id"]] == "accepted"]
    labels = np.array([int(row["label"]) for row in index])
    groups = np.array([row["source_id"] for row in index])
    for group in np.unique(groups):
        if sorted(labels[groups == group]) != [0, 1]:
            raise ValueError(f"Expected one original and one edit for {group}.")
    matrix = np.array([
        [float(features[row["image_id"]][name]) for name in FEATURE_NAMES]
        for row in index
    ])
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite value in feature cache.")
    return index, matrix, labels, groups


def load_main_dataset(batch_path):
    from batch import read_attempts, read_batch

    jobs = read_batch(batch_path)
    attempts = read_attempts(batch_path, jobs)
    source_rows = read_csv(batch_path / "sources.csv")
    sources = {row["source_id"]: row for row in source_rows}
    reviews = read_csv(batch_path / "review.csv")
    review = {row["job_id"]: row["status"] for row in reviews}
    if len(review) != len(reviews) or set(review) != {job["job_id"] for job in jobs}:
        raise ValueError("review.csv must contain one decision for every render job.")
    if set(review.values()) - {"accepted", "rejected"}:
        raise ValueError("Valid review statuses are accepted and rejected.")

    index, features, splits = [], [], []
    group_splits = {}
    for source in source_rows:
        group = source["group_id"]
        split = source["split"]
        if split not in {"train", "validation", "test"} or group_splits.setdefault(group, split) != split:
            raise ValueError(f"Invalid or inconsistent split for group {group}.")
    for job in jobs:
        if review[job["job_id"]] != "accepted":
            continue
        history = attempts.get(job["job_id"], [])
        if not history or history[-1]["status"] != "ok":
            raise ValueError(f"Accepted job has no verified output: {job['job_id']}.")
        if job["category"] != "paint":
            continue
        source = sources[job["source_id"]]
        for label, relative_path in enumerate((source["prepared"], history[-1]["output"])):
            path = batch_path / relative_path
            index.append({
                "image_id": f"{source['source_id']}:{label}",
                "source_id": source["source_id"], "group_id": source["group_id"],
                "path": path.resolve().as_posix(), "label": str(label), "prepared": "yes",
            })
            features.append(extract_features(load_rgb(path)))
            splits.append(source["split"])
    groups = np.array([row["group_id"] for row in index])
    labels = np.array([int(row["label"]) for row in index])
    splits = np.array(splits)
    if any(len(set(groups[splits == split])) < minimum
           for split, minimum in (("train", 4), ("validation", 1), ("test", 1))):
        raise ValueError("Need at least four accepted training groups and one per other split.")
    return index, np.array(features), labels, groups, splits


def recode(image, quality):
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recoded:
            return recoded.convert("RGB")


def challenge_features(index):
    transformed = {name: {} for name in CHALLENGES}
    for i, row in enumerate(index):
        path = ROOT / row["path"]
        original = row["label"] == "0"
        image = prepare_original(path) if original and not row.get("prepared") else load_rgb(path)
        width, height = image.size
        images = {
            "jpeg85": recode(image, 85),
            "jpeg65": recode(image, 65),
            "resize75": image.resize(
                (round(width * 0.75), round(height * 0.75)), Image.Resampling.LANCZOS
            ),
        }
        if original:
            images["brightness"] = ImageEnhance.Brightness(image).enhance(1.15)
            images["sharpen"] = ImageEnhance.Sharpness(image).enhance(1.75)
        for name, derived in images.items():
            transformed[name][i] = extract_features(derived)
    return transformed


def score_predictions(rows):
    labels = [row[4] for row in rows]
    scores = [row[5] for row in rows]
    decisions = [row[6] for row in rows]
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    both_classes = len(set(labels)) == 2
    return [
        len(rows),
        balanced_accuracy_score(labels, decisions) if both_classes else "",
        roc_auc_score(labels, scores) if both_classes else "",
        precision_score(labels, decisions, zero_division=0) if both_classes else "",
        recall_score(labels, decisions, zero_division=0) if both_classes else "",
        f1_score(labels, decisions, zero_division=0) if both_classes else "",
        fp / (tn + fp), tn, fp, fn, tp,
    ]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_grouped(features, labels, groups):
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        make_model(), {"logisticregression__C": C_VALUES}, cv=inner,
        scoring="balanced_accuracy", error_score="raise",
    )
    search.fit(features, labels, groups=groups)
    return search.best_estimator_, search.best_params_["logisticregression__C"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-rejected", action="store_true", help="Run a QA sensitivity check.")
    parser.add_argument("--batch", type=Path, help="Evaluate a reviewed main-study batch.")
    args = parser.parse_args()
    if args.batch and args.include_rejected:
        parser.error("--include-rejected applies to the pilot only")
    if args.batch:
        index, matrix, labels, groups, splits = load_main_dataset(args.batch)
        train, validation, test = (
            np.flatnonzero(splits == split) for split in ("train", "validation", "test")
        )
        partitions = [(train, test)]
    else:
        index, matrix, labels, groups = load_dataset(args.include_rejected)
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        partitions = outer.split(matrix, labels, groups)
    challenges = challenge_features(index)
    families = {
        "dummy": np.arange(matrix.shape[1]),
        "rgb": [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("rgb_")],
        "residual": [
            i for i, name in enumerate(FEATURE_NAMES) if name.startswith("residual_")
        ],
        "edges": [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("edge_")],
        "all": np.arange(matrix.shape[1]),
    }
    predictions = {}
    folds = []
    selections = []
    validation_scores = []
    fitted = {}
    for fold, (train, test) in enumerate(partitions, 1):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Source leakage between training and evaluation.")
        folds.extend((group, fold) for group in sorted(set(groups[test])))
        for family, columns in families.items():
            features = matrix[:, columns]
            threshold = 0.5
            if family == "dummy":
                model = DummyClassifier(strategy="most_frequent")
                model.fit(features[train], labels[train])
            else:
                model, selected_c = fit_grouped(features[train], labels[train], groups[train])
                if args.batch:
                    scores = model.predict_proba(features[validation])[:, 1]
                    candidates = [0.5, 0.4, 0.6, 0.3, 0.7]
                    threshold = max(candidates, key=lambda value: balanced_accuracy_score(
                        labels[validation], scores >= value
                    ))
                    fitted[family] = (model, columns, threshold, selected_c)
                    validation_scores.append((family, balanced_accuracy_score(
                        labels[validation], scores >= threshold
                    )))
                selections.append((fold, family, selected_c, threshold))
            conditions = {"matched": (test, features[test])}
            if family == "all" or (args.batch and family != "dummy"):
                for name, vectors in challenges.items():
                    subset = [i for i in test if i in vectors]
                    conditions[name] = (subset, np.array([vectors[i][columns] for i in subset]))
            for condition, (subset, values) in conditions.items():
                probabilities = model.predict_proba(values)[:, 1]
                decisions = (probabilities >= threshold).astype(int)
                rows = predictions.setdefault((family, condition), [])
                for i, probability, decision in zip(subset, probabilities, decisions, strict=True):
                    rows.append((fold, index[i]["image_id"], groups[i], index[i]["path"],
                                 labels[i], probability, decision))

    run_name = "pilot_all" if args.include_rejected else "pilot"
    output = args.batch / "results" if args.batch else ROOT / "results" / run_name
    csv_output = args.batch / "run-data" if args.batch else RUN_DATA / run_name
    output.mkdir(parents=True, exist_ok=True)
    csv_output.mkdir(parents=True, exist_ok=True)
    write_csv(csv_output / "folds.csv", ("source_id", "held_out_fold"), sorted(folds))
    write_csv(csv_output / "tuning.csv", ("fold", "family", "selected_c", "threshold"), selections)
    write_csv(
        csv_output / "predictions.csv",
        ("family", "condition", "fold", "image_id", "source_id", "path", "label",
         "probability", "prediction"),
        ((family, condition, *row) for (family, condition), rows in predictions.items()
         for row in rows),
    )
    write_csv(
        csv_output / "metrics.csv",
        ("family", "condition", "n", "balanced_accuracy", "roc_auc", "precision",
         "recall", "f1", "false_positive_rate", "tn", "fp", "fn", "tp"),
        ((family, condition, *score_predictions(rows))
         for (family, condition), rows in predictions.items()),
    )
    write_error_gallery(output, predictions["all", "matched"])
    if args.batch:
        from detector import save_model
        from batch import CONFIG

        winner = max(validation_scores, key=lambda row: row[1])[0]
        model, columns, threshold, selected_c = fitted[winner]
        save_model(output / "detector.json", model, columns, threshold, CONFIG["max_side"],
                   winner, selected_c, "held-out-main", {
                       split: sorted(set(groups[splits == split]))
                       for split in ("train", "validation", "test")
                   })
    if args.batch:
        manifest = []
        for row, split in zip(index, splits, strict=True):
            path = Path(row["path"])
            manifest.append((
                row["image_id"], path.relative_to(args.batch.resolve()).as_posix(),
                image_hash(path), row["source_id"], row["group_id"], split, row["label"],
                CONFIG["model"] if row["label"] == "1" else "original",
            ))
        write_csv(
            csv_output / "analysis_manifest.csv",
            ("image_id", "path", "sha256", "source_id", "group_id", "split", "label", "editor"),
            manifest,
        )
        write_csv(
            csv_output / "features.csv", ("image_id", "sha256", "feature_version", *FEATURE_NAMES),
            ((row[0], row[2], FEATURE_VERSION, *values)
             for row, values in zip(manifest, matrix, strict=True)),
        )
        write_csv(csv_output / "validation.csv", ("family", "balanced_accuracy"), validation_scores)
        write_csv(
            csv_output / "split_counts.csv", ("split", "groups", "sources", "images"),
            ((split, len(set(groups[splits == split])),
              len({index[i]["source_id"] for i in np.flatnonzero(splits == split)}),
              int(np.sum(splits == split)))
             for split in ("train", "validation", "test")),
        )
    config = {
        "include_rejected": args.include_rejected,
        "protocol": "fixed train/validation/test" if args.batch else "nested grouped cross-validation",
        "validation_selected_family": max(validation_scores, key=lambda row: row[1])[0]
        if validation_scores else None,
        "feature_version": FEATURE_VERSION, "seed": SEED, "outer_folds": 1 if args.batch else 5,
        "inner_folds": 4, "c_values": C_VALUES,
        "score_for_tuning": "balanced_accuracy", "threshold": "see ../run-data/tuning.csv" if args.batch else 0.5,
        "versions": {name: version(name) for name in ("numpy", "scipy", "Pillow", "scikit-learn")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print((csv_output / "metrics.csv").read_text(encoding="utf-8"))


def write_error_gallery(output, matched):
    from html import escape
    from os.path import relpath
    from urllib.parse import quote

    rows = []
    for fold, image_id, _, path, label, score, decision in matched:
        if label == decision:
            continue
        kind = "Miss" if label else "False positive"
        rows.append(
            f'<figure><img loading="lazy" width="220" src="{escape(quote(relpath(ROOT / path, output)))}" '
            f'alt="{escape(image_id)}"><figcaption>{kind}, fold {fold}, '
            f'score {score:.3f}<br>{escape(image_id.split("_")[0])}</figcaption></figure>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Pilot errors</title>'
        '<h1>Pilot errors</h1><main style="display:flex;gap:1rem;flex-wrap:wrap">'
        + "".join(rows) + "</main></html>"
    )
    (output / "errors.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
