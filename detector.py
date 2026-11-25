"""Fit an exploratory pilot detector or score a single candidate from a frozen model."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from extract_features import (
    FEATURE_CONFIG, FEATURE_NAMES, MAX_SIDE, ROOT, extract_features, prepare_original,
)


def save_model(path, model, columns, threshold, max_side, family, regularization, protocol, splits):
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Expected positive class 1 and negative class 0.")
    artifact = {
        "model_version": 1,
        "feature_config": FEATURE_CONFIG,
        "decoder": {"exif_transpose": True, "mode": "RGB", "max_side": max_side,
                    "resample": "LANCZOS", "crop_to_multiple": 8},
        "protocol": protocol,
        "splits": splits,
        "family": family,
        "regularization_c": regularization,
        "feature_names": [FEATURE_NAMES[i] for i in columns],
        "threshold": threshold,
        "labels": ["unedited", "ai_edited"],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "versions": {name: version(name) for name in ("Pillow", "numpy", "scipy", "scikit-learn")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_image(path, artifact):
    if artifact["model_version"] != 1 or artifact["feature_config"] != FEATURE_CONFIG:
        raise ValueError("Unsupported model or feature configuration.")
    decoder = artifact["decoder"]
    if decoder != {"exif_transpose": True, "mode": "RGB", "max_side": decoder["max_side"],
                   "resample": "LANCZOS", "crop_to_multiple": 8}:
        raise ValueError("Unsupported decoder configuration.")
    indices = [FEATURE_NAMES.index(name) for name in artifact["feature_names"]]
    features = extract_features(prepare_original(path, decoder["max_side"]))[indices]
    standardized = (features - artifact["mean"]) / np.asarray(artifact["scale"])
    score = float(expit(standardized @ artifact["weights"] + artifact["intercept"]))
    label = artifact["labels"][int(score >= artifact["threshold"])]
    return {"edit_score": score, "label": label}


def fit_pilot(output):
    from evaluate_pilot import fit_grouped, load_dataset

    _, features, labels, groups = load_dataset(include_rejected=False)
    columns = np.arange(len(FEATURE_NAMES))
    model, regularization = fit_grouped(features, labels, groups)
    save_model(output, model, columns, 0.5, MAX_SIDE, "all", regularization,
               "exploratory-pilot-fit", {"train": sorted(set(groups))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fit = actions.add_parser("fit-pilot")
    fit.add_argument("--output", type=Path, default=ROOT / "results/pilot/detector.json")
    score = actions.add_parser("score")
    score.add_argument("image", type=Path)
    score.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit-pilot":
        fit_pilot(args.output)
        print(args.output)
    else:
        artifact = json.loads(args.model.read_text(encoding="utf-8"))
        print(json.dumps(score_image(args.image, artifact)))
