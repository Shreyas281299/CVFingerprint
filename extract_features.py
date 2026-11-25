"""Extract image-level regional features for the recorded 256px pilot."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, sobel

ROOT = Path(__file__).resolve().parent
RUN_DATA = ROOT / "run-data"
FEATURE_VERSION = "regional-v1"
GRID_SIZE = 4
MAX_SIDE = 256
GAUSSIAN_SIGMA = 1.0
EDGE_THRESHOLD = 0.08
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
PATCH_NAMES = (
    "rgb_r_mean", "rgb_g_mean", "rgb_b_mean",
    "rgb_r_std", "rgb_g_std", "rgb_b_std",
    "residual_std", "residual_corr_horizontal", "residual_corr_vertical",
    "edge_mean_magnitude", "edge_density",
)
FEATURE_NAMES = tuple(
    f"{name}_patch_{stat}" for stat in ("mean", "std") for name in PATCH_NAMES
)


FEATURE_CONFIG = {
    "version": FEATURE_VERSION,
    "grid_size": GRID_SIZE,
    "intensity_divisor": 255.0,
    "luminance_weights": LUMA_WEIGHTS.tolist(),
    "gaussian_sigma": GAUSSIAN_SIGMA,
    "gaussian_truncate": 4.0,
    "border_mode": "reflect",
    "sobel_divisor": 8.0,
    "edge_threshold": EDGE_THRESHOLD,
    "correlation_min_scale": 1e-12,
    "pooling": ["mean", "population_std"],
}


def image_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_original(path, max_side=MAX_SIDE):
    image = load_rgb(path)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width, height = image.size
    return image.crop((0, 0, width // 8 * 8, height // 8 * 8))


def correlation(left, right):
    left = left.ravel() - left.mean()
    right = right.ravel() - right.mean()
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / scale) if scale > 1e-12 else 0.0


def patch_streams(image):
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    height, width = rgb.shape[:2]
    if min(height, width) < 2 * GRID_SIZE:
        raise ValueError("Each 4×4 grid cell needs at least two pixels per side.")
    luminance = rgb @ LUMA_WEIGHTS
    residual = luminance - gaussian_filter(
        luminance, sigma=GAUSSIAN_SIGMA, mode="reflect", truncate=4.0
    )
    dx = sobel(luminance, axis=1, mode="reflect") / 8.0
    dy = sobel(luminance, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(dx, dy)

    appearance = []
    structure = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            patch = (
                slice(row * height // GRID_SIZE, (row + 1) * height // GRID_SIZE),
                slice(col * width // GRID_SIZE, (col + 1) * width // GRID_SIZE),
            )
            color = rgb[patch]
            noise = residual[patch]
            edges = magnitude[patch]
            appearance.append([*color.mean(axis=(0, 1)), *color.std(axis=(0, 1))])
            structure.append([
                noise.std(),
                correlation(noise[:, :-1], noise[:, 1:]),
                correlation(noise[:-1, :], noise[1:, :]),
                edges.mean(),
                (edges >= EDGE_THRESHOLD).mean(),
            ])

    return np.asarray(appearance), np.asarray(structure)


def extract_features(image):
    appearance, structure = patch_streams(image)
    patches = np.concatenate((appearance, structure), axis=1)
    return np.concatenate((patches.mean(axis=0), patches.std(axis=0)))


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    temporary.replace(path)


def main():
    with (RUN_DATA / "pilot.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if not pairs:
        raise ValueError("pilot.csv contains no pairs.")

    feature_rows = []
    index_rows = []
    seen_originals = set()
    seen_edits = set()
    for pair in pairs:
        source_id = Path(pair["original"]).stem
        if source_id in seen_originals or pair["edited"] in seen_edits:
            raise ValueError(f"Duplicate source or edited path for {source_id}.")
        seen_originals.add(source_id)
        seen_edits.add(pair["edited"])

        original = prepare_original(ROOT / pair["original"])
        edited = load_rgb(ROOT / pair["edited"])
        if original.size != edited.size:
            raise ValueError(f"Prepared original and edit dimensions differ for {source_id}.")
        if np.ptp(np.asarray(edited)) == 0:
            raise ValueError(f"Blank edited image for {source_id}.")

        for label, key, image in ((0, "original", original), (1, "edited", edited)):
            image_id = f"{source_id}:{key}"
            features = extract_features(image)
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite features for {image_id}.")
            digest = image_hash(ROOT / pair[key])
            feature_rows.append([image_id, digest, FEATURE_VERSION, *features])
            index_rows.append([image_id, source_id, label, pair[key], digest, FEATURE_VERSION])

    write_csv(RUN_DATA / "pilot_features.csv", ("image_id", "sha256", "feature_version", *FEATURE_NAMES), feature_rows)
    write_csv(
        RUN_DATA / "pilot_feature_index.csv",
        ("image_id", "source_id", "label", "path", "sha256", "feature_version"),
        index_rows,
    )
    print(f"Extracted {len(FEATURE_NAMES)} features per image for {len(feature_rows)} images.")


if __name__ == "__main__":
    main()
