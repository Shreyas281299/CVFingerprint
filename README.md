# AI image-edit fingerprint detection

This project tests whether regional RGB, residual, and edge statistics can identify InstructPix2Pix edits from a single car image. The detector consumes only pixel-derived features. Source and group IDs determine the evaluation partitions.

## Reproduce the pilot

From the project directory on the processing laptop, using Python 3 with Pillow, NumPy, SciPy, and scikit-learn:

```sh
.venv/bin/python -B extract_features.py
.venv/bin/python -B evaluate_pilot.py
.venv/bin/python -B evaluate_pilot.py --include-rejected
```

The extractor recreates the renderer's 256-pixel input preprocessing for each original. It pools 11 patch measurements over a 4×4 grid into 22 image features. Five outer folds hold out complete source groups. Within each outer fold, four grouped inner folds tune L2 regularization for each feature subset. The pipeline fits scaling on each training fold. Pilot decisions use probability 0.5.

The [visual review](run-data/pilot_review.csv) excluded three of 20 source pairs. All images remain in the dataset. The [contact sheet](results/pilot/pairs.jpg) shows every pair and the [error gallery](results/pilot/errors.html) shows errors for the combined model. [Pilot metrics](run-data/pilot/metrics.csv), per-image predictions, held-out folds, and tuning results are in `run-data/pilot/`. The sensitivity-check tables are in `run-data/pilot_all/`. Model artifacts, dependency versions, and galleries are under `results/`.

| Feature set | Balanced accuracy | ROC-AUC | Original false positives | Missed edits |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | 50.0% | 0.500 | 0/17 | 17/17 |
| RGB | 91.2% | 0.986 | 1/17 | 2/17 |
| Residual | 79.4% | 0.827 | 4/17 | 3/17 |
| Edges | 50.0% | 0.502 | 6/17 | 11/17 |
| Combined | 85.3% | 0.927 | 2/17 | 3/17 |

The combined model reached 82.4% after matched JPEG quality 85, 79.4% after quality 65, and 67.6% after 0.75× resizing. Brightness and sharpening controls produced 3/17 and 1/17 false positives respectively. The strong RGB score is consistent with learning the shared purple-paint request. The study still needs varied target colors and conventional recolors matched to the requested edits to distinguish this effect from model traces. The pilot contains one editor, one prompt, and 17 accepted groups.

## Run the main study on two laptops

1. On the processing laptop, curate independent photographs and record provenance in an inventory CSV. Use columns `source_id,group_id,path,creator,source_url,license,target_color`. Paths are relative to the inventory. Group session photos and near-duplicates before preparation. Keep pilot groups outside the main study and choose a target color that differs from the original. Confirm the archive's provenance and usage terms before relying on it for the main study.
2. Prepare a new batch directory. This writes lossless prepared inputs, SHA-256 checksums, two prompt jobs per source, and a fixed group split of approximately 60% train, 20% validation, and 20% test.

   ```sh
   .venv/bin/python -B batch.py prepare batches/main --inventory inventory.csv
   ```

3. Copy the batch folder and `batch.py` to the RTX laptop. Run this with its CUDA Python environment. The renderer loads the pinned checkpoint once, writes one PNG per successful job, and resumes from its attempt log. Runtime errors or blank outputs receive one retry with a predetermined seed.

   ```powershell
   .\.venv\Scripts\python.exe batch.py render batches/main
   ```

4. Copy the batch back and verify it. The command reports nonzero status for incomplete batches. Retain failed job records. Record a visual decision per job in `review.csv` inside the batch, using columns `job_id,status,reason` and statuses `accepted` or `rejected`. Reject jobs that lack the requested change, damage the car's identity or framing, or substantially alter preserved regions. Mark jobs without usable outputs rejected.

   ```sh
   .venv/bin/python -B batch.py verify batches/main
   .venv/bin/python -B evaluate_pilot.py --batch batches/main
   ```

5. The main evaluator uses accepted paint jobs, tunes regularization on training groups, and uses validation balanced accuracy to choose a threshold per feature family and identify the best family. It evaluates each predeclared ablation on the fixed test set and writes CSV tables to the batch’s `run-data/` and model artifacts to its `results/`. Robustness controls use the same fitted models and thresholds as the matched evaluation. The validation-selected detector is exported to the batch’s results directory with its preprocessing, feature schema, coefficients, and threshold. Background jobs are prepared for a separate prompt-generalization experiment.

The main pipeline was smoke-tested with fixture outputs through preparation, checksum verification, review import, and fixed-split evaluation. Main-set image provenance, CUDA rendering on the other laptop, conventional recoloring, prompt generalization, and final study results are pending. See [plan.md](plan.md) for reporting and submission milestones.

## Score one candidate image

The [image architecture](image_architecture.md) defines two aligned streams. The feature extractor forms a 16 × 6 appearance matrix and a 16 × 5 residual/edge matrix, joins them by patch index, and pools the joined matrix into the classifier input. The artifact records the decoder, pinned feature settings, feature order, scaler, logistic coefficients, class mapping, and threshold in JSON.

```sh
.venv/bin/python -B detector.py fit-pilot
.venv/bin/python -B detector.py score candidate.jpg --model results/pilot/detector.json
```

The pilot artifact is fitted on all accepted pilot groups for demonstration. Use the grouped pilot results for performance estimates. For the main study, pass the detector JSON generated by the batch evaluator instead. The command returns a JSON object containing `edit_score` and `label`.

## Test the frozen detector on fresh archive photos

The 20 archive photos in `test-data/original/` are listed in `run-data/archive_test.csv` and are labeled `0` (unedited). Selection excluded pilot images by SHA-256 and excluded their filename-derived make/model groups. These group names are a proxy for independence. The archive provides no edit history or session IDs for this check.

```sh
.venv/bin/python -B score_test_set.py
```

This uses the existing pilot model and writes `run-data/archive_test_predictions.csv`. The archive set measures the false-positive rate. To evaluate recall and balanced accuracy too, supply a separate manifest with both originals (`0`) and verified AI edits (`1`). Include `image_id,path,label`, with paths relative to the manifest. Include `source_id,group_id,sha256` for overlap and integrity checks. Keep this set and its source groups out of training and tuning. Use `--manifest`, `--model`, and `--output` to select alternate files.
