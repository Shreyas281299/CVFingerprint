# Detecting and Attributing AI Image Edits Using Low-Level Statistical Fingerprints

CS543/ECE549 · Shreyas Sharma · November 24, 2026

## Scope and starting point

Measure whether regional color, residual, and edge statistics can distinguish car photos edited by InstructPix2Pix from unedited car photos. At inference, the detector receives a single candidate image. Retain the original for dataset construction and diagnostic comparisons. A controlled negative result satisfies the core research goal. Attribution and cross-editor transfer are extensions.

The proposal was submitted November 16. The RTX 4060 laptop has a working generation setup and a reported benchmark. Recover its settings and logs if available, and capture missing throughput and memory measurements during the pilot. Check solo-project approval status in parallel. The milestone dates and submission requirements below come from the supplied brief.

| Machine | Responsibilities |
| --- | --- |
| Other laptop, RTX 4060 | Generate AI edits from prepared inputs, record every attempt, and export batches. Reuse the existing CUDA environment and benchmark script. |
| This laptop | Collect and prepare photos, assign splits, build and verify transfer bundles, review edits, extract features, train and tune CPU models, and produce results. Keep the authoritative manifest and dataset here. |

## 1. Collect and inventory source photos

- [ ] Start with personal photos and fill gaps with original-size car photos from Wikimedia Commons. Sample varied cars, colors, viewpoints, cameras, and backgrounds. Record creator, source URL, license, and attribution requirements per photo. Keep original downloads intact. [Wikimedia Commons reuse guide](https://commons.wikimedia.org/wiki/Commons:First_steps/Reuse)
- [ ] Assign a stable `source_id` and a `group_id` for photos of the same car/session or near-duplicates. Use file hashes for exact duplicates and perceptual hashing plus manual review for near-duplicates. Exclude obvious composites or unusable photos. Record exclusions and count independent groups separately from photos and derivatives.
- [ ] Use 20 sources for the pilot. If it passes, provisionally aim for 150–300 independent sources total in the main study, subject to collection time, render throughput, and storage. Keep pilot or development groups out of the final test split. Once the main inventory is ready, freeze an approximately 60/20/20 split by group, with both labels represented in every split. Each derivative inherits its source's split.

## 2. Prepare a matched dataset

- [ ] Decode with EXIF orientation applied and convert consistently to RGB. Prepare a lossless PNG per source using one documented resize/crop recipe. Start from the benchmark dimensions, roughly a 512-pixel maximum side with dimensions divisible by eight. Use the exact prepared input pixels for the corresponding unedited control.
- [ ] Start with two fixed prompt templates, such as changing the paint color and replacing the background. Choose target paint colors different from the input, using a consistent selection rule. Save exact prompt text and prompt category. Use the same source and matched requests if a second editor is added.
- [ ] Add ordinary edits as a separate negative challenge set. Include simple brightness or sharpening and, for a subset, conventional recoloring matched to the paint request so the detector can be challenged on the same semantic change. Export all analysis classes with a shared encoding and dimensions while keeping raw inputs and native outputs. Earlier JPEG history can survive PNG export.
- [ ] Use `sources.csv` for IDs, group, split, paths, hashes, and provenance. Use `jobs.csv` for job ID, source ID, prepared input path, prompt, seed, and configuration ID. Use `attempts.jsonl` for attempt number, editor/checkpoint revision, settings, dimensions, output path/hash, elapsed time, and failure or QA outcome. Use relative paths in all manifests.

## 3. Render on the other laptop and verify the handoff

- [ ] Copy prepared PNGs, the job manifest, and pinned generation configuration to the RTX laptop in a ZIP or plain file copy. Load the Diffusers pipeline once and generate one image at a time. Begin with the existing benchmark settings where available. Record package versions, precision, scheduler, steps, guidance scales, seeds, and model revision.
- [ ] Give each attempt a unique output path. Write the PNG before marking it complete. On restart, skip verified successful attempts and rerun only incomplete or failed jobs under the documented retry rule. Keep failed attempts in the log. Save raw outputs as lossless PNGs.
- [ ] Return images, attempt log, configuration, and SHA-256 checksums in a batch folder. On this laptop, verify hashes, expected job IDs, image decoding, dimensions, and absence of missing or duplicate outputs. Reconcile failures before marking the batch complete. Keep at least one verified copy on each laptop until import is complete.
- [ ] Review the 40 pilot attempts and 20 originals. Before review, define success as a visible requested change, preservation of the car's identity and framing, and absence of severe corruption. Record failure reasons and use at most one retry per job with a predetermined second seed. For the main paired baseline, use one accepted edit per original and use the second prompt category as a separate check or in a report that accounts for repeated sources. Report acceptance rates and excluded sources by prompt.
- [ ] Use measured elapsed time per attempt, observed retry rate, and bytes per job to estimate overnight rendering time and space. Scale only after the round trip verifies, edit quality is usable, and both laptop stages run end to end on the pilot.

## 4. Extract one feature vector per image on this laptop

- [ ] Divide each prepared image into a fixed 4×4 non-overlapping grid. Compute residual and gradient maps on the whole image before slicing patches to avoid artificial patch borders. Convert pixel intensities to a consistent floating-point range.
- [ ] **Color:** RGB means and standard deviations per patch. **Residual:** subtract a fixed Gaussian-smoothed luminance image, then measure residual spread and horizontal/vertical neighbor correlations. **Edges:** use Sobel gradients for mean magnitude and edge density under a fixed threshold. Define finite values for flat patches where correlation is undefined.
- [ ] Start by pooling each statistic across patches with its mean and standard deviation. Compare ordered patch concatenation or a second grid size on development data if time allows. Leave frequency-band energy for a later ablation.
- [ ] Cache one row per image with named columns and a feature configuration version. Keep filenames, paths, split IDs, EXIF, and encoder information outside the model inputs. Check deterministic output, finite values, correct label joins, and zero group overlap before training.

## 5. Train, tune, and check for shortcuts

- [ ] Fit a dummy majority-class baseline and a scikit-learn pipeline with `StandardScaler` and L2 logistic regression. Train on matched original/edit pairs from the training groups. Compare RGB-only, residual-only, edge-only, and all-feature models on the same groups and splits. Add a shallow random forest only if the schedule allows.
- [ ] Tune a small regularization grid, for example `C = 0.01, 0.1, 1, 10`, using grouped cross-validation within training. Fit the scaler independently inside each fold. Use validation to select the final feature settings and decision threshold by balanced accuracy. Freeze the configuration before scoring test groups. [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ ] On development data, check whether dimensions, original encoding history, or photo provenance differ systematically by label and whether these variables alone predict class. Standardize the analysis exports and evaluate both classes under the same JPEG and resizing transforms. Example fixed challenges are JPEG quality 85, quality 65, and a 0.75× downscale. Reuse the trained model and threshold for these challenges.
- [ ] If time allows, make a VAE encode/decode-only diagnostic from the same checkpoint with matching preprocessing. Score these outputs separately to estimate whether reconstruction alone triggers the detector. Keep ordinary-edit negatives separate as well, so their false-positive rate is visible.

## 6. Evaluate and report

- [ ] Score the frozen detector on the held-out source groups. Report balanced accuracy as the primary metric, plus ROC-AUC, precision, recall, F1, a confusion matrix, and false-positive rates for originals and conventional edits. Report counts of independent groups, sources, and outputs by split. Use group-level bootstrap intervals if the test set is large enough, and show raw counts when it is small.
- [ ] Produce a dataset flow table, feature-ablation table, robustness table, and a small gallery of false positives and misses. Break down results by prompt and export condition where there are enough examples. Record excluded generations and explain how restricting the analysis to successful edits affects what the numbers represent.
- [ ] Discuss whether performance survives the conventional-edit and re-encoding controls. State the limits of a car-only, single-editor study and whether the evidence could reflect the generation/export pipeline. Present systematic negative findings if the detector performs near chance or fails under the controls.
- [ ] If the core evaluation is complete and time permits, render matched jobs with editor B on the same source groups. Test A → B and B → A on held-out originals plus edits from the target editor, then try known-editor attribution with original/A/B labels. Count each original once and report macro-F1 and a confusion matrix. Describe attribution at the pipeline level unless controls isolate the underlying model.
- [ ] Save split manifests, versioned configs, dependency versions, predictions, and figure scripts. Do a clean reproduction on the pilot and regenerate the final tables from frozen predictions. Prepare the research-style PDF and a ZIP of project-authored source code, excluding external libraries. Cite data, dependencies, model checkpoints, related work, and assistance. The supplied brief specifies at least six single-spaced pages in 10-point type for the final report.

## Code and research to reuse

| Resource | Use in this project |
| --- | --- |
| [Diffusers InstructPix2Pix pipeline](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix), [checkpoint](https://huggingface.co/timbrooks/instruct-pix2pix), and [authors' code](https://github.com/timothybrooks/instruct-pix2pix) | Adapt the existing benchmark into a resumable generator. Reuse the official inference pipeline and pin the checkpoint revision. |
| [Pillow](https://pillow.readthedocs.io/), [NumPy](https://numpy.org/doc/stable/), and [OpenCV](https://docs.opencv.org/) | Reuse orientation, resize, PNG/JPEG encoding, statistics, Gaussian smoothing, and Sobel operators. Write the regional aggregation and experimental comparisons. |
| [scikit-learn](https://scikit-learn.org/stable/) | Reuse preprocessing pipelines, dummy and logistic baselines, grouped validation, and metrics. |
| [TruFor code and paper](https://github.com/grip-unina/TruFor) | Related work for combining RGB with noise-sensitive features. Optional external comparison if its license, compute cost, and localization outputs fit the study. |
| [EditTrack](https://arxiv.org/abs/2510.01173) and [Learning Rich Features for Image Manipulation Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Learning_Rich_Features_CVPR_2018_paper.html) | Cite EditTrack as a reference-pair method and Zhou et al. as prior work on RGB/noise fusion. |

## Milestones

| Date | Target |
| --- | --- |
| November 24–26 | Inventory the pilot photos, define provenance and splits, and verify one batch across both laptops. |
| November 27–30 | Complete pilot QA and the first binary baseline. Submit the November 30 progress report with real counts, throughput, examples, and risks through Canvas. |
| December 1–7 | Expand and freeze the single-editor dataset. Extract features, tune on development data, and finalize the control conditions. |
| December 8–12 | Run held-out evaluation, ablations, and robustness checks. Assemble tables and analyze errors. |
| December 13–16 | Reproduce the tables, finish the PDF and source ZIP, and submit by December 16. Prioritize the single-editor study if time gets tight. |
