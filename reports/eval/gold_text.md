# Gold page-text accuracy (letters-only)

Ground truth: `Gold Dataset/Text pages/pg-N.txt`, decoded via the ids in `Gold Dataset/symbols guide.jpeg` -- letters (a-z) and space only; capital/number/punctuation ids are skipped (unverified dot patterns, see eval_gold_text.py docstring), not counted as errors either way.

Predicted: `recognize_page(backend="cells", lang="en")`, sorted by (line, col), same letters/space-only filter applied.

| page | cells detected | gt chars | acc (with spaces) | acc (letters only) |
|---|---|---|---|---|
| pg-1 | 348 | 290 | 0.603 | 0.764 |
| pg-2 | 296 | 238 | 0.660 | 0.863 |
| pg-3 | 329 | 276 | 0.627 | 0.805 |
| pg-4 | 391 | 290 | 0.569 | 0.797 |
| pg-5 | 314 | 264 | 0.576 | 0.727 |
| pg-6 | 332 | 287 | 0.669 | 0.807 |

**Overall: acc_with_spaces=0.616  acc_letters_only=0.793**

acc_with_spaces is lower than acc_letters_only mostly because the cell-detector YOLO model only proposes boxes where it sees raised dots -- it has no mechanism to box a blank word-gap cell, so most spaces are structurally missing from the prediction regardless of classifier accuracy. That's a detection-recall gap, not a classification error.

This run uses `--cell-conf 0.30 --spine-boost`, plus `--drop-ruler-lines`
(now the default -- pass `--no-drop-ruler-lines` to disable). Without it:
acc_with_spaces=0.609, acc_letters_only=0.792 -- a real gain on both
numbers, no regression on any of the 6 pages. Validated against all 12 gold
pages (not just these 6) at the box level -- see
reports/eval/gold_cell_detector_finetune.md's "Ruler-line filter" section
for the full table this threshold was tuned and validated against.

## Tried and rejected: classify-based grid-gap recovery

Idea from the Ovodov CNN paper's fixed-grid framing: instead of assuming a
wide within-line gap is a blank space (`recognize.py`'s `_insert_word_gaps`,
still in use), crop the grid-implied slot(s) directly from the page and run
them through the classifier for a real answer -- could recover genuine
missed letters, not just spaces.

Tested against the checkpoint pair/split active at the time: made both
numbers worse (acc_with_spaces 0.404->0.337, acc_letters_only 0.640->0.594).
Inspecting the recovered slots directly on pg-1 showed why: alongside correct
high-confidence `space` predictions, a large fraction of slots got
confidently classified as a real letter (`capital_sign` 0.964, `f` 0.957,
`a` 1.0, ...) that wasn't actually there -- softmax overconfidence on a crop
that's noise or a slightly misaligned sliver of a neighboring cell, not a
genuine missed detection. A confidence threshold can't cleanly separate these
since the wrong ones are often just as confident as the right ones. Also,
pg-1 alone had ~444 cells already detected against only 290 ground-truth
characters -- the page is already substantially over-fragmented, so
gap-ratio-based slot counting had a noisy pitch estimate to work from. Not
adopted -- reverted in `recognize.py`; `_insert_word_gaps`' plain "assume
blank" heuristic remains in place.

## Update: camera-degradation-trained detector (`braille_cell_gold_degraded.pt`)

Re-ran with the same flags (`--cell-conf 0.30 --spine-boost --drop-ruler-lines`)
against today's codebase, to compare the original gold detector against the
camera-degradation-augmented one (see `cell_detect/CAMERA_DEGRADATION.md`)
now wired into the Flutter app's `app_config.dart`.

| cell weights | acc (with spaces) | acc (letters only) |
|---|---|---|
| `braille_cell_gold.pt` (previous default) | 0.598 | 0.767 |
| `braille_cell_gold_degraded.pt` (current app default) | **0.671** | **0.834** |

Note the first row (0.767) is measurably lower than the 0.793 recorded above
for the same checkpoint -- not yet root-caused; flagged for investigation,
not dismissed as noise. The degraded checkpoint improves on both numbers
regardless, and on every one of the 6 pages individually (no regressions):

| page | gt chars | acc_letters_only (`gold.pt`) | acc_letters_only (`gold_degraded.pt`) |
|---|---|---|---|
| pg-1 | 290 | 0.704 | 0.837 |
| pg-2 | 238 | 0.838 | 0.843 |
| pg-3 | 276 | 0.814 | 0.891 |
| pg-4 | 290 | 0.784 | 0.819 |
| pg-5 | 264 | 0.660 | 0.809 |
| pg-6 | 287 | 0.803 | 0.807 |

Command: `py -3.11 -m braille_cnn.eval_gold_text --cell-conf 0.30 --spine-boost --checkpoint braille_cnn/checkpoints/braille_cnn_gold_finetuned.pt --cell-weights cell_detect/weights/braille_cell_gold_degraded.pt`

## Update: hard-negatives fine-tuned on top of the teammate's lighting-robust checkpoint

`cell_detect/weights/braille_cell_lighting.pt` (teammate's lighting/negatives work,
built on top of our camera-degradation checkpoint -- see
`lighting_robust/results/`) is now tracked in the repo, resolving the earlier
blocker (only its ONNX export existed before). Fine-tuned our proven 12-image
positive set (8 high-quality + 4 low-quality-lighting pages) plus 39
hard-negative images (`cell_detect/make_synthetic_negatives.py` + "Non-braille
images/") on top of it via `cell_detect/gold_finetune_2.py`, same proven
hyperparameters, no other changes. Training converged at 16 epochs (patience
stop), val mAP50 0.945 -- no regression.

Held-out test (pg-10/11, high+low, n=4, 1097 instances): mAP50 0.940 (lighting
alone) vs 0.926 (+ negatives) -- a negligible difference, both in the same
healthy range. pg-1..6 box counts (the anomaly `braille_cell_gold_negatives.pt`
showed, ~40-160 boxes/page instead of ~300) are completely absent here: 273-312
boxes/page, normal for every checkpoint compared. That anomaly was specific to
the old checkpoint's base lineage, not a general hard-negative-training issue.

Full pipeline (detect + classify + reconstruct text), vs. the teammate's
checkpoint alone:

| cell weights | acc (with spaces) | acc (letters only) |
|---|---|---|
| `braille_cell_lighting.pt` (teammate's, current app default) | 0.761 | 0.882 |
| `braille_cell_lighting_negatives.pt` (ours, on top of theirs) | **0.783** | **0.899** |

Per-page acc_letters_only, `lighting.pt` -> `lighting_negatives.pt`: pg-1
0.871->0.901, pg-2 0.853->0.843 (negligible dip), pg-3 0.937->0.964, pg-4
0.832->0.871, pg-5 0.880->0.928, pg-6 0.917->0.886 (small dip). Net gain on
4 of 6 pages, both dips small -- a real, if modest, improvement stacked on an
already-strong base. The best full-pipeline result measured this session
(up from the original ~79.3% recorded at the top of this file, 76.7% on
today's codebase's gold.pt baseline, 83.4% on the degraded-only checkpoint,
88.2% on the teammate's lighting-robust checkpoint alone).

False-positive re-check (VS Code screenshot, ID card, laptop/keyboard photo)
not yet re-run against this checkpoint -- the original test images expired
from the session scratchpad; need them resent before repeating that check.

Command: `py -3.11 -m braille_cnn.eval_gold_text --cell-conf 0.30 --spine-boost --checkpoint braille_cnn/checkpoints/braille_cnn_gold_finetuned.pt --cell-weights cell_detect/weights/braille_cell_lighting_negatives.pt`
