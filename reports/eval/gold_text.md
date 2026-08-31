# Gold page-text accuracy (letters-only)

Ground truth: `Gold Dataset/Text pages/pg-N.txt`, decoded via the ids in `Gold Dataset/symbols guide.jpeg` -- letters (a-z) and space only; capital/number/punctuation ids are skipped (unverified dot patterns, see eval_gold_text.py docstring), not counted as errors either way.

Predicted: `recognize_page(backend="cells", lang="en")`, sorted by (line, col), same letters/space-only filter applied.

| page | cells detected | gt chars | acc (with spaces) | acc (letters only) |
|---|---|---|---|---|
| pg-1 | 348 | 290 | 0.603 | 0.764 |
| pg-2 | 324 | 238 | 0.605 | 0.853 |
| pg-3 | 329 | 276 | 0.627 | 0.805 |
| pg-4 | 391 | 290 | 0.569 | 0.797 |
| pg-5 | 314 | 264 | 0.576 | 0.727 |
| pg-6 | 332 | 287 | 0.669 | 0.807 |

**Overall: acc_with_spaces=0.609  acc_letters_only=0.792**

acc_with_spaces is lower than acc_letters_only mostly because the cell-detector YOLO model only proposes boxes where it sees raised dots -- it has no mechanism to box a blank word-gap cell, so most spaces are structurally missing from the prediction regardless of classifier accuracy. That's a detection-recall gap, not a classification error.

This run uses `--cell-conf 0.30 --spine-boost` (`recognize_page(spine_boost=True)`,
`CellDetector.detect_boxes(spine_boost=True)`) -- a second detection pass on
the spine-proximal strip of the page, merged with the full-page pass, that
recovers cells page curvature near an open book's spine otherwise suppresses
the confidence of. See reports/eval/gold_cell_detector_finetune.md's failure
analysis for the full diagnosis. Without it, at the same cell-conf=0.30: 0.587
/ 0.755 -- a real gain on both numbers (0.587->0.609, 0.755->0.792), same
checkpoints, same pages.

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
