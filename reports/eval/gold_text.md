# Gold page-text accuracy (letters-only)

Ground truth: `Gold Dataset/Text pages/pg-N.txt`, decoded via the ids in `Gold Dataset/symbols guide.jpeg` -- letters (a-z) and space only; capital/number/punctuation ids are skipped (unverified dot patterns, see eval_gold_text.py docstring), not counted as errors either way.

Predicted: `recognize_page(backend="cells", lang="en")`, sorted by (line, col), same letters/space-only filter applied.

| page | cells detected | gt chars | acc (with spaces) | acc (letters only) |
|---|---|---|---|---|
| pg-1 | 462 | 290 | 0.455 | 0.665 |
| pg-2 | 448 | 238 | 0.319 | 0.563 |
| pg-3 | 430 | 276 | 0.388 | 0.615 |
| pg-4 | 444 | 290 | 0.366 | 0.621 |
| pg-5 | 400 | 264 | 0.405 | 0.684 |
| pg-6 | 425 | 287 | 0.477 | 0.684 |

**Overall: acc_with_spaces=0.404  acc_letters_only=0.640**

acc_with_spaces is lower than acc_letters_only mostly because the cell-detector YOLO model only proposes boxes where it sees raised dots -- it has no mechanism to box a blank word-gap cell, so most spaces are structurally missing from the prediction regardless of classifier accuracy. That's a detection-recall gap, not a classification error.
