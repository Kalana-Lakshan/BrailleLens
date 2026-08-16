# Stage 2b cleaning log

Input rows: **157,559**
Output rows: **157,559** (100.00% retained)

| Rule | Rows removed | Note |
|---|---:|---|
| Angelina markout (code 63) | 0 | already filtered by _read_csv_annotation, kept as a guard |
| Zero-area or inverted boxes | 0 |  |
| Boxes outside the page | 0 | tolerance 50% of box size |
| Duplicate cell annotations | 0 |  |
| Implausible aspect ratio | 0 | kept 0.15 to 6.0 |
| Decorative divider rows | 0 | 3,514 flagged but kept; pass --drop-rulers to remove |
| Missing page image | 0 |  |
| dots / code disagreement | 0 |  |
