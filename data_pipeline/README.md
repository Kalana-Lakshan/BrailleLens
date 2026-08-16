# Stage 1–3 — Data pipeline

One cell-level contract for every later stage. DSBI (flatbed scans) and
Angelina (handheld photos) are the training sources now. Gold (Sinhala
page photos) is wired in but stays empty until you label it.

```
data DBSI/  +  data Angelina/  [+ Gold Dataset/ later]
        |
        v
  2a  integrate.py     ->  manifests/manifest_raw.csv
  2b  clean.py         ->  manifests/manifest_clean.csv
  2c  reduce.py        ->  crops/crops_{train,val,test}.npz
  2d  transform.py     ->  used at load time (CropDataset)
  3   analyze.py       ->  reports/eda/
        |
        v
  Stage 4 reads only the clean manifest and the crop archives
```

The handoff file is `manifests/manifest_clean.csv`. One row = one Braille
cell. Schema is defined in [`contracts.py`](contracts.py).

---

## What each file does

| File | Stage | Role |
|---|---|---|
| `contracts.py` | — | Shared schema: columns, `page_group` leakage key, dot-code helpers |
| `integrate.py` | 2a | Parse DSBI + Angelina (+ Gold JSON) into one CSV |
| `clean.py` | 2b | Drop bad boxes; write `reports/cleaning_log.md` |
| `reduce.py` | 2c | Crop every cell to 64×64 grayscale `.npz` |
| `transform.py` | 2d | Margins, `normalize_crop`, real-crop augmentation |
| `crop_dataset.py` | 2d | Torch `Dataset` over the `.npz` archives |
| `analyze.py` | 3 | EDA plots + `reports/eda/README.md` |
| `transfer_gold_labels.py` | 1b | After labelling: copy High-quality boxes onto Low-quality twins |

---

## Run (from repo root)

```bash
# 2a  official splits (DSBI train/test, Angelina train/val)
py -3.11 -m data_pipeline.integrate --sources dbsi angelina

# 2a  recommended for training: 70/15/15 by page_group (gives a real val set)
py -3.11 -m data_pipeline.integrate --sources dbsi angelina --split-mode rebalance

# 2b
py -3.11 -m data_pipeline.clean

# 3   look at crop_samples.png first
py -3.11 -m data_pipeline.analyze

# 2c  after EDA looks right
py -3.11 -m data_pipeline.reduce
```

Gold later (no labels yet — this prints 0 cells and that is expected):

```bash
py -3.11 -m data_pipeline.integrate --sources dbsi angelina gold --split-mode rebalance
```

---

## Manifest columns

`source, image_path, page_group, book, page, side, split, x0, y0, x1, y1, code, dots, img_w, img_h, dot_pitch_px`

- `code` is 0–63 (dot *i* → bit *i*−1). `dots` is the same thing as `"1345"`.
- `page_group` is the split key. High/Low Gold twins of `pg-3` share `gold:pg-3`.
- Splits are `train | val | Angelina has no official test; official mode leaves Angelina test empty.

---

## Defaults that matter

| Setting | Value | Why |
|---|---|---|
| DSBI box margin in integrate | 0.35 | Matches Angelina full-cell boxes; the old 0.8 crop ate neighbouring lines |
| Angelina path | `data Angelina/books` | Not `AngelinaDataset-master/books` |
| DSBI path | `data DBSI/data` | Images live under `data/`; split files are copied there |
| Crop size | 64×64 uint8 | Same as `SimpleBrailleCNN` |

---

## Next stage

- Cell detector: `py -3.11 -m cell_detect.prepare_cell_dataset`
- Cell classifier: `py -3.11 -m braille_cnn.train_classifier`
