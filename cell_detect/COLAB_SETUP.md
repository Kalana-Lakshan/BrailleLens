# Train the cell detector on Colab / Kaggle

This machine has **CPU-only PyTorch**. A real 80-epoch run belongs on a GPU.

## What to upload

From the repo root, after `prepare_cell_dataset` has run:

```
cell_detect/datasets/braille_cells/   (images + labels + data.yaml)
cell_detect/configs/cells.yaml
cell_detect/train_detector.py
```

Zip the dataset if Drive is easier:

```bash
# from repo root
tar -a -c -f cell_detect/braille_cells.zip -C cell_detect/datasets braille_cells
```

## Colab sketch

```python
# GPU runtime required
!pip install -q ultralytics pyyaml opencv-python-headless

# unzip braille_cells.zip so data.yaml's path still points at the folder
# then:
!python train_detector.py --data /content/braille_cells/data.yaml --device 0 --epochs 80 --imgsz 1280 --batch 4
```

Copy `runs/detect/braille_cell_yolo26/weights/best.pt` back to

`cell_detect/weights/braille_cell_best.pt`

## Notes

- `imgsz 1280` is intentional (cells shrink too far at 640 on a full page).
- If Colab OOMs, drop to `imgsz 960` or `batch 2`, not tiling — cells are
  large enough that tiling is the *dot* pipeline's trick, not this one.
- Never enable `fliplr` / `flipud`. Braille is not mirror-symmetric.
