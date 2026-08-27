# Gold cell-detector fine-tune: before vs after (held-out test page)

Train pages: pg-[1, 2, 3, 4, 5, 6, 7, 8] (high quality) + low-quality-lighting pg-[1, 2, 3, 4] | val (checkpoint selection only): pg-[9, 12] | **test (held out, never trained/monitored on): pg-[10, 11]**

Trained on Colab GPU (T4), 40 epochs, imgsz=1280, batch=8, mosaic=0.5 -- an
earlier CPU-only local attempt at this same train/val/test split (high-quality
pages only, no mosaic, since mosaic segfaulted natively on this machine's
CPU/OpenCV build) only reached mAP50=0.3764/precision=0.5783/recall=0.5891;
adding the low-quality-lighting training images plus mosaic augmentation on a
real GPU closed most of the remaining gap.

| model | mAP50 | precision | recall |
|---|---|---|---|
| baseline (`braille_cell_best.pt`) | 0.3713 | 0.5582 | 0.4805 |
| gold fine-tuned, Colab GPU (`braille_cell_gold.pt`) | **0.6526** | **0.7300** | **0.7436** |
