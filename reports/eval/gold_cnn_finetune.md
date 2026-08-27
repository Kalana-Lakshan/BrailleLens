# Gold CNN classifier fine-tune: before vs after (held-out test page)

Train pages: pg-[1, 2, 3, 4, 5, 6, 7, 8] (mixed with live DBSI crops, equal draw rate) | val (checkpoint selection only): pg-[9, 12] | **test (held out, never trained/monitored on): pg-[10, 11]**

Ground-truth crops (LabelMe boxes via data_pipeline.transform.extract_crop, margin=SOURCE_MARGINS['gold']=0.15), not detector output -- isolates classifier accuracy from cell-detection error.

| model | gold test acc |
|---|---|
| baseline (`braille_cnn_mixed.pt`) | 0.6774 |
| gold fine-tuned (`braille_cnn_gold_finetuned.pt`) | 0.9593 |

Val-set snapshot, start -> best epoch: gold 0.5711 -> 0.9526 (checkpoint-selection metric), DBSI 0.9925 -> 0.9942 (regression check only, not used for checkpoint selection, 6000-cell subset of DBSI test).

Angelina accuracy could not be checked on this machine (raw Angelina data and the Stage 2c crops_*.npz archives braille_cnn_mixed.pt was actually trained on aren't present locally) -- re-check there before trusting this checkpoint on handheld-photo input.
