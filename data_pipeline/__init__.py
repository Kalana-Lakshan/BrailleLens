"""BrailleLens data pipeline (data-science-lifecycle stages 1-3).

Stage 1  collection   - datasets already on disk, described in README.md
Stage 2a integration  - integrate.py  -> manifest_raw.csv
Stage 2b cleaning     - clean.py      -> manifest_clean.csv
Stage 2c reduction    - reduce.py     -> crops_<split>.npz
Stage 2d transform    - transform.py  -> applied at load time by CropDataset
Stage 3  analysis     - analyze.py    -> reports/eda/

Everything downstream reads the manifest defined in contracts.py.
"""

from .contracts import (  # noqa: F401
    MANIFEST_COLUMNS,
    CellRow,
    code_to_dot_string,
    dot_string_to_code,
    dots_to_code,
    read_manifest,
    write_manifest,
)
