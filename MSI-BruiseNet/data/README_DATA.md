# Data placeholder

Put your MSI `.npy` files into:
- `data/images/`: `(H, W, 9)`
- `data/masks/`: `(H, W)` with values `{0,1}`

Use `scripts/prepare_splits.py` to generate fold JSON under `data/splits/`.
