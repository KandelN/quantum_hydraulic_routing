# Direct h-u QUBO research code

This folder is intentionally kept simple.

Files:
- `run_hu_qubo.py` : main script
- `data/data_h.csv` : depth grid
- `data/data_u.csv` : velocity grid

The script:
- reads the `h` and `u` space-time grids
- treats numeric entries as known and `?` entries as unknown
- binary-encodes the unknown `h` and `u` values
- builds the direct nonlinear `h-u` residual objective
- applies Rosenberg reduction to obtain a QUBO
- solves the QUBO by brute force for very small examples
- decodes the recovered `h` and `u` values

Run:

```bash
python hu_qubo/run_hu_qubo.py
```

Consistent options with the main repo style:

```bash
python hu_qubo/run_hu_qubo.py \
  --data-dir hu_qubo/data \
  --output-root outputs \
  --qubo-m 2 \
  --direct-sh 3.5 \
  --direct-su 3.5
```

Outputs are saved in `outputs/hu_qubo/`.
