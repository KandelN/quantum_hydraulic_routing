# Quantum SWE Thesis Repository

1. Classical implicit SWE solve using Newton iterations (`run_swe.py`)
2. HHL-based quantum linear solve for Jacobian systems (`hhl_solver.py`)
3. Newton-linearized QUBO solve for small linear systems (`newton_qubo_solver.py`)
4. Direct nonlinear binary/QUBO-style solve for the one-step SWE residual objective (`direct_qubo_system.py`, `direct_qubo_solver.py`)

## Structure
```text
|-- src /
| |-- run_swe . py
| |-- hhl_solver . py
| |-- newton_qubo_solver . py
| |-- direct_qubo_system . py
| |-- direct_qubo_solver . py
| +-- hu_form /
|     +-- run_hu_direct_qubo . py
|-- data /
| |-- swe /
| | |-- geometry-clipped . csv
| | +-- hydrograph . csv
| +-- hu_form /
| |-- data_h . csv
| +-- data_u . csv
+-- outputs /
```

## Python workflow

Run the SWE solver in one of four modes:

```bash
python run_swe.py --mode classical
python run_swe.py --mode hhl
python run_swe.py --mode newton_qubo --qubo-m 2
python run_swe.py --mode direct_qubo --direct-mq 2 --direct-my 2
```

Backward-compatible aliases are also accepted for the linearized QUBO mode:

```bash
python run_swe.py --mode qubo
python run_swe.py --mode qubo_with_newton
```

Outputs are written to:

```text
outputs/classical/
outputs/hhl/
outputs/newton_qubo/
outputs/direct_qubo/
```

Each run saves the same CSV format:
- `output_Q.csv`
- `output_depth.csv`
- `mass_balance.csv`

For non-classical runs, the plots also include the classical solution as a dashed reference line.
The corresponding reference CSV files are also saved in the same folder.

## Notes

- `run_swe.py` uses the uploaded geometry and hydrograph files from `data/`.
- `hhl_solver.py` keeps the older HHL workflow and expects an older Qiskit setup plus a local install of the matching `quantum_linear_solvers` package.
- `newton_qubo_solver.py` keeps the small-system Newton-linearized QUBO workflow with brute-force search.
- `direct_qubo_system.py` builds the direct nonlinear residual objective and `direct_qubo_solver.py` handles the binary encoding and brute-force minimization.
- The direct mode follows the uploaded script structure: it fixes the new inflow discharge from the hydrograph and carries the upstream depth from the previous time step while solving the remaining unknowns directly from the nonlinear residual objective.
- Both QUBO-style modes are intended for toy or very small settings. Start with low bit counts.

## Minimal Python install

```bash
pip install -r requirements.txt
```

For the HHL path, install the matching older Qiskit environment separately.


# Direct h-u QUBO package

This package builds a direct binary optimization model for a small 1D shallow-water test problem in the state variables `(h, u)`.

What it does:
- reads space-time hydraulic grids from two CSV files
- treats any non-missing grid entry as a known hydraulic state
- binary-encodes the unknown `h` and `u` values
- builds continuity and momentum residuals in polynomial form
- squares and sums the residuals to form the global objective
- applies Rosenberg reduction to obtain a quadratic QUBO matrix
- exports the QUBO matrix and bit labels for downstream solvers

What it does not do:
- it does not solve the QUBO
- it does not call QuantumAnnealing.jl
- it does not include a brute-force solver

## CSV format

Each CSV should have the layout

```text
t\x,0,100
0,1.0,1.0
100,2.5,?
```

The first row stores the spatial coordinates.
The first column stores the time coordinates.
Use `?` for unknown hydraulic states.

## Example

Run the demo:

```bash
python -m hu_qubo.demo
```

Build and save the QUBO explicitly:

```bash
python -m hu_qubo.cli \
  --h-file hu_qubo/example/data_h.csv \
  --u-file hu_qubo/example/data_u.csv \
  --output-dir hu_qubo/example/output \
  --m 2 --h-max 3.5 --u-max 3.5
```

Outputs:
- `qubo_matrix.npy`
- `qubo_matrix.csv`
- `bit_labels.txt`
- `summary.json`

## Modeling notes

The residual templates follow the direct h-u construction in your Julia prototype:
- continuity residual uses linear terms in `h` and bilinear terms in `h*u`
- momentum residual uses bilinear `h*u`, cubic `h*u^2`, quadratic `h^2`, and linear slope terms

Unknown states are detected directly from the input grids rather than through a hard-coded boundary-index rule.
