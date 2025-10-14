# High-dimensional BEKK — Code Overview

This repository implements **high-dimensional conditional covariance modeling** and **portfolio backtesting**.  
Your **main research methods are in `core/`**. Methods under `backtest/benchmarks/` are **secondary (comparison only)**.

---

## Repository Layout

```text
High-dim-BEKK/
├─ core/                  # Main methods (research contributions)
├─ backtest/              # Rolling evaluation & metrics
│  ├─ bekk_pipeline.py    # Main pipeline: rolling Σ_t + GMV backtests
│  └─ benchmarks/         # Comparison methods (secondary)
├─ data/                  # Raw datasets + optional loaders/preprocessing
└─ utils/                 # Generic helpers (device, sync, hashing, etc.)
```

---

## Module Guide

### 1) `core/` (Main Methods)

- **`matrix_ops.py`**  
  Foundational linear-algebra helpers:  
  - `vech` / `vech_to_mat` (upper-triangular vectorization of symmetric matrices and its inverse)  
  - `construct_D_N` (mapping consistent with `vech`)  
  - `project_to_psd` (stable PSD projection + symmetrization)

- **`vech_utils.py`**  
  VECH/BEKK-related dataset builders and utilities (e.g., truncated datasets, return-to-design transformations).

- **`kron_transform.py`**  
  Kronecker and permutation transforms used by BEKK/VECH:  
  - `build_kron_indices`, `transformation_kron_torch`, `permutation_torch`

- **`projection.py`**  
  Constraints/regularizers for robustness and structure:  
  - Hard truncation, row top-k pruning, approximate quantiles, nuclear-norm tools

- **`optimization.py`**  
  Core solvers tying objectives to structure operators:  
  - `padding`, `fista_algorithm`, `_power_L_op`  
  - Integrates `kron_transform` / `projection` / `matrix_ops`

- **`estimation.py`**  
  Parameter/path estimation & recursion:  
  - e.g., `estimate_A_dict_from_R_list`, `estimate_K_dict_from_Psi_list`,  
    `compute_sigma_sequence_from_bekk_arch`, `omega_from_B_est`  
  - Produces one-step-ahead covariance paths $\,\Sigma_t\,$ consumed by backtests

> **Summary:** `core/` holds the **research implementation** (structure + regularization + optimization + estimation). Its output is $\,\Sigma_t\,$ used directly by the backtesting layer.

---

### 2) `backtest/` (Backtests & Evaluation)

- **`bekk_pipeline.py`** — Main pipeline  
  - Calls `core/` methods to produce rolling (expanding/fixed window) **one-step-ahead $\,\Sigma_t\,$**  
  - Provides **GMV** backtests and metrics (AV/SD/IR), often via a robust
    `gmv_weights_from_cov_torch` (adaptive diagonal loading + `pinv` fallback)

- **`benchmarks/`** — Comparison methods (secondary)  
  - **`ccc_dcc.py`**  
    - Per-asset **GARCH(1,1)** (EBE)  
    - Empirical correlation + **nonlinear shrinkage** (QuEST/RIE proxy)  
    - **DCC-NL** (corrected recursion) for $R_t$  
    - Rolling backtests: `rolling_backtest_cccnl` / `rolling_backtest_dccnl`  
      ($\Sigma_t = D_t\,R_t\,D_t \;\rightarrow\;$ GMV $\rightarrow$ AV/SD/IR)  
    - Note: `rolling_backtest_cccnl` is aligned with the `dccnl` workflow; the legacy `ebe_ccc_pipeline` was removed.
  - **`factor_garch.py`**  
    - **Latent (PCA) factor** route (Ahn–Horenstein ER/GR for $K$; `_pca_factors_from_returns`)  
    - Factor-level **GARCH(1,1)+CCC** → factor variances $H$ & correlation $\Gamma$  
    - Residual covariance **adaptive thresholding** to ensure PSD ($\Sigma_u$)  
    - Assemble observed-space $\Sigma_y(t)$ as **$B\,D_t\,\Gamma\,D_t\,B^\top + \Sigma_u$**  
    - Rolling backtest: `rolling_backtest_factor_garch_latent(...)`  
      (same signature and rolling style as `dccnl`)

> **Summary:** items in `backtest/benchmarks/` are **for comparison only** and intentionally not at the same level as your main methods.

---

### 3) `data/`

This folder also contains the **raw datasets used in the real-data experiments**:

- `100_Portfolios_daily.csv` — raw data for the 100 size–investment portfolios (as downloaded).  
- `17_Industry_Portfolios.csv` — raw data for the 17 industry portfolios (as downloaded).

These files are read by the loaders and then processed during experiments (date alignment, centering, and, when needed, percentage→decimal conversion).

---

### 4) `utils/`

Generic helpers: device selection (e.g., `get_device()`), CUDA/MPS sync (`sync()`), hashing for caches (`hash_tensor`, `hash_obj`), and a lightweight in-memory cache.
