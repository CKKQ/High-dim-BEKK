# High-dimensional BEKK — Code Overview

This repository implements **high-dimensional conditional covariance modeling** and **portfolio backtesting**.  
Our **main research methods are in `core/`**. Methods under `backtest/benchmarks/` are **secondary (comparison only)**.

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
    - **`padding(...)` — Optimized padded operator**  
      Returns the **optimized padded matrix** with loss_type can be choosen as `nuclear` or `eigsplit`; for `eigsplit` the loss is $L = -\sum_{j=1}^{K} lambda_j(\cdot) + \sum_{j=K+1}^{n^2} lambda_j(\cdot)^2$.

    - **`fista_algorithm(...)` — VECH estimator**  
      Returns the **VECH coefficients** (accelerated proximal gradient on the VECH regression built from `vech(rrᵀ)`), ready for `estimation.py` to produce the path of \( \Sigma_t \).
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
    `gmv_weights_from_cov_torch` 

- **`benchmarks/`** — Comparison methods
  - **`ccc_dcc.py`**  
    - Per-asset **GARCH(1,1)** (EBE)  
    - Empirical correlation + **nonlinear shrinkage**
    - **DCC-NL** for $R_t$  
    - Rolling backtests: `rolling_backtest_cccnl` / `rolling_backtest_dccnl`  
      ($\Sigma_t = D_t\,R_t\,D_t \;\rightarrow\;$ GMV $\rightarrow$ AV/SD/IR)  
    - Note: `rolling_backtest_cccnl` is aligned with the `dccnl` workflow
  - **`factor_garch.py`**  
    - **Latent (PCA) factor** route (Ahn–Horenstein ER/GR for $K$; `_pca_factors_from_returns` for ​​factor extraction​​)
    - Factor-level **GARCH(1,1)+CCC** → factor variances $H$ & correlation $\Gamma$  
    - Residual covariance **adaptive thresholding** to ensure PSD ($\Sigma_u$)  
    - Assemble observed-space $\Sigma_y(t)$ as **$B\,D_t\,\Gamma\,D_t\,B^\top + \Sigma_u$**  
    - Rolling backtest: `rolling_backtest_factor_garch_latent(...)`  
      (same signature and rolling style as `dccnl`)

> **Summary:** items in `backtest/benchmarks/` are **for comparison only**.

---

### 3) `data/`

- **`preprocess.py`** — Data helpers  
  - Loads/cleans CSV returns (flexible `Date`, fill/dropna, **percent→decimal**), builds VECH design rows, and supports optional subsampling.

This folder also contains the **raw datasets used in the real-data experiments**:

- `100_Portfolios_daily.csv` — raw data for the 100 size–investment portfolios.  
- `17_Industry_Portfolios.csv` — raw data for the 17 industry portfolios.

These files are read by the loaders and then processed during experiments (date alignment, centering, and, when needed, percentage→decimal conversion).

---

### 4) `utils/`

Generic helpers: device selection (e.g., `get_device()`), CUDA/MPS sync (`sync()`), hashing for caches (`hash_tensor`, `hash_obj`), and a lightweight in-memory cache.
