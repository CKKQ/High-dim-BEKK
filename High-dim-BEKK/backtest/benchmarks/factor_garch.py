import pandas as pd
import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional, Any
from bekk_pipeline import gmv_weights_from_cov_torch
from ccc_dcc import fit_ebe_garch11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

@dataclass
class UnivariateGarchResult:
    omega: float
    alpha: float
    beta: float
    cond_var: pd.Series
    std_resid: pd.Series
    success: bool
    message: str

# ==================== Latent (PCA) Factor-GARCH ====================

from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict

@dataclass
class FactorGARCHLatentConfig:
    n_factors: int = 3            # number of latent factors (K); ignored if auto_select_k=True
    omega_mult: float = 0.3       # adaptive threshold for Σ_u (ω_T = c * sqrt(log p / T))
    annualization: int = 12       # 12 for monthly
    min_train: int = 200
    mode: str = "expanding"       # "expanding" | "fixed"
    window: Optional[int] = None
    reestimate_step: int = 1
    n_jobs: int = 1
    verbose: bool = True
    # --- ER/GR (Ahn–Horenstein, 2013) options for latent K selection ---
    auto_select_k: bool = True  # if True, choose K each window via ER/GR; else use n_factors
    k_rule: str = "er"            # "er" or "gr"
    kmax: Optional[int] = None    # cap for K search; default min(12, min(T,N)-1)
    demean: str = "both"         # "both" | "time" | "none" for pre-demeaning

# ---------- Ahn–Horenstein (2013) ER / GR selectors for the number of latent factors ----------
def _double_demean_matrix(X: np.ndarray) -> np.ndarray:
    """
    Double-demean a T x N matrix: X - col_means - row_means + grand_mean.
    Returns a copy (float64).
    """
    X = np.asarray(X, dtype=float)
    row_means = X.mean(axis=1, keepdims=True)
    col_means = X.mean(axis=0, keepdims=True)
    grand_mean = float(X.mean())
    return (X - col_means) - (row_means - grand_mean)

def select_num_factors_er(df_returns: pd.DataFrame,
                          kmax: int | None = None,
                          demean: str = "both") -> int:
    """
    Ahn–Horenstein (2013) Eigenvalue Ratio (ER) selector for latent K.
    df_returns: T x N (decimal returns). ER(k)=mu_k/mu_{k+1} (k>=1) with a mock mu_0 allowing r=0.
    """
    if not isinstance(df_returns, pd.DataFrame):
        raise ValueError("df_returns must be a DataFrame (T x N)")
    X = np.asarray(df_returns.values, dtype=float)
    Tn, N = X.shape
    m = int(min(Tn, N))
    if m < 2:
        return 0
    if demean == "both":
        Xd = _double_demean_matrix(X)
    elif demean == "time":
        Xd = X - X.mean(axis=0, keepdims=True)
    elif demean == "none":
        Xd = X.copy()
    else:
        raise ValueError("demean must be one of {'both','time','none'}")
    if Tn <= N:
        S = (Xd @ Xd.T) / (Tn * N)
    else:
        S = (Xd.T @ Xd) / (Tn * N)
    w = np.linalg.eigvalsh(S)
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    w = np.maximum(w, 1e-15)
    if kmax is None:
        kmax = min(12, m - 1)
    else:
        kmax = int(max(1, min(kmax, m - 1)))
    V0 = float(np.sum(w))
    mu0 = V0 / max(np.log(m), 1.0)
    er_vals = []
    er_vals.append(mu0 / max(w[0], 1e-15))
    for k in range(1, kmax + 1):
        num = w[k - 1]
        den = w[k] if (k < len(w)) else 1e-15
        er_vals.append(float(num / max(den, 1e-15)))
    return int(np.argmax(er_vals))

def select_num_factors_gr(df_returns: pd.DataFrame,
                          kmax: int | None = None,
                          demean: str = "both") -> int:
    """
    Ahn–Horenstein (2013) Growth Ratio (GR) selector for latent K.
    """
    if not isinstance(df_returns, pd.DataFrame):
        raise ValueError("df_returns must be a DataFrame (T x N)")
    X = np.asarray(df_returns.values, dtype=float)
    Tn, N = X.shape
    m = int(min(Tn, N))
    if m < 2:
        return 0
    if demean == "both":
        Xd = _double_demean_matrix(X)
    elif demean == "time":
        Xd = X - X.mean(axis=0, keepdims=True)
    elif demean == "none":
        Xd = X.copy()
    else:
        raise ValueError("demean must be one of {'both','time','none'}")
    if Tn <= N:
        S = (Xd @ Xd.T) / (Tn * N)
    else:
        S = (Xd.T @ Xd) / (Tn * N)
    w = np.sort(np.linalg.eigvalsh(S))[::-1]
    w = np.maximum(w, 1e-15)
    if kmax is None:
        kmax = min(12, m - 1)
    else:
        kmax = int(max(1, min(kmax, m - 1)))
    V_list = [float(np.sum(w))]
    V_list.extend([float(np.sum(w[k:])) for k in range(1, len(w) + 1)])
    V_arr = np.asarray(V_list, dtype=float)
    mu0 = V_arr[0] / max(np.log(m), 1.0)
    gr_vals = [mu0 / max(w[0], 1e-15)]
    for k in range(1, kmax + 1):
        if k + 1 >= len(V_arr):
            gr_vals.append(0.0)
            continue
        num = np.log(max(V_arr[k - 1] / max(V_arr[k], 1e-15), 1e-15))
        den = np.log(max(V_arr[k] / max(V_arr[k + 1], 1e-15), 1e-15))
        gr_vals.append(float(num / max(den, 1e-15)))
    return int(np.argmax(gr_vals))


@torch.no_grad()
def _adaptive_threshold_cov(U: pd.DataFrame, omega_mult: float) -> pd.DataFrame:
    """Adaptive thresholding of residual covariance (Cai & Liu, 2011 style), as used in Li et al. (2022)"""
    Uv = U.values.astype(float)
    Tn, p = Uv.shape
    Su = (Uv.T @ Uv) / float(Tn)
    Vij = np.empty((p, p), dtype=float)
    for i in range(p):
        ui = Uv[:, i]
        for j in range(i, p):
            uj = Uv[:, j]
            sij = Su[i, j]
            vhat = np.mean((ui * uj - sij) ** 2)
            Vij[i, j] = vhat
            Vij[j, i] = vhat
    omega_T = float(omega_mult) * np.sqrt(np.log(max(p, 2)) / max(Tn, 2))
    Thr = np.sqrt(np.maximum(Vij, 0.0) * omega_T)
    Su_thr = Su * (np.abs(Su) >= Thr)
    w, V = np.linalg.eigh(0.5 * (Su_thr + Su_thr.T))
    w = np.clip(w, 1e-12, None)
    Su_psd = (V @ np.diag(w) @ V.T)
    Su_psd = 0.5 * (Su_psd + Su_psd.T)
    return pd.DataFrame(Su_psd, index=U.columns, columns=U.columns)


def _pca_factors_from_returns(train_Y: pd.DataFrame, K: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PCA on TRAIN (time × assets) -> latent factor series X (T×K) and PCA loadings B0 (p×K)
    """
    Y = train_Y.sort_index().astype(float).copy()
    Yc = Y - Y.mean(axis=0)
    U, S, Vt = np.linalg.svd(Yc.values, full_matrices=False)
    K = int(min(K, len(S)))
    if K <= 0:
        raise ValueError("n_factors must be >= 1")
    X = pd.DataFrame(U[:, :K] * S[:K], index=Yc.index, columns=[f"PC{k+1}" for k in range(K)])
    B0 = pd.DataFrame(Vt.T[:, :K], index=Yc.columns, columns=X.columns)
    return X, B0


def fit_ccc_garch_on_factors(factors: pd.DataFrame, n_jobs: int = 1, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, UnivariateGarchResult]]:
    """
    Step A (Li et al., 2022): Fit CCC-GARCH on observable factors x_t.
      - Per-factor univariate GARCH(1,1) to obtain conditional variances h_{k,t}
      - Standardize to get u_{k,t} = x_{k,t} / sqrt(h_{k,t})
      - Constant correlation Gamma from standardized residuals (sample correlation with min_periods)
    """
    X = factors.sort_index().astype(float).copy()
    K = X.shape[1]
    # Fit univariate GARCH per factor (reuse existing robust pipeline)
    res = fit_ebe_garch11(X, n_jobs=n_jobs)
    garch_results, std_resid_df, _, params_df = res
    # Conditional variances from results (aligned to intersection of time)
    common_index = None
    for k, r in garch_results.items():
        if len(r.cond_var) > 0:
            common_index = r.cond_var.index if common_index is None else common_index.intersection(r.cond_var.index)
    if common_index is None:
        raise RuntimeError("Factor GARCH failed: no valid conditional variance series.")
    H = pd.DataFrame(index=common_index, columns=list(X.columns), dtype=float)
    for k in X.columns:
        r = garch_results[k]
        if len(r.cond_var) == 0:
            raise RuntimeError(f"Factor GARCH failed on factor {k}: empty cond_var.")
        H[k] = r.cond_var.reindex(common_index).astype(float).values
    # Standardized residuals for correlation
    U = pd.DataFrame(index=common_index, columns=list(X.columns), dtype=float)
    for k in X.columns:
        eps = X[k].reindex(common_index).astype(float).values
        hk = H[k].values
        U[k] = (eps / np.sqrt(np.maximum(hk, 1e-12)))
    Gamma = U.corr(min_periods=100).astype(float)
    np.fill_diagonal(Gamma.values, 1.0)
    return H, Gamma, garch_results


def estimate_factor_loadings_LSE(returns: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step B: Estimate factor loadings B via least squares Y = B X + U.
    """
    Y = returns.sort_index().astype(float).copy()
    X = factors.sort_index().astype(float).copy()
    # align by intersection of dates
    idx = Y.index.intersection(X.index)
    Y = Y.loc[idx]
    X = X.loc[idx]
    # OLS: B_hat = Y X^T (X X^T)^{-1}
    Xt = X.to_numpy().T
    XXt = Xt @ Xt.T
    B_hat = (Y.to_numpy().T @ X.to_numpy()) @ np.linalg.pinv(XXt)
    B_df = pd.DataFrame(B_hat, index=Y.columns, columns=X.columns)
    U_hat = Y - X @ B_df.T
    return B_df, U_hat


def build_Sigma_y_path_factor_garch(B: pd.DataFrame,
                                    H: pd.DataFrame,
                                    Gamma: pd.DataFrame,
                                    Sigma_u: pd.DataFrame) -> np.ndarray:
    """
    Construct Sigma_y(t) = B D_t Gamma D_t B^T + Sigma_u, where D_t = diag(h_{1t}^{1/2},...,h_{Kt}^{1/2}) * I * diag(...) (but Gamma already correlation).
    """
    idx = H.index
    p = B.shape[0]; K = B.shape[1]
    G = Gamma.values.astype(float)
    sigmas = []
    for t in range(len(idx)):
        d = np.sqrt(np.maximum(H.iloc[t].values.astype(float), 1e-12))
        D = np.diag(d)
        Sigma_x = D @ G @ D
        Sy = B.values @ Sigma_x @ B.values.T + Sigma_u.values
        # symmetrize numeric
        Sy = 0.5 * (Sy + Sy.T)
        sigmas.append(Sy)
    return np.stack(sigmas, axis=0)


def rolling_backtest_factor_garch_latent(raw_returns: pd.DataFrame,
                                                min_train: int = 200,
                                                mode: str = "expanding",
                                                window: int | None = None,
                                                reestimate_step: int = 1,
                                                annualization: int = 12,
                                                n_jobs: int = 1,
                                                verbose: bool = True
                                                ) -> Dict[str, Any]:
    """
    Rolling OOS for Factor-GARCH with PCA latent factors.
    For each rolling window:
        - Fit PCA on returns to get latent factors X and loadings B.
        - Fit GARCH(1,1) + CCC on factors X.
        - Estimate residual covariance with adaptive threshold.
        - Construct Sigma_y(t) = B D_t Gamma D_t B^T + Σ_u.
        - Compute GMV portfolio weights and returns.
    """
    df_all = raw_returns.sort_index().astype(float).copy()
    Tn = len(df_all)
    min_train = int(max(min_train, 50))
    if Tn < min_train + 1:
        raise ValueError(f"Not enough observations: T={Tn}, need >= {min_train+1}")

    oos_dates, port_lr, ew_lr = [], [], []
    last_cache = None
    total_core_seconds: float = 0.0
    core_calls: int = 0

    for t in range(min_train, Tn):
        if mode == "expanding":
            train_Y = df_all.iloc[:t]
        elif mode == "fixed":
            if window is None or window < min_train:
                raise ValueError("window must be provided and >= min_train when mode='fixed'.")
            train_Y = df_all.iloc[t - window:t]
        else:
            raise ValueError("mode must be 'expanding' or 'fixed'")

        tr_centered = train_Y - train_Y.mean(axis=0)

        refit = ((t - min_train) % max(1, reestimate_step) == 0) or (last_cache is None)
        if refit:
            if verbose:
                print(f"[Factor-GARCH (latent)] t={t}/{Tn-1} | TRAIN={len(tr_centered)} | PCA + GARCH(factors) + Σ_u")
            t0_core = time.perf_counter()
            K_sel = select_num_factors_er(tr_centered, demean="both")
            if verbose:
                print(f"[LATENT] ER selector -> K={K_sel}")
            if K_sel <= 0:
                if verbose:
                    print("[LATENT] ER suggests K=0; setting K=1 for model compatibility.")
                K_sel = 1
            X_latent, _ = _pca_factors_from_returns(tr_centered, K_sel)
            H, Gamma, _ = fit_ccc_garch_on_factors(X_latent, n_jobs=n_jobs, verbose=verbose)
            if H.index[-1] != tr_centered.index[-1]:
                H = H.reindex(tr_centered.index).ffill().bfill()
            h_next = H.iloc[-1]
            B_hat, U_hat = estimate_factor_loadings_LSE(tr_centered, X_latent)
            Sigma_u = _adaptive_threshold_cov(U_hat, omega_mult=0.3)
            use_assets = list(tr_centered.columns)
            last_cache = (B_hat, Gamma, Sigma_u, h_next, use_assets)
            t1_core = time.perf_counter()
            total_core_seconds += (t1_core - t0_core)
            core_calls += 1
        else:
            B_hat, Gamma, Sigma_u, h_next, use_assets = last_cache

        d = np.sqrt(np.maximum(h_next.values.astype(float), 1e-12))
        D = np.diag(d)
        Sigma_x = D @ Gamma.values.astype(float) @ D
        Sy = B_hat.values @ Sigma_x @ B_hat.values.T + Sigma_u.values
        Sy = 0.5 * (Sy + Sy.T)
        Sigma_t = torch.tensor(Sy, dtype=torch.float64, device=device)

        r_t = torch.tensor(df_all.iloc[t][use_assets].values.astype(float), dtype=torch.float64, device=device)
        w_t = gmv_weights_from_cov_torch(Sigma_t)
        port_lr.append(float((w_t * r_t).sum().item()))
        ew_lr.append(float(r_t.mean().item()))
        oos_dates.append(df_all.index[t])

    port_lr_t = torch.tensor(port_lr, dtype=torch.float64, device=device)
    ew_lr_t = torch.tensor(ew_lr, dtype=torch.float64, device=device)
    AV_gmv = float((port_lr_t.mean() * annualization).item())
    SD_gmv = float((port_lr_t.std(unbiased=True) * np.sqrt(annualization)).item())
    IR_gmv = AV_gmv / SD_gmv if SD_gmv > 0 else float("nan")
    AV_ew = float((ew_lr_t.mean() * annualization).item())
    SD_ew = float((ew_lr_t.std(unbiased=True) * np.sqrt(annualization)).item())
    IR_ew = AV_ew / SD_ew if SD_ew > 0 else float("nan")
    avg_core_seconds = float(total_core_seconds / core_calls) if core_calls > 0 else float("nan")

    return {
        "dates": oos_dates,
        "T_eff": len(oos_dates),
        "GMV": {"AV": AV_gmv, "SD": SD_gmv, "IR": IR_gmv},
        "EW": {"AV": AV_ew, "SD": SD_ew, "IR": IR_ew},
        "_series": {"gmv": port_lr, "ew": ew_lr},
        "config": {
            "min_train": min_train,
            "mode": mode,
            "window": window,
            "reestimate_step": reestimate_step,
            "annualization": annualization,
            "n_jobs": n_jobs,
            "verbose": verbose,
        },
        "avg_core_seconds": avg_core_seconds,
        "total_core_seconds": float(total_core_seconds),
        "core_refits": int(core_calls),
    }