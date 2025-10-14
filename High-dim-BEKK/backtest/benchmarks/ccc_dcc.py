from re import VERBOSE
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import warnings
import torch
import time
from scipy.optimize import minimize, Bounds, LinearConstraint
from bekk_pipeline import gmv_weights_from_cov_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Verbosity & progress controls =====
VERBOSE = True                 # master switch for prints
GARCH_PRINT_EVERY = 200        # print every N steps inside torch GARCH
DCC_PRINT_EVERY = 50           # print every N iters inside DCC optimizer
SHOW_TOP_K_SLOWEST = 20        # number of slowest GARCH assets to display

@dataclass
class UnivariateGarchResult:
    omega: float
    alpha: float
    beta: float
    cond_var: pd.Series
    std_resid: pd.Series
    success: bool
    message: str

@dataclass
class DCCResultNL:
    alpha: float
    beta: float
    S: pd.DataFrame
    success: bool
    message: str
    stats: Dict[str, Any]


@dataclass
class EBEOutput:
    garch_results: Dict[str, UnivariateGarchResult]
    std_resid: pd.DataFrame
    R_ccc: pd.DataFrame
    dcc_nl: Optional[DCCResultNL]
    timings: Dict[str, float]
    details: Dict[str, Any]



# ==================== GARCH(1,1) QML ====================
def _fit_garch11_torch(y: pd.Series, scale: float = 100.0, max_steps: int = 2000, lr: float = 0.05) -> UnivariateGarchResult:
    y_np = y.values.astype(float)
    T = y_np.shape[0]
    # scale for stability
    y_scaled = torch.tensor(y_np * scale, dtype=torch.float64, device=device)

    p_omega = torch.tensor([-2.0], dtype=torch.float64, device=device, requires_grad=True)
    p_alpha_raw = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
    p_beta_raw = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([p_omega, p_alpha_raw, p_beta_raw], lr=lr)

    def transform_params():
        omega = torch.exp(p_omega)
        ab = torch.softmax(torch.stack([p_alpha_raw, p_beta_raw]), dim=0)
        eps_c = 1e-3
        alpha = (1.0 - eps_c) * ab[0]
        beta  = (1.0 - eps_c) * ab[1]
        return omega, alpha, beta

    def neg_ll():
        omega, alpha, beta = transform_params()
        eps = y_scaled
        h = torch.empty(T, dtype=torch.float64, device=device)
        h0 = omega / torch.clamp(1.0 - (alpha + beta), min=1e-6)
        h[0] = torch.clamp(h0, min=1e-10)
        for t in range(1, T):
            h[t] = omega + alpha * eps[t-1] * eps[t-1] + beta * h[t-1]
            h[t] = torch.clamp(h[t], min=1e-12)
        nll = 0.5 * torch.sum(torch.log(h) + eps * eps / h)
        pen = 1e3 * torch.relu(alpha + beta - 0.998)
        return nll + pen

    last_loss = None
    for _ in range(max_steps):
        opt.zero_grad()
        loss = neg_ll()
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if VERBOSE and (_ % GARCH_PRINT_EVERY == 0):
            print(f"[GARCH torch] step={_}, loss={last_loss:.4f}")

    omega, alpha, beta = transform_params()
    with torch.no_grad():
        eps = y_scaled
        h = torch.empty(T, dtype=torch.float64, device=device)
        h0 = omega / torch.clamp(1.0 - (alpha + beta), min=1e-6)
        h[0] = torch.clamp(h0, min=1e-10)
        for t in range(1, T):
            h[t] = omega + alpha * eps[t-1] * eps[t-1] + beta * h[t-1]
            h[t] = torch.clamp(h[t], min=1e-12)
        cond_var_np = (h / (scale**2)).detach().cpu().numpy()
        mu = 0.0
        std_resid_np = (y_np - mu) / np.sqrt(cond_var_np)

    cond_var = pd.Series(cond_var_np, index=y.index)
    std_resid = pd.Series(std_resid_np, index=y.index)
    _sync()
    return UnivariateGarchResult(
        omega=float((omega / (scale**2)).item()),
        alpha=float(alpha.item()),
        beta=float(beta.item()),
        cond_var=cond_var,
        std_resid=std_resid,
        success=True,
        message=f"torch QML done (final loss={last_loss:.3f})"
    )


def _fit_garch11_arch(y: pd.Series) -> UnivariateGarchResult:
    """Fit univariate GARCH(1,1) with Gaussian QML using 'arch'."""
    # Numerically stable scaling; fit on scaled series then scale back
    scale = 100.0
    y_scaled = (y.values.astype(float) * scale)
    am = arch_model(y_scaled, vol='GARCH', p=1, q=1, dist='t', mean='zero')
    res = am.fit(update_freq=0, disp='off', show_warning=False)
    params = res.params
    omega_scaled = float(params['omega'])
    alpha = float(params['alpha[1]'])
    beta = float(params['beta[1]'])
    cond_var_scaled = (res.conditional_volatility**2)
    cond_var = pd.Series(cond_var_scaled / (scale**2), index=y.index)
    std_resid = (y - 0.0) / np.sqrt(cond_var)
    omega = omega_scaled / (scale**2)
    return UnivariateGarchResult(
        omega=omega, alpha=alpha, beta=beta,
        cond_var=cond_var, std_resid=std_resid,
        success=True, message="arch success (scaled)"
    )

def _fit_garch11_qml_fallback(y: pd.Series) -> UnivariateGarchResult:
    """
    Simple QML (Gaussian) fallback without 'arch'.
    Model: y_t = mu + eps_t,  eps_t = sigma_t * z_t,
           sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    Constraints: omega>0, alpha>=0, beta>=0, alpha+beta<1
    """
    yv = y.values.astype(float)
    n = len(yv)
    var = np.var(yv)
    x0 = np.array([max(1e-6, 0.05*var), 0.05, 0.9])

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 1e-12 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e20
        eps = yv  # zero mean
        h = np.empty(n)
        h[0] = np.maximum(omega/(1 - max(1e-6, alpha+beta)), 1e-6)
        for t in range(1, n):
            h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]
            if not np.isfinite(h[t]) or h[t] <= 1e-20:
                return 1e20
        ll = -0.5 * np.sum(np.log(h) + eps**2 / h)
        return -ll

    bnds = Bounds([1e-12, 0.0, 0.0], [np.inf, 1.0, 1.0])
    A = np.array([[0, 1, 1]])
    lc = LinearConstraint(A, [-np.inf], [0.999])

    res = minimize(neg_ll, x0, method='SLSQP', bounds=bnds, constraints=[lc], options={'maxiter': 2000})
    omega, alpha, beta = res.x
    eps = yv
    h = np.empty(n)
    h[0] = np.maximum(omega/(1 - max(1e-6, alpha+beta)), 1e-6)
    for t in range(1, n):
        h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]
        if h[t] <= 1e-20:
            h[t] = 1e-20
    cond_var = pd.Series(h, index=y.index)
    std_resid = (y - 0.0) / np.sqrt(cond_var)
    return UnivariateGarchResult(
        omega=float(omega), alpha=float(alpha), beta=float(beta),
        cond_var=cond_var, std_resid=std_resid,
        success=res.success, message=res.message
    )


def fit_ebe_garch11(returns: pd.DataFrame, n_jobs: int = -1) -> Tuple[Dict[str, UnivariateGarchResult], pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Step 1 (EBE): Fit univariate GARCH(1,1) to each column (asset).
    """
    returns = returns.sort_index()
    cols = list(returns.columns)

    def _fit(col):
        y = returns[col].dropna()
        if len(y) < 200:
            warnings.warn(f"[{col}] has fewer than 200 observations; estimates might be unstable.")
        t0 = time.perf_counter()
        try:
            if torch.cuda.is_available():
                try:
                    res = _fit_garch11_torch(y)
                except Exception as e_torch:
                    if VERBOSE:
                        print(f"[EBE][{col}] torch GPU fit failed: {e_torch}. Falling back to 'arch'/'qml'.")
                    if _HAS_ARCH:
                        try:
                            res = _fit_garch11_arch(y)
                        except Exception as e_arch:
                            if VERBOSE:
                                print(f"[EBE][{col}] arch fallback failed: {e_arch}. Falling back to simple QML.")
                            res = _fit_garch11_qml_fallback(y)
                    else:
                        res = _fit_garch11_qml_fallback(y)
            else:
                if _HAS_ARCH:
                    try:
                        res = _fit_garch11_arch(y)
                    except Exception as e_arch:
                        if VERBOSE:
                            print(f"[EBE][{col}] arch fit failed: {e_arch}. Falling back to simple QML.")
                        res = _fit_garch11_qml_fallback(y)
                else:
                    res = _fit_garch11_qml_fallback(y)
        except Exception as e_final:
            if VERBOSE:
                print(f"[EBE][{col}] all fit methods failed: {e_final}")
            _sync()
            return col, UnivariateGarchResult(np.nan, np.nan, np.nan, pd.Series(dtype=float), pd.Series(dtype=float), False, f"error: {e_final}"), 0.0
        _sync()
        t1 = time.perf_counter()
        return col, res, (t1 - t0)

    if n_jobs == 1:
        pairs = []
        total = len(cols)
        for idx, c in enumerate(cols, 1):
            if VERBOSE:
                print(f"[EBE] ({idx}/{total}) fitting GARCH for asset: {c}")
            pairs.append(_fit(c))
    else:
        pairs = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_fit)(c) for c in cols)
    results = {k: v for (k, v, s) in pairs}
    per_asset_seconds = {k: s for (k, v, s) in pairs}
    params_df = pd.DataFrame(
        {
            "omega": {k: v.omega for k, v in results.items()},
            "alpha": {k: v.alpha for k, v in results.items()},
            "beta":  {k: v.beta for k, v in results.items()},
            "success": {k: v.success for k, v in results.items()},
            "message": {k: v.message for k, v in results.items()},
            "seconds": per_asset_seconds,
        }
    )
    # Guard: ensure at least one asset produced standardized residuals
    nonempty = {k: v for k, v in results.items() if isinstance(v, UnivariateGarchResult) and hasattr(v, "std_resid") and getattr(v.std_resid, "size", 0) > 0}
    if len(nonempty) == 0:
        raise RuntimeError("All per-asset GARCH fits failed to produce standardized residuals. "
                           "This can happen if GPU/torch encounters a device-specific error. "
                           "Try setting CUDA_VISIBLE_DEVICES='' to force CPU, or keep VERBOSE=TRUE to inspect per-asset fallbacks.")
    std_resid_df = pd.concat({k: v.std_resid for k, v in nonempty.items()}, axis=1).dropna(how='all')
    return results, std_resid_df, per_asset_seconds, params_df


def pairwise_empirical_corr(std_resid_df: pd.DataFrame, min_pair: int = 100) -> pd.DataFrame:
    return std_resid_df.corr(min_periods=min_pair)



# ==================== Nonlinear eigenvalue shrinkage (QuEST-style) ====================

def _rie_bulk_clip(eigs: np.ndarray, c: float) -> np.ndarray:
    eigs = np.asarray(eigs, dtype=float)
    lam_plus = (1.0 + np.sqrt(max(c, 1e-12))) ** 2
    lam_minus = (1.0 - np.sqrt(max(c, 1e-12))) ** 2
    bulk_mask = (eigs >= lam_minus) & (eigs <= lam_plus)
    if not np.any(bulk_mask):
        return eigs
    bulk_mean = float(np.mean(eigs[bulk_mask]))
    shrunk = eigs.copy()
    shrunk[bulk_mask] = bulk_mean
    return shrunk


def _quest_inverse_population_eigs(sample_eigs: np.ndarray, n: int, T: int) -> np.ndarray:
    lam = np.asarray(sample_eigs, dtype=float)
    lam = np.maximum(lam, 1e-15)
    lam_sorted = np.sort(lam)
    n = int(n)
    T = int(T)
    c = float(n) / float(T)
    if n > 1:
        gaps = np.diff(lam_sorted)
        med_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else np.mean(lam_sorted) * 1e-3
    else:
        med_gap = lam_sorted[0] * 1e-3
    eta = float(np.clip(0.5 * med_gap, 1e-6, 1e-1 * max(1.0, np.max(lam_sorted))))
    ieta = 1j * eta
    diff = lam_sorted[None, :] - lam_sorted[:, None] - ieta
    m_vals_sorted = np.sum(1.0 / diff, axis=1) / float(n)
    order = np.argsort(lam)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    m_vals = m_vals_sorted[inv_order]
    denom = (1.0 - c + c * lam * m_vals)
    tau_hat = lam / np.maximum(np.abs(denom) ** 2, 1e-15)
    tau_hat = np.maximum(tau_hat, 1e-15)
    scale = float(n) / float(np.sum(tau_hat))
    return tau_hat * scale


def _quest_shrink_eigs(sample_eigs: np.ndarray, n: int, T: int, max_iter: int = 0, tol: float = 0.0) -> np.ndarray:
    lam = np.asarray(sample_eigs, dtype=float)
    lam = np.maximum(lam, 1e-15)
    lam_sorted = np.sort(lam)
    n = int(n); T = int(T)
    c = float(n) / float(T)
    if n > 1:
        gaps = np.diff(lam_sorted)
        pos = gaps[gaps > 0]
        med_gap = np.median(pos) if pos.size > 0 else np.mean(lam_sorted) * 1e-3
    else:
        med_gap = lam_sorted[0] * 1e-3
    eta = float(np.clip(0.5 * med_gap, 1e-6, 1e-1 * max(1.0, float(np.max(lam_sorted)))))
    ieta = 1j * eta
    diff = lam_sorted[None, :] - lam_sorted[:, None] - ieta
    m_vals_sorted = np.sum(1.0 / diff, axis=1) / float(n)
    order = np.argsort(lam)
    inv_order = np.empty_like(order); inv_order[order] = np.arange(len(order))
    m_vals = m_vals_sorted[inv_order]
    denom = (1.0 - c + c * lam * m_vals)
    denom_sq = np.abs(denom) ** 2
    d_hat = lam / np.maximum(denom_sq, 1e-15)
    def _pav_increasing(x: np.ndarray) -> np.ndarray:
        x = x.astype(float).copy()
        n_ = x.size
        g = x.copy(); w = np.ones(n_, dtype=float)
        i = 0
        while i < n_ - 1:
            if g[i] > g[i + 1]:
                s = g[i] * w[i] + g[i + 1] * w[i + 1]
                w[i] = w[i] + w[i + 1]
                g[i] = s / w[i]
                g = np.delete(g, i + 1)
                w = np.delete(w, i + 1)
                i = max(i - 1, 0)
            else:
                i += 1
        out = np.empty(n_, dtype=float); idx = 0
        for val, ww in zip(g, w):
            k = int(ww); out[idx: idx + k] = val; idx += k
        return out
    try:
        d_hat = _pav_increasing(d_hat)
    except Exception:
        pass
    d_hat = np.maximum(d_hat, 1e-15)
    scale = float(n) / float(np.sum(d_hat))
    return d_hat * scale


def nonlinear_shrinkage_corr_quest(R: pd.DataFrame, T: int, method: str = "quest") -> pd.DataFrame:
    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a pandas DataFrame")
    Rv = np.asarray(R.values, dtype=float)
    Rv = 0.5 * (Rv + Rv.T)
    eigvals, eigvecs = np.linalg.eigh(Rv)
    n = eigvals.shape[0]
    if T is None or T <= 0:
        raise ValueError("T must be a positive integer for nonlinear shrinkage.")
    tau_hat = _quest_inverse_population_eigs(np.maximum(eigvals, 1e-15), n, int(T))
    if method == "rie":
        shrunk_eigs = _rie_bulk_clip(np.maximum(eigvals, 1e-15), n / float(T))
    else:
        shrunk_lw = _quest_shrink_eigs(np.maximum(eigvals, 1e-15), n, int(T))
        eps = 0.0
        shrunk_eigs = (1.0 - eps) * shrunk_lw + eps * tau_hat
    R_shrunk = (eigvecs @ np.diag(shrunk_eigs) @ eigvecs.T)
    R_shrunk = 0.5 * (R_shrunk + R_shrunk.T)
    d = np.sqrt(np.clip(np.diag(R_shrunk), 1e-12, None))
    R_shrunk = R_shrunk / (d[:, None] * d[None, :])
    np.fill_diagonal(R_shrunk, 1.0)
    return pd.DataFrame(R_shrunk, index=R.index, columns=R.columns)

# ---------------------------------------------

def fit_corrected_dcc_nl(std_resid_df: pd.DataFrame,
                         alpha0: float = 0.02,
                         beta0: float = 0.97,
                         maxiter: int = 500,
                         tol: float = 1e-3) -> DCCResultNL:
    """
    Corrected DCC (Aielli, 2013) with nonlinear-shrunk long-run correlation C (a.k.a. DCC-NL).
    Steps:
      1) Sample correlation of standardized residuals (TRAIN).
      2) Nonlinear eigenvalue shrinkage (QuEST/LW) -> C (PSD, unit diagonal).
      3) Optimize (a,b) under a>=0, b>=0, a+b<1 with Gaussian QML objective as in fit_corrected_dcc.
    Returns DCCResultNL including diagnostics of C.
    """
    U_np = std_resid_df.dropna().values
    Tn, m = U_np.shape
    if Tn < 200:
        warnings.warn("DCC-NL step with <200 observations may be unstable.")

    U = torch.tensor(U_np, dtype=torch.float64, device=device)
    Uc = U - U.mean(dim=0, keepdim=True)
    cov = (Uc.T @ Uc) / (Uc.shape[0] - 1)
    std = torch.sqrt(torch.diag(cov)).clamp_min(1e-12)
    S_sample = (cov / (std[:, None] * std[None, :])).detach().cpu().numpy()
    S_df_sample = pd.DataFrame(S_sample, index=std_resid_df.columns, columns=std_resid_df.columns)

    C_df = nonlinear_shrinkage_corr_quest(S_df_sample, T=Tn, method="quest")
    C_np = C_df.values
    evals_np = np.linalg.eigvalsh(C_np)
    evals_min = float(np.min(evals_np))
    evals_max = float(np.max(evals_np))

    C = torch.tensor(C_np, dtype=torch.float64, device=device)
    a = torch.tensor(alpha0, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor(beta0,  dtype=torch.float64, device=device, requires_grad=True)

    def dcc_nll_nl(a_, b_):
        if (a_ < 0) or (b_ < 0) or (a_ + b_ >= 0.999):
            return torch.tensor(1e20, dtype=torch.float64, device=device)
        Q = C.clone()
        nll = torch.tensor(0.0, dtype=torch.float64, device=device)
        I = torch.eye(m, dtype=torch.float64, device=device)
        for t in range(Tn):
            q_diag = torch.diag(torch.diag(Q))
            qstar_inv_sqrt = torch.linalg.inv(torch.sqrt(q_diag))
            Rt = qstar_inv_sqrt @ Q @ qstar_inv_sqrt
            evals, evecs = torch.linalg.eigh(Rt)
            evals = torch.clamp(evals, min=1e-12)
            Rt = (evecs @ torch.diag(evals) @ evecs.T)
            Rt = Rt + 1e-12 * I
            Rt.fill_diagonal_(1.0)
            ut = U[t]
            invR = torch.linalg.inv(Rt)
            sign, logdet = torch.linalg.slogdet(Rt)
            if sign.le(0):
                return torch.tensor(1e20, dtype=torch.float64, device=device)
            nll = nll + 0.5 * (logdet + ut @ invR @ ut)
            Qt_star_half = torch.sqrt(q_diag)
            outer = Qt_star_half @ (torch.ger(ut, ut)) @ Qt_star_half
            Q = (1.0 - a_ - b_) * C + a_ * outer + b_ * Q
        return nll

    opt = torch.optim.Adam([a, b], lr=0.05)
    iters = 0
    best = (float('inf'), None, None)
    for k in range(maxiter):
        opt.zero_grad()
        loss = dcc_nll_nl(a, b)
        loss.backward()
        opt.step()
        iters += 1
        with torch.no_grad():
            a.clamp_(min=0.0)
            b.clamp_(min=0.0)
            s = a + b
            if s >= 0.999:
                a.mul_(0.999 / (s + 1e-12))
                b.mul_(0.999 / (s + 1e-12))
        if loss.item() < best[0]:
            best = (loss.item(), float(a.item()), float(b.item()))
        if VERBOSE and (iters % DCC_PRINT_EVERY == 0):
            print(f"[DCC-NL] iter={iters}, loss={loss.item():.4f}, a={float(a.item()):.5f}, b={float(b.item()):.5f}")
        if loss.item() <= tol:
            break

    a_hat = best[1] if best[1] is not None else float('nan')
    b_hat = best[2] if best[2] is not None else float('nan')
    final_nll = float(dcc_nll_nl(a, b).detach().cpu().item())

    stats = {
        "min_eig_C": evals_min,
        "max_eig_C": evals_max,
        "used_nonlinear_shrinkage_for_C": True,
        "a_hat": a_hat,
        "b_hat": b_hat,
        "nll": final_nll,
        "iters": iters,
        "T": int(Tn),
        "N": int(m),
    }
    _sync()
    return DCCResultNL(
        alpha=a_hat,
        beta=b_hat,
        S=C_df,
        success=np.isfinite(a_hat) and np.isfinite(b_hat),
        message=f"DCC-NL done; nll={final_nll:.3f}; iters={iters}",
        stats=stats,
    )


@torch.no_grad()
def _dcc_last_corr_from_train(std_resid_df: pd.DataFrame, a: float, b: float, C: np.ndarray) -> np.ndarray:
    U_np = std_resid_df.dropna().values
    Tn, m = U_np.shape
    U = torch.tensor(U_np, dtype=torch.float64, device=device)
    C_t = torch.tensor(C, dtype=torch.float64, device=device)
    Q = C_t.clone()
    I = torch.eye(m, dtype=torch.float64, device=device)
    for t in range(Tn):
        q_diag = torch.diag(torch.diag(Q)).clamp_min(1e-12)
        qstar_inv_sqrt = torch.linalg.inv(torch.sqrt(q_diag))
        Rt = qstar_inv_sqrt @ Q @ qstar_inv_sqrt
        evals, evecs = torch.linalg.eigh(Rt)
        evals = torch.clamp(evals, min=1e-12)
        Rt = (evecs @ torch.diag(evals) @ evecs.T)
        Rt = Rt + 1e-12 * I
        Rt.fill_diagonal_(1.0)
        ut = U[t]
        Qt_star_half = torch.sqrt(q_diag)
        outer = Qt_star_half @ (torch.ger(ut, ut)) @ Qt_star_half
        Q = (1.0 - a - b) * C_t + a * outer + b * Q
    q_diag = torch.diag(torch.diag(Q)).clamp_min(1e-12)
    qstar_inv_sqrt = torch.linalg.inv(torch.sqrt(q_diag))
    Rt = qstar_inv_sqrt @ Q @ qstar_inv_sqrt
    evals, evecs = torch.linalg.eigh(Rt)
    evals = torch.clamp(evals, min=1e-12)
    Rt = (evecs @ torch.diag(evals) @ evecs.T)
    Rt = Rt + 1e-12 * I
    Rt.fill_diagonal_(1.0)
    return Rt.detach().cpu().numpy()


def rolling_backtest_cccnl(
    raw_returns: pd.DataFrame,
    min_train: int = 200,
    mode: str = "expanding",
    window: Optional[int] = None,
    reestimate_step: int = 1,
    annualization: int = 252,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Rolling OOS backtest for CCC-NL Σ_t and GMV.
    At each OOS time t, fit EBE GARCH on TRAIN, compute CCC correlation (with nonlinear shrinkage) on TRAIN standardized residuals,
    compute one-step-ahead h_next for each asset, build Σ_t = D_t R_t D_t, and compute GMV portfolio return.
    """
    df = raw_returns.sort_index().copy()
    assets_all = list(df.columns)
    Tn = len(df)
    if Tn < min_train + 1:
        raise ValueError(f"Not enough observations: T={Tn}, need >= {min_train+1}")
    if mode not in ("expanding", "fixed"):
        raise ValueError("mode must be 'expanding' or 'fixed'")
    if mode == "fixed":
        if window is None:
            raise ValueError("window must be provided for fixed mode")
        if window < min_train:
            raise ValueError("window must be >= min_train")

    oos_dates: list = []
    port_lr: list = []
    ew_lr: list = []

    total_core_seconds: float = 0.0
    core_calls: int = 0

    last_garch: Optional[Dict[str, UnivariateGarchResult]] = None
    last_R: Optional[pd.DataFrame] = None
    last_assets: Optional[list] = None

    for t in range(min_train, Tn):
        # TRAIN slice
        if mode == "expanding":
            train_df = df.iloc[:t]
        else:
            train_df = df.iloc[max(0, t - window):t]
        mu_tr = train_df.mean(axis=0)
        tr_centered = (train_df - mu_tr).astype(float)

        need_refit = ((t - min_train) % int(max(1, reestimate_step)) == 0) or (last_garch is None)
        if need_refit:
            if verbose:
                print(f"[ROLL-CCC-NL] t={t}/{Tn-1} | TRAIN size={len(tr_centered)} | re-fit EBE + CCC-NL")
            t0_core = time.perf_counter()
            garch_results, std_resid, _, _ = fit_ebe_garch11(tr_centered, n_jobs=n_jobs)
            R_ccc = pairwise_empirical_corr(std_resid, min_pair=100)
            T_eff = int(std_resid.shape[0])
            R_ccc_shrunk = nonlinear_shrinkage_corr_quest(R_ccc, T=T_eff, method="quest")
            common_assets = sorted(set(garch_results.keys()).intersection(R_ccc_shrunk.columns).intersection(assets_all))
            R_sub = R_ccc_shrunk.loc[common_assets, common_assets].astype(float).copy()
            np.fill_diagonal(R_sub.values, 1.0)
            R = torch.tensor(R_sub.values, dtype=torch.float64, device=device)
            evals, evecs = torch.linalg.eigh(R)
            evals = torch.clamp(evals, min=1e-12)
            R_psd = evecs @ torch.diag(evals) @ evecs.T
            d = torch.sqrt(torch.diag(R_psd))
            R_psd = R_psd / (d[:, None] * d[None, :])
            last_garch, last_R, last_assets = garch_results, pd.DataFrame(R_psd.detach().cpu().numpy(), index=common_assets, columns=common_assets), common_assets
            _sync()
            t1_core = time.perf_counter()
            total_core_seconds += (t1_core - t0_core)
            core_calls += 1
        else:
            garch_results, R_sub, common_assets = last_garch, last_R, last_assets

        if (common_assets is None) or (len(common_assets) == 0):
            continue

        h_next = np.empty(len(common_assets), dtype=float)
        valid_mask = np.ones(len(common_assets), dtype=bool)
        for j, c in enumerate(common_assets):
            gr = garch_results.get(c)
            if gr is None or not np.isfinite(gr.omega) or not np.isfinite(gr.alpha) or not np.isfinite(gr.beta) or (len(gr.cond_var) == 0):
                valid_mask[j] = False
                h_next[j] = np.nan
                continue
            omega = max(1e-12, float(gr.omega))
            alpha = max(0.0, float(gr.alpha))
            beta  = max(0.0, float(gr.beta))
            rho = alpha + beta
            if rho >= 0.999:
                alpha *= 0.999 / (rho + 1e-12)
                beta  *= 0.999 / (rho + 1e-12)
            eps_last = float(tr_centered[c].iloc[-1])
            h_last   = float(np.maximum(1e-12, gr.cond_var.iloc[-1]))
            h_next[j] = max(1e-12, omega + alpha * (eps_last ** 2) + beta * h_last)

        use_assets = [a for a, m in zip(common_assets, valid_mask) if m]
        if len(use_assets) == 0:
            continue
        R_use = R_sub.loc[use_assets, use_assets].values
        h_use = h_next[valid_mask]
        D = np.diag(np.sqrt(h_use))
        Sigma_t = torch.tensor(D @ R_use @ D, dtype=torch.float64, device=device)

        r_t = torch.tensor(df.loc[df.index[t], use_assets].values.astype(float), dtype=torch.float64, device=device)
        w_t = gmv_weights_from_cov_torch(Sigma_t)
        pr_t = float((w_t * r_t).sum().item())
        ew_t = float(r_t.mean().item())

        oos_dates.append(df.index[t])
        port_lr.append(pr_t)
        ew_lr.append(ew_t)

    port_lr_t = torch.tensor(port_lr, dtype=torch.float64, device=device)
    ew_lr_t   = torch.tensor(ew_lr,   dtype=torch.float64, device=device)

    AV_gmv = float((port_lr_t.mean() * annualization).item())
    SD_gmv = float((port_lr_t.std(unbiased=True) * np.sqrt(annualization)).item())
    IR_gmv = AV_gmv / SD_gmv if SD_gmv > 0 else float("nan")

    AV_ew = float((ew_lr_t.mean() * annualization).item())
    SD_ew = float((ew_lr_t.std(unbiased=True) * np.sqrt(annualization)).item())
    IR_ew = AV_ew / SD_ew if SD_ew > 0 else float("nan")

    avg_core_seconds = float(total_core_seconds / core_calls) if core_calls > 0 else float("nan")
    if verbose:
        print(f"[TIME] Core estimation avg per refit: {avg_core_seconds:.2f} s over {core_calls} refits (total {total_core_seconds:.2f} s)")

    return {
        "dates": oos_dates,
        "T_eff": len(oos_dates),
        "assets": use_assets if len(oos_dates) > 0 else [],
        "GMV": {"AV": AV_gmv, "SD": SD_gmv, "IR": IR_gmv},
        "EW":  {"AV": AV_ew,  "SD": SD_ew,  "IR": IR_ew},
        "_series": {"gmv": port_lr, "ew": ew_lr},
        "config": {
            "mode": mode,
            "min_train": min_train,
            "window": window,
            "reestimate_step": reestimate_step,
            "annualization": annualization,
        },
        "avg_core_seconds": avg_core_seconds,
        "total_core_seconds": float(total_core_seconds),
        "core_refits": int(core_calls),
    }


def rolling_backtest_dccnl(
    raw_returns: pd.DataFrame,
    min_train: int = 200,
    mode: str = "expanding",
    window: Optional[int] = None,
    reestimate_step: int = 1,
    annualization: int = 12,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Rolling OOS backtest for DCC-NL Σ_t and GMV.
    At each OOS time t, fit EBE GARCH on TRAIN, fit DCC-NL on TRAIN standardized residuals to get (a,b,C),
    compute one-step-ahead R_t from TRAIN via corrected DCC recursion, then build Σ_t = D_t R_t D_t.
    """
    df = raw_returns.sort_index().copy()
    assets_all = list(df.columns)
    Tn = len(df)
    if Tn < min_train + 1:
        raise ValueError(f"Not enough observations: T={Tn}, need >= {min_train+1}")
    if mode not in ("expanding", "fixed"):
        raise ValueError("mode must be 'expanding' or 'fixed'")
    if mode == "fixed":
        if window is None:
            raise ValueError("window must be provided for fixed mode")
        if window < min_train:
            raise ValueError("window must be >= min_train")

    oos_dates: list = []
    port_lr: list = []

    total_core_seconds: float = 0.0
    core_calls: int = 0

    last_garch: Optional[Dict[str, UnivariateGarchResult]] = None
    last_R: Optional[pd.DataFrame] = None
    last_assets: Optional[list] = None

    for t in range(min_train, Tn):
        if mode == "expanding":
            train_df = df.iloc[:t]
        else:
            train_df = df.iloc[max(0, t - window):t]
        mu_tr = train_df.mean(axis=0)
        tr_centered = (train_df - mu_tr).astype(float)

        need_refit = ((t - min_train) % int(max(1, reestimate_step)) == 0) or (last_garch is None)
        if need_refit:
            if verbose:
                print(f"[ROLL-DCC-NL] t={t}/{Tn-1} | TRAIN size={len(tr_centered)} | re-fit EBE + DCC-NL")
            t0_core = time.perf_counter()
            garch_results, std_resid, _, _ = fit_ebe_garch11(tr_centered, n_jobs=n_jobs)
            dcc_nl_res = fit_corrected_dcc_nl(std_resid)
            C_df = dcc_nl_res.S
            a_hat, b_hat = float(dcc_nl_res.alpha), float(dcc_nl_res.beta)
            Rt_np = _dcc_last_corr_from_train(std_resid, a_hat, b_hat, C_df.values)
            common_assets = sorted(set(garch_results.keys()).intersection(C_df.columns).intersection(assets_all))
            R_sub = pd.DataFrame(Rt_np, index=C_df.index, columns=C_df.columns).loc[common_assets, common_assets].astype(float).copy()
            np.fill_diagonal(R_sub.values, 1.0)
            last_garch, last_R, last_assets = garch_results, R_sub, common_assets
            _sync()
            t1_core = time.perf_counter()
            total_core_seconds += (t1_core - t0_core)
            core_calls += 1
        else:
            garch_results, R_sub, common_assets = last_garch, last_R, last_assets

        if (common_assets is None) or (len(common_assets) == 0):
            continue

        h_next = np.empty(len(common_assets), dtype=float)
        valid_mask = np.ones(len(common_assets), dtype=bool)
        for j, c in enumerate(common_assets):
            gr = garch_results.get(c)
            if gr is None or not np.isfinite(gr.omega) or not np.isfinite(gr.alpha) or not np.isfinite(gr.beta) or (len(gr.cond_var) == 0):
                valid_mask[j] = False
                h_next[j] = np.nan
                continue
            omega = max(1e-12, float(gr.omega))
            alpha = max(0.0, float(gr.alpha))
            beta  = max(0.0, float(gr.beta))
            rho = alpha + beta
            if rho >= 0.999:
                alpha *= 0.999 / (rho + 1e-12)
                beta  *= 0.999 / (rho + 1e-12)
            eps_last = float(tr_centered[c].iloc[-1])
            h_last   = float(np.maximum(1e-12, gr.cond_var.iloc[-1]))
            h_next[j] = max(1e-12, omega + alpha * (eps_last ** 2) + beta * h_last)

        use_assets = [a for a, m in zip(common_assets, valid_mask) if m]
        if len(use_assets) == 0:
            continue
        R_use = R_sub.loc[use_assets, use_assets].values
        h_use = h_next[valid_mask]
        D = np.diag(np.sqrt(h_use))
        Sigma_t = torch.tensor(D @ R_use @ D, dtype=torch.float64, device=device)

        r_t = torch.tensor(df.loc[df.index[t], use_assets].values.astype(float), dtype=torch.float64, device=device)
        w_t = gmv_weights_from_cov_torch(Sigma_t)
        pr_t = float((w_t * r_t).sum().item())

        oos_dates.append(df.index[t])
        port_lr.append(pr_t)

    port_lr_t = torch.tensor(port_lr, dtype=torch.float64, device=device)
    AV = float((port_lr_t.mean() * annualization).item())
    SD = float((port_lr_t.std(unbiased=True) * np.sqrt(annualization)).item())
    IR = AV / SD if SD > 0 else float("nan")
    avg_core_seconds = float(total_core_seconds / core_calls) if core_calls > 0 else float("nan")

    return {
        "dates": oos_dates,
        "T_eff": len(oos_dates),
        "assets": last_assets if len(oos_dates) > 0 else [],
        "GMV": {"AV": AV, "SD": SD, "IR": IR},
        "avg_core_seconds": avg_core_seconds,
        "total_core_seconds": float(total_core_seconds),
        "core_refits": int(core_calls),
    }