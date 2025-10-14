import torch
import numpy as np
import time
import math
from core.matrix_ops import vech,vech_to_mat,construct_D_N,project_to_psd
from core.vech_utils import _build_Xval_from_raw_vech,create_truncated_dataset
from core.optimization import padding,fista_algorithm
from core.estimation import estimate_A_dict_from_R_list,estimate_K_dict_from_Psi_list,omega_from_B_est,compute_sigma_sequence_from_bekk_arch


def predict_vech_sigma(X, B_est):
    return X @ B_est


@torch.no_grad()
def sigma_hat_sequence(B_est, X_rows, D_N, N):
    d = N * (N + 1) // 2
    assert B_est.shape[0] == X_rows.shape[1] and B_est.shape[1] == d
    Y_hat = predict_vech_sigma(X_rows, B_est)
    sigmas = []
    for i in range(Y_hat.shape[0]):
        S = vech_to_mat(Y_hat[i], D_N, N)
        sigmas.append(S)
    return torch.stack(sigmas, dim=0)


@torch.no_grad()
def gmv_weights_from_cov_torch(S, jitter=1e-8, max_tries=6, tol=1e-14):
    """GMV weights w = S^{-1}1 / (1'S^{-1}1)"""
    S = 0.5 * (S + S.T)
    N = S.shape[0]
    ones = torch.ones(N, dtype=S.dtype, device=S.device)
    I = torch.eye(N, dtype=S.dtype, device=S.device)
    jit = float(jitter)
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(S + jit * I)
            x = torch.cholesky_solve(ones.unsqueeze(1), L).squeeze(1)
            denom = torch.sum(x)
            if torch.isfinite(denom) and abs(float(denom)) > tol:
                return x / denom
        except RuntimeError:
            pass
        jit *= 10.0

    x = torch.linalg.pinv(S + jit * I) @ ones
    denom = torch.sum(x)
    if torch.isfinite(denom) and abs(float(denom)) > tol:
        return x / denom
    

@torch.no_grad()
def backtest_equal_weight(oos_returns: torch.Tensor, annualization: int = 252):
    T_eff, N = oos_returns.shape
    w = torch.full((N,), 1.0 / N, dtype=oos_returns.dtype, device=oos_returns.device)
    port_lr = (oos_returns @ w)
    AV = port_lr.mean() * annualization
    SD = port_lr.std(unbiased=True) * torch.sqrt(torch.tensor(float(annualization), dtype=oos_returns.dtype, device=oos_returns.device))
    IR = AV / SD
    weights = w.expand(T_eff, -1).clone()
    return {
        "AV": float(AV.item()),
        "SD": float(SD.item()),
        "IR": float(IR.item()),
        "weights": weights,
        "portfolio_log_returns": port_lr,
    }

@torch.no_grad()
def backtest_gmv_from_sigma_hat(sigmas, oos_returns, annualization=252, jitter=1e-8):
    T_eff, N, N2 = sigmas.shape
    assert N == N2 and oos_returns.shape == (T_eff, N)
    weights = torch.empty((T_eff, N), dtype=sigmas.dtype, device=sigmas.device)
    port_lr = torch.empty((T_eff,), dtype=sigmas.dtype, device=sigmas.device)
    for t in range(T_eff):
        w_t = gmv_weights_from_cov_torch(sigmas[t], jitter=jitter)
        weights[t] = w_t
        port_lr[t] = (w_t * oos_returns[t]).sum()
    AV = port_lr.mean() * annualization
    SD = port_lr.std(unbiased=True) * torch.sqrt(torch.tensor(float(annualization), dtype=sigmas.dtype, device=sigmas.device))
    IR = AV / SD
    return {
        "AV": float(AV.item()),
        "SD": float(SD.item()),
        "IR": float(IR.item()),
        "weights": weights,
        "portfolio_log_returns": port_lr
    }


def bekk_backtest_from_B(B_hat: torch.Tensor,
                         returns_centered: torch.Tensor,
                         returns_raw: torch.Tensor,
                         p: int,
                         D_N: torch.Tensor,
                         sparse_level: int = 10,
                         epsilon_rule: float = 0.5,
                         Kbar: int = 5,
                         annualization: int = 252,
                         use_omega_from_B: bool = True,
                         loss_type: str = "nuclear",
                         K_user_dict: dict | None = None):
    """Pipeline: (1) get Phi_i from B_hat; (2) padding → R_i; (3) estimate K_i;
    (4) estimate A_hat per lag; (5) get Omega via B (or fallback to moment matching);
    (6) Sigma_t path; (7) GMV/1N metrics.
    K_user_dict example: {1: 2, 2: 2, 3: 1}
    """
    t_pad = 0.0
    t_estK = 0.0
    t_estA = 0.0
    t_omega = 0.0
    t_sigma = 0.0
    device_ = returns_centered.device
    dtype_ = returns_centered.dtype
    T, N = returns_centered.shape

    Psi_list = []
    _t0 = time.perf_counter()
    d = N * (N + 1) // 2
    assert B_hat.shape == (1 + p * d, d)
    for lag in range(1, p + 1):
        row_s = 1 + (lag - 1) * d
        row_e = row_s + d
        Phi_i = B_hat[row_s:row_e, :].T
        top_k_for_loss = None
        if loss_type == "eigsplit" and (K_user_dict is not None) and (lag in K_user_dict):
            top_k_for_loss = int(K_user_dict.get(lag))
        R_i =  padding(
                Phi_i,
                opt_method="adam",
                num_epochs_stage=2000,
                lr=1e-2,
                warm_start=False,
                svd_every=25,
                enforce_symmetry=False,
                sym_penalty=1e-3,
                ht_s=sparse_level, 
                ht_row_mult=4.0,
                loss_type=loss_type,
                top_k=top_k_for_loss,
            )
        try:
            if loss_type == "eigsplit" and (K_user_dict is not None):
                print(f"[BEKK][loss=eigsplit] lag={lag}: using user-specified K={top_k_for_loss}")
        except Exception:
            pass
        Psi_list.append(R_i)
    t_pad += (time.perf_counter() - _t0)
    
    _t0 = time.perf_counter()
    if (loss_type == "eigsplit") and (K_user_dict is not None) and (len(K_user_dict) > 0):
        K_hat_dict = {int(lag): int(K_user_dict[lag]) for lag in K_user_dict if 1 <= int(lag) <= p}
        missing = [lag for lag in range(1, p + 1) if lag not in K_hat_dict]
        if len(missing) > 0:
            K_hat_fallback, _ = estimate_K_dict_from_Psi_list(Psi_list, Kbar=Kbar, T=T, epsilon=epsilon_rule)
            for lag in missing:
                K_hat_dict[lag] = int(K_hat_fallback.get(lag, 1))
        t_estK += (time.perf_counter() - _t0)
        try:
            ks = ", ".join([f"K_{lag}={K_hat_dict[lag]}" for lag in sorted(K_hat_dict.keys())])
            print(f"[BEKK] Using user-specified K_i per lag (loss=eigsplit): {ks}")
        except Exception as e:
            print(f"[BEKK] (info) Unable to format user K_i dict: {e}")
    else:
        K_hat_dict, _ = estimate_K_dict_from_Psi_list(Psi_list, Kbar=Kbar, T=T, epsilon=epsilon_rule)
        t_estK += (time.perf_counter() - _t0)
        try:
            ks = ", ".join([f"K_{lag}={K_hat_dict[lag]}" for lag in sorted(K_hat_dict.keys())])
            print(f"[BEKK] Estimated K_i per lag: {ks}")
        except Exception as e:
            print(f"[BEKK] (info) Unable to format K_i dict: {e}")

    _t0 = time.perf_counter()
    A_hat_dict = estimate_A_dict_from_R_list(Psi_list, K_hat_dict, N)
    t_estA += (time.perf_counter() - _t0)

    _t0 = time.perf_counter()
    if use_omega_from_B:
        Omega_hat = omega_from_B_est(B_hat, D_N, N, eps=1e-10)
    else:
        T_eff = T - p
        S_emp = torch.zeros((N, N), dtype=dtype_, device=device_)
        for t in range(p, T):
            eps = returns_centered[t]
            S_emp += torch.outer(eps, eps)
        S_emp = S_emp / float(T_eff)
        avg_arch = torch.zeros_like(S_emp)
        for t in range(p, T):
            arch_t = torch.zeros_like(S_emp)
            for lag, mats in A_hat_dict.items():
                eps = returns_centered[t - lag]
                outer = torch.outer(eps, eps)
                for Aik in mats:
                    Aik_ = Aik.to(dtype_).to(device_)
                    arch_t = arch_t + Aik_ @ outer @ Aik_.T
            avg_arch += arch_t
        avg_arch = avg_arch / float(T_eff)
        C_hat = S_emp - avg_arch
        C_hat = 0.5 * (C_hat + C_hat.T)
        evals, evecs = torch.linalg.eigh(C_hat)
        evals = torch.clamp(evals, min=0.0)
        Omega_hat = evecs @ torch.diag(evals) @ evecs.T
    t_omega += (time.perf_counter() - _t0)

    _t0 = time.perf_counter()
    sigmas_bekk = compute_sigma_sequence_from_bekk_arch(returns_centered, Omega_hat, A_hat_dict, p)
    t_sigma += (time.perf_counter() - _t0)
    rr_t = returns_raw[p:]

    gmv = backtest_gmv_from_sigma_hat(sigmas_bekk, rr_t, annualization=annualization)
    eqw = backtest_equal_weight(rr_t, annualization=annualization)

    T_eff = rr_t.shape[0]
    result = {
        "K_hat": K_hat_dict,
        "A_hat_dict": A_hat_dict,
        "Omega_hat": Omega_hat,
        "Assets": N,
        "T_eff": T_eff,
        "GMV": gmv,
        "EW": eqw,
        "sigmas": sigmas_bekk,
        "timing": {
        "padding": float(t_pad),
        "estimate_K": float(t_estK),
        "estimate_A": float(t_estA),
        "omega": float(t_omega),
        "sigma_seq": float(t_sigma),
        "total": float(t_pad + t_estK + t_estA + t_omega + t_sigma),
        },
    }
    return result


def rolling_bekk_lastval_using_Bhat(
    returns_centered: torch.Tensor,
    returns_raw: torch.Tensor,
    p: int,
    lambda_: float,
    tau: float,
    varrho: float = 1e-6,
    annualization: int = 252,
    reestimate_step: int = 1,
    verbose: bool = True,
    val_pct: float = 0.2,
    per_step_log: bool = False,
    bekk_enable_padding: bool = True,
):
    """
    Rolling expanding-window OOS evaluation on the last `val_pct` of the sample.
    At each OOS time t, fit B_hat on centered data up to t (using (p, lambda, tau)),
    then call `bekk_backtest_from_B` on the same window to recover (A_hat_dict, Omega_hat),
    and form a **one-step-ahead** Sigma_t^BEKK using eps_{t-1..t-p} and Omega. In parallel, build
    **one-step-ahead** Sigma_t^VECH via X_t(B_hat) from raw vech(rr) lags. Compute GMV/1N
    returns on r_t and accumulate. This function thus already produces BOTH BEKK and VECH.
    """
    T_all, N = returns_centered.shape
    assert returns_raw.shape == (T_all, N)
    if T_all <= p + 5:
        return {"T_eff": 0, "GMV": {"AV": float("nan"), "SD": float("nan"), "IR": float("nan")},
                "EW": {"AV": float("nan"), "SD": float("nan"), "IR": float("nan")}}

    val_pct = float(val_pct)
    val_pct = 0.0 if val_pct < 0 else (1.0 if val_pct > 1.0 else val_pct)
    t0 = max(p + 5, int(np.floor((1.0 - val_pct) * T_all)))
    D_N_local = construct_D_N(N)

    gmv_ret, ew_ret, idx = [], [], []
    vech_gmv_ret = []
    vech_total_seconds: float = 0.0
    vech_refits: int = 0
    bekk_total_seconds: float = 0.0
    bekk_refits: int = 0

    B_cache = None
    last_refit_t = None

    for t in range(t0, T_all):
        need_refit = (B_cache is None) or (last_refit_t is None) or ((t - last_refit_t) % max(1, int(reestimate_step)) == 0)
        if need_refit:
            Rc = returns_centered[:t]
            X_tau, Y_tau = create_truncated_dataset(Rc, p, float(tau))
            r_tau_local = Rc.clamp(min=float(-tau), max=float(tau))
            outer_all_local = r_tau_local[:, :, None] * r_tau_local[:, None, :]
            Y_full_tau_local = vech(outer_all_local)
            _t0_fista = time.perf_counter()
            B_hat_t = fista_algorithm(
                Y_tau, X_tau, float(lambda_), float(varrho), p=p, N=N,
                Y_full_tau_for_L=Y_full_tau_local,
                power_iters=6,
            )
            _t1_fista = time.perf_counter()
            vech_total_seconds += (_t1_fista - _t0_fista)
            vech_refits += 1
            if verbose:
                print(f"[TIME][VECH] avg per refit: {vech_total_seconds/vech_refits:.3f}s over {vech_refits} refits (total {vech_total_seconds:.3f}s)")
            B_cache = B_hat_t.detach()
            last_refit_t = t
            if verbose:
                print(f"[ROLL-BEKK] refit at t={t}/{T_all-1}: TRAIN={t}, p={p}, lambda={lambda_}, tau={tau}")
        else:
            B_hat_t = B_cache

        if t - p >= 0:
            X_val_vech = _build_Xval_from_raw_vech(returns_raw, t, p)
            Y_hat_vech = X_val_vech @ B_hat_t
            S_vech = vech_to_mat(Y_hat_vech.view(-1), D_N_local, N)
            S_vech = project_to_psd(S_vech, eps=1e-10)
            r_t = returns_raw[t]
            w_gmv_vech = gmv_weights_from_cov_torch(S_vech)
            gmv_rt_vech = float((w_gmv_vech * r_t).sum().item())
        else:
            gmv_rt_vech = None

        win_centered = returns_centered[:t]
        win_raw = returns_raw[:t]
        if bekk_enable_padding:
            t0_bekk = time.perf_counter()
            bekk_win = bekk_backtest_from_B(
                B_hat=B_hat_t,
                returns_centered=win_centered,
                returns_raw=win_raw,
                p=p,
                D_N=D_N_local,
                sparse_level=10,
                epsilon_rule=0.5,
                Kbar=5,
                annualization=annualization,
                use_omega_from_B=True,
                loss_type="eigsplit",
                K_user_dict={1: 2, 2: 2, 3: 1}
            )
            t1_bekk = time.perf_counter()
            bekk_total_seconds += (t1_bekk - t0_bekk)
            bekk_refits += 1
            if verbose:
                print(f"[TIME][BEKK] avg per refit: {bekk_total_seconds/bekk_refits:.3f}s over {bekk_refits} refits (total {bekk_total_seconds:.3f}s)")
            A_hat_dict = bekk_win.get("A_hat_dict", {})
            Omega_hat = bekk_win.get("Omega_hat")
        else:
            if verbose:
                print("[TIME][BEKK] skipped padding/A recovery (--bekk_padding=False)")
            A_hat_dict = {}
            Omega_hat = omega_from_B_est(B_hat_t, D_N_local, N, eps=1e-10)

        if t - p < 0:
            continue
        eps_hist = [returns_centered[t - j] for j in range(1, p + 1)]
        Sigma_t = Omega_hat.clone()
        for lag, mats in A_hat_dict.items():
            rt_im1 = eps_hist[lag - 1].reshape(-1, 1)
            outer = rt_im1 @ rt_im1.T
            for Aik in mats:
                Aik_ = Aik.to(returns_centered.dtype).to(returns_centered.device)
                Sigma_t = Sigma_t + Aik_ @ outer @ Aik_.T
        Sigma_t = project_to_psd(0.5 * (Sigma_t + Sigma_t.T), eps=1e-10)

        r_t = returns_raw[t]
        w_gmv = gmv_weights_from_cov_torch(Sigma_t)
        gmv_rt = float((w_gmv * r_t).sum().item())
        ew_rt = float(r_t.mean().item())

        gmv_ret.append(gmv_rt)
        if gmv_rt_vech is not None:
            vech_gmv_ret.append(gmv_rt_vech)
        else:
            pass
        ew_ret.append(ew_rt)
        idx.append(t)

        if per_step_log and verbose:
            g = torch.tensor(gmv_ret, dtype=torch.float64)
            e = torch.tensor(ew_ret,  dtype=torch.float64)
            n_used = len(gmv_ret)
            AV_bekk = float((g.mean() * annualization).item())
            SD_bekk = float((g.std(unbiased=True) * math.sqrt(annualization)).item())
            AV_ew   = float((e.mean() * annualization).item())
            SD_ew   = float((e.std(unbiased=True) * math.sqrt(annualization)).item())
            vech_str = ""
            if len(vech_gmv_ret) >= 1:
                v = torch.tensor(vech_gmv_ret, dtype=torch.float64)
                AV_vech = float((v.mean() * annualization).item())
                SD_vech = float((v.std(unbiased=True) * math.sqrt(annualization)).item())
                vech_str = f" | VECH AV={AV_vech:.6f}, SD={SD_vech:.6f}"
            print(f"[BEKK][t={t}] used={n_used}  AV={AV_bekk:.6f}, SD={SD_bekk:.6f}{vech_str}  ||  1/N AV={AV_ew:.6f}, SD={SD_ew:.6f}")
            try:
                import sys as _sys
                _sys.stdout.flush()
            except Exception:
                pass

    if len(gmv_ret) == 0:
        return {
            "T_eff": 0,
            "BEKK": {"AV": float("nan"), "SD": float("nan"), "IR": float("nan")},
            "VECH": {"AV": float("nan"), "SD": float("nan"), "IR": float("nan")},
            "EW":   {"AV": float("nan"), "SD": float("nan"), "IR": float("nan")},
            "_series": {"bekk_gmv": [], "vech_gmv": [], "ew": []},
        }

    g_bekk = torch.tensor(gmv_ret, dtype=torch.float64)
    e_bm   = torch.tensor(ew_ret,  dtype=torch.float64)

    AV_bekk = float((g_bekk.mean() * annualization).item())
    SD_bekk = float((g_bekk.std(unbiased=True) * math.sqrt(annualization)).item())
    IR_bekk = AV_bekk / SD_bekk if SD_bekk > 0 else float('nan')

    if len(vech_gmv_ret) == len(gmv_ret) and len(vech_gmv_ret) > 0:
        g_vech = torch.tensor(vech_gmv_ret, dtype=torch.float64)
        AV_vech = float((g_vech.mean() * annualization).item())
        SD_vech = float((g_vech.std(unbiased=True) * math.sqrt(annualization)).item())
        IR_vech = AV_vech / SD_vech if SD_vech > 0 else float('nan')
    else:
        AV_vech = SD_vech = IR_vech = float('nan')

    AV_ew = float((e_bm.mean() * annualization).item())
    SD_ew = float((e_bm.std(unbiased=True) * math.sqrt(annualization)).item())
    IR_ew = AV_ew / SD_ew if SD_ew > 0 else float('nan')

    bekk_avg_seconds = float(bekk_total_seconds / bekk_refits) if bekk_refits > 0 else float("nan")
    if vech_refits > 0:
        print(f"[TIME][VECH][FINAL AVG] {vech_total_seconds/vech_refits:.3f}s over {vech_refits} refits (total {vech_total_seconds:.3f}s)")
    if bekk_refits > 0:
        print(f"[TIME][BEKK][FINAL AVG] {bekk_total_seconds/bekk_refits:.3f}s over {bekk_refits} refits (total {bekk_total_seconds:.3f}s)")

    return {
        "config": {
            "p": p, "lambda": lambda_, "tau": tau, "annualization": annualization,
            "reestimate_step": reestimate_step, "val_pct": val_pct
        },
        "bekk_avg_seconds": bekk_avg_seconds,
        "bekk_total_seconds": float(bekk_total_seconds),
        "bekk_refits": int(bekk_refits),
        "T_eff": len(gmv_ret),
        "BEKK": {"AV": AV_bekk, "SD": SD_bekk, "IR": IR_bekk},
        "VECH": {"AV": AV_vech, "SD": SD_vech, "IR": IR_vech},
        "EW":   {"AV": AV_ew,   "SD": SD_ew,   "IR": IR_ew},
        "_series": {"bekk_gmv": gmv_ret, "vech_gmv": vech_gmv_ret, "ew": ew_ret}
    }
