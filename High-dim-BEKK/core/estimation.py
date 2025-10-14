import torch
from matrix_ops import vech
from vech_utils import create_truncated_dataset
from kron_transform import permutation_torch
from optimization import fista_algorithm

@torch.no_grad()
def compute_bic_for_fixed_p(returns: torch.Tensor, p: int, tau: float, lambda_: float, varrho: float, epsilon: float, iota_d: float):
    T, N = returns.shape
    r_tau_full = returns.clamp(min=float(-tau), max=float(tau))
    outer_all_tau = r_tau_full[:, :, None] * r_tau_full[:, None, :]
    Y_full_tau = vech(outer_all_tau)
    X_tau_p, Y_tau_p = create_truncated_dataset(returns, p, tau)
    B_hat = fista_algorithm(
            Y_tau_p, X_tau_p, lambda_, varrho, p=p, N=N,
            Y_full_tau_for_L=Y_full_tau,
            power_iters=6,
        )
    T_eff = T / (torch.log(torch.tensor(T, dtype=torch.float32, device=returns.device)) ** 2)
    resid = Y_tau_p - X_tau_p @ B_hat
    loss = (1 / (2 * T)) * torch.norm(resid, p='fro')**2
    logL = torch.log(loss)
    d_dim = Y_tau_p.shape[1]
    penalty_exp = (torch.log(torch.tensor(p * d_dim + 1, dtype=torch.float32, device=returns.device)) / T_eff) ** ((1 + 2 * epsilon) / (1 + epsilon))
    penalty = iota_d * penalty_exp * torch.log(torch.tensor(T, dtype=torch.float32, device=returns.device))
    bic_val = logL + penalty
    return float(bic_val.item()), B_hat


@torch.no_grad()
def estimate_K_dict_from_Psi_list(Psi_list: list, Kbar: int, T: int, epsilon: float, alpha: float = 0.001):
    K_hats_dict = {}
    ratios_dict = {}
    if len(Psi_list) == 0:
        return K_hats_dict, ratios_dict
    n2 = Psi_list[0].shape[0]
    N = int(torch.sqrt(torch.tensor(n2, dtype=torch.float32)).item())
    p = len(Psi_list)
    T_eff = T / (torch.log(torch.tensor(T, dtype=torch.float32)) ** 2)
    c = alpha * N * (N * p * torch.log(torch.tensor(T, dtype=torch.float32)) / T_eff) ** (epsilon / (1 + epsilon))
    for i, Psi in enumerate(Psi_list):
        S = 0.5 * (Psi + Psi.T)
        S = S.to(torch.float32)
        n2 = S.shape[0]
        need = min(Kbar + 1, n2)
        k = min(need + 2, max(1, n2 - 1))
        try:
            X0 = torch.randn(n2, k, device=S.device, dtype=S.dtype)
            vals, _ = torch.lobpcg(S, k=k, B=None, X=X0, niter=100, tol=1e-4, largest=True)
            lam_sorted = torch.sort(vals, descending=True).values
        except Exception:
            try:
                vals, _ = torch.lobpcg(S.cpu(), k=k, niter=200, tol=1e-4, largest=True)
                lam_sorted = torch.sort(vals, descending=True).values.to(S.device)
            except Exception:
                if n2 <= 4096:
                    lam_sorted = torch.linalg.eigvalsh(S.cpu())
                    lam_sorted = torch.sort(lam_sorted, descending=True).values.to(S.device)
                else:
                    v = torch.randn(n2, 1, device=S.device, dtype=S.dtype)
                    v = v / (torch.norm(v) + 1e-12)
                    for _ in range(50):
                        v = S @ v
                        v = v / (torch.norm(v) + 1e-12)
                    lam1 = (v.T @ (S @ v)).squeeze()
                    lam_sorted = lam1.repeat(need)
        lam_use = lam_sorted[:need]
        Lm = lam_use.shape[0]
        max_k = min(Kbar, Lm - 1)
        if max_k >= 1:
            r_list = [((lam_use[k] + c) / (lam_use[k - 1] + c)) for k in range(1, max_k + 1)]
            # print(f"[estimate_K_dict] lag={i+1}, top eigenvalues: {lam_use[:min(5,Lm)].cpu().numpy()}, c={c:.3e}, ratios: {[float(r.item()) for r in r_list]}")
            r = torch.stack(r_list)
            ratios_dict[i + 1] = r.detach().cpu().tolist()
            k_star = int(torch.argmin(r).item()) + 1
        else:
            ratios_dict[i + 1] = []
            k_star = 1
        K_hats_dict[i + 1] = int(k_star)
    return K_hats_dict, ratios_dict


@torch.no_grad()
def estimate_A_dict_from_R_list(imputation_list, K_dict, N):
    est_A_dict = {}
    for idx, lag in enumerate(sorted(K_dict.keys())):
        R = imputation_list[idx]
        Rperm = permutation_torch(R, N)
        Rperm = 0.5 * (Rperm + Rperm.T)
        R64 = Rperm.to(torch.float64)
        try:
            evals, evecs = torch.linalg.eigh(R64)
        except Exception:
            eye = torch.eye(R64.shape[0], dtype=R64.dtype, device=R64.device)
            evals, evecs = torch.linalg.eigh(R64 + 1e-8 * eye)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]

        mats = []
        K_i = K_dict[lag]
        for k in range(K_i):
            lam = torch.clamp(evals[k], min=0.0)
            vk = evecs[:, k]
            Ak = torch.sqrt(lam) * vk.reshape(N, N)
            mats.append(Ak.to(torch.float32))
        est_A_dict[lag] = mats
    return est_A_dict


@torch.no_grad()
def omega_from_B_est(B_hat: torch.Tensor, D_N: torch.Tensor, N: int, eps: float = 1e-10) -> torch.Tensor:
    d = N * (N + 1) // 2
    assert B_hat.shape[0] >= 1 and B_hat.shape[1] == d
    omega_vech = B_hat[0]
    Omega_sym = (D_N @ omega_vech).view(N, N)
    # Omega_sym = 0.5 * (Omega_sym + Omega_sym.T)
    evals, evecs = torch.linalg.eigh(Omega_sym)
    evals = torch.clamp(evals, min=0.0)
    Omega_hat = evecs @ torch.diag(evals) @ evecs.T

    return Omega_hat


@torch.no_grad()
def compute_sigma_sequence_from_bekk_arch(returns_centered: torch.Tensor,
                                          Omega_hat: torch.Tensor,
                                          A_hat_dict: dict,
                                          p: int) -> torch.Tensor:
    T, N = returns_centered.shape
    assert T > p
    I = torch.eye(N, dtype=returns_centered.dtype, device=returns_centered.device)

    base = 0.5 * (Omega_hat + Omega_hat.T)

    seq = []
    for t in range(p, T):
        Sigma_t = base.clone()
        for lag, mats in A_hat_dict.items():
            eps = returns_centered[t - lag]
            outer = torch.outer(eps, eps)
            for Aik in mats:
                Aik_ = Aik.to(returns_centered.dtype).to(returns_centered.device)
                Sigma_t = Sigma_t + Aik_ @ outer @ Aik_.T
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T) + 1e-10 * I
        seq.append(Sigma_t)

    return torch.stack(seq, dim=0)