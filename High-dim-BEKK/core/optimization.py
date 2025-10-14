import torch
from kron_transform import transformation_kron_torch,permutation_torch
from projection import _hard_truncate_inplace,_nuclear_norm_slq,_row_topk_prune_inplace
from utils.hash_cache import _hash_tensor_cpu_fp32,_PADDING_RESULT_CACHE

def padding(mat: torch.Tensor,
            num_epochs_stage: int = 1000,
            lr: float = 1e-2,
            enforce_symmetry: bool = False,
            use_amp: bool = True,
            svd_every: int = 10,
            sym_penalty: float = 1e-3,
            loss_type: str = "nuclear",
            top_k: int | None = None,
            hard_truncate: bool = True,
            ht_mode: str = "median_kappa",
            ht_kappa: float = 1e-3,
            ht_q_floor: float = 0.02,
            ht_abs_eps: float = 0.0,
            ht_every: int = 1,
            ht_enable_from_N: int = 40,
            ht_row_cap: bool = True,
            ht_row_mult: float = 4.0,
            ht_s: int | None = None):
    """
    Padding via **exact nuclear norm** minimization on Rperm with Adam.
    loss_type can be "nuclear" or "eigsplit"; for "eigsplit" the loss is L = -sum_{j=1}^{K} lambda_j(S) + sum_{j=K+1}^{n} lambda_j(S)^2 on S = 0.5*(Rperm+Rperm^T), where K=top_k must be provided by the caller.
    """
    try:
        cache_key_r = (
            _hash_tensor_cpu_fp32(mat),
        )
    except Exception:
        pass
    assert mat.shape[0] == mat.shape[1]
    d = mat.shape[0]
    N = int(-0.5 + 0.5 * (1 + 8 * d)**0.5)
    L = N * (N - 1) // 2

    device_ = mat.device
    dtype_ = mat.dtype

    try:
        cache_key_r = (
            _hash_tensor_cpu_fp32(mat), N, L,
            int(enforce_symmetry), float(lr),
            float(sym_penalty), int(num_epochs_stage)
        )
        if cache_key_r in _PADDING_RESULT_CACHE:
            R_cached = _PADDING_RESULT_CACHE[cache_key_r]
            return R_cached.to(device=mat.device, dtype=mat.dtype).detach().clone()
    except Exception:
        pass

    W_raw = torch.nn.Parameter(torch.zeros(L, L, device=device_, dtype=dtype_), requires_grad=True)
    if float(torch.norm(W_raw).item()) == 0.0:
        eps = 1e-6
        with torch.no_grad():
            Z = eps * torch.randn_like(W_raw)
            W_raw.add_(0.5 * (Z + Z.T))

    def W_expr():
        return 0.5 * (W_raw + W_raw.T) if enforce_symmetry else W_raw

    optimizer = torch.optim.Adam([W_raw], lr=lr)
    best_loss = float('inf')
    epochs_since_improvement = 0
    patience = 10
    delta = 1e-3

    from torch import amp as _amp
    if use_amp:
        autocast_ctx = _amp.autocast(device_type=('cuda' if mat.is_cuda else 'cpu'))
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast_ctx = _NoOp()

    if loss_type == "nuclear":
        U_cache = None
        V_cache = None
        for it in range(num_epochs_stage):
            W_use = W_expr()
            with autocast_ctx:
                R = transformation_kron_torch(mat, W_use)
                Rperm = permutation_torch(R, N)
                Ssym = 0.5 * (Rperm + Rperm.T)

            need_exact = (it % max(1, (5 if it < 50 else svd_every)) == 0) or (U_cache is None or V_cache is None)
            if need_exact:
                with torch.no_grad():
                    S32 = Ssym.to(torch.float32)
                    U32, Svals, Vh32 = torch.linalg.svd(S32, full_matrices=False)
                    energy = 0.995
                    max_k = 2048
                    s = Svals.clamp(min=0)
                    if s.numel() == 0:
                        k_keep = 1
                    else:
                        cum = torch.cumsum(s, dim=0)
                        total = float(s.sum().item()) + 1e-12
                        k_energy = int((cum / total <= energy).sum().item())
                        thresh = 1e-3 * float(s[0].item())
                        k_rel = int((s >= thresh).sum().item())
                        k_keep = max(1, min(max_k, max(k_energy, k_rel)))
                    U_cache = U32[:, :k_keep].detach()
                    V_cache = Vh32[:k_keep, :].transpose(-2, -1).detach()
                    true_loss_val = _nuclear_norm_slq(S32, probes=16, iters=80)
            else:
                true_loss_val = None

            G = U_cache @ V_cache.T
            g_norm = float(torch.norm(G).item())
            if g_norm > 0:
                G = G / g_norm
            lin_loss = (Ssym.to(torch.float32) * G).sum()
            if sym_penalty > 0.0:
                asym = Rperm - Rperm.T
                lin_loss = lin_loss + sym_penalty * (asym * asym).sum()

            optimizer.zero_grad(set_to_none=True)
            lin_loss.backward()
            torch.nn.utils.clip_grad_norm_([W_raw], max_norm=5.0)

            W_prev = W_raw.data.clone()
            optimizer.step()

            if hard_truncate and N >= ht_enable_from_N and (it + 1) % max(1, ht_every) == 0:
                try:
                    thr = _hard_truncate_inplace(W_raw.data,
                                                 mode=ht_mode,
                                                 kappa=ht_kappa,
                                                 q_floor=ht_q_floor,
                                                 abs_eps=ht_abs_eps,
                                                 per_iter_cap=0.99)
                    if (it + 1) % 10 == 0:
                        nnz = int((W_raw.data != 0).sum().item())
                        print(f"[padding] hard-trunc it={it+1}: thr={thr:.3e}, nnz(W)={nnz}")
                except Exception as e:
                    print(f"[padding] hard-trunc skipped due to: {e}")
            if hard_truncate and ht_row_cap and (ht_s is not None):
                try:
                    k_row = int(max(1, round(ht_row_mult * (float(ht_s) ** 2))))
                    _row_topk_prune_inplace(W_raw.data, k_row)
                    if (it + 1) % 10 == 0:
                        nnz = int((W_raw.data != 0).sum().item())
                        print(f"[padding] per-row cap it={it+1}: k_row={k_row}, nnz(W)={nnz}")
                except Exception as e:
                    print(f"[padding] per-row cap skipped due to: {e}")

            if need_exact:
                with torch.no_grad():
                    R_chk = transformation_kron_torch(mat, W_expr())
                    Rperm_chk = permutation_torch(R_chk, N)
                    Ssym_chk = 0.5 * (Rperm_chk + Rperm_chk.T)
                    cur_true = _nuclear_norm_slq(Ssym_chk.to(torch.float32), probes=16, iters=80)
                if cur_true > true_loss_val + 1e-6:
                    W_raw.data.copy_(W_prev)
                    for pg in optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * 0.5, 1e-5)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_ctx:
                        R_bt = transformation_kron_torch(mat, W_expr())
                        Rperm_bt = permutation_torch(R_bt, N)
                        Ssym_bt = 0.5 * (Rperm_bt + Rperm_bt.T)
                    G = U_cache @ V_cache.T
                    g_norm = float(torch.norm(G).item())
                    if g_norm > 0:
                        G = G / g_norm
                    lin_loss_bt = (Ssym_bt.to(torch.float32) * G).sum()
                    if sym_penalty > 0.0:
                        asym_bt = Rperm_bt - Rperm_bt.T
                        lin_loss_bt = lin_loss_bt + sym_penalty * (asym_bt * asym_bt).sum()
                    lin_loss_bt.backward()
                    torch.nn.utils.clip_grad_norm_([W_raw], max_norm=5.0)
                    optimizer.step()
                    if hard_truncate and N >= ht_enable_from_N and (it + 1) % max(1, ht_every) == 0:
                        try:
                            thr = _hard_truncate_inplace(W_raw.data,
                                                         mode=ht_mode,
                                                         kappa=ht_kappa,
                                                         q_floor=ht_q_floor,
                                                         abs_eps=ht_abs_eps,
                                                         per_iter_cap=0.99)
                            if (it + 1) % 10 == 0:
                                nnz = int((W_raw.data != 0).sum().item())
                                print(f"[padding] hard-trunc it={it+1}: thr={thr:.3e}, nnz(W)={nnz}")
                        except Exception as e:
                            print(f"[padding] hard-trunc skipped due to: {e}")
                    if hard_truncate and ht_row_cap and (ht_s is not None):
                        try:
                            k_row = int(max(1, round(ht_row_mult * (float(ht_s) ** 2))))
                            _row_topk_prune_inplace(W_raw.data, k_row)
                            if (it + 1) % 10 == 0:
                                nnz = int((W_raw.data != 0).sum().item())
                                print(f"[padding] per-row cap it={it+1}: k_row={k_row}, nnz(W)={nnz}")
                        except Exception as e:
                            print(f"[padding] per-row cap skipped due to: {e}")
                    with torch.no_grad():
                        R_chk = transformation_kron_torch(mat, W_expr())
                        Rperm_chk = permutation_torch(R_chk, N)
                        Ssym_chk = 0.5 * (Rperm_chk + Rperm_chk.T)
                        cur_true = _nuclear_norm_slq(Ssym_chk.to(torch.float32), probes=16, iters=80)
                    del R_bt, Rperm_bt, Ssym_bt
                    if 'asym_bt' in locals():
                        del asym_bt
                    del lin_loss_bt
                cur = cur_true
            else:
                cur = float(lin_loss.detach().item())

            del R, Rperm, Ssym, G
            if 'asym' in locals():
                del asym
            if torch.cuda.is_available() and (it % 20 == 0):
                torch.cuda.empty_cache()

            if cur + delta < best_loss:
                best_loss, epochs_since_improvement = cur, 0
            else:
                epochs_since_improvement += 1

            if need_exact and epochs_since_improvement >= patience // 2:
                svd_every = max(5, svd_every // 2)
                for pg in optimizer.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.8, 1e-5)

            if epochs_since_improvement >= patience:
                break

    elif loss_type == "eigsplit":
        for it in range(num_epochs_stage):
            W_use = W_expr()
            with autocast_ctx:
                R = transformation_kron_torch(mat, W_use)
                Rperm = permutation_torch(R, N)
                Ssym = 0.5 * (Rperm + Rperm.T)
            S32 = Ssym.to(torch.float32)
            evals = torch.linalg.eigvalsh(S32)
            lam = torch.flip(evals, dims=[0])
            if (top_k is None) or (top_k < 1):
                raise ValueError("padding(...): top_k must be provided and >=1 when loss_type='eigsplit'.")
            K = min(top_k, lam.numel())
            lead = lam[:K]
            tail = lam[K:]
            loss = (-lead.sum()) + (tail * tail).sum()
            if sym_penalty > 0.0:
                asym = Rperm - Rperm.T
                loss = loss + sym_penalty * (asym * asym).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W_raw], max_norm=5.0)
            optimizer.step()
            if hard_truncate and N >= ht_enable_from_N and (it + 1) % max(1, ht_every) == 0:
                try:
                    thr = _hard_truncate_inplace(W_raw.data,
                                                 mode=ht_mode,
                                                 kappa=ht_kappa,
                                                 q_floor=ht_q_floor,
                                                 abs_eps=ht_abs_eps,
                                                 per_iter_cap=0.99)
                    if (it + 1) % 10 == 0:
                        nnz = int((W_raw.data != 0).sum().item())
                        print(f"[padding] hard-trunc it={it+1}: thr={thr:.3e}, nnz(W)={nnz}")
                except Exception as e:
                    print(f"[padding] hard-trunc skipped due to: {e}")
            if hard_truncate and ht_row_cap and (ht_s is not None):
                try:
                    k_row = int(max(1, round(ht_row_mult * (float(ht_s) ** 2))))
                    _row_topk_prune_inplace(W_raw.data, k_row)
                    if (it + 1) % 10 == 0:
                        nnz = int((W_raw.data != 0).sum().item())
                        print(f"[padding] per-row cap it={it+1}: k_row={k_row}, nnz(W)={nnz}")
                except Exception as e:
                    print(f"[padding] per-row cap skipped due to: {e}")
            cur = float(loss.detach().item())
            if cur + delta < best_loss:
                best_loss, epochs_since_improvement = cur, 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    with torch.no_grad():
        W_final = W_expr().detach()
        if hard_truncate and ht_row_cap and (ht_s is not None):
            try:
                k_row = int(max(1, round(ht_row_mult * (float(ht_s) ** 2))))
                _row_topk_prune_inplace(W_final, k_row)
            except Exception:
                pass
        R_final = transformation_kron_torch(mat, W_final).detach()
    try:
        nnz_final = int((W_final != 0).sum().item())
        print(f"[padding] final nnz(W)={nnz_final}")
    except Exception:
        pass
    return R_final



@torch.no_grad()
def _power_L_op(Y_full_tau: torch.Tensor, p: int, num: int, d: int, c: int, iters: int = 6):
    """Estimate L = spectral_norm((1/num) * X^T X) for the implicit design X with p lags"""
    device_ = Y_full_tau.device
    dtype_ = Y_full_tau.dtype
    B = torch.randn(1 + p * d, c, device=device_, dtype=dtype_)
    B = B / (torch.norm(B) + 1e-12)

    def XB(Bmat):
        b0 = Bmat[0:1, :]
        Yhat = b0.expand(num, c).clone()
        for i in range(1, p + 1):
            Bi = Bmat[1 + (i - 1) * d : 1 + i * d, :]
            Ylag = Y_full_tau[p - i : p - i + num]
            Yhat = Yhat + Ylag @ Bi
        return Yhat

    def XT_R(R):
        g0 = R.sum(dim=0, keepdim=True) / num
        G = [g0]
        for i in range(1, p + 1):
            Ylag = Y_full_tau[p - i : p - i + num]
            Gi = (Ylag.T @ R) / num
            G.append(Gi)
        return torch.cat(G, dim=0)

    L_est = torch.tensor(0.0, device=device_, dtype=dtype_)
    for _ in range(max(3, iters)):
        Yhat = XB(B)
        HB = XT_R(Yhat)
        nrm = torch.norm(HB)
        if float(nrm) == 0.0:
            return torch.tensor(1.0, device=device_, dtype=dtype_)
        B = HB / (nrm + 1e-12)
        L_est = nrm
    return L_est

def fista_algorithm(
    Y_tau: torch.Tensor,
    X_tau: torch.Tensor,
    lambda_: float,
    varrho: float,
    p: int,
    N: int,
    Y_full_tau_for_L: torch.Tensor | None = None,
    power_iters: int = 6,
    chunk_cols: int = 256,
):
    d = N * (N + 1) // 2
    m = 1 + p * d
    T_samples = X_tau.shape[0]

    num = T_samples
    c_probe = min(chunk_cols, d)
    if Y_full_tau_for_L is None:
        L_fallback = torch.linalg.norm(X_tau, ord=2) ** 2 / max(1, T_samples)
        step_size = float(1.0 / (float(L_fallback.item()) + 1e-12))
    else:
        L_est = _power_L_op(Y_full_tau_for_L, p, num, d, c_probe, iters=power_iters)
        step_size = 1.0 / (float(L_est.item()) + 1e-12)

    B_out = torch.zeros((m, d), dtype=X_tau.dtype, device=X_tau.device)

    for j0 in range(0, d, chunk_cols):
        j1 = min(j0 + chunk_cols, d)
        c = j1 - j0
        Y_tgt = Y_tau[:, j0:j1]

        Bk = torch.zeros((m, c), dtype=X_tau.dtype, device=X_tau.device)
        Uk = Bk.clone()
        tk = torch.tensor(1.0, dtype=X_tau.dtype, device=X_tau.device)

        def XB_block(Bmat):
            b0 = Bmat[0:1, :]
            Yhat = b0.expand(num, c).clone()
            for i in range(1, p + 1):
                Bi = Bmat[1 + (i - 1) * d : 1 + i * d, :]
                Ylag = Y_full_tau_for_L[p - i : p - i + num]
                Yhat = Yhat + Ylag @ Bi
            return Yhat

        def grad_block(Bmat):
            Yhat = XB_block(Bmat)
            R = Yhat - Y_tgt
            g0 = R.sum(dim=0, keepdim=True) / num
            Grads = [g0]
            for i in range(1, p + 1):
                Ylag = Y_full_tau_for_L[p - i : p - i + num]
                Gi = (Ylag.T @ R) / num
                Grads.append(Gi)
            return torch.cat(Grads, dim=0), R

        max_iter = 10000
        tol = varrho
        for _ in range(max_iter):
            Gk, R = grad_block(Uk)
            Bnext = torch.nn.functional.softshrink(Uk - step_size * Gk, lambda_ * step_size)
            tnext = (1 + torch.sqrt(1 + 4 * tk * tk)) / 2
            Uk = Bnext + ((tk - 1) / tnext) * (Bnext - Bk)
            if torch.norm(Bnext - Bk) < tol:
                Bk = Bnext
                break
            Bk = Bnext
            tk = tnext

        B_out[:, j0:j1] = Bk

        del Bk, Uk, tk, Y_tgt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return B_out.detach()