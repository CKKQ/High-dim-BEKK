import torch

@torch.no_grad()
def _hard_truncate_inplace(W: torch.Tensor,
                           mode: str = "quantile",
                           kappa: float = 3.0,
                           q_floor: float = 0.9,
                           abs_eps: float = 0.0,
                           per_iter_cap: float = 0.99) -> float:
    if W.numel() == 0:
        return 0.0

    W_abs = torch.nan_to_num(W.abs(), nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "mad":
        med = torch.median(W)
        mad = torch.median((W - med).abs()) + 1e-12
        thr = max(float(kappa * mad.item()), float(abs_eps))
    elif mode == "absolute":
        thr = float(abs_eps)
    else:
        try:
            qv = torch.quantile(W_abs.view(-1), q_floor)
        except Exception:
            qv = _approx_quantile_1d(W_abs.view(-1), q_floor, sample_cap=1_000_000)
        thr = max(float(qv.item()), float(abs_eps))

    if 0.0 < per_iter_cap < 1.0:
        n = W_abs.numel()
        k_keep = int(per_iter_cap * n)
        if k_keep > 0:
            kth = W_abs.view(-1).kthvalue(max(1, n - k_keep + 1)).values
            thr = max(thr, float(kth.item()))

    W.masked_fill_(W_abs < thr, 0.0)
    return float(thr)


@torch.no_grad()
def _row_topk_prune_inplace(W: torch.Tensor, k_row: int):
    if k_row <= 0:
        W.zero_()
        return
    k_row = int(k_row)
    m, n = W.shape
    k_eff = min(k_row, n)
    # topk on abs per row
    vals, idx = torch.topk(W.abs(), k_eff, dim=1, largest=True)
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    W.masked_fill_(~mask, 0.0)


@torch.no_grad()
def _approx_quantile_1d(x: torch.Tensor, q: float, sample_cap: int = 1_000_000):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = x.numel()
    if n == 0 or not (0.0 < q < 1.0):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if n > sample_cap:
        idx = torch.randint(0, n, (sample_cap,), device=x.device)
        x = x.index_select(0, idx)
        n = sample_cap
    k = max(1, int(q * n))
    return x.kthvalue(k).values


@torch.no_grad()
def _nuclear_norm_slq(S: torch.Tensor, probes: int = 8, iters: int = 40) -> float:
    """Lightweight proxy for nuclear norm. If SLQ not available, use exact svdvals sum on float32."""
    S32 = S.to(torch.float32)
    try:
        s = torch.linalg.svdvals(S32)
        return float(s.sum().item())
    except Exception:
        evals = torch.linalg.eigvalsh(S32.T @ S32)
        return float(torch.sqrt(torch.clamp(evals, min=0.0)).sum().item())