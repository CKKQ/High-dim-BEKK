import torch
from matrix_ops import vech

def create_truncated_dataset(returns, p, tau):
    T, N = returns.shape
    d = N * (N + 1) // 2
    r_tau = returns.clamp(min=-tau, max=tau)
    outer_all = r_tau[:, :, None] * r_tau[:, None, :]
    Y_full_tau = vech(outer_all)

    num_samples = T - p
    X_tau = torch.ones((num_samples, 1 + p * d), dtype=returns.dtype, device=returns.device)
    X_tau[:, 0] = torch.tensor(1.0, dtype=returns.dtype, device=returns.device)
    Y_tau = Y_full_tau[p:]

    for lag in range(1, p + 1):
        X_tau[:, 1 + (lag - 1) * d : 1 + lag * d] = Y_full_tau[p - lag : T - lag]

    return X_tau, Y_tau


@torch.no_grad()
def _build_Xval_from_raw_vech(raw_returns: torch.Tensor, t: int, p: int) -> torch.Tensor:
    outer_all_raw = raw_returns[:, :, None] * raw_returns[:, None, :]
    Y_full_raw = vech(outer_all_raw)
    cols = [torch.ones(1, dtype=raw_returns.dtype, device=raw_returns.device)]
    for lag_i in range(1, p + 1):
        cols.append(Y_full_raw[t - lag_i])
    X_val = torch.cat(cols).reshape(1, -1)
    return X_val