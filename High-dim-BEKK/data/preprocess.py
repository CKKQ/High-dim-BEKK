import pandas as pd
import torch
import random
import math
from core.matrix_ops import vech

def load_centered_log_returns_from_csv(
    csv_path: str,
    keep_first_k: int | None = None,
    fill_method: str = "ffill",
    dropna: bool = True,
):
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Input must contain a 'Date' column.")
    def _parse_date_col(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        mask_yyyymm = s.str.match(r"^\d{6}$")
        s.loc[mask_yyyymm] = s.loc[mask_yyyymm].str.slice(0, 4) + "-" + s.loc[mask_yyyymm].str.slice(4, 6) + "-01"
        mask_yyyy_mm = s.str.match(r"^\d{4}[-/]\d{2}$")
        s.loc[mask_yyyy_mm] = s.loc[mask_yyyy_mm].str.replace("/", "-", regex=False) + "-01"
        dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if dt.isna().any():
            dt_fallback = pd.to_datetime(s, errors="coerce")
            dt = dt.fillna(dt_fallback)
        return dt
    df["Date"] = _parse_date_col(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    if keep_first_k is not None:
        k = int(keep_first_k)
        all_cols = list(df.columns)
        sel_cols = all_cols[:k]
        df = df[sel_cols]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    if dropna:
        df = df.dropna(how="any")

    df = df * 0.01
    try:
        print("[LOAD] Scaled returns by 0.01: treating input as percent, using decimal downstream.")
    except Exception:
        pass
    df_ret = df.copy()

    return df_ret

@torch.no_grad()
def _build_Xval_from_raw_vech(raw_returns: torch.Tensor, t: int, p: int) -> torch.Tensor:
    outer_all_raw = raw_returns[:, :, None] * raw_returns[:, None, :]
    Y_full_raw = vech(outer_all_raw)
    cols = [torch.ones(1, dtype=raw_returns.dtype, device=raw_returns.device)]
    for lag_i in range(1, p + 1):
        cols.append(Y_full_raw[t - lag_i])
    X_val = torch.cat(cols).reshape(1, -1)
    return X_val


def maybe_subsample_returns(returns, N_threshold=None, T_threshold=None, frac_cols=1.0, frac_time=1.0, seed=None):
    """
    Optionally subsample the returns tensor along time (T) and columns (N).
    This speeds up (lambda, tau) tuning on large problems.
    """
    T, N = returns.shape
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    # Columns
    if (N_threshold is not None) and (N_threshold > 0) and (N > N_threshold) and (0 < frac_cols < 1.0):
        N_keep = max(1, math.ceil(frac_cols * N))
        perm_cols = torch.randperm(N, device=returns.device)
        kept_j = perm_cols[:N_keep].cpu().tolist()
        kept_j.sort()
    else:
        kept_j = list(range(N))

    # Time
    if (T_threshold is not None) and (T_threshold > 0) and (T > T_threshold) and (0 < frac_time < 1.0):
        T_keep = max(1, math.ceil(frac_time * T))
        perm_t = torch.randperm(T, device=returns.device)
        kept_t = perm_t[:T_keep].cpu().tolist()
        kept_t.sort()
    else:
        kept_t = list(range(T))

    returns_sub = returns[kept_t][:, kept_j]
    return returns_sub, kept_t, kept_j