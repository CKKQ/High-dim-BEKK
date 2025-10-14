import torch

# --- kron transformation cache ---
_kron_cache = {}

def build_kron_indices(N, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = N * (N + 1) // 2
    n2 = N * N
    src_mat_idx = []
    src_w_idx = []
    dst_matcol_idx = []
    use_w_mask = []
    
    def edge_idx(u, v):
        u, v = min(u, v), max(u, v)
        return (u - 1) * (2 * N - u) // 2 + (v - u - 1)

    for l in range(1, N + 1):
        idx1 = ((l - 1) * (2 * N - l + 2)) // 2
        for row in range(d):
            src_mat_idx.append(row * d + idx1)
            src_w_idx.append(-1)
            dst_matcol_idx.append(row * n2 + (l - 1) * N + l - 1)
            use_w_mask.append(False)
        for j in range(1, l):
            idx2 = ((j - 1) * (2 * N - j + 2)) // 2 + (l - j)
            e_jl = edge_idx(j, l)
            for h in range(1, N + 1):
                idx3 = ((h - 1) * (2 * N - h + 2)) // 2
                src_mat_idx.append(idx3 * d + idx2)
                src_w_idx.append(-1)
                dst_matcol_idx.append(idx3 * n2 + (j - 1) * N + l - 1)
                use_w_mask.append(False)
                src_mat_idx.append(idx3 * d + idx2)
                src_w_idx.append(-1)
                dst_matcol_idx.append(idx3 * n2 + (l - 1) * N + j - 1)
                use_w_mask.append(False)
                for k in range(1, h):
                    idx4 = ((k - 1) * (2 * N - k + 2)) // 2 + (h - k)
                    e_kh = edge_idx(k, h)
                    src_mat_idx.append(idx4 * d + idx2)
                    src_w_idx.append(e_jl * (N * (N - 1) // 2) + e_kh)
                    dst_matcol_idx.append(idx4 * n2 + (j - 1) * N + l - 1)
                    use_w_mask.append(True)
                    src_mat_idx.append(idx4 * d + idx2)
                    src_w_idx.append(e_jl * (N * (N - 1) // 2) + e_kh)
                    dst_matcol_idx.append(idx4 * n2 + (l - 1) * N + j - 1)
                    use_w_mask.append(False)

    src_matcol_idx = []
    dst_R_idx = []
    for l in range(1, N + 1):
        idx1 = ((l - 1) * (2 * N - l + 2)) // 2
        for col in range(n2):
            src = idx1 * n2 + col
            dst = (l - 1) * N + l - 1
            src_matcol_idx.append(src)
            dst_R_idx.append(dst * n2 + col)
        for j in range(1, l):
            idx2 = ((j - 1) * (2 * N - j + 2)) // 2 + (l - j)
            for h in range(1, N + 1):
                base_idx = idx2 * n2 + (h - 1) * N + h - 1
                row1 = (j - 1) * N + l - 1
                row2 = (l - 1) * N + j - 1
                col1 = (h - 1) * N + h - 1
                src_matcol_idx.append(base_idx)
                dst_R_idx.append(row1 * n2 + col1)
                src_matcol_idx.append(base_idx)
                dst_R_idx.append(row2 * n2 + col1)
                for k in range(1, h):
                    idxA = idx2 * n2 + (k - 1) * N + h - 1
                    idxB = idx2 * n2 + (h - 1) * N + k - 1
                    rowA = (l - 1) * N + j - 1
                    rowB = (j - 1) * N + l - 1
                    colA = (k - 1) * N + h - 1
                    colB = (h - 1) * N + k - 1
                    src_matcol_idx.extend([idxA, idxB, idxB, idxA])
                    dst_R_idx.extend([rowA * n2 + colA, rowA * n2 + colB,
                                      rowB * n2 + colA, rowB * n2 + colB])

    return (
        torch.tensor(src_mat_idx, dtype=torch.long, device=device),
        torch.tensor(src_w_idx, dtype=torch.long, device=device),
        torch.tensor(dst_matcol_idx, dtype=torch.long, device=device),
        torch.tensor(use_w_mask, dtype=torch.bool, device=device),
        torch.tensor(src_matcol_idx, dtype=torch.long, device=device),
        torch.tensor(dst_R_idx, dtype=torch.long, device=device),
        d, n2
    )


def transformation_kron_torch(mat, W):
    d = mat.shape[0]
    N = int((-1 + (1 + 8 * d) ** 0.5) // 2)
    device = mat.device

    if N not in _kron_cache:
        _kron_cache[N] = build_kron_indices(N, device=device)

    (src_mat_idx, src_w_idx, dst_matcol_idx,
     use_w_mask, src_matcol_idx, dst_R_idx, d, n2) = _kron_cache[N]

    mat_flat = mat.flatten()
    W_flat = W.flatten()

    v_mat = mat_flat[src_mat_idx]
    values = v_mat.clone()

    if use_w_mask.any():
        mask_idx = use_w_mask.nonzero(as_tuple=True)[0]
        w_indices = src_w_idx[mask_idx]
        values[mask_idx] = v_mat[mask_idx] - W_flat[w_indices]

    w_assign_idx = (~use_w_mask).nonzero(as_tuple=True)[0]
    w_valid = src_w_idx[w_assign_idx] >= 0
    if w_valid.any():
        w_pos = w_assign_idx[w_valid]
        values[w_pos] = W_flat[src_w_idx[w_pos]]

    mat_col_flat = torch.zeros(d * n2, dtype=mat.dtype, device=device)
    mat_col_flat[dst_matcol_idx] = values

    R_flat = torch.zeros(n2 * n2, dtype=mat.dtype, device=device)
    R_flat[dst_R_idx] = mat_col_flat[src_matcol_idx]
    R = R_flat.view(n2, n2)
    return R


def permutation_torch(X, N):
    blocks = X.view(N, N, N, N)
    blocks_t = blocks.permute(0, 2, 3, 1)
    blocks_col = blocks_t.permute(1, 0, 2, 3)
    permuted = blocks_col.reshape(N*N, N*N).T
    return permuted