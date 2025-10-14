import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def vech(mat):
    if mat.dim() == 2:
        N = mat.shape[0]
        triu_indices = torch.triu_indices(N, N)
        return mat[triu_indices[0], triu_indices[1]]
    elif mat.dim() == 3:
        T, N, _ = mat.shape
        triu_indices = torch.triu_indices(N, N)
        return mat[:, triu_indices[0], triu_indices[1]]
    else:
        raise ValueError("Input to vech must be 2D or 3D tensor.")
    

def vech_to_mat(y_vec, D_N, N):
    mat = (D_N @ y_vec).view(N, N)
    return 0.5 * (mat + mat.T)


def construct_D_N(N):
    d = N * (N + 1) // 2
    D = torch.zeros((N**2, d), dtype=torch.float32, device=device)
    for j in torch.arange(1, N + 1):
        for l in torch.arange(j, N + 1):
            f = 0.5 * (j - 1) * (2 * N - j + 2) + (l - j + 1)
            col = int(f - 1)
            if l == j:
                row = (l - 1) + (j - 1) * N
                D[row, col] = 1.0
            else:
                row1 = (l - 1) + (j - 1) * N
                row2 = (j - 1) + (l - 1) * N
                D[row1, col] = 1.0
                D[row2, col] = 1.0
    return D


def project_to_psd(M, eps=0.0):
    M_sym = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M_sym)
    evals_clipped = torch.clamp(evals, min=(eps if eps is not None else 0.0))
    return evecs @ torch.diag(evals_clipped) @ evecs.T