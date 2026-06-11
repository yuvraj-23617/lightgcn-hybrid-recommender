"""
graph.py
========
Builds the bipartite user-item interaction graph and computes the
symmetrically normalised adjacency matrix used by LightGCN.

The full adjacency matrix has shape  (n_users + n_items) × (n_users + n_items):

    A = [  0    R  ]
        [  R^T  0  ]

where R is the n_users × n_items binary interaction matrix.

Normalised form: Ã = D^{-1/2} · A · D^{-1/2}
"""

import numpy as np
import scipy.sparse as sp
import torch


def build_interaction_matrix(train_df, n_users: int, n_items: int) -> sp.csr_matrix:
    """
    Build the binary user-item interaction matrix R of shape (n_users, n_items)
    from the training DataFrame.
    """
    rows = train_df["user_idx"].values.astype(np.int32)
    cols = train_df["item_idx"].values.astype(np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return R


def build_adjacency_matrix(R: sp.csr_matrix) -> sp.csr_matrix:
    """
    Build the full bipartite adjacency matrix A of shape
    (n_users + n_items) × (n_users + n_items).

        A = [  0    R  ]
            [  R^T  0  ]
    """
    n_users, n_items = R.shape
    n = n_users + n_items

    # Upper-right block: R
    # Lower-left  block: R.T
    upper = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    lower = sp.hstack([R.T, sp.csr_matrix((n_items, n_items))])
    A = sp.vstack([upper, lower]).tocsr()
    return A


def normalize_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute the symmetrically normalised adjacency matrix:
        Ã = D^{-1/2} · A · D^{-1/2}

    where D is the diagonal degree matrix of A.
    Zero-degree nodes are handled gracefully (D^{-1/2} = 0 for degree-0 nodes).
    """
    rowsum = np.array(A.sum(axis=1)).flatten()          # degree vector
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0, out=np.zeros_like(rowsum))
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.tocsr()


def sparse_to_torch(A: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """
    Convert a scipy sparse CSR matrix to a PyTorch sparse COO tensor.
    """
    A = A.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([A.row, A.col])).long()
    values  = torch.from_numpy(A.data)
    shape   = torch.Size(A.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)
    return sparse_tensor


def build_norm_adj(train_df, n_users: int, n_items: int, device: torch.device):
    """
    Convenience function: build R → A → Ã → torch sparse tensor.

    Returns
    -------
    norm_adj : torch.Tensor  sparse COO tensor of shape (n_users+n_items, n_users+n_items)
    R        : sp.csr_matrix  raw interaction matrix
    """
    R = build_interaction_matrix(train_df, n_users, n_items)
    A = build_adjacency_matrix(R)
    A_norm = normalize_adjacency(A)
    norm_adj = sparse_to_torch(A_norm, device)
    return norm_adj, R
