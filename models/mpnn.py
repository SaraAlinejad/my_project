"""
models/mpnn.py — MPNN baseline adapted for PyG's QM9 dataset format.

This is the invariant baseline model that uses:
  - One-hot encoded atomic numbers as node features
  - Gaussian RBF expansion of bond distances as edge features
  - NNConv message passing layers with residual connections
  - Mean + Sum pooling → MLP readout

It operates on the **bond graph** (edges from QM9's covalent bonds) and
adds RBF-expanded distance features. This keeps the 2D/topological inductive
bias of the original MPNN while adding minimal 3D information through
the bond lengths.

Reference:
    Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion of distances.

    Expands scalar distances into a vector of Gaussian basis functions
    centered at evenly spaced values between 0 and cutoff.

    Args:
        n_rbf: Number of Gaussian basis functions.
        cutoff: Maximum distance (Å). Centers are spaced in [0, cutoff].
    """

    def __init__(self, n_rbf=20, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # Evenly spaced Gaussian centers
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)

        # Width parameter (inverse of variance)
        width = (cutoff / n_rbf) * 0.5
        self.register_buffer("width", torch.tensor(width))

    def forward(self, dist):
        """
        Args:
            dist: Interatomic distances, shape (n_edges,).
        Returns:
            Gaussian RBF expansion, shape (n_edges, n_rbf).
        """
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))


class MPNN(nn.Module):
    """Invariant Message Passing Neural Network for QM9.

    Uses one-hot atomic number embeddings and Gaussian RBF bond features
    with NNConv layers. Compatible with PyG's QM9 dataset format
    (uses data.z, data.pos, data.edge_index, data.batch).

    Args:
        hidden_dim: Hidden layer dimension. Default: 128.
        n_conv_layers: Number of NNConv message passing layers. Default: 3.
        n_targets: Number of prediction targets. Default: 1.
        dropout: Dropout rate in prediction head. Default: 0.2.
        n_rbf: Number of Gaussian RBF basis functions. Default: 20.
        cutoff: Distance cutoff for RBF expansion (Å). Default: 10.0.
        max_z: Maximum atomic number (+1) for embedding. Default: 10.
    """

    def __init__(self, hidden_dim=128, n_conv_layers=3, n_targets=1,
                 dropout=0.2, n_rbf=20, cutoff=10.0, max_z=10):
        super().__init__()

        self.n_conv_layers = n_conv_layers
        self.dropout = dropout
        self.cutoff = cutoff

        # Atom embedding: atomic number → hidden_dim
        self.atom_embed = nn.Embedding(max_z, hidden_dim)

        # RBF expansion of bond distances
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        edge_dim = n_rbf

        # NNConv message passing layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_conv_layers):
            edge_nn = Sequential(
                Linear(edge_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr="mean"))
            self.bns.append(BatchNorm1d(hidden_dim))

        # Prediction head (mean + sum pooling → 2 * hidden_dim)
        self.lin1 = Linear(2 * hidden_dim, hidden_dim)
        self.bn_head = BatchNorm1d(hidden_dim)
        self.lin2 = Linear(hidden_dim, n_targets)

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Batch with z, pos, edge_index, batch.
                  edge_index should be the bond graph from QM9.

        Returns:
            Predictions of shape (batch_size, n_targets).
        """
        z = data.z  # atomic numbers (n_atoms,)
        pos = data.pos  # 3D coordinates (n_atoms, 3)
        edge_index = data.edge_index  # bond connectivity (2, n_edges)
        batch = data.batch  # graph membership (n_atoms,)

        # Node features: embed atomic numbers
        h = self.atom_embed(z)  # (n_atoms, hidden_dim)

        # Edge features: RBF expansion of bond distances
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)  # (n_edges,)
        edge_attr = self.rbf(dist)  # (n_edges, n_rbf)

        # Message passing with residual connections
        for i in range(self.n_conv_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.bns[i](h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # Residual connection

        # Graph-level readout: concatenate mean and sum pooling
        h_mean = global_mean_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_sum], dim=1)

        # Prediction head
        out = self.lin1(h_graph)
        out = self.bn_head(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)

        return out.squeeze(-1)  # (batch_size,)
