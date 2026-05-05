"""
models/schnet.py — SchNet: Continuous-Filter Convolutional Neural Network.

SchNet is an invariant GNN that operates on 3D molecular structures using:
  - Gaussian RBF expansion of interatomic distances
  - Continuous-filter convolution (cfconv): filter weights are generated
    from RBF-expanded distances via an MLP, then applied to atom features
  - Interaction blocks with residual connections
  - Scalar readout via sum pooling (extensive) or mean pooling (intensive)

Key insight: instead of fixed graph convolution filters, SchNet *generates*
the filter weights continuously from interatomic distances. This makes the
model naturally handle varying molecular geometries without a fixed grid.

Reference:
    Schütt et al., "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions", JCP 2018.

Expected QM9 performance:
    - μ MAE ≈ 0.033 D (33 mD)
    - Δε MAE ≈ 63 meV
"""

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_mean_pool, global_add_pool


# =============================================================================
# Building blocks
# =============================================================================

class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion.

    Expands scalar distances into n_gaussians basis functions evenly spaced
    between 0 and cutoff.

    Args:
        n_gaussians: Number of Gaussian centers.
        cutoff: Maximum distance (Å).
    """

    def __init__(self, n_gaussians=50, cutoff=10.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_gaussians)
        self.register_buffer("centers", centers)
        width = (cutoff / n_gaussians)
        self.register_buffer("width", torch.tensor(width))

    def forward(self, dist):
        """Expand distances into Gaussian basis.

        Args:
            dist: Interatomic distances (n_edges,).
        Returns:
            RBF expansion (n_edges, n_gaussians).
        """
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.centers) / self.width) ** 2)


class ShiftedSoftplus(nn.Module):
    """Shifted softplus activation: ssp(x) = ln(1 + exp(x)) - ln(2).

    The shift ensures ssp(0) = 0 (unlike standard softplus where f(0) = ln(2)).
    Used in SchNet instead of ReLU for smoother gradients with respect to
    atomic positions.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.log(torch.tensor(2.0)))

    def forward(self, x):
        return nn.functional.softplus(x) - self.shift


class ContinuousFilterConv(nn.Module):
    """Continuous-filter convolution (cfconv) layer.

    Instead of using fixed convolution weights, cfconv generates filter weights
    as a continuous function of interatomic distances:
        x_i' = Σ_j x_j ⊙ W(||r_ij||)

    where W(·) is a filter-generating network that takes RBF-expanded distances
    and outputs per-feature filter weights.

    Args:
        n_gaussians: Number of Gaussian RBF inputs.
        n_filters: Number of output filter dimensions.
        hidden_channels: Feature dimension of atom embeddings.
    """

    def __init__(self, n_gaussians, n_filters, hidden_channels):
        super().__init__()

        # Filter-generating network: RBF → filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters),
        )

        # Atom feature transform (applied after aggregation)
        self.atom_net = nn.Sequential(
            nn.Linear(hidden_channels, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, hidden_channels),
        )

    def forward(self, x, edge_index, rbf):
        """
        Args:
            x: Atom features (n_atoms, hidden_channels).
            edge_index: Graph connectivity (2, n_edges).
            rbf: RBF expansion of distances (n_edges, n_gaussians).

        Returns:
            Updated atom features (n_atoms, hidden_channels).
        """
        src, dst = edge_index

        # Generate distance-dependent filter weights
        W = self.filter_net(rbf)  # (n_edges, n_filters)

        # Gather source atom features and apply filter
        x_src = x[src]  # (n_edges, hidden_channels)

        # Truncate or pad to match filter dimension if needed
        # (in practice hidden_channels == n_filters in SchNet)
        filtered = x_src * W  # (n_edges, n_filters)

        # Aggregate filtered messages to destination atoms
        n_atoms = x.size(0)
        agg = torch.zeros(n_atoms, filtered.size(-1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(filtered), filtered)

        # Transform aggregated features
        return self.atom_net(agg)


class SchNetInteraction(nn.Module):
    """One SchNet interaction block.

    Applies continuous-filter convolution and adds the result as a
    residual connection to the input features.

    Args:
        hidden_channels: Atom feature dimension.
        n_gaussians: Number of RBF basis functions.
        n_filters: Number of cfconv filter channels.
    """

    def __init__(self, hidden_channels, n_gaussians, n_filters):
        super().__init__()
        self.cfconv = ContinuousFilterConv(n_gaussians, n_filters, hidden_channels)

    def forward(self, x, edge_index, rbf):
        """
        Args:
            x: Atom features (n_atoms, hidden_channels).
            edge_index: Graph connectivity (2, n_edges).
            rbf: RBF expansion (n_edges, n_gaussians).

        Returns:
            Updated atom features with residual (n_atoms, hidden_channels).
        """
        return x + self.cfconv(x, edge_index, rbf)


# =============================================================================
# Full SchNet model
# =============================================================================

class SchNet(nn.Module):
    """SchNet: Continuous-Filter Convolutional Neural Network.

    An invariant GNN that uses distance-dependent continuous filter
    convolutions on 3D molecular structures. Unlike PaiNN, SchNet
    only maintains scalar (invariant) per-atom features.

    Args:
        hidden_channels: Feature dimensionality. Default: 128.
        n_filters: Number of cfconv filters. Default: 128.
        n_interactions: Number of interaction blocks. Default: 6.
        n_gaussians: Number of Gaussian RBF functions. Default: 50.
        cutoff: Distance cutoff (Å). Default: 10.0.
        max_z: Maximum atomic number for embedding. Default: 10.
        readout: 'mean' for intensive properties (gap), 'sum' for extensive.
        max_neighbors: Maximum neighbors per atom. Default: 32.
    """

    def __init__(self, hidden_channels=128, n_filters=128, n_interactions=6,
                 n_gaussians=50, cutoff=10.0, max_z=10, readout="mean",
                 max_neighbors=32):
        super().__init__()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # Atom embedding: Z → hidden_channels
        self.atom_embed = nn.Embedding(max_z, hidden_channels)

        # Gaussian RBF expansion
        self.rbf = GaussianRBF(n_gaussians=n_gaussians, cutoff=cutoff)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, n_gaussians, n_filters)
            for _ in range(n_interactions)
        ])

        # Output network: atom features → scalar prediction
        self.output_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, 1),
        )

        # Pooling type
        self.readout_type = readout

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -1.0, 1.0)

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Batch with z, pos, batch.

        Returns:
            Predictions (batch_size,).
        """
        z = data.z
        pos = data.pos
        batch = data.batch

        # Build radius graph
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_neighbors,
        )

        # Compute pairwise distances
        src, dst = edge_index
        dist = (pos[dst] - pos[src]).norm(dim=-1)

        # RBF expansion
        rbf = self.rbf(dist)

        # Atom embeddings
        x = self.atom_embed(z)

        # Interaction blocks
        for interaction in self.interactions:
            x = interaction(x, edge_index, rbf)

        # Per-atom output
        x = self.output_net(x).squeeze(-1)  # (n_atoms,)

        # Pool to molecular level
        if self.readout_type == "mean":
            return global_mean_pool(x.unsqueeze(-1), batch).squeeze(-1)
        else:
            return global_add_pool(x.unsqueeze(-1), batch).squeeze(-1)
