"""
models/schnet_wrapper.py — SchNet wrapper using PyG's built-in implementation.

SchNet (Schütt et al., 2018) is an invariant GNN that uses:
  - Continuous-filter convolution (cfconv) on 3D coordinates
  - Gaussian RBF distance expansion
  - Interaction blocks with residual connections
  - Optional dipole readout via learned atomic charges

This wrapper provides a clean interface consistent with the other models
and handles the dipole vs. scalar readout switching.

Reference:
    Schütt et al., "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions", NeurIPS 2017 / JCP 2018.

Expected QM9 performance:
    - μ MAE ≈ 0.033 D (with dipole=True)
    - Δε MAE ≈ 63 meV
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet


class SchNetWrapper(nn.Module):
    """Thin wrapper around PyG's SchNet for consistent interface.

    Args:
        hidden_channels: Feature dimensionality. Default: 128.
        num_filters: Number of filters in cfconv. Default: 128.
        num_interactions: Number of interaction blocks. Default: 6.
        num_gaussians: Number of Gaussian RBF functions. Default: 50.
        cutoff: Distance cutoff (Å). Default: 10.0.
        dipole: If True, predict dipole moment as vector norm
                (uses learned charges × positions). Default: False.
        max_num_neighbors: Max neighbors per atom. Default: 32.
        mean: Target mean for output scaling. Default: None.
        std: Target std for output scaling. Default: None.
    """

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=10.0,
                 dipole=False, max_num_neighbors=32, mean=None, std=None):
        super().__init__()

        self.model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            dipole=dipole,
            max_num_neighbors=max_num_neighbors,
            mean=mean,
            std=std,
            atomref=None,  # No atomref for μ/gap (intensive properties)
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Batch with z, pos, batch.

        Returns:
            Predictions of shape (batch_size,).
        """
        out = self.model(data.z, data.pos, data.batch)
        return out.squeeze(-1)
