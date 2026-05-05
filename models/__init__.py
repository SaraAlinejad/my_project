"""
models/ — GNN architectures for QM9 molecular property prediction.

All three models are implemented from scratch for this project:
    - MPNN: Invariant Message Passing Neural Network (NNConv, bond graph)
    - SchNet: Invariant continuous-filter convolution (3D, radius graph)
    - PaiNN: Equivariant Polarizable Atom Interaction Network (3D, vector dipole)

Usage:
    from models import get_model
    model = get_model("painn", target="dipole")
"""

from models.mpnn import MPNN
from models.schnet import SchNet
from models.painn import PaiNN


def get_model(name, target="dipole", **kwargs):
    """Factory function to create a model by name.

    Args:
        name: One of 'mpnn', 'schnet', 'painn'.
        target: One of 'dipole', 'gap'.  Controls readout head.
        **kwargs: Override default hyperparameters.

    Returns:
        nn.Module instance.
    """
    name = name.lower().strip()

    if name == "mpnn":
        defaults = dict(
            hidden_dim=128,
            n_conv_layers=3,
            n_targets=1,
            dropout=0.2,
            n_rbf=20,
            cutoff=10.0,
        )
        defaults.update(kwargs)
        return MPNN(**defaults)

    elif name == "schnet":
        defaults = dict(
            hidden_channels=128,
            n_filters=128,
            n_interactions=6,
            n_gaussians=50,
            cutoff=10.0,
        )
        defaults.update(kwargs)
        # Use mean pooling for intensive properties (dipole, gap)
        return SchNet(readout="mean", **defaults)

    elif name == "painn":
        defaults = dict(
            hidden_dim=128,
            n_interactions=3,
            n_rbf=20,
            cutoff=5.0,
            max_z=10,
        )
        defaults.update(kwargs)
        readout = "dipole" if target == "dipole" else "scalar"
        return PaiNN(readout=readout, **defaults)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: mpnn, schnet, painn"
        )


__all__ = ["get_model", "MPNN", "SchNet", "PaiNN"]
