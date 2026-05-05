"""
models/painn.py — Polarizable Atom Interaction Neural Network (PaiNN).

PaiNN is an E(3)-equivariant message passing neural network that maintains
both scalar (invariant) and vector (equivariant) per-atom representations.
The key advantage for dipole prediction is the **vector dipole readout**:
per-atom partial charges + equivariant atomic dipoles, summed and normed.

Architecture overview:
    1. Embed atomic numbers into scalar features s (F dims) and zero vectors V (F×3)
    2. For each interaction block:
       a. Message: propagate scalar and vector features via distance-filtered messages
       b. Update: mix scalar/vector channels via gated equivariant block
    3. Readout:
       - Dipole: q_i·r_i + μ_i (vector), then ||·||₂ → scalar
       - Scalar: linear on scalar features → mean pool

Reference:
    Schütt, Unke, Gastegger, "Equivariant message passing for the prediction
    of tensorial properties and molecular spectra", ICML 2021.

Expected QM9 performance:
    - μ MAE ≈ 0.012 D (12 mD) — 3× better than SchNet due to vector readout
    - Δε MAE ≈ 45.7 meV
    - ~600k parameters with F=128, 3 interactions, 20 RBFs
"""

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_mean_pool


# =============================================================================
# Building blocks
# =============================================================================

class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope.

    f_cut(r) = 0.5 * (cos(π * r / r_cut) + 1)  for r < r_cut
             = 0                                  for r >= r_cut
    """

    def __init__(self, cutoff):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float))

    def forward(self, dist):
        """Apply cutoff envelope.

        Args:
            dist: Interatomic distances (n_edges,).
        Returns:
            Cutoff values (n_edges,).
        """
        return 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion.

    Expands distances into n_rbf Gaussian basis functions evenly spaced
    between 0 and cutoff.

    Args:
        n_rbf: Number of basis functions.
        cutoff: Maximum distance (Å).
    """

    def __init__(self, n_rbf=20, cutoff=5.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", torch.tensor(cutoff / n_rbf))

    def forward(self, dist):
        """
        Args:
            dist: Distances (n_edges,).
        Returns:
            RBF expansion (n_edges, n_rbf).
        """
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.centers) / self.widths) ** 2)


class RadialFilter(nn.Module):
    """Distance filter network: RBF → Linear → SiLU → Linear.

    Maps Gaussian RBF features to output_dim filter weights,
    modulated by a cosine cutoff envelope.

    Matches SchNetPack: intermediate dimension is n_atom_basis (F),
    NOT the output dimension (3F). This keeps the filter network
    appropriately sized.

    Args:
        n_rbf: Number of RBF inputs.
        n_atom_basis: Intermediate feature dimension (F).
        output_dim: Output filter dimension (typically 3F).
        cutoff: Cutoff distance for the envelope.
    """

    def __init__(self, n_rbf, n_atom_basis, output_dim, cutoff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, output_dim),
        )
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, dist, rbf):
        """
        Args:
            dist: Raw distances (n_edges,).
            rbf: RBF expansion (n_edges, n_rbf).
        Returns:
            Filter weights (n_edges, hidden_dim).
        """
        W = self.net(rbf)  # (n_edges, F)
        C = self.cutoff_fn(dist)  # (n_edges,)
        return W * C.unsqueeze(-1)  # (n_edges, F)


# =============================================================================
# PaiNN interaction blocks
# =============================================================================

class PaiNNMessage(nn.Module):
    """PaiNN message passing sub-layer.

    Computes scalar and vector messages from neighbors:
        Δs_i = Σ_j φ_s(s_j) ⊙ W_s(||r_ij||)
        ΔV_i = Σ_j V_j ⊙ W_vv(||r_ij||) + s_j · W_vs(||r_ij||) · r̂_ij

    Args:
        hidden_dim: Feature dimension (F).
        n_rbf: Number of RBF basis functions.
        cutoff: Distance cutoff (Å).
    """

    def __init__(self, hidden_dim, n_rbf, cutoff):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Scalar feature projection: s_j → 3F dims
        # SchNetPack uses a SINGLE linear (no activation, no bias)
        # The nonlinearity comes from the filter × projection product
        self.scalar_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Distance filter: RBF → 3F filter weights
        # Intermediate dim = F (not 3F) — matches SchNetPack
        self.dist_filter = RadialFilter(n_rbf, hidden_dim, 3 * hidden_dim, cutoff)

    def forward(self, s, V, edge_index, dist, rbf, dir_ij):
        """Compute scalar and vector messages from neighbors.

        Args:
            s: Scalar features (n_atoms, F).
            V: Vector features (n_atoms, 3, F).
            edge_index: Edge connectivity (2, n_edges).
            dist: Interatomic distances (n_edges,).
            rbf: RBF expansion (n_edges, n_rbf).
            dir_ij: Unit direction vectors r̂_ij (n_edges, 3).

        Returns:
            ds: Scalar updates (n_atoms, F).
            dV: Vector updates (n_atoms, 3, F).
        """
        src, dst = edge_index
        n_atoms = s.size(0)

        # Distance filter
        W = self.dist_filter(dist, rbf)  # (n_edges, 3F)

        # Scalar feature projection from sources (single linear, no activation)
        phi = self.scalar_proj(s[src])  # (n_edges, 3F)

        # Combined: element-wise product of message and filter
        combined = phi * W  # (n_edges, 3F)
        W_s, W_vv, W_vs = combined.split(self.hidden_dim, dim=-1)

        # --- Scalar messages: aggregate W_s to destination ---
        ds = torch.zeros(n_atoms, self.hidden_dim, device=s.device, dtype=s.dtype)
        ds.scatter_add_(0, dst.unsqueeze(-1).expand_as(W_s), W_s)

        # --- Vector messages ---
        # Part 1: V_j ⊙ W_vv (filter existing vectors from neighbors)
        V_src = V[src]  # (n_edges, 3, F)
        dV_vv = V_src * W_vv.unsqueeze(1)  # (n_edges, 3, F)

        # Part 2: W_vs · r̂_ij (create new vectors from scalar info + direction)
        dV_vs = W_vs.unsqueeze(1) * dir_ij.unsqueeze(-1)  # (n_edges, 3, F)

        # Sum both parts
        dV_total = dV_vv + dV_vs  # (n_edges, 3, F)

        # Aggregate to destination atoms
        dV = torch.zeros(n_atoms, 3, self.hidden_dim, device=s.device, dtype=s.dtype)
        dV.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(dV_total), dV_total)

        return ds, dV


class PaiNNUpdate(nn.Module):
    """PaiNN update sub-layer (gated equivariant block).

    Mixes scalar and vector channels:
        U_Vi = U · V_i     (linear transform of vectors)
        V_Vi = V_mat · V_i (another linear transform)

        a_vv, a_sv, a_ss = split(Linear([||U_Vi||, s_i]))

        ΔV_i = a_vv · U_Vi
        Δs_i = a_ss + a_sv · ⟨U_Vi, V_Vi⟩

    Args:
        hidden_dim: Feature dimension (F).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear transforms for vector channels (no bias — equivariant)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_mat = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP for gating: [||U·V||, s] → [a_vv, a_sv, a_ss]
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s, V):
        """
        Args:
            s: Scalar features (n_atoms, F).
            V: Vector features (n_atoms, 3, F).

        Returns:
            ds: Scalar updates (n_atoms, F).
            dV: Vector updates (n_atoms, 3, F).
        """
        # Apply linear transforms to vector channels
        # V is (n_atoms, 3, F), we transform along the F dimension
        U_V = self.U(V)  # (n_atoms, 3, F)
        V_V = self.V_mat(V)  # (n_atoms, 3, F)

        # Compute ||U·V_i|| — norm of each atom's transformed vector
        U_V_norm = U_V.norm(dim=1)  # (n_atoms, F)

        # Gating MLP: concat [||U·V||, s] → 3F outputs
        gate_input = torch.cat([U_V_norm, s], dim=-1)  # (n_atoms, 2F)
        gates = self.gate_mlp(gate_input)  # (n_atoms, 3F)
        a_vv, a_sv, a_ss = gates.split(self.hidden_dim, dim=-1)

        # Vector update: a_vv · U·V
        dV = a_vv.unsqueeze(1) * U_V  # (n_atoms, 3, F)

        # Scalar update: a_ss + a_sv · ⟨U·V, V·V⟩
        # Inner product over the spatial dimension (dim=1)
        inner = (U_V * V_V).sum(dim=1)  # (n_atoms, F)
        ds = a_ss + a_sv * inner  # (n_atoms, F)

        return ds, dV


class PaiNNInteraction(nn.Module):
    """One PaiNN interaction block = Message + Update.

    Args:
        hidden_dim: Feature dimension (F).
        n_rbf: Number of RBF basis functions.
        cutoff: Distance cutoff (Å).
    """

    def __init__(self, hidden_dim, n_rbf, cutoff):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, n_rbf, cutoff)
        self.update = PaiNNUpdate(hidden_dim)

    def forward(self, s, V, edge_index, dist, rbf, dir_ij):
        """
        Args:
            s: Scalar features (n_atoms, F).
            V: Vector features (n_atoms, 3, F).
            edge_index: Graph connectivity (2, n_edges).
            dist: Distances (n_edges,).
            rbf: RBF expansion (n_edges, n_rbf).
            dir_ij: Unit direction vectors (n_edges, 3).

        Returns:
            s_out: Updated scalar features (n_atoms, F).
            V_out: Updated vector features (n_atoms, 3, F).
        """
        # Message passing
        ds_msg, dV_msg = self.message(s, V, edge_index, dist, rbf, dir_ij)
        s = s + ds_msg
        V = V + dV_msg

        # Update (gated equivariant mixing)
        ds_upd, dV_upd = self.update(s, V)
        s = s + ds_upd
        V = V + dV_upd

        return s, V


# =============================================================================
# Readout heads
# =============================================================================

class ScalarReadout(nn.Module):
    """Scalar (intensive) property readout.

    Per-atom scalar → linear → sum over atoms, then divide by n_atoms.
    Suitable for HOMO-LUMO gap and other intensive properties.

    Args:
        hidden_dim: Input feature dimension.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, V, pos, batch):
        """
        Args:
            s: Scalar features (n_atoms, F).
            V: Vector features (n_atoms, 3, F). Unused for scalar readout.
            pos: Atom positions (n_atoms, 3). Unused.
            batch: Batch indices (n_atoms,).

        Returns:
            Predictions (batch_size,).
        """
        out = self.net(s).squeeze(-1)  # (n_atoms,)
        return global_mean_pool(out.unsqueeze(-1), batch).squeeze(-1)


class DipoleMomentReadout(nn.Module):
    """Vector dipole moment readout.

    Predicts the dipole moment as:
        μ_mol = Σ_i (q_i · r_i + μ_i)
        |μ| = ||μ_mol||₂

    where q_i is a learned partial charge (scalar) and μ_i is a learned
    per-atom equivariant dipole (3D vector from L=1 features).

    This is the KEY architectural advantage of PaiNN for dipole prediction —
    it uses the correct physical inductive bias that dipole = charge × position
    plus local atomic dipoles.

    Args:
        hidden_dim: Input feature dimension.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Predict per-atom partial charge from scalar features
        self.charge_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Project vector features to per-atom dipole (F → 1 along feature dim)
        self.dipole_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, s, V, pos, batch):
        """
        Args:
            s: Scalar features (n_atoms, F).
            V: Vector features (n_atoms, 3, F).
            pos: Atom positions (n_atoms, 3).
            batch: Batch indices (n_atoms,).

        Returns:
            Predicted dipole magnitude (batch_size,).
        """
        batch_size = batch.max().item() + 1

        # --- Center positions per molecule ---
        # This is critical: without centering, q*r depends on the
        # choice of origin when learned charges don't sum to zero.
        # SchNetPack centers at the geometric center before computing dipole.
        atom_counts = torch.zeros(batch_size, 1, device=pos.device, dtype=pos.dtype)
        atom_counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(pos[:, :1]))
        pos_sum = torch.zeros(batch_size, 3, device=pos.device, dtype=pos.dtype)
        pos_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(pos), pos)
        center = pos_sum / atom_counts  # (batch_size, 3)
        pos_centered = pos - center[batch]  # (n_atoms, 3)

        # Per-atom partial charge (raw)
        q = self.charge_net(s)  # (n_atoms, 1)

        # --- Enforce charge neutrality per molecule ---
        # Without this, Σq_i ≠ 0 and the dipole q*r is origin-dependent.
        # Subtracting the per-molecule mean charge ensures Σq_i = 0.
        # This is what SchNetPack's DipoleMoment does internally.
        q_sum = torch.zeros(batch_size, 1, device=q.device, dtype=q.dtype)
        q_sum.scatter_add_(0, batch.unsqueeze(-1), q)
        q_mean = q_sum / atom_counts  # (batch_size, 1)
        q = q - q_mean[batch]  # now Σ_i q_i = 0 per molecule

        # Charge × centered position contribution to dipole
        mu_charge = q * pos_centered  # (n_atoms, 3)

        # Per-atom equivariant dipole from vector features
        # V is (n_atoms, 3, F), project F → 1
        mu_atomic = self.dipole_proj(V).squeeze(-1)  # (n_atoms, 3)

        # Total per-atom dipole contribution
        mu_atom = mu_charge + mu_atomic  # (n_atoms, 3)

        # Sum over atoms per molecule to get molecular dipole vector
        mu_mol = torch.zeros(batch_size, 3, device=s.device, dtype=s.dtype)
        mu_mol.scatter_add_(0, batch.unsqueeze(-1).expand_as(mu_atom), mu_atom)

        # Return L2 norm (scalar dipole moment)
        # eps inside sqrt prevents NaN gradient when ||μ|| → 0
        # (grad of sqrt(x) = 1/(2√x) → ∞ as x → 0)
        return (mu_mol ** 2).sum(dim=-1).add(1e-8).sqrt()  # (batch_size,)


# =============================================================================
# Full PaiNN model
# =============================================================================

class PaiNN(nn.Module):
    """Polarizable Atom Interaction Neural Network (PaiNN).

    E(3)-equivariant message passing network with scalar (invariant) and
    vector (equivariant) per-atom representations. Supports both scalar
    property prediction and vector dipole moment readout.

    Args:
        hidden_dim: Feature dimension (F). Default: 128.
        n_interactions: Number of PaiNN interaction blocks. Default: 3.
        n_rbf: Number of Gaussian RBF basis functions. Default: 20.
        cutoff: Radius cutoff for neighbor graph (Å). Default: 5.0.
        max_z: Maximum atomic number for embedding. Default: 10.
        readout: 'dipole' for vector dipole, 'scalar' for intensive properties.
        max_neighbors: Maximum neighbors per atom. Default: 32.
    """

    def __init__(self, hidden_dim=128, n_interactions=3, n_rbf=20,
                 cutoff=5.0, max_z=10, readout="dipole", max_neighbors=32):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # Atom embedding
        self.atom_embed = nn.Embedding(max_z, hidden_dim)

        # Radial basis functions
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_dim, n_rbf, cutoff)
            for _ in range(n_interactions)
        ])

        # Readout head
        if readout == "dipole":
            self.readout = DipoleMomentReadout(hidden_dim)
        elif readout == "scalar":
            self.readout = ScalarReadout(hidden_dim)
        else:
            raise ValueError(f"Unknown readout: {readout}. Use 'dipole' or 'scalar'.")

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters following best practices."""
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
            data: PyG Batch with z (atomic numbers), pos (3D coordinates),
                  batch (graph membership).

        Returns:
            Predictions of shape (batch_size,).
        """
        z = data.z        # (n_atoms,)
        pos = data.pos    # (n_atoms, 3)
        batch = data.batch  # (n_atoms,)

        # Build radius graph
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_neighbors,
        )

        # Compute pairwise distances and unit direction vectors
        src, dst = edge_index
        diff = pos[dst] - pos[src]  # r_ij = r_j - r_i  (n_edges, 3)
        dist = diff.norm(dim=-1, keepdim=False)  # (n_edges,)
        # Avoid division by zero
        dir_ij = diff / (dist.unsqueeze(-1) + 1e-8)  # (n_edges, 3)

        # RBF expansion
        rbf = self.rbf(dist)  # (n_edges, n_rbf)

        # Initial features
        s = self.atom_embed(z)  # (n_atoms, F)
        V = torch.zeros(z.size(0), 3, self.hidden_dim,
                        device=z.device, dtype=s.dtype)  # (n_atoms, 3, F)

        # Interaction blocks
        for interaction in self.interactions:
            s, V = interaction(s, V, edge_index, dist, rbf, dir_ij)

        # Readout
        return self.readout(s, V, pos, batch)
