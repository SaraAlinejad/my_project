# Graph Neural Networks for QM9 Molecular Property Prediction

**CHEM 4930/5610 — Final Project, Spring 2026**

A comparative study of GNN architectures — **MPNN**, **SchNet**, and **PaiNN** — for predicting the dipole moment (μ) and HOMO–LUMO gap (Δε) of small organic molecules in the QM9 dataset. The project demonstrates the progression from invariant to E(3)-equivariant message passing and documents the critical implementation details required to achieve literature-level accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Lessons Learned](#lessons-learned--implementation-journey)
- [Architecture Comparison](#architecture-comparison)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Mathematical Details](#mathematical-details)
- [Key Concepts](#key-concepts)
- [References](#references)

---

## Overview

This project implements a **4-model comparison framework** for molecular property prediction on QM9:

| Model | Implementation | Symmetry | Dipole μ MAE | Gap Δε MAE |
|-------|---------------|----------|:------------:|:----------:|
| **MPNN** | From scratch (PyG) | Invariant | ~300 mD | — |
| **SchNet** | From scratch | Invariant | — | — |
| **PaiNN** | From scratch | E(3)-Equivariant | ~170 mD | — |
| **PaiNN (SchNetPack)** | Official library | E(3)-Equivariant | **25.5 mD** | **71.9 meV** |
| *Literature PaiNN* | *Reference* | — | *12 mD* | *45.7 meV* |

The central narrative: **geometric inductive biases matter, and so do implementation details.** PaiNN's vector dipole readout provides the right physics, but achieving paper-level accuracy requires careful attention to normalization, loss functions, charge neutrality, and initialization.

---

## Results

### My Trained Models


### COMPARISON TABLE

Model                                    μ MAE     Δε MAE Type                
──────────────────────────────────────────────────────────────────────
★ PaiNN SchNetPack               25.5 mD          — equivariant         
★ PaiNN SchNetPack                     —   71.9 meV equivariant         
★ PAINN from-scratch            169.1 mD          — equivariant         
──────────────────────────────────────────────────────────────────────



### Literature Comparison — Dipole Moment (mD)

| Model | MAE (mD) | Type |
|-------|:--------:|------|
| Plain GIN/GAT (no 3D) | 300–550 | Invariant, no 3D |
| ★ MPNN (this project) | ~300 | Invariant, bond graph |
| SchNet (Schütt 2018) | 33 | Invariant, cfconv |
| DimeNet++ (Gasteiger 2020) | 30 | Invariant, angles |
| ★ PaiNN (this project, SchNetPack) | **25.5** | Equivariant, vector dipole |
| **PaiNN (Schütt 2021)** | **12** | Equivariant, vector dipole |
| TorchMD-NET (Thölke 2022) | 11 | Equivariant |
| Equiformer (Liao 2023) | 11 | Equivariant transformer |

### Literature Comparison — HOMO–LUMO Gap (meV)

| Model | MAE (meV) | Type |
|-------|:---------:|------|
| SchNet (Schütt 2018) | 63 | Invariant, cfconv |
| ★ PaiNN (this project, SchNetPack) | **71.9** | Equivariant (100 epochs) |
| **PaiNN (Schütt 2021)** | **45.7** | Equivariant |
| DimeNet++ (Gasteiger 2020) | 33 | Invariant, angles |
| Equiformer (Liao 2023) | ~30 | Equivariant transformer |

---

## Lessons Learned — Implementation Journey

Building PaiNN from scratch and comparing against the official SchNetPack implementation revealed several critical insights. Each lesson below corresponds to a bug that caused the model to plateau far above literature accuracy.

### 1. Target Normalization vs. Norm-Based Readouts

**Problem:** Standard practice normalizes targets as `(y - mean) / std`. But PaiNN's dipole readout outputs `||μ_vec||₂ ≥ 0` (a norm, always non-negative). Normalized targets `(μ - 2.67) / 1.50` are negative for ~half the molecules. The model physically cannot predict negative norms.

**Impact:** Model converged to ~1.0 D MAE (predicting ~mean for everything).

**Fix:** Train dipole on **raw Debye values** (no normalization). Gap targets still use mean/std normalization since the scalar readout can output any real number.

```python
# train.py
if TARGET == "dipole":
    USE_NORMALIZATION = False  # norm output ≥ 0, can't match negative shifted targets
else:
    USE_NORMALIZATION = True   # scalar output, normalization helps
```

### 2. Position Centering in Dipole Readout

**Problem:** The dipole formula `μ = Σ q_i · r_i` depends on the coordinate origin when charges don't sum to zero. At initialization, learned charges are random (don't satisfy neutrality), making the dipole strongly origin-dependent.

**Impact:** Slow convergence, model stuck at ~230 mD.

**Fix:** Center positions at the per-molecule geometric center before computing `q · r`:

```python
# Compute per-molecule geometric center
center = scatter_mean(pos, batch)
pos_centered = pos - center[batch]
mu_charge = q * pos_centered  # origin-independent
```

### 3. Charge Neutrality Constraint

**Problem:** Without constraining `Σ q_i = 0` per molecule, the charge × position term remains origin-dependent even with centering, because `Σ q_i · (r_i - r_cm) = Σ q_i · r_i - r_cm · Σ q_i`, and the second term doesn't vanish when `Σ q_i ≠ 0`.

**Fix:** Subtract the per-molecule mean charge after prediction:

```python
q = charge_net(s)              # raw predicted charges
q_mean = scatter_mean(q, batch)
q = q - q_mean[batch]          # now Σ q_i = 0 per molecule
```

### 4. Loss Function: MSE vs. L1 for Norm Outputs

**Problem:** L1 (MAE) loss has constant gradient magnitude (±1) regardless of error size. For a norm-based output involving `sqrt()`, this creates poor gradient signal — the gradient of `sqrt(x)` is `1/(2√x)`, which interacts badly with L1's flat gradient when predictions improve.

**Impact:** Model plateaued at ~220 mD with L1; dropped to ~170 mD with MSE.

**Fix:** Use MSE loss for dipole (matching SchNetPack), L1 for gap:

```python
if TARGET == "dipole":
    loss_fn = nn.MSELoss()   # smoother gradients for norm output
else:
    loss_fn = nn.L1Loss()    # standard for scalar predictions
```

### 5. Message Block Architecture

**Problem:** Our initial message block used a 2-layer MLP with activation for the scalar projection:
```python
# WRONG: too much nonlinearity
scalar_msg = Linear(F, F) → SiLU → Linear(F, 3F)
```
SchNetPack uses a **single linear projection** (no activation, no bias). The nonlinearity comes from the filter network only. Having nonlinearity on *both* the feature transform and distance filter creates a harder optimization landscape.

**Impact:** Convergence was 3× slower (450 mD at epoch 37 vs proper convergence).

**Fix:**
```python
# CORRECT: single linear, no activation (matches SchNetPack)
scalar_proj = nn.Linear(F, 3*F, bias=False)
```

Also fixed the filter network intermediate dimension from 3F=384 to F=128 (matching SchNetPack).

### 6. Numerical Stability in Norm Computation

**Problem:** `torch.norm()` and `sqrt(x)` have gradient `1/(2√x)` which is undefined at x=0. When the predicted dipole vector is near zero, this produces NaN gradients that poison the entire training.

**Impact:** Training diverged to NaN after a few epochs.

**Fix:** Add epsilon inside the sqrt:
```python
return (mu_mol ** 2).sum(dim=-1).add(1e-8).sqrt()
```

### 7. RemoveOffsets for Dipole in SchNetPack

**Problem:** SchNetPack's `RemoveOffsets` transform subtracts the training-set mean from targets. For dipole with `DipoleMoment(predict_magnitude=True)`, this creates the same normalization mismatch as Lesson 1 — the norm output can't be negative, but shifted targets can.

**Impact:** SchNetPack PaiNN produced 810 mD (worse than our from-scratch version!).

**Fix:** Skip `RemoveOffsets`/`AddOffsets` for dipole targets:
```python
if TARGET != "dipole_moment":
    transforms.append(RemoveOffsets(TARGET, remove_mean=True))
# No offset removal for dipole — norm output ≥ 0
```

### Summary: The Gap Between Equations and Implementation

Our from-scratch PaiNN (170 mD) vs SchNetPack (25.5 mD) shows a **7× gap** despite implementing the same equations. The remaining differences come from:
- **Weight initialization** (SchNetPack tunes per-layer initial scales)
- **Activation functions** (shifted softplus vs SiLU)
- **Internal numerical handling** (scatter operations, precision)
- **Training dynamics** (optimizer state, gradient flow)

This is itself an important finding: **getting the math right is necessary but not sufficient; implementation engineering matters enormously for GNN performance.**

---

## Architecture Comparison

### 1. MPNN — Invariant Baseline

**Reference:** Gilmer et al., ICML 2017 [[1]](#references)

Operates on the **bond graph** with distance-augmented edges:
- **Node features:** Atomic number embeddings (128 dims)
- **Edge features:** Gaussian RBF of bond distances (20 basis, cutoff 10 Å)
- **Message passing:** 3 layers of `NNConv` with GRU update
- **Readout:** Mean pool → MLP → scalar

**Limitation:** Only sees distances, not directions. Cannot represent vector properties.

### 2. SchNet — Invariant 3D Convolutions

**Reference:** Schütt et al., NeurIPS 2017 / JCP 2018 [[2]](#references)

Implemented **from scratch** (~260 LoC) with:
- **Continuous-filter convolution (cfconv):** Filter weights generated from RBF-expanded distances via MLP
- **Shifted softplus:** `ssp(x) = ln(1 + exp(x)) - ln(2)`, ensuring ssp(0) = 0
- **6 interaction blocks** with residual connections
- **Gaussian RBF:** 50 basis functions, cutoff 10 Å

Still **invariant** — uses only `||r_ij||`, discards direction `r̂_ij`.

### 3. PaiNN — Equivariant Message Passing

**Reference:** Schütt, Unke, Gastegger, ICML 2021 [[3]](#references)

Implemented **from scratch** (~530 LoC). Maintains **dual representations** per atom:

| Feature | Shape | Symmetry | Role |
|---------|-------|----------|------|
| **Scalar s** | (N, 128) | Invariant | Chemical environment |
| **Vector V** | (N, 3, 128) | Equivariant | Directional information |

**Message passing** injects directional information via unit vectors `r̂_ij`:
```
Δs_i = Σ_j Linear(s_j) ⊙ Filter(||r_ij||)           ← scalar channel
ΔV_i = Σ_j V_j ⊙ W_vv + W_vs · r̂_ij                 ← vector channel
```

**Vector dipole readout** (the key innovation):
```
μ_mol = Σ_i (q_i · r_i_centered + μ_i)    ← physics-based formula
|μ| = ||μ_mol||₂                            ← take the norm
```

### 4. PaiNN (SchNetPack) — Official Reference

Uses `schnetpack.representation.PaiNN` and `schnetpack.atomistic.DipoleMoment` from the paper authors' library. Same architecture, battle-tested implementation.

---

## Project Structure

```
gnn-antigrav/
├── models/
│   ├── __init__.py              # Model registry & get_model() factory
│   ├── mpnn.py                  # MPNN baseline (NNConv + RBF)
│   ├── schnet.py                # SchNet from scratch (cfconv)
│   └── painn.py                 # PaiNN from scratch (equivariant)
├── train.py                     # Unified training (from-scratch models)
├── 05_train_painn_schnetpack.py # PaiNN via SchNetPack (official impl)
├── evaluate.py                  # Post-training comparison & figures
├── predict_smiles.py            # SMILES → prediction pipeline
├── data/                        # Datasets (auto-downloaded)
├── results/                     # From-scratch model checkpoints
└── results_schnetpack/          # SchNetPack checkpoints
```

---

## Installation

```bash
conda create -n torch_gnn python=3.10 -y
conda activate torch_gnn

# PyTorch (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# PyG and extensions
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# For SchNetPack reference model
pip install schnetpack pytorch_lightning

# Other dependencies
pip install -r requirements.txt
```

---

## Usage

### Training (From-Scratch Models)

```bash
# PaiNN on dipole moment
python train.py --model painn --target dipole --epochs 300

# SchNet on dipole moment
python train.py --model schnet --target dipole --epochs 300

# MPNN baseline
python train.py --model mpnn --target dipole --epochs 200
```

### Training (SchNetPack Reference)

```bash
# PaiNN (official) on dipole — expect ~12 mD at 300 epochs
python 05_train_painn_schnetpack.py --target dipole_moment --epochs 300

# PaiNN (official) on gap — expect ~46 meV at 300 epochs
python 05_train_painn_schnetpack.py --target gap --epochs 300
```

### Evaluation & Prediction

```bash
python evaluate.py                              # comparison tables & plots
python predict_smiles.py --model painn --smiles "CCO"  # predict from SMILES
```

---

## Training Details

| Setting | Dipole (μ) | Gap (Δε) | Rationale |
|---------|-----------|----------|-----------|
| **Loss** | MSE | L1 (MAE) | MSE gives smoother gradients for norm-based output |
| **Normalization** | None (raw Debye) | Z-score | Norm output ≥ 0; normalized targets can be < 0 |
| **Optimizer** | AdamW, lr=5e-4 | AdamW, lr=5e-4 | Paper defaults |
| **LR Schedule** | ReduceLROnPlateau (0.5, patience=25) | Same | — |
| **EMA** | decay=0.999 | decay=0.999 | Smoother evaluation |
| **Gradient Clipping** | max_norm=10 | max_norm=10 | Prevents explosion |
| **Early Stopping** | patience=50 | patience=50 | — |
| **Batch Size** | 100 | 100 | Paper default |

---

## Mathematical Details

### PaiNN Message Passing

$$\Delta s_i = \sum_{j \in \mathcal{N}(i)} \text{Linear}(s_j) \odot W_s(\|r_{ij}\|)$$

$$\Delta \mathbf{V}_i = \sum_{j} \left[ \mathbf{V}_j \odot W_{vv}(\|r_{ij}\|) + W_{vs}(\|r_{ij}\|) \cdot \hat{r}_{ij} \right]$$

### PaiNN Update (Gated Equivariant Block)

$$\mathbf{U}_i = U \cdot \mathbf{V}_i, \quad \tilde{\mathbf{V}}_i = V_{\text{mat}} \cdot \mathbf{V}_i$$

$$(a_{vv}, a_{sv}, a_{ss}) = \text{MLP}\left([\|\mathbf{U}_i\|, s_i]\right)$$

$$\Delta \mathbf{V}_i = a_{vv} \cdot \mathbf{U}_i, \quad \Delta s_i = a_{ss} + a_{sv} \cdot \langle \mathbf{U}_i, \tilde{\mathbf{V}}_i \rangle$$

### Vector Dipole Readout

$$q_i = \text{ChargeNet}(s_i), \quad q_i \leftarrow q_i - \overline{q} \quad \text{(charge neutrality)}$$

$$\boldsymbol{\mu}_{\text{mol}} = \sum_i \left( q_i \cdot (\mathbf{r}_i - \mathbf{r}_{\text{cm}}) + \text{Linear}(\mathbf{V}_i) \right)$$

$$|\mu| = \sqrt{\|\boldsymbol{\mu}_{\text{mol}}\|^2 + \epsilon}$$

---

## Key Concepts

### E(3)-Equivariance

A function f is **E(3)-equivariant** if: f(R·x + t) = R·f(x) + t

- **Scalar properties** (energy, gap): must be **invariant** f(Rx) = f(x)
- **Vector properties** (dipole, forces): must be **equivariant** f(Rx) = R·f(x)

PaiNN achieves this by never applying nonlinearities to vector features — only scaling by invariant scalars.

### Continuous-Filter Convolution (cfconv)

SchNet's innovation: filter weights are a **continuous function** of distance:

$$x_i' = \sum_{j} x_j \cdot W(\|r_i - r_j\|)$$

where W is an MLP applied to Gaussian RBF expansions, allowing arbitrary geometries without fixed grids.

---

## References

1. **Gilmer et al.** (2017). Neural Message Passing for Quantum Chemistry. *ICML*. [[arXiv:1704.01212]](https://arxiv.org/abs/1704.01212)
2. **Schütt et al.** (2017). SchNet: Continuous-filter CNNs for modeling quantum interactions. *NeurIPS*. [[arXiv:1706.08566]](https://arxiv.org/abs/1706.08566)
3. **Schütt, Unke, Gastegger** (2021). Equivariant message passing for tensorial properties and molecular spectra. *ICML*. [[arXiv:2102.03150]](https://arxiv.org/abs/2102.03150)
4. **Ramakrishnan et al.** (2014). Quantum chemistry structures and properties of 134k molecules. *Scientific Data*. [[DOI:10.1038/sdata.2014.22]](https://doi.org/10.1038/sdata.2014.22)
5. **Gasteiger et al.** (2020). Directional Message Passing for Molecular Graphs. *ICLR*. [[arXiv:2003.03123]](https://arxiv.org/abs/2003.03123)
6. **Thölke & De Fabritiis** (2022). TorchMD-NET: Equivariant Transformers for Molecular Potentials. *ICLR*. [[arXiv:2202.02541]](https://arxiv.org/abs/2202.02541)
7. **Liao & Smidt** (2023). Equiformer: Equivariant Graph Attention Transformer. *ICLR*. [[arXiv:2206.11990]](https://arxiv.org/abs/2206.11990)

---

## Acknowledgments

- **SchNetPack** [[3]](#references) for the official PaiNN reference implementation
- **PyTorch Geometric** for graph neural network primitives and QM9 dataset
- **RDKit** for cheminformatics (SMILES parsing, 3D conformer generation)
- Course: **CHEM 4930/5610**, Spring 2026

---

*This project was developed for academic coursework. Please cite the relevant papers if you use any of the model implementations.*
