#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_train_painn_schnetpack.py — PaiNN via SchNetPack 2.x (official implementation)

CHEM 4930/5610 Final Project

WHY THIS EXISTS
===============
- models/painn.py: hand-written PaiNN in pure PyG. Pedagogically clear,
  demonstrates understanding of the architecture.
- This script: uses SchNetPack 2.x, the OFFICIAL PaiNN implementation from
  the paper authors (Schütt et al., ICML 2021). ~150 LoC, battle-tested.
  Uses schnetpack.atomistic.DipoleMoment(predict_magnitude=True) for the
  correct vector dipole readout.

Having both validates the architecture: if SchNetPack gives ~12 mD,
we know PaiNN *works* on QM9, and any gap from our from-scratch version
is implementation-level, not architectural.

USAGE
=====
    pip install schnetpack==2.0.4 pytorch_lightning
    python 05_train_painn_schnetpack.py                    # dipole (default)
    python 05_train_painn_schnetpack.py --target gap
    python 05_train_painn_schnetpack.py --target dipole --seed 1
"""

# %% [markdown]
# ## 0. Imports & Config

# %%
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.datasets import QM9
from schnetpack.task import AtomisticTask

# %%
# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train PaiNN (SchNetPack) on QM9")
    parser.add_argument("--target", type=str, default="dipole_moment",
                        help="QM9 target property (e.g. dipole_moment, gap, "
                             "homo, lumo, energy_U0). Run --list_targets to see all.")
    parser.add_argument("--list_targets", action="store_true",
                        help="List available QM9 properties and exit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=250, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    if not hasattr(sys, 'ps1') and 'google.colab' not in sys.modules:
        return parser.parse_args()
    return parser.parse_args([])

args = parse_args()

# ==================== CONFIG ====================
TARGET         = args.target         # 'dipole_moment' or 'homo_lumo_gap'
SEED           = args.seed
EPOCHS         = args.epochs
BATCH_SIZE     = args.batch_size

# PaiNN hyperparameters (paper defaults)
F_DIM          = 128
N_INTERACTIONS = 3
N_RBF          = 20
CUTOFF         = 5.0                 # Å
LR             = 5e-4
WEIGHT_DECAY   = 0.01
LR_PATIENCE    = 25
LR_FACTOR      = 0.5
EARLY_STOP     = 50

OUT_DIR = Path(f"results_schnetpack/painn_{TARGET}_seed{SEED}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ================================================

pl.seed_everything(SEED, workers=True)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device_str}")
print(f"Target: {TARGET} | Seed: {SEED} | Epochs: {EPOCHS}")


# %% [markdown]
# ## 1. Load QM9 via SchNetPack's built-in dataset

# %%
# SchNetPack's QM9 handles download, unit conversion, and splitting
#
# IMPORTANT: RemoveOffsets must NOT be used for dipole_moment!
# DipoleMoment output module returns ||μ_vec|| ≥ 0 (a norm),
# but RemoveOffsets subtracts the mean, creating negative targets
# that the norm can never match. Only use for scalar targets (gap).
transforms = [trn.ASENeighborList(cutoff=CUTOFF), trn.CastTo32()]
if TARGET != "dipole_moment":
    transforms.insert(1, trn.RemoveOffsets(TARGET, remove_mean=True, remove_atomrefs=False))
    print("Using RemoveOffsets for scalar target")
else:
    print("⚡ Skipping RemoveOffsets for dipole (norm output ≥ 0, shifted targets < 0)")

qm9_data = QM9(
    datapath="data/qm9_schnetpack.db",
    batch_size=BATCH_SIZE,
    num_train=110000,
    num_val=10000,
    transforms=transforms,
    num_workers=4,
    pin_memory=True,
    split_file=str(OUT_DIR / "split.npz"),
)
qm9_data.prepare_data()
qm9_data.setup()

print(f"Train: {len(qm9_data.train_dataset)} | "
      f"Val: {len(qm9_data.val_dataset)} | "
      f"Test: {len(qm9_data.test_dataset)}")


# %% [markdown]
# ## 2. Build PaiNN Model
#
# Key architectural choices:
# - **Dipole**: DipoleMoment output with `predict_magnitude=True` and
#   `use_vector_representation=True` — uses PaiNN's L=1 equivariant features
#   to predict the dipole VECTOR, then takes its norm. This is THE reason
#   PaiNN achieves 12 mD vs SchNet's 33 mD.
# - **Gap**: Atomwise output — standard scalar readout with sum pooling.

# %%
# Pairwise distances (input module for PaiNN)
pairwise_distance = spk.atomistic.PairwiseDistances()

# PaiNN representation (the equivariant message-passing backbone)
representation = spk.representation.PaiNN(
    n_atom_basis=F_DIM,
    n_interactions=N_INTERACTIONS,
    radial_basis=spk.nn.GaussianRBF(n_rbf=N_RBF, cutoff=CUTOFF),
    cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
)

# Output head — depends on target
if TARGET == "dipole_moment":
    output_module = spk.atomistic.DipoleMoment(
        n_in=F_DIM,
        predict_magnitude=True,            # predict ||μ||, not the vector
        use_vector_representation=True,    # use PaiNN's L=1 equivariant features
    )
    print("Using DipoleMoment readout (vector → norm)")
else:
    output_module = spk.atomistic.Atomwise(
        n_in=F_DIM,
        output_key=TARGET,
    )
    print("Using Atomwise readout (scalar)")

# Full model
# Only use AddOffsets postprocessor for scalar targets (not dipole)
postprocessors = [trn.CastTo32()]
if TARGET != "dipole_moment":
    postprocessors.append(trn.AddOffsets(TARGET, add_mean=True, add_atomrefs=False))

model = spk.model.NeuralNetworkPotential(
    representation=representation,
    input_modules=[pairwise_distance],
    output_modules=[output_module],
    postprocessors=postprocessors,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# Wrap in AtomisticTask for Lightning training
import torchmetrics

task = AtomisticTask(
    model=model,
    outputs=[
        spk.task.ModelOutput(
            name=TARGET,
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={"MAE": torchmetrics.MeanAbsoluteError()},
        )
    ],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": LR, "weight_decay": WEIGHT_DECAY},
    scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_args={"mode": "min", "factor": LR_FACTOR, "patience": LR_PATIENCE},
    scheduler_monitor=f"val_{TARGET}_MAE",
)


# %% [markdown]
# ## 3. Train

# %%
callbacks = [
    pl.callbacks.ModelCheckpoint(
        dirpath=str(OUT_DIR),
        filename="best",
        monitor=f"val_{TARGET}_MAE",
        mode="min",
        save_top_k=1,
    ),
    pl.callbacks.EarlyStopping(
        monitor=f"val_{TARGET}_MAE",
        patience=EARLY_STOP,
        mode="min",
    ),
    pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
]

logger = pl.loggers.CSVLogger(save_dir=str(OUT_DIR), name="logs")

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    max_epochs=EPOCHS,
    callbacks=callbacks,
    logger=logger,
    gradient_clip_val=5.0,
    log_every_n_steps=50,
)

print(f"\n{'='*60}")
print(f"Training PaiNN (SchNetPack) on {TARGET}")
print(f"{'='*60}")

trainer.fit(task, datamodule=qm9_data)


# %% [markdown]
# ## 4. Test & Report

# %%
# Load best checkpoint and test
best_ckpt = OUT_DIR / "best.ckpt"
print(f"\nLoading best checkpoint: {best_ckpt}")
test_results = trainer.test(task, datamodule=qm9_data, ckpt_path=str(best_ckpt))

test_mae = test_results[0][f"test_{TARGET}_MAE"]

# Unit conversion for display
if TARGET == "dipole_moment":
    unit = "D"
    display_unit = "mD"
    display_mae = test_mae * 1000
    lit_painn = "12 mD"
    lit_schnet = "33 mD"
else:
    # SchNetPack QM9 stores gap in HARTREE (not eV!)
    # 1 Ha = 27.2114 eV = 27211.4 meV
    unit = "Ha"
    display_unit = "meV"
    display_mae = test_mae * 27211.386  # Ha → meV
    lit_painn = "45.7 meV"
    lit_schnet = "63 meV"

print(f"\n{'='*60}")
print(f"PaiNN (SchNetPack) — {TARGET}")
print(f"{'='*60}")
print(f"  Test MAE: {test_mae:.6f} {unit}  ({display_mae:.1f} {display_unit})")
print(f"  Literature PaiNN: {lit_painn}")
print(f"  Literature SchNet: {lit_schnet}")
print(f"  Parameters: {n_params:,}")
print(f"  Seed: {SEED}")
print(f"{'='*60}")

# Save metrics
metrics = {
    "model": "PaiNN (SchNetPack)",
    "target": TARGET,
    "seed": SEED,
    "test_mae": float(test_mae),
    "test_mae_display": f"{display_mae:.1f} {display_unit}",
    "n_params": n_params,
    "config": {
        "F_dim": F_DIM, "n_interactions": N_INTERACTIONS,
        "n_rbf": N_RBF, "cutoff": CUTOFF, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
    },
}
metrics_path = OUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")


# %% [markdown]
# ## 5. Inference helper (for predict_smiles.py)

# %%
def predict_schnetpack(smiles: str, target: str = "dipole_moment",
                       seed: int = 0, cutoff: float = 5.0):
    """Predict a property from SMILES using the trained SchNetPack PaiNN."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    from schnetpack.interfaces import AtomsConverter

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])

    atoms = Atoms(numbers=z, positions=pos, pbc=False)
    converter = AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=cutoff),
        dtype=torch.float32,
        device=device_str,
    )
    inputs = converter(atoms)

    ckpt_path = f"results_schnetpack/painn_{target}_seed{seed}/best.ckpt"
    loaded_task = AtomisticTask.load_from_checkpoint(ckpt_path, map_location=device_str)
    loaded_task.eval()

    with torch.no_grad():
        result = loaded_task(inputs)
    return result[target].cpu().item()


# Example (run after training completes):
# print(predict_schnetpack("CCO", target="dipole_moment"))  # → ~1.69 D
# print(predict_schnetpack("CCO", target="homo_lumo_gap"))  # → in eV
