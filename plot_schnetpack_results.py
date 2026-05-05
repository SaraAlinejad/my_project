#!/usr/bin/env python
"""Plot training curves and summarize results from SchNetPack runs."""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

RESULTS_DIR = Path("results_schnetpack")
HA_TO_MEV = 27211.386

# ── 1. Collect all metrics ──────────────────────────────────────────────
print("=" * 60)
print("SchNetPack PaiNN — Results Summary")
print("=" * 60)

metrics_files = sorted(RESULTS_DIR.glob("*/metrics.json"))
for mf in metrics_files:
    with open(mf) as f:
        m = json.load(f)
    print(f"\n  {m['target']} (seed {m['seed']}):")
    print(f"    Test MAE: {m['test_mae_display']}")
    print(f"    Parameters: {m['n_params']:,}")
    print(f"    Checkpoint: {mf.parent / 'best.ckpt'}")

# ── 2. Plot training curves from Lightning CSV logs ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

targets_info = {
    "dipole_moment": {"label": "Dipole μ", "unit": "mD", "scale": 1000, "lit": 12},
    "gap": {"label": "HOMO-LUMO Gap Δε", "unit": "meV", "scale": HA_TO_MEV, "lit": 45.7},
}

for ax_idx, (target, info) in enumerate(targets_info.items()):
    ax = axes[ax_idx]

    # Find the CSV log
    log_dirs = sorted(RESULTS_DIR.glob(f"painn_{target}_seed*/logs/painn_{target}_seed*/version_*/metrics.csv"))
    if not log_dirs:
        # Try alternative path structure
        log_dirs = sorted(RESULTS_DIR.glob(f"painn_{target}_seed*/logs/*/version_*/metrics.csv"))
    if not log_dirs:
        log_dirs = sorted(RESULTS_DIR.glob(f"painn_{target}_seed*/**/metrics.csv"))

    if not log_dirs:
        ax.text(0.5, 0.5, f"No training logs found\nfor {target}",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(info["label"])
        continue

    csv_path = log_dirs[-1]  # latest version
    print(f"\n  Reading logs: {csv_path}")
    df = pd.read_csv(csv_path)

    # Lightning logs train and val metrics on separate rows
    # Find the val MAE column
    val_col = f"val_{target}_MAE"
    train_col = "train_loss"

    if val_col in df.columns:
        val_data = df[["epoch", val_col]].dropna()
        val_mae = val_data[val_col].values * info["scale"]
        val_epochs = val_data["epoch"].values

        ax.plot(val_epochs, val_mae, 'b-', lw=2, label="Val MAE")
        ax.axhline(y=info["lit"], color='r', ls='--', lw=1.5,
                   label=f'Literature PaiNN ({info["lit"]} {info["unit"]})')

        best_idx = np.argmin(val_mae)
        ax.plot(val_epochs[best_idx], val_mae[best_idx], 'r*', ms=15,
                label=f'Best: {val_mae[best_idx]:.1f} {info["unit"]}')

    if train_col in df.columns:
        train_data = df[["epoch", train_col]].dropna()
        if len(train_data) > 0:
            ax2 = ax.twinx()
            ax2.plot(train_data["epoch"].values, train_data[train_col].values,
                     'g-', alpha=0.3, lw=1, label="Train loss")
            ax2.set_ylabel("Train Loss", color='g', alpha=0.5)
            ax2.tick_params(axis='y', labelcolor='g')

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Val MAE ({info['unit']})")
    ax.set_title(f"PaiNN (SchNetPack) — {info['label']}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR / "training_curves.png"
plt.savefig(fig_path, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {fig_path}")

# ── 3. Combined comparison table ────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FULL COMPARISON TABLE")
print(f"{'=' * 70}")
print(f"{'Model':<35} {'μ MAE':>10} {'Δε MAE':>10} {'Type':<20}")
print(f"{'─' * 70}")

# Our results
for mf in metrics_files:
    with open(mf) as f:
        m = json.load(f)
    if m["target"] == "dipole_moment":
        print(f"{'★ PaiNN SchNetPack (ours)':<35} {m['test_mae_display']:>10} {'—':>10} {'equivariant':<20}")
    elif m["target"] == "gap":
        mae_mev = m["test_mae"] * HA_TO_MEV
        print(f"{'★ PaiNN SchNetPack (ours)':<35} {'—':>10} {f'{mae_mev:.1f} meV':>10} {'equivariant':<20}")

# From-scratch results (if available)
from_scratch = sorted(Path("results").glob("*_best.pt"))
for ckpt in from_scratch:
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    name = c.get("model_name", "?").upper()
    target = c.get("target", "?")
    mae = c.get("test_mae", 0)
    if target == "dipole":
        print(f"{'★ ' + name + ' from-scratch (ours)':<35} {f'{mae*1000:.1f} mD':>10} {'—':>10} {'equivariant':<20}")
    elif target == "gap":
        print(f"{'★ ' + name + ' from-scratch (ours)':<35} {'—':>10} {f'{mae*1000:.1f} meV':>10} {'equivariant':<20}")

# Literature
print(f"{'─' * 70}")
print(f"{'MPNN (Gilmer 2017)':<35} {'~300 mD':>10} {'—':>10} {'invariant':<20}")
print(f"{'SchNet (Schütt 2018)':<35} {'33 mD':>10} {'63 meV':>10} {'invariant':<20}")
print(f"{'DimeNet++ (Gasteiger 2020)':<35} {'30 mD':>10} {'33 meV':>10} {'invariant':<20}")
print(f"{'PaiNN (Schütt 2021)':<35} {'12 mD':>10} {'45.7 meV':>10} {'equivariant':<20}")
print(f"{'TorchMD-NET (Thölke 2022)':<35} {'11 mD':>10} {'36 meV':>10} {'equivariant':<20}")
print(f"{'Equiformer (Liao 2023)':<35} {'11 mD':>10} {'~30 meV':>10} {'equivariant':<20}")
print(f"{'=' * 70}")

print(f"\n🎉 All plots saved to {RESULTS_DIR}/")
