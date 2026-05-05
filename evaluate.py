#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py — Post-training comparison of MPNN, SchNet, and PaiNN on QM9.

Loads checkpoints from all trained models and generates:
  - Unified comparison table (MAE, RMSE, R² per model per target)
  - Side-by-side parity plots
  - Literature comparison
  - Publication-quality figures

Usage (Colab):
    Run cells sequentially after training all models.

Usage (script):
    python evaluate.py --results_dir results
"""

# %% [markdown]
# # Model Comparison: MPNN vs SchNet vs PaiNN
#
# This notebook loads all trained checkpoints and generates a comprehensive
# comparison across architectures and targets.

# %%
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model

# Config
RESULTS_DIR = "results"
BATCH_SIZE = 100
SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Dataset (same split as training)

# %%
TARGET_MAP = {
    "dipole": {"idx": 0, "name": "Dipole Moment μ", "unit": "D", "display_unit": "mD", "scale": 1000},
    "gap":    {"idx": 4, "name": "HOMO-LUMO Gap Δε", "unit": "eV", "display_unit": "meV", "scale": 1000},
}

dataset = QM9(root="data/QM9")
N = len(dataset)
n_train, n_val = 110000, 10000
perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
test_idx = perm[n_train + n_val:]
test_dataset = dataset[test_idx]
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Test set size: {len(test_dataset)}")

# %% [markdown]
# ## 2. Load All Checkpoints

# %%
checkpoints = {}
ckpt_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_best.pt")))

if not ckpt_files:
    print(f"⚠️  No checkpoints found in {RESULTS_DIR}/")
    print("   Train models first: python train.py --model painn --target dipole")
else:
    for f in ckpt_files:
        ckpt = torch.load(f, map_location=device)
        key = f"{ckpt['model_name']}_{ckpt['target']}"
        checkpoints[key] = ckpt
        print(f"  Loaded: {key} (test MAE: {ckpt.get('test_mae', '?')})")

# %% [markdown]
# ## 3. Evaluate All Models

# %%
@torch.no_grad()
def full_evaluate(model, loader, target_idx):
    """Get predictions and targets for the full test set."""
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data)
        preds.append(pred.cpu())
        targets.append(data.y[:, target_idx].cpu())
    return torch.cat(preds), torch.cat(targets)


results_rows = []
predictions = {}

for key, ckpt in checkpoints.items():
    model_name = ckpt['model_name']
    target = ckpt['target']
    target_idx = ckpt['target_idx']
    target_mean = ckpt['target_mean']
    target_std = ckpt['target_std']
    info = TARGET_MAP[target]

    # Rebuild model
    model = get_model(model_name, target=target).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get raw predictions
    pred_raw, true_vals = full_evaluate(model, test_loader, target_idx)

    # Unnormalize only if gap (dipole models output raw values)
    if target == "gap":
        pred_vals = pred_raw * target_std + target_mean
    else:
        pred_vals = pred_raw  # dipole: already in original units

    # Store
    predictions[key] = (pred_vals, true_vals)

    # Compute metrics
    p_np = pred_vals.numpy()
    t_np = true_vals.numpy()
    mae = mean_absolute_error(t_np, p_np)
    rmse = np.sqrt(mean_squared_error(t_np, p_np))
    r2 = r2_score(t_np, p_np)

    results_rows.append({
        'Model': model_name.upper(),
        'Target': info['name'],
        f'MAE ({info["display_unit"]})': f"{mae * info['scale']:.1f}",
        f'RMSE ({info["display_unit"]})': f"{rmse * info['scale']:.1f}",
        'R²': f"{r2:.4f}",
        'Params': f"{ckpt['n_params']:,}",
        'Best Epoch': ckpt['best_epoch'],
    })

    print(f"\n  {model_name.upper()} on {info['name']}:")
    print(f"    MAE  = {mae * info['scale']:.1f} {info['display_unit']}  ({mae:.4f} {info['unit']})")
    print(f"    RMSE = {rmse * info['scale']:.1f} {info['display_unit']}")
    print(f"    R²   = {r2:.4f}")

if results_rows:
    results_df = pd.DataFrame(results_rows)
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_results.csv'), index=False)
    print(f"\nSaved to {RESULTS_DIR}/comparison_results.csv")

# %% [markdown]
# ## 4. Side-by-Side Parity Plots

# %%
if predictions:
    # Group by target
    targets_seen = set()
    for key in predictions:
        target = key.split('_')[-1]
        targets_seen.add(target)

    for target in sorted(targets_seen):
        info = TARGET_MAP[target]
        models_for_target = {k: v for k, v in predictions.items() if k.endswith(f'_{target}')}

        if not models_for_target:
            continue

        n_models = len(models_for_target)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        colors = {'mpnn': '#E74C3C', 'schnet': '#3498DB', 'painn': '#2ECC71'}

        for ax, (key, (pred, true)) in zip(axes, models_for_target.items()):
            model_name = key.split('_')[0]
            color = colors.get(model_name, '#95A5A6')

            mae = mean_absolute_error(true.numpy(), pred.numpy())

            ax.scatter(true.numpy(), pred.numpy(), s=3, alpha=0.2, c=color, rasterized=True)

            lo = min(true.min().item(), pred.min().item())
            hi = max(true.max().item(), pred.max().item())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', lw=1.5, alpha=0.5)

            ax.set_xlabel(f'Actual ({info["unit"]})', fontsize=12)
            ax.set_ylabel(f'Predicted ({info["unit"]})', fontsize=12)
            ax.set_title(f'{model_name.upper()}\n'
                         f'MAE = {mae * info["scale"]:.1f} {info["display_unit"]}',
                         fontsize=13)
            ax.set_aspect('equal', 'box')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{info["name"]} — Test Set Parity Comparison', fontsize=15, y=1.02)
        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f'comparison_parity_{target}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {fig_path}")

# %% [markdown]
# ## 5. Literature Comparison Table

# %%
print(f"\n{'='*80}")
print("LITERATURE COMPARISON")
print(f"{'='*80}")

literature = {
    "dipole": {
        "unit": "mD",
        "models": [
            ("Plain GIN/GAT (no 3D)", "300-550"),
            ("SchNet (Schütt 2018)", "33"),
            ("DimeNet++ (Gasteiger 2020)", "30"),
            ("SphereNet (Liu 2022)", "26"),
            ("PaiNN (Schütt 2021)", "12"),
            ("TorchMD-NET (Thölke 2022)", "11"),
            ("Equiformer L=2 (Liao 2023)", "11"),
        ],
    },
    "gap": {
        "unit": "meV",
        "models": [
            ("SchNet (Schütt 2018)", "63"),
            ("Cormorant (Anderson 2019)", "61"),
            ("PaiNN (Schütt 2021)", "45.7"),
            ("TorchMD-NET (Thölke 2022)", "36"),
            ("DimeNet++ (Gasteiger 2020)", "33"),
            ("SphereNet (Liu 2022)", "32"),
            ("Equiformer L=2 (Liao 2023)", "~30"),
        ],
    },
}

for target in ["dipole", "gap"]:
    info = TARGET_MAP[target]
    lit = literature[target]

    print(f"\n  {info['name']} ({lit['unit']})")
    print(f"  {'─'*55}")
    print(f"  {'Model':<40} {'MAE':>10}")
    print(f"  {'─'*55}")

    # Our models
    for key in sorted(predictions.keys()):
        if key.endswith(f'_{target}'):
            model_name = key.split('_')[0]
            pred, true = predictions[key]
            mae = mean_absolute_error(true.numpy(), pred.numpy())
            mae_display = mae * info['scale']
            print(f"  ★ {model_name.upper() + ' (ours)':<38} {mae_display:>10.1f}  ◀")

    # Literature
    for name, mae in lit["models"]:
        print(f"    {name:<38} {mae:>10}")

print(f"\n{'='*80}")

# %%
print("\n🎉 Evaluation complete!")
print(f"   All figures and tables saved to {RESULTS_DIR}/")
