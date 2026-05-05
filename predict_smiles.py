#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_smiles.py — Predict molecular properties from SMILES using trained models.

CHEM 4930/5610 Final Project
Loads trained MPNN/SchNet/PaiNN and predicts dipole moment & HOMO-LUMO gap.
Generates 3D conformers via RDKit for 3D models (SchNet, PaiNN).

Usage (Colab):
    Run cells sequentially.

Usage (script):
    python predict_smiles.py --model painn --smiles "CCO"
"""

# %% [markdown]
# # Predict Molecular Properties from SMILES
#
# Load a trained model (MPNN, SchNet, or PaiNN) and predict
# **dipole moment (μ)** and **HOMO-LUMO gap (Δε)** for any molecule.
#
# For SchNet/PaiNN (3D models), we generate a 3D conformer using RDKit's
# `EmbedMolecule` with the ETKDG method.

# %%
import os
import sys
import torch
import numpy as np
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from torch_geometric.data import Data, Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %%
# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model

# %% [markdown]
# ## 1. SMILES to 3D Graph Conversion

# %%
def smiles_to_3d_graph(smiles, max_z=10):
    """Convert SMILES to a PyG Data object with 3D coordinates.

    Uses RDKit's ETKDG method to generate a 3D conformer, then extracts
    atomic numbers and positions for use with SchNet/PaiNN.

    Args:
        smiles: SMILES string.
        max_z: Maximum atomic number (for validation).

    Returns:
        PyG Data object with z, pos, or None if conversion fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Fallback: try without constraints
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status != 0:
            print(f"  ⚠️  Could not generate 3D conformer for {smiles}")
            return None

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # Continue with un-optimized geometry

    # Extract atomic numbers and positions
    conf = mol.GetConformer()
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float
    )

    return Data(z=z, pos=pos)


# %% [markdown]
# ## 2. Applicability Domain Checks

# %%
def check_applicability(smiles):
    """Check if a SMILES string falls within the QM9 applicability domain.

    Returns:
        tuple: (is_valid: bool, messages: list of str)
    """
    messages = []
    is_valid = True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ["ERROR: Invalid SMILES — RDKit could not parse it."]

    # Check elements (QM9: H, C, N, O, F only)
    allowed = {1, 6, 7, 8, 9}
    mol_h = Chem.AddHs(mol)
    found = set(a.GetAtomicNum() for a in mol_h.GetAtoms())
    bad = found - allowed
    if bad:
        syms = [Chem.GetPeriodicTable().GetElementSymbol(z) for z in bad]
        messages.append(f"WARNING: Contains atoms not in QM9: {', '.join(syms)}.")
        is_valid = False

    # Check heavy atom count (QM9 ≤ 9 heavy atoms)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > 9:
        messages.append(f"WARNING: {n_heavy} heavy atoms (QM9 max is 9).")
        is_valid = False
    if n_heavy == 0:
        messages.append("WARNING: No heavy atoms.")
        is_valid = False

    # Check formal charges
    max_q = max(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    if max_q > 1:
        messages.append(f"WARNING: Large formal charges (|q|>{max_q}).")

    # Molecular weight check
    mw = Descriptors.MolWt(mol)
    if mw < 10 or mw > 200:
        messages.append(f"WARNING: MW ({mw:.1f}) outside QM9 range (~16-180 Da).")

    if is_valid and not messages:
        messages.append("OK: Within QM9 applicability domain.")

    return is_valid, messages


# %% [markdown]
# ## 3. Load Models

# %%
RESULTS_DIR = "results"

def load_trained_model(model_name, target, results_dir=RESULTS_DIR):
    """Load a trained model from checkpoint.

    Args:
        model_name: 'mpnn', 'schnet', or 'painn'.
        target: 'dipole' or 'gap'.
        results_dir: Directory containing checkpoints.

    Returns:
        (model, target_mean, target_std, target_info) or None if not found.
    """
    ckpt_path = os.path.join(results_dir, f"{model_name}_{target}_best.pt")
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    model = get_model(model_name, target=target).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return {
        'model': model,
        'target_mean': ckpt['target_mean'],
        'target_std': ckpt['target_std'],
        'target_name': ckpt['target_name'],
        'target_unit': ckpt['target_unit'],
        'test_mae': ckpt.get('test_mae', None),
        'n_params': ckpt.get('n_params', None),
    }


# Try to load all available models
loaded_models = {}
for model_name in ['mpnn', 'schnet', 'painn']:
    for target in ['dipole', 'gap']:
        result = load_trained_model(model_name, target)
        if result is not None:
            loaded_models[f"{model_name}_{target}"] = result
            test_mae = result.get('test_mae')
            mae_str = f" (test MAE: {test_mae:.4f})" if test_mae else ""
            print(f"  ✅ Loaded {model_name.upper()} for {target}{mae_str}")

if not loaded_models:
    print(f"\n⚠️  No trained models found in {RESULTS_DIR}/")
    print("   Train models first: python train.py --model painn --target dipole")


# %% [markdown]
# ## 4. Prediction Function

# %%
@torch.no_grad()
def predict_from_smiles(smiles, model_name="painn", show_molecule=True):
    """Predict molecular properties from a SMILES string.

    Args:
        smiles: SMILES string.
        model_name: Which model to use ('mpnn', 'schnet', 'painn').
        show_molecule: If True, show the 2D molecule structure.

    Returns:
        dict of {target_name: predicted_value} or None if prediction fails.
    """
    print(f"\n{'='*60}")
    print(f"SMILES: {smiles}")
    print(f"Model:  {model_name.upper()}")
    print(f"{'='*60}")

    # Applicability check
    is_valid, messages = check_applicability(smiles)
    print("\n📋 Applicability Domain:")
    for msg in messages:
        prefix = "❌" if msg.startswith("ERROR") else ("⚠️ " if msg.startswith("WARNING") else "✅")
        print(f"  {prefix} {msg}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Cannot proceed — invalid SMILES.")
        return None

    if show_molecule:
        print(f"\nFormula: {rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"Heavy atoms: {mol.GetNumHeavyAtoms()}")
        try:
            from IPython.display import display
            display(Draw.MolToImage(mol, size=(300, 300)))
        except Exception:
            pass

    # Generate 3D graph
    graph = smiles_to_3d_graph(smiles)
    if graph is None:
        print("Cannot generate 3D structure.")
        return None

    batch = Batch.from_data_list([graph]).to(device)

    # Predict for each available target
    results = OrderedDict()
    print(f"\n🎯 Predictions ({model_name.upper()}):")

    for target in ['dipole', 'gap']:
        key = f"{model_name}_{target}"
        if key not in loaded_models:
            continue

        info = loaded_models[key]
        model = info['model']
        pred_norm = model(batch)
        pred = pred_norm.item() * info['target_std'] + info['target_mean']

        name = info['target_name']
        unit = info['target_unit']
        results[name] = pred

        # Display in convenient units
        if target == 'dipole':
            print(f"  {name}: {pred:.4f} {unit} ({pred*1000:.1f} mD)")
        else:
            print(f"  {name}: {pred:.4f} {unit} ({pred*1000:.1f} meV)")

    if not is_valid:
        print("\n⚠️  Outside training domain — predictions may be unreliable.")

    return results


# %% [markdown]
# ## 5. Example Predictions

# %%
# Determine which model to use (prefer PaiNN > SchNet > MPNN)
available_models = set(k.split('_')[0] for k in loaded_models.keys())
default_model = 'painn' if 'painn' in available_models else \
                'schnet' if 'schnet' in available_models else \
                'mpnn' if 'mpnn' in available_models else None

if default_model:
    print(f"Using {default_model.upper()} for predictions")

    # Molecules in QM9 domain
    predict_from_smiles("C", model_name=default_model)        # Methane
    predict_from_smiles("O", model_name=default_model)        # Water
    predict_from_smiles("CCO", model_name=default_model)      # Ethanol
    predict_from_smiles("CC(=O)O", model_name=default_model)  # Acetic acid
    predict_from_smiles("CF", model_name=default_model)       # Fluoromethane

    # %% [markdown]
    # ## 6. Cross-Model Comparison

    # %%
    test_smiles = ["C", "O", "CCO", "CC(=O)O", "NCC(=O)O", "CF"]
    test_names = ["Methane", "Water", "Ethanol", "Acetic acid", "Glycine", "Fluoromethane"]

    print(f"\n{'='*80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*80}")

    for target in ['dipole', 'gap']:
        target_models = {k: v for k, v in loaded_models.items() if k.endswith(f'_{target}')}
        if not target_models:
            continue

        unit_info = {'dipole': ('D', 'mD', 1000), 'gap': ('eV', 'meV', 1000)}
        unit, disp_unit, scale = unit_info[target]

        header_models = [k.split('_')[0].upper() for k in sorted(target_models.keys())]
        header = f"  {'Molecule':<20}" + "".join(f"{m:>12}" for m in header_models) + f"  ({disp_unit})"
        print(f"\n  {list(target_models.values())[0]['target_name']}")
        print(f"  {'─'*len(header)}")
        print(header)
        print(f"  {'─'*len(header)}")

        for smi, name in zip(test_smiles, test_names):
            graph = smiles_to_3d_graph(smi)
            if graph is None:
                continue
            batch = Batch.from_data_list([graph]).to(device)

            row = f"  {name:<20}"
            for key in sorted(target_models.keys()):
                info = target_models[key]
                pred_norm = info['model'](batch)
                pred = pred_norm.item() * info['target_std'] + info['target_mean']
                row += f"{pred * scale:>12.1f}"
            print(row)

    print(f"\n{'='*80}")
else:
    print("No models loaded. Train models first!")

# %% [markdown]
# ## 7. Your Custom Prediction
#
# Change the SMILES below to predict for any molecule:

# %%
if default_model:
    my_smiles = "CCO"  # <-- Change this!
    predict_from_smiles(my_smiles, model_name=default_model)
