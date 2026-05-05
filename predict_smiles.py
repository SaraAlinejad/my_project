#!/usr/bin/env python
"""
predict_molecule.py — Predict molecular properties from SMILES using trained PaiNN (SchNetPack).
Uses the trained SchNetPack PaiNN checkpoint to predict dipole moment (μ)
and/or HOMO-LUMO gap (Δε) for arbitrary molecules given as SMILES strings.
Usage:
    python predict_molecule.py CCO                          # ethanol dipole
    python predict_molecule.py "CC=O" "c1ccccc1" "O"        # multiple molecules
    python predict_molecule.py CCO --target gap              # HOMO-LUMO gap
    python predict_molecule.py CCO --target both             # both properties
"""
import argparse
import sys
import warnings
import numpy as np
import torch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from ase import Atoms
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.task import AtomisticTask
RDLogger.DisableLog('rdApp.*')
# ── Config ──────────────────────────────────────────────────────────────
CUTOFF = 5.0
HA_TO_MEV = 27211.386
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINTS = {
    "dipole_moment": "results_schnetpack/painn_dipole_moment_seed0/best.ckpt",
    "gap": "results_schnetpack/painn_gap_seed0/best.ckpt",
}
# QM9 applicability domain
QM9_ELEMENTS = {1, 6, 7, 8, 9}  # H, C, N, O, F
QM9_MAX_HEAVY = 9
QM9_MAX_ATOMS = 29
# Cache loaded models to avoid reloading for each molecule
_model_cache = {}
# ── Core prediction ─────────────────────────────────────────────────────
def smiles_to_atoms(smiles):
    """Convert SMILES to ASE Atoms with 3D coordinates via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        # Fallback for difficult molecules
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    return Atoms(numbers=z, positions=pos, pbc=False), mol
def check_applicability(mol, smiles):
    """Check if molecule is within QM9's chemical space."""
    warnings = []
    elements = set(a.GetAtomicNum() for a in mol.GetAtoms())
    outside = elements - QM9_ELEMENTS
    if outside:
        names = {el: Chem.GetPeriodicTable().GetElementSymbol(el) for el in outside}
        warnings.append(f"Contains elements outside QM9: {names}")
    n_heavy = Descriptors.HeavyAtomCount(mol)
    if n_heavy > QM9_MAX_HEAVY:
        warnings.append(f"Too many heavy atoms ({n_heavy} > {QM9_MAX_HEAVY})")
    n_atoms = mol.GetNumAtoms()
    if n_atoms > QM9_MAX_ATOMS:
        warnings.append(f"Too many total atoms ({n_atoms} > {QM9_MAX_ATOMS})")
    return warnings
def predict(smiles, target="dipole_moment", seed=0):
    """Predict a molecular property from SMILES.
    Args:
        smiles: SMILES string.
        target: 'dipole_moment' or 'gap'.
        seed: Model seed (default 0).
    Returns:
        Predicted value in original units (D for dipole, Ha for gap).
    """
    atoms, mol = smiles_to_atoms(smiles)
    # Check applicability domain
    domain_warnings = check_applicability(mol, smiles)
    # Convert to SchNetPack input
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=CUTOFF),
        dtype=torch.float32,
        device=DEVICE,
    )
    inputs = converter(atoms)
    # Load model (cached)
    cache_key = f"{target}_seed{seed}"
    if cache_key not in _model_cache:
        ckpt_path = CHECKPOINTS.get(target)
        if ckpt_path is None:
            raise ValueError(f"No checkpoint for target '{target}'. "
                             f"Available: {list(CHECKPOINTS.keys())}")
        task = AtomisticTask.load_from_checkpoint(ckpt_path, map_location=DEVICE)
        task.eval()
        _model_cache[cache_key] = task
    task = _model_cache[cache_key]
    with torch.no_grad():
        result = task(inputs)
    return result[target].cpu().item(), domain_warnings
# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from SMILES using PaiNN (SchNetPack)")
    parser.add_argument("smiles", nargs="+", help="SMILES string(s)")
    parser.add_argument("--target", default="dipole",
                        choices=["dipole", "gap", "both"],
                        help="Property to predict (default: dipole)")
    args = parser.parse_args()
    # Map short names to SchNetPack property keys
    target_map = {
        "dipole": ["dipole_moment"],
        "gap": ["gap"],
        "both": ["dipole_moment", "gap"],
    }
    targets = target_map[args.target]
    print(f"\n{'='*65}")
    print(f" PaiNN (SchNetPack) — Molecular Property Prediction")
    print(f"{'='*65}")
    print(f" Device: {DEVICE}")
    print(f" Targets: {', '.join(targets)}")
    print(f"{'='*65}\n")
    for smi in args.smiles:
        name = Chem.MolToSmiles(Chem.MolFromSmiles(smi))  # canonical SMILES
        print(f" Molecule: {name}")
        for target in targets:
            try:
                value, warnings = predict(smi, target=target)
                if target == "dipole_moment":
                    print(f"   Dipole μ  = {value:.3f} D  ({value*1000:.1f} mD)")
                elif target == "gap":
                    value_meV = value * HA_TO_MEV
                    value_eV = value * 27.2114
                    print(f"   Gap Δε    = {value_eV:.3f} eV  ({value_meV:.1f} meV)")
                if warnings:
                    for w in warnings:
                        print(f"     {w}")
            except Exception as e:
                print(f"    Error: {e}")
        print()
if __name__ == "__main__":
    main()
