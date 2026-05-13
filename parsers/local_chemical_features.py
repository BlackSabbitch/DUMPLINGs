from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdchem


DEFAULT_LOCAL_CHEMICAL_FEATURE_FLAGS: dict[str, bool] = {
    "formal_charge": True,
    "donor_acceptor": True,
    "pair_contact_flags": True,
    "residue_chemistry": True,
    "sidechain_backbone": True,
    "aromaticity": True,
    "hybridization": True,
}

RESIDUE_CLASS_ORDER = [
    "acidic",
    "basic",
    "polar",
    "aromatic",
    "hydrophobic",
    "sulfur",
]

RESIDUE_CLASS_MAP = {
    "ASP": "acidic",
    "GLU": "acidic",
    "LYS": "basic",
    "ARG": "basic",
    "HIS": "basic",
    "SER": "polar",
    "THR": "polar",
    "ASN": "polar",
    "GLN": "polar",
    "TYR": "polar",
    "CYS": "polar",
    "PHE": "aromatic",
    "TYR": "aromatic",
    "TRP": "aromatic",
    "HIS": "aromatic",
    "ALA": "hydrophobic",
    "VAL": "hydrophobic",
    "LEU": "hydrophobic",
    "ILE": "hydrophobic",
    "PRO": "hydrophobic",
    "MET": "hydrophobic",
    "PHE": "hydrophobic",
    "TRP": "hydrophobic",
    "TYR": "hydrophobic",
    "CYS": "sulfur",
    "MET": "sulfur",
}

AROMATIC_PROTEIN_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

POSITIVE_PROTEIN_ATOMS = {
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
}

NEGATIVE_PROTEIN_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

PROTEIN_DONOR_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "TRP": {"NE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}

PROTEIN_ACCEPTOR_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}

_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


@dataclass(frozen=True)
class LocalChemicalFeatureConfig:
    enabled: bool
    features: dict[str, bool]


def normalize_local_chemical_features_config(raw_cfg: Dict[str, Any] | None) -> LocalChemicalFeatureConfig:
    if not isinstance(raw_cfg, dict):
        return LocalChemicalFeatureConfig(enabled=False, features=dict(DEFAULT_LOCAL_CHEMICAL_FEATURE_FLAGS))

    enabled = bool(raw_cfg.get("enabled", False))
    raw_features = raw_cfg.get("features", {})
    merged = dict(DEFAULT_LOCAL_CHEMICAL_FEATURE_FLAGS)
    if isinstance(raw_features, dict):
        for key, default_value in DEFAULT_LOCAL_CHEMICAL_FEATURE_FLAGS.items():
            if key in raw_features:
                merged[key] = bool(raw_features[key])
    return LocalChemicalFeatureConfig(enabled=enabled, features=merged)


def feature_names_from_config(cfg: LocalChemicalFeatureConfig) -> list[str]:
    names: list[str] = []
    if cfg.features.get("formal_charge", False):
        names.append("formal_charge")
    if cfg.features.get("donor_acceptor", False):
        names.extend(["is_hbond_donor", "is_hbond_acceptor"])
    if cfg.features.get("aromaticity", False):
        names.append("is_aromatic")
    if cfg.features.get("hybridization", False):
        names.extend(["hybrid_sp", "hybrid_sp2", "hybrid_sp3", "hybrid_other"])
    if cfg.features.get("sidechain_backbone", False):
        names.extend(["is_backbone_atom", "is_sidechain_atom"])
    if cfg.features.get("residue_chemistry", False):
        names.extend([f"residue_class_{name}" for name in RESIDUE_CLASS_ORDER])
    if cfg.features.get("pair_contact_flags", False):
        names.extend(
            [
                "contact_hbond_count",
                "contact_salt_bridge_count",
                "contact_hydrophobic_count",
                "contact_aromatic_count",
            ]
        )
    return names


def build_local_chemical_node_features(
    lig_mol: Chem.Mol,
    lig_coords: np.ndarray,
    pocket_atom_records: Sequence[dict[str, Any]],
    pocket_coords: np.ndarray,
    *,
    dist_threshold: float,
    cfg: LocalChemicalFeatureConfig,
) -> np.ndarray:
    if not cfg.enabled:
        return np.zeros((lig_mol.GetNumAtoms() + len(pocket_atom_records), 0), dtype=np.float32)

    donor_ids, acceptor_ids = _ligand_donor_acceptor_sets(lig_mol)
    ligand_props = [_ligand_atom_props(atom, donor_ids, acceptor_ids) for atom in lig_mol.GetAtoms()]
    pocket_props = [_protein_atom_props(record) for record in pocket_atom_records]

    ligand_feats = [_node_feature_vector(props, cfg) for props in ligand_props]
    pocket_feats = [_node_feature_vector(props, cfg) for props in pocket_props]

    if cfg.features.get("pair_contact_flags", False) and len(pocket_props) > 0 and lig_coords.size > 0 and pocket_coords.size > 0:
        counts = _contact_participation_counts(
            ligand_props,
            pocket_props,
            lig_coords,
            pocket_coords,
            dist_threshold=dist_threshold,
        )
        for feat, count_vec in zip(ligand_feats, counts["ligand"], strict=False):
            feat.extend(count_vec)
        for feat, count_vec in zip(pocket_feats, counts["pocket"], strict=False):
            feat.extend(count_vec)

    all_feats = ligand_feats + pocket_feats
    if not all_feats:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(all_feats, dtype=np.float32)


def _ligand_donor_acceptor_sets(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    donor_ids: set[int] = set()
    acceptor_ids: set[int] = set()
    for feature in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        atom_ids = set(feature.GetAtomIds())
        if family == "Donor":
            donor_ids.update(atom_ids)
        elif family == "Acceptor":
            acceptor_ids.update(atom_ids)
    return donor_ids, acceptor_ids


def _ligand_atom_props(atom: Chem.Atom, donor_ids: set[int], acceptor_ids: set[int]) -> dict[str, Any]:
    atom_idx = atom.GetIdx()
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    hybrid = atom.GetHybridization()
    return {
        "formal_charge": float(formal_charge),
        "is_hbond_donor": atom_idx in donor_ids,
        "is_hbond_acceptor": atom_idx in acceptor_ids,
        "is_aromatic": bool(atom.GetIsAromatic()),
        "hybridization": _hybridization_bucket(hybrid),
        "is_backbone_atom": False,
        "is_sidechain_atom": False,
        "residue_class": None,
        "is_positive": formal_charge > 0,
        "is_negative": formal_charge < 0,
        "is_hydrophobic": _is_ligand_hydrophobic_atom(atomic_num, formal_charge, atom_idx in donor_ids, atom_idx in acceptor_ids, atom.GetIsAromatic()),
    }


def _protein_atom_props(record: dict[str, Any]) -> dict[str, Any]:
    residue_name = str(record.get("residue_name", "")).upper()
    atom_name = str(record.get("atom_name", "")).upper()
    element = str(record.get("element", "")).upper()
    residue_class = RESIDUE_CLASS_MAP.get(residue_name)
    is_backbone = atom_name in BACKBONE_ATOM_NAMES
    is_sidechain = not is_backbone

    is_donor = False
    is_acceptor = False
    if atom_name == "N" and residue_name != "PRO":
        is_donor = True
    if atom_name in {"O", "OXT"}:
        is_acceptor = True
    is_donor = is_donor or atom_name in PROTEIN_DONOR_ATOMS.get(residue_name, set())
    is_acceptor = is_acceptor or atom_name in PROTEIN_ACCEPTOR_ATOMS.get(residue_name, set())

    formal_charge = 0.0
    if atom_name in POSITIVE_PROTEIN_ATOMS.get(residue_name, set()):
        formal_charge = 1.0
    elif atom_name in NEGATIVE_PROTEIN_ATOMS.get(residue_name, set()):
        formal_charge = -1.0

    is_aromatic = atom_name in AROMATIC_PROTEIN_ATOMS.get(residue_name, set())
    is_positive = formal_charge > 0
    is_negative = formal_charge < 0
    is_hydrophobic = residue_class in {"hydrophobic", "aromatic"} and element in {"C", "S"}

    return {
        "formal_charge": formal_charge,
        "is_hbond_donor": is_donor,
        "is_hbond_acceptor": is_acceptor,
        "is_aromatic": is_aromatic,
        "hybridization": "other",
        "is_backbone_atom": is_backbone,
        "is_sidechain_atom": is_sidechain,
        "residue_class": residue_class,
        "is_positive": is_positive,
        "is_negative": is_negative,
        "is_hydrophobic": is_hydrophobic,
    }


def _node_feature_vector(props: dict[str, Any], cfg: LocalChemicalFeatureConfig) -> list[float]:
    feat: list[float] = []
    if cfg.features.get("formal_charge", False):
        feat.append(float(props["formal_charge"]))
    if cfg.features.get("donor_acceptor", False):
        feat.extend(
            [
                1.0 if props["is_hbond_donor"] else 0.0,
                1.0 if props["is_hbond_acceptor"] else 0.0,
            ]
        )
    if cfg.features.get("aromaticity", False):
        feat.append(1.0 if props["is_aromatic"] else 0.0)
    if cfg.features.get("hybridization", False):
        hybrid = props["hybridization"]
        feat.extend(
            [
                1.0 if hybrid == "sp" else 0.0,
                1.0 if hybrid == "sp2" else 0.0,
                1.0 if hybrid == "sp3" else 0.0,
                1.0 if hybrid == "other" else 0.0,
            ]
        )
    if cfg.features.get("sidechain_backbone", False):
        feat.extend(
            [
                1.0 if props["is_backbone_atom"] else 0.0,
                1.0 if props["is_sidechain_atom"] else 0.0,
            ]
        )
    if cfg.features.get("residue_chemistry", False):
        residue_class = props["residue_class"]
        feat.extend([1.0 if residue_class == name else 0.0 for name in RESIDUE_CLASS_ORDER])
    return feat


def _contact_participation_counts(
    ligand_props: Sequence[dict[str, Any]],
    pocket_props: Sequence[dict[str, Any]],
    lig_coords: np.ndarray,
    pocket_coords: np.ndarray,
    *,
    dist_threshold: float,
) -> dict[str, list[list[float]]]:
    ligand_counts = [[0.0, 0.0, 0.0, 0.0] for _ in ligand_props]
    pocket_counts = [[0.0, 0.0, 0.0, 0.0] for _ in pocket_props]

    dist_mat = np.linalg.norm(lig_coords[:, None, :] - pocket_coords[None, :, :], axis=2)
    lig_idx, pock_idx = np.where(dist_mat < dist_threshold)
    for i, j in zip(lig_idx.tolist(), pock_idx.tolist(), strict=False):
        d = float(dist_mat[i, j])
        lig = ligand_props[i]
        pock = pocket_props[j]

        hbond = _possible_hbond(lig, pock, d)
        salt = _possible_salt_bridge(lig, pock, d)
        hydrophobic = _possible_hydrophobic_contact(lig, pock, d)
        aromatic = _possible_aromatic_contact(lig, pock, d)

        flags = [hbond, salt, hydrophobic, aromatic]
        for idx, flag in enumerate(flags):
            if flag:
                ligand_counts[i][idx] += 1.0
                pocket_counts[j][idx] += 1.0

    return {"ligand": ligand_counts, "pocket": pocket_counts}


def _possible_hbond(lig: dict[str, Any], pock: dict[str, Any], distance: float) -> bool:
    if distance > 3.5:
        return False
    return (lig["is_hbond_donor"] and pock["is_hbond_acceptor"]) or (
        lig["is_hbond_acceptor"] and pock["is_hbond_donor"]
    )


def _possible_salt_bridge(lig: dict[str, Any], pock: dict[str, Any], distance: float) -> bool:
    if distance > 4.0:
        return False
    return (lig["is_positive"] and pock["is_negative"]) or (lig["is_negative"] and pock["is_positive"])


def _possible_hydrophobic_contact(lig: dict[str, Any], pock: dict[str, Any], distance: float) -> bool:
    return distance <= 4.5 and lig["is_hydrophobic"] and pock["is_hydrophobic"]


def _possible_aromatic_contact(lig: dict[str, Any], pock: dict[str, Any], distance: float) -> bool:
    return distance <= 5.0 and lig["is_aromatic"] and pock["is_aromatic"]


def _is_ligand_hydrophobic_atom(
    atomic_num: int,
    formal_charge: int,
    is_donor: bool,
    is_acceptor: bool,
    is_aromatic: bool,
) -> bool:
    if formal_charge != 0:
        return False
    if is_donor or is_acceptor:
        return False
    if is_aromatic:
        return True
    return atomic_num in {6, 9, 17, 35, 53, 16}


def _hybridization_bucket(hybridization: rdchem.HybridizationType) -> str:
    if hybridization == rdchem.HybridizationType.SP:
        return "sp"
    if hybridization == rdchem.HybridizationType.SP2:
        return "sp2"
    if hybridization == rdchem.HybridizationType.SP3:
        return "sp3"
    return "other"
