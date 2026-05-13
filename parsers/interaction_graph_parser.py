# parsers/interaction_graph_parser.py

import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from ._base_parser import BaseParser
from .local_chemical_features import (
    feature_names_from_config,
    build_local_chemical_node_features,
    normalize_local_chemical_features_config,
)
from logger import log_info, log_warn


class InteractionGraphParser(BaseParser):
    """
    Parser that builds a single interaction graph combining ligand and pocket atoms.

    Produces a graph dictionary with node features, 3D positions, and edge indices.
    """

    def __init__(
        self,
        dist_threshold: float = 5.0,
        ca_only: bool = False,
        local_chemical_features: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.parser_version = 2
        self.dist_threshold = dist_threshold
        self.ca_only = ca_only
        self.local_chemical_features_cfg = normalize_local_chemical_features_config(local_chemical_features)
        self.local_chemical_features_enabled = self.local_chemical_features_cfg.enabled
        self.local_chemical_feature_flags = dict(self.local_chemical_features_cfg.features)
        self.local_chemical_feature_names = feature_names_from_config(self.local_chemical_features_cfg)
        self.pdb_parser = PDBParser(QUIET=True)
        log_info(
            "Initialized "
            f"dist_threshold={dist_threshold}, ca_only={ca_only}, "
            f"local_chemical_features_enabled={self.local_chemical_features_enabled}, "
            f"local_chemical_feature_count={len(self.local_chemical_feature_names)}",
            stage="InteractionGraphParser",
        )

    @staticmethod
    def _stabilize_duplicate_coordinates(coords: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """
        DimeNet++ can become numerically unstable when different nodes share
        exactly the same 3D position. We keep the graph topology intact but
        nudge repeated coordinates by a tiny deterministic offset.
        """
        if coords.shape[0] < 2:
            return coords

        adjusted = coords.astype(np.float32, copy=True)
        seen: Dict[Tuple[float, float, float], int] = {}
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        for idx, coord in enumerate(adjusted):
            key = tuple(np.round(coord, 6).tolist())
            dup_count = seen.get(key, 0)
            if dup_count > 0:
                direction = directions[(dup_count - 1) % len(directions)]
                adjusted[idx] = coord + direction * (eps * dup_count)
            seen[key] = dup_count + 1

        return adjusted

    def parse_file(self, lig_path: str, pock_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse ligand and pocket files from disk.
        """
        if pock_path is None:
            return None, "InteractionGraphParser requires lig_path and pock_path"
        try:
            return self._build_complex_graph(lig_path, pock_path, is_file=True)
        except Exception as e:
            log_warn(f"parse_file failure: {e}", stage="InteractionGraphParser")
            return None, str(e)

    def _build_complex_graph(self, lig_data: Any, pock_data: Any, is_file: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if is_file:
            lig_mol = Chem.MolFromMolFile(lig_data, sanitize=False)
        else:
            lig_content = lig_data.decode('utf-8') if isinstance(lig_data, bytes) else lig_data
            lig_mol = Chem.MolFromMolBlock(lig_content, sanitize=False)

        if not lig_mol:
            return None, "ligand_load_error"

        try:
            Chem.SanitizeMol(
                lig_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass

        lig_coords = lig_mol.GetConformer().GetPositions()
        lig_x = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic()), 1] for a in lig_mol.GetAtoms()]

        pock_coords: List[List[float]] = []
        pock_x: List[List[int]] = []
        pock_atom_records: List[Dict[str, Any]] = []

        if is_file:
            struct = self.pdb_parser.get_structure("pock", pock_data)
        else:
            stream = io.StringIO(pock_data.decode('utf-8') if isinstance(pock_data, bytes) else pock_data)
            struct = self.pdb_parser.get_structure("pock", stream)

        for model in struct:
            for chain in model:
                for residue in chain:
                    atoms_to_process = [residue['CA']] if self.ca_only and 'CA' in residue else residue.get_atoms()
                    for atom in atoms_to_process:
                        if not self.ca_only and atom.element == 'H':
                            continue
                        coord = atom.get_coord()
                        pock_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
                        residue_name = getattr(residue, "get_resname", lambda: "UNK")()
                        atom_name = atom.get_name().strip()
                        element = str(atom.element).upper()
                        pock_atom_records.append(
                            {
                                "residue_name": str(residue_name).upper(),
                                "atom_name": atom_name.upper(),
                                "element": element,
                            }
                        )
                        atomic_num = 6 if atom.element == 'C' else 7 if atom.element == 'N' else 8 if atom.element == 'O' else 16 if atom.element == 'S' else 0
                        pock_x.append([atomic_num, 0, 0, 0])
            break

        if not pock_coords:
            return None, "empty_pocket_ca" if self.ca_only else "empty_pocket"

        all_x = np.array(lig_x + pock_x)
        all_coords = np.concatenate([lig_coords, np.array(pock_coords)], axis=0)
        all_coords = self._stabilize_duplicate_coordinates(all_coords)
        local_chemical_x = None
        if self.local_chemical_features_enabled:
            local_chemical_x = build_local_chemical_node_features(
                lig_mol,
                lig_coords,
                pock_atom_records,
                np.asarray(pock_coords, dtype=np.float32),
                dist_threshold=self.dist_threshold,
                cfg=self.local_chemical_features_cfg,
            )

        edges: List[List[int]] = []
        for bond in lig_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])

        dist_mat = cdist(lig_coords, pock_coords)
        lig_idx, pock_idx = np.where(dist_mat < self.dist_threshold)
        for i, j in zip(lig_idx, pock_idx):
            p_idx_shifted = j + len(lig_x)
            edges.extend([[i, p_idx_shifted], [p_idx_shifted, i]])

        graph_dict = {
            'x': all_x.tolist(),
            'pos': all_coords.tolist(),
            'edge_index': edges,
        }
        if local_chemical_x is not None:
            graph_dict["local_chemical_x"] = local_chemical_x.tolist()
            graph_dict["local_chemical_feature_names"] = tuple(self.local_chemical_feature_names)
        return graph_dict, None

    def _process_ligand(self, mol: Any):
        pass

    def _process_protein(self, path_or_bytes: Any, is_file: bool = True):
        pass
