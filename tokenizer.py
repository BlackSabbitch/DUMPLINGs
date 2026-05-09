# tokenizer.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, Any, Optional, Union, Tuple
from logger import logger


class UniversalPDBBindDataset(Dataset):
    """
    Dataset adapter that converts dataframe rows into model-ready objects.

    Historically this project experimented with multiple representation
    families, so the dataset class still carries a "universal" shape. In the
    current A1/A2/A3 line, however, its main job is simpler:

    - convert cached graph dictionaries into PyG `Data`,
    - attach metadata such as `pdb_id`, protein sequence, and ligand SMILES,
    - preserve legacy sequence tensors for compatibility with older branches.

    Attributes:
        df: DataFrame containing molecular data and affinity labels.
        prot_vocab: Vocabulary mapping for protein sequences.
        lig_vocab: Vocabulary mapping for ligand sequences.
        max_prot: Maximum protein sequence length.
        max_lig: Maximum ligand sequence length.
        max_pock: Maximum pocket atoms.

    Example:
        >>> dataset = UniversalPDBBindDataset('data/refined.parquet', config)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     # batch contains (prot, lig, pock, target)
        ...     pass
    """

    def __init__(self, df: pd.DataFrame, config_dict: Dict[str, Any]) -> None:
        """
        Initialize the dataset.

        Args:
            df: Prepared dataframe containing parsed complexes and labels.
            config_dict: Configuration dictionary with dataset parameters.
        """
        self.df = df
        ds_cfg = config_dict['dataset']
        self.prot_vocab = {c: i for i, c in enumerate(ds_cfg['prot_vocab'])}
        self.lig_vocab = {c: i for i, c in enumerate(ds_cfg['lig_vocab'])}

        self.max_prot = ds_cfg.get('max_prot', 1000)
        self.max_lig = ds_cfg.get('max_lig', 150)
        self.max_pock = ds_cfg.get('max_pock', 63)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def _encode_sequence(self, text: Optional[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
        """
        Encode a legacy sequence/text representation into padded token ids.

        Args:
            text: Input text sequence or None.
            vocab: Vocabulary dictionary.
            max_len: Maximum sequence length.

        Returns:
            Encoded tensor with padding.
        """
        if text is None or pd.isna(text):
            # Return zero tensor (padding)
            return torch.zeros(max_len, dtype=torch.long)
        tokens = [vocab.get(c, 0) for c in str(text)[:max_len]]
        padded = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def _encode_graph(self, graph_dict: Optional[Dict[str, Any]]) -> Data:
        """
        Convert a cached graph dictionary into a PyG `Data` object.

        Args:
            graph_dict: Dictionary containing graph data or None.

        Returns:
            PyG Data object with encoded graph.
        """
        if graph_dict is None or not isinstance(graph_dict, dict):
            # Fallback for corrupted data
            return Data(x=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))
            
        tensor_kwargs = {}
        for key, value in graph_dict.items():
            if key == 'edge_index':
                ei = torch.tensor(value, dtype=torch.long)
                # Normalize edge-index layout to PyG's [2, num_edges].
                if ei.numel() > 0 and ei.shape[1] == 2 and ei.shape[0] != 2:
                    ei = ei.t().contiguous()
                tensor_kwargs[key] = ei
            elif isinstance(value, (list, np.ndarray)):
                # Integer-valued features such as atomic numbers stay in `long`,
                # while coordinates or continuous features are cast to float32.
                arr = np.array(value)
                dtype = torch.long if arr.dtype.kind in 'iu' else torch.float32
                tensor_kwargs[key] = torch.tensor(arr, dtype=dtype)
            else:
                tensor_kwargs[key] = value
                
        return Data(**tensor_kwargs)

    def _process_item(self, item: Any, vocab: Optional[Dict[str, int]] = None, max_len: Optional[int] = None) -> Union[torch.Tensor, Data]:
        """
        Route one dataframe cell to the appropriate encoder.

        Args:
            item: Input data (dict for graphs, string for sequences).
            vocab: Vocabulary for sequence encoding.
            max_len: Maximum length for sequence encoding.

        Returns:
            Encoded data (tensor for sequences, Data for graphs).
        """
        if isinstance(item, dict):
            return self._encode_graph(item)
        else:
            return self._encode_sequence(item, vocab, max_len)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Data], Union[torch.Tensor, Data], Union[torch.Tensor, Data], Data, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (protein_data, ligand_data, pocket_data, complex_data, target).
            In the current graph-first pipeline, `complex_data` is the primary
            geometric carrier used by A1/A2/A3.
        """
        row = self.df.iloc[idx]

        # These legacy slots are still returned for compatibility, even though
        # the active A1/A2/A3 family primarily consumes `complex_data`.
        prot_data = self._process_item(row['protein'], self.prot_vocab, self.max_prot)
        lig_data  = self._process_item(row['ligand'], self.lig_vocab, self.max_lig)
        pock_data = self._process_item(row['pocket'], self.prot_vocab, self.max_pock)

        # The fused ligand-pocket graph is the current main geometric input.
        complex_data = None
        if 'complex_graph' in row and row['complex_graph'] is not None:
            complex_data = self._encode_graph(row['complex_graph'])
        else:
            # Defensive fallback for malformed cached rows.
            complex_data = self._encode_graph(None)
        complex_data.pdb_id = str(row['pdb_id'])
        if 'protein' in row and row['protein'] is not None and not pd.isna(row['protein']):
            complex_data.protein_sequence = str(row['protein'])
        ligand_smiles = None
        if 'ligand_smiles' in row and row['ligand_smiles'] is not None and not pd.isna(row['ligand_smiles']):
            ligand_smiles = str(row['ligand_smiles'])
        elif 'ligand' in row and isinstance(row['ligand'], str) and not pd.isna(row['ligand']):
            ligand_smiles = str(row['ligand'])
        if ligand_smiles is not None:
            complex_data.ligand_smiles = ligand_smiles

        # Targets are already normalized upstream by the experiment runner.
        target = torch.tensor(row['pkd'], dtype=torch.float32)
        
        return prot_data, lig_data, pock_data, complex_data, target
