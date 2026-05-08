# parsers/__init__.py

from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser
from parsers.interaction_graph_parser import InteractionGraphParser


class ParserFactory:
    @classmethod
    def _build_one(cls, mode, is_ligand=False, **kwargs):
        """
        Build a single parser instance from a one-letter parser code.

        Args:
            mode: Parser family code: `C` (sequence/SMILES), `G` (graph),
                or `E` (fused interaction graph).
            is_ligand: Whether the parser should interpret its input as a
                ligand rather than a protein-side object.
            **kwargs: Parser-specific keyword arguments.

        Returns:
            A configured parser instance.
        """
        if mode == 'C':
            return CNNParser(is_ligand=is_ligand)
        if mode == 'G':
            return GNNParser(
                is_ligand=is_ligand,
                dist_threshold=kwargs.get('dist_threshold', 10.0),
                ca_only=kwargs.get('ca_only', True))
        if mode in ['E']:
            return InteractionGraphParser(**kwargs)
        raise ValueError(f"Unknown parser mode: {mode}")

    @classmethod
    def build_chain(cls, config_dict):
        """
        Build the parser chain described by the selected graph-encoder config.

        The configuration uses a short string such as `CGC` or `NE` to
        describe which parser should be attached to the protein, ligand, and
        pocket/complex slots.

        Args:
            config_dict: Full experiment configuration dictionary.

        Returns:
            Ordered list of parser instances matching the configured signature.
        """
        pairing = {'C': "cnn_params", 'G': "gnn_params", 'E': "egnn_params"}
        mode = config_dict['model']['graph_encoder']['selected']
        available_cfg = config_dict['model']['graph_encoder']['available'][mode]
        config_str = available_cfg['protein_ligand_pocket_encoders']

        parsers = []
        for i, char in enumerate(config_str):
            if char == 'N':
                parsers.append(None)
                continue
            kwargs = available_cfg[pairing[char]]
            p = cls._build_one(
                mode=char, 
                is_ligand=(i == 1),
                **kwargs
            )
            parsers.append(p)
            
        return parsers
