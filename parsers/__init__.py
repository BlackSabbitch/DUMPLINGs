from models.graph_components import get_global_graph_config
from parsers.interaction_graph_parser import InteractionGraphParser


class ParserFactory:
    @classmethod
    def build_chain(cls, config_dict):
        """
        Build the parser chain used by the current DUMPLINGs pipeline.

        The modern A1/A2/A3 path always starts from a single fused interaction
        graph over `ligand + pocket`. Protein sequence extraction and ligand
        SMILES extraction are optional side channels handled by the extractor,
        so they are not represented as primary parsers here.
        """

        global_graph_cfg = get_global_graph_config(config_dict)
        return [
            None,
            InteractionGraphParser(
                dist_threshold=global_graph_cfg.get("dist_threshold", 5.0),
                ca_only=global_graph_cfg.get("ca_only", False),
            ),
        ]
