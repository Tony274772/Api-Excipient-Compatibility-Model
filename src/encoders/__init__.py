"""Encoder registry – Section 5.1.

config.encoder selects the key. model.py reads encoder.is_sequence_capable
to decide cross-attention vs concat fusion.

Imports are lazy to avoid failing when optional dependencies (e.g. dgl) are missing.
"""


def _get_molformer():
    from src.encoders.molformer_encoder import MoLFormerEncoder
    return MoLFormerEncoder


def _get_gin():
    from src.encoders.gin_encoder import PretrainedGINEncoder
    return PretrainedGINEncoder


def _get_chemberta():
    from src.encoders.chemberta_encoder import ChemBERTaEncoder
    return ChemBERTaEncoder


def _get_fixed_vector():
    from src.encoders.fixed_vector_encoder import FixedVectorEncoder
    return FixedVectorEncoder


def _get_dmpnn():
    from src.encoders.dmpnn_encoder import DMPNNEncoder
    return DMPNNEncoder


def _get_dmpnn_scratch():
    from src.encoders.dmpnn_encoder import DMPNNEncoder
    return DMPNNEncoder


def _get_attentivefp():
    from src.encoders.attentivefp_encoder import AttentiveFPEncoder
    return AttentiveFPEncoder


def _get_gine():
    from src.encoders.gine_encoder import GINEEncoder
    return GINEEncoder


def _get_gatv2():
    from src.encoders.gatv2_encoder import GATv2Encoder
    return GATv2Encoder


def _get_pna():
    from src.encoders.pna_encoder import PNAEncoder
    return PNAEncoder


def _get_pretrained_gat():
    from src.encoders.stanford_gat_encoder import StanfordPretrainedGATEncoder
    return StanfordPretrainedGATEncoder


class _LazyRegistry(dict):
    """Dict that resolves encoder classes lazily on access."""

    _factories = {
        "molformer": _get_molformer,
        "pretrained_gin": _get_gin,
        "chemberta": _get_chemberta,
        "fixed_vector": _get_fixed_vector,
        "dmpnn_chemprop": _get_dmpnn,
        "dmpnn_scratch": _get_dmpnn_scratch,
        "attentivefp": _get_attentivefp,
        "gine": _get_gine,
        "gatv2": _get_gatv2,
        "pna": _get_pna,
        "pretrained_gat": _get_pretrained_gat,
    }

    def __getitem__(self, key):
        if key in self._factories:
            return self._factories[key]()
        raise KeyError(f"Unknown encoder: {key}")

    def __contains__(self, key):
        return key in self._factories

    def keys(self):
        return self._factories.keys()

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]


ENCODER_REGISTRY = _LazyRegistry()
