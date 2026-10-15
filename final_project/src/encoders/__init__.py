"""Encoder registry, trimmed to the one encoder the final Morgan-PGB model uses.

The Morgan-PGB model builds its fingerprints from SMILES inside PGBDataset, so the
fixed-vector encoder only supplies `is_sequence_capable = False` (concat path, no
structural branch) and holds no weights.
"""


def _get_fixed_vector():
    from src.encoders.fixed_vector_encoder import FixedVectorEncoder
    return FixedVectorEncoder


class _LazyRegistry(dict):
    _factories = {"fixed_vector": _get_fixed_vector}

    def __getitem__(self, key):
        if key in self._factories:
            return self._factories[key]()
        raise KeyError(f"Unknown encoder: {key} (final_project only ships 'fixed_vector')")

    def __contains__(self, key):
        return key in self._factories

    def keys(self):
        return self._factories.keys()


ENCODER_REGISTRY = _LazyRegistry()
