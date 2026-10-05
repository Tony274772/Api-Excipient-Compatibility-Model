"""Excipient prior block for the PGB architecture.

Implements Section 3.3 of the instructions document:
- Exact key: stereo-blind canonical SMILES, salt pieces sorted, hydrate kept
- Family key: hydrate / solvent stripped, same smoothing
- Neighbour key: Morgan-512 Tanimoto ≥ 0.6, similarity-weighted
- Leave-own-API-cluster-out prior for training rows
- Val / test / 24-pair: prior from all train rows

Output per excipient: [p_exact, p_exact − 0.094, p_family, log(1+n), unseen_flag]
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, rdFingerprintGenerator
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical key functions
# ═══════════════════════════════════════════════════════════════════════════════

def _stereo_blind_canonical(smiles: str) -> str:
    """Return a stereo-blind canonical SMILES.

    Steps:
    1. Parse each dot-separated fragment.
    2. Remove stereo (chiral tags, E/Z).
    3. Canonicalize each fragment.
    4. Sort the fragments alphabetically.
    5. Rejoin with '.'.
    """
    if not smiles or smiles == "nan":
        return ""

    parts = smiles.split(".")
    canon_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            canon_parts.append(part)
            continue
        Chem.RemoveStereochemistry(mol)
        canon = Chem.MolToSmiles(mol, canonical=True)
        canon_parts.append(canon)

    canon_parts.sort()
    return ".".join(canon_parts)


# Common solvents / hydrate fragments to strip for the family key
_HYDRATE_SOLVENT_SMILES = {"O", "[2H]O[2H]", "O=CO", "CCO", "ClCCl", "CO"}


def _family_key(smiles: str) -> str:
    """Hydrate / solvent-stripped family key.

    Removes water, deuterated water, ethanol, methanol, formic acid,
    DCM fragments from the salt string, then canonicalizes.
    """
    if not smiles or smiles == "nan":
        return ""

    parts = smiles.split(".")
    kept = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            kept.append(part)
            continue
        Chem.RemoveStereochemistry(mol)
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon not in _HYDRATE_SOLVENT_SMILES:
            kept.append(canon)

    if not kept:
        # Everything was solvent — keep original
        return _stereo_blind_canonical(smiles)

    kept.sort()
    return ".".join(kept)


# ═══════════════════════════════════════════════════════════════════════════════
# Morgan-512 fingerprint for Tanimoto nearest-neighbour lookup
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _morgan_fp_for_tanimoto(smiles: str):
    """Compute Morgan-512 fingerprint for Tanimoto comparison.

    Returns an RDKit ExplicitBitVect or None.
    """
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    return gen.GetFingerprint(mol)


# ═══════════════════════════════════════════════════════════════════════════════
# Prior table builder
# ═══════════════════════════════════════════════════════════════════════════════

class ExcipientPriorTable:
    """Precomputed excipient prior table.

    Built once from training data; queried per-row during dataset
    construction.

    Attributes:
        global_rate: global incompatibility rate across all train rows.
        exact_stats: {stereo_blind_key: (n_pos, n_total)}
        family_stats: {family_key: (n_pos, n_total)}
        cluster_stats: {cluster_id: {stereo_blind_key: (n_pos, n_total)}}
        nn_fps: {stereo_blind_key: RDKit ExplicitBitVect}
    """

    def __init__(
        self,
        train_exc_smiles: list[str],
        train_labels: list[int],
        train_api_clusters: Optional[list] = None,
        laplace_alpha: int = 4,
        tanimoto_threshold: float = 0.6,
        global_rate: float = 0.094,
    ):
        self.laplace_alpha = laplace_alpha
        self.tanimoto_threshold = tanimoto_threshold
        self.global_rate = global_rate

        # ── Build exact-key stats ────────────────────────────────────────
        self.exact_stats: dict[str, tuple[int, int]] = {}
        self.family_stats: dict[str, tuple[int, int]] = {}
        self.cluster_stats: dict[int, dict[str, tuple[int, int]]] = {}
        self.nn_fps: dict[str, DataStructs.ExplicitBitVect] = {}

        # Per-key accumulators
        exact_acc: dict[str, list[int]] = {}
        family_acc: dict[str, list[int]] = {}
        cluster_exact_acc: dict[int, dict[str, list[int]]] = {}

        for i, (exc_smi, label) in enumerate(zip(train_exc_smiles, train_labels)):
            ekey = _stereo_blind_canonical(exc_smi)
            fkey = _family_key(exc_smi)
            cluster = train_api_clusters[i] if train_api_clusters is not None else None

            exact_acc.setdefault(ekey, []).append(label)
            family_acc.setdefault(fkey, []).append(label)

            if cluster is not None:
                cluster_exact_acc.setdefault(cluster, {})
                cluster_exact_acc[cluster].setdefault(ekey, []).append(label)

        # Summarize
        for ekey, labels_list in exact_acc.items():
            n_pos = sum(labels_list)
            n_total = len(labels_list)
            self.exact_stats[ekey] = (n_pos, n_total)

        for fkey, labels_list in family_acc.items():
            n_pos = sum(labels_list)
            n_total = len(labels_list)
            self.family_stats[fkey] = (n_pos, n_total)

        for cluster, ekey_dict in cluster_exact_acc.items():
            self.cluster_stats[cluster] = {}
            for ekey, labels_list in ekey_dict.items():
                n_pos = sum(labels_list)
                n_total = len(labels_list)
                self.cluster_stats[cluster][ekey] = (n_pos, n_total)

        # ── Build Morgan FPs for nearest-neighbour lookup ────────────────
        seen_smiles = set()
        for exc_smi in train_exc_smiles:
            ekey = _stereo_blind_canonical(exc_smi)
            if ekey not in self.nn_fps and ekey not in seen_smiles:
                fp = _morgan_fp_for_tanimoto(exc_smi)
                if fp is not None:
                    self.nn_fps[ekey] = fp
                seen_smiles.add(ekey)

    def _laplace_smooth(self, n_pos: int, n_total: int) -> float:
        """Laplace-smoothed rate: (pos + alpha * base) / (total + alpha)."""
        alpha = self.laplace_alpha
        return (n_pos + alpha * self.global_rate) / (n_total + alpha)

    def _get_nn_prior(self, exc_smiles: str, exclude_key: Optional[str] = None) -> float:
        """Nearest-neighbour prior via Morgan-512 Tanimoto.

        Returns similarity-weighted rate of excipients with Tanimoto ≥ threshold.
        Falls back to global rate if no neighbour found.
        """
        query_fp = _morgan_fp_for_tanimoto(exc_smiles)
        if query_fp is None:
            return self.global_rate

        weighted_sum = 0.0
        weight_total = 0.0

        for ekey, ref_fp in self.nn_fps.items():
            if exclude_key is not None and ekey == exclude_key:
                continue
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            if sim >= self.tanimoto_threshold:
                stats = self.exact_stats.get(ekey)
                if stats is not None:
                    rate = self._laplace_smooth(stats[0], stats[1])
                    weighted_sum += sim * rate
                    weight_total += sim

        if weight_total > 0:
            return weighted_sum / weight_total
        return self.global_rate

    def get_prior_block(
        self,
        exc_smiles: str,
        api_cluster: Optional[int] = None,
        is_train: bool = False,
    ) -> np.ndarray:
        """Compute the 5-dim prior block for one excipient.

        For training rows: leave-own-API-cluster-out.
        For val / test / 24-pair: use all train rows.

        Returns:
            [p_exact, p_exact - 0.094, p_family, log(1+n), unseen_flag]
        """
        ekey = _stereo_blind_canonical(exc_smiles)
        fkey = _family_key(exc_smiles)

        # ── Exact prior ──────────────────────────────────────────────────
        if is_train and api_cluster is not None and api_cluster in self.cluster_stats:
            # Leave-own-API-cluster-out: subtract this cluster's contribution
            full_stats = self.exact_stats.get(ekey, (0, 0))
            cluster_stats = self.cluster_stats[api_cluster].get(ekey, (0, 0))
            n_pos = full_stats[0] - cluster_stats[0]
            n_total = full_stats[1] - cluster_stats[1]
        else:
            stats = self.exact_stats.get(ekey)
            if stats is not None:
                n_pos, n_total = stats
            else:
                n_pos, n_total = 0, 0

        if n_total > 0:
            p_exact = self._laplace_smooth(n_pos, n_total)
        else:
            # Try nearest-neighbour
            p_nn = self._get_nn_prior(exc_smiles, exclude_key=ekey if is_train else None)
            p_exact = p_nn
            n_total = 0  # still unseen under exact key

        # ── Family prior ─────────────────────────────────────────────────
        fstats = self.family_stats.get(fkey)
        if fstats is not None and fstats[1] > 0:
            p_family = self._laplace_smooth(fstats[0], fstats[1])
        else:
            p_family = self.global_rate

        # ── Unseen flag ──────────────────────────────────────────────────
        full_stats = self.exact_stats.get(ekey, (0, 0))
        unseen = 1.0 if full_stats[1] == 0 else 0.0

        return np.array([
            p_exact,
            p_exact - self.global_rate,
            p_family,
            np.log1p(full_stats[1]),  # log(1 + n)
            unseen,
        ], dtype=np.float32)

    def get_logit_offset_p(self, exc_smiles: str) -> float:
        """Return p_exact for use in the logit offset.

        The offset uses p_exact only (not p_family or p_nn).
        """
        ekey = _stereo_blind_canonical(exc_smiles)
        stats = self.exact_stats.get(ekey)
        if stats is not None and stats[1] > 0:
            return self._laplace_smooth(stats[0], stats[1])
        # Nearest-neighbour fallback
        return self._get_nn_prior(exc_smiles)
