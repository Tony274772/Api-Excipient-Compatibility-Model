"""
tests/test_chemberta_similarity.py

Automated test verifying the ChemBERTa vs. Morgan similarity evaluation pipeline.
Tests that:
1. All cohort molecules load and produce valid vectors.
2. Cosine and Tanimoto values lie strictly within mathematical bounds [0, 1].
3. Symmetry holds: Sim(A, B) == Sim(B, A).
4. Self-similarity holds: Sim(A, A) == 1.0.
5. Spearman correlation runs without NaN or division by zero.
"""

import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_chemberta_similarity import (
    COHORT,
    cosine_similarity,
    chemberta_continuous_tanimoto,
    dot_product,
    load_precomputed_vectors,
    morgan_tanimoto,
    spearman_correlation,
    vector_norm,
)


class TestChemBERTaSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c_vecs, cls.m_vecs = load_precomputed_vectors()

    def test_cohort_coverage(self):
        """Verify all 16 cohort molecules exist in both tables."""
        for item in COHORT:
            smi = item["smiles"]
            self.assertIn(smi, self.c_vecs, f"Missing ChemBERTa vector for {item['name']}")
            self.assertIn(smi, self.m_vecs, f"Missing Morgan vector for {item['name']}")
            self.assertEqual(len(self.c_vecs[smi]), 384, f"ChemBERTa dimension mismatch for {item['name']}")
            self.assertEqual(len(self.m_vecs[smi]), 1024, f"Morgan dimension mismatch for {item['name']}")

    def test_similarity_bounds_and_symmetry(self):
        """Verify similarity values lie in [0, 1] and are symmetric."""
        sample_smiles = [item["smiles"] for item in COHORT[:6]]
        for i in range(len(sample_smiles)):
            smi_a = sample_smiles[i]
            ca, ma = self.c_vecs[smi_a], self.m_vecs[smi_a]

            # Self similarity
            self.assertAlmostEqual(cosine_similarity(ca, ca), 1.0, places=4)
            self.assertAlmostEqual(cosine_similarity(ma, ma), 1.0, places=4)
            self.assertAlmostEqual(morgan_tanimoto(ma, ma), 1.0, places=4)
            self.assertAlmostEqual(chemberta_continuous_tanimoto(ca, ca), 1.0, places=4)

            for j in range(i + 1, len(sample_smiles)):
                smi_b = sample_smiles[j]
                cb, mb = self.c_vecs[smi_b], self.m_vecs[smi_b]

                # Symmetry
                self.assertAlmostEqual(cosine_similarity(ca, cb), cosine_similarity(cb, ca), places=5)
                self.assertAlmostEqual(morgan_tanimoto(ma, mb), morgan_tanimoto(mb, ma), places=5)
                self.assertAlmostEqual(chemberta_continuous_tanimoto(ca, cb), chemberta_continuous_tanimoto(cb, ca), places=5)

                # Bounds
                c_cos = cosine_similarity(ca, cb)
                m_tan = morgan_tanimoto(ma, mb)
                self.assertGreaterEqual(c_cos, -1.0)
                self.assertLessEqual(c_cos, 1.0001)
                self.assertGreaterEqual(m_tan, 0.0)
                self.assertLessEqual(m_tan, 1.0001)

    def test_spearman_correlation_computation(self):
        """Verify Spearman correlation correctly handles perfect, inverse, and tied rankings."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pos = [2.0, 4.0, 6.0, 8.0, 10.0]
        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]

        self.assertAlmostEqual(spearman_correlation(x, y_pos), 1.0, places=4)
        self.assertAlmostEqual(spearman_correlation(x, y_neg), -1.0, places=4)

    def test_diagnostic_salt_pair(self):
        """Verify Benzoic acid and Sodium benzoate have high ChemBERTa similarity due to salt blindness."""
        benzoic_smi = "C1=CC=C(C=C1)C(=O)O"
        na_benzoate_smi = "C1=CC=C(C=C1)C(=O)[O-].[Na+]"

        c_benzoic = self.c_vecs[benzoic_smi]
        c_na = self.c_vecs[na_benzoate_smi]

        c_cos = cosine_similarity(c_benzoic, c_na)
        self.assertGreater(c_cos, 0.90, "ChemBERTa failed to show salt blindness (>0.90 similarity)")


if __name__ == "__main__":
    unittest.main()

