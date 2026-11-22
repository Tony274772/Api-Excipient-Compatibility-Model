"""Unit tests for dmpnn__x_maccs ablation suite."""

import unittest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.ablations_dmpnn_maccs import variants as V
from experiments.ablations_dmpnn_maccs.run_seeds import build_queue
from experiments.ablations_dmpnn_maccs.subsets import is_salt_or_ion, is_sugar_or_polyol, pair_class


class TestDmpnnMaccsAblations(unittest.TestCase):

    def test_all_plan_variants_registered(self):
        """Every variant in PLAN must be registered in ABLATIONS and DESCRIPTIONS."""
        self.assertEqual(len(V.PLAN), 23)
        self.assertEqual(V.PLAN[0], "full")
        for name in V.PLAN:
            parts = V.parse_ablation(name)
            self.assertTrue(len(parts) >= 1)
            desc = V.describe(name)
            self.assertTrue(len(desc) > 5)

    def test_combined_ablation_parsing(self):
        """Combinations like prior1+no_w or no_flags_metal+no_salt must parse cleanly."""
        parts1 = V.parse_ablation("prior1+no_w")
        self.assertEqual(parts1, ["prior1", "no_w"])

        parts2 = V.parse_ablation("no_flags_metal+no_salt")
        self.assertEqual(parts2, ["no_flags_metal", "no_salt"])

        with self.assertRaises(ValueError):
            V.parse_ablation("non_existent_ablation")

        with self.assertRaises(ValueError):
            V.parse_ablation("full+no_w")

    def test_build_settings(self):
        """Variants requiring build modifications must return correct settings dicts."""
        self.assertEqual(V.build_settings("no_family"), {"prior_lookup": "no_family"})
        self.assertEqual(V.build_settings("mech_v2"), {"mech_set": "v2"})
        self.assertEqual(V.build_settings("pool_gated"), {"pooling": "global_gated_attention"})
        self.assertEqual(V.build_settings("shuf_maccs"), {"secondary": "shuf_maccs"})
        self.assertEqual(V.build_settings("dmpnn_parent"), {"parent_smiles": True})
        self.assertEqual(V.build_settings("no_w"), {})

    def test_weight_pairing_flags(self):
        """Only mech_v2 should be flagged as not weight-paired."""
        self.assertTrue(V.weight_paired("full"))
        self.assertTrue(V.weight_paired("no_w"))
        self.assertTrue(V.weight_paired("no_secondary"))
        self.assertTrue(V.weight_paired("no_dmpnn"))
        self.assertFalse(V.weight_paired("mech_v2"))

    def test_run_seeds_queue(self):
        """Step 0 ('full') must always be first in the queue."""
        queue = build_queue(V.PLAN, [42, 43])
        self.assertEqual(len(queue), 23 * 2)
        self.assertEqual(queue[0], ("full", 42))
        self.assertEqual(queue[1], ("full", 43))

    def test_subsets_chemical_classification(self):
        """Verify chemical classification rules for salts and sugars."""
        # Salts / ions
        self.assertTrue(is_salt_or_ion("c1ccccc1C(=O)[O-].[Na+]"))  # Sodium benzoate
        self.assertTrue(is_salt_or_ion("[Ca+2].[O-]S(=O)(=O)[O-]"))  # Calcium sulfate
        self.assertFalse(is_salt_or_ion("CC(=O)Oc1ccccc1C(=O)O"))   # Neutral Aspirin

        # Sugars / polyols
        self.assertTrue(is_sugar_or_polyol("C(C1C(C(C(C(O1)O)O)O)O)O"))  # Glucose
        self.assertTrue(is_sugar_or_polyol("C(C(C(C(C(CO)O)O)O)O)O"))    # Sorbitol / Mannitol
        self.assertFalse(is_sugar_or_polyol("c1ccccc1C(=O)O"))            # Benzoic acid

        # Pair classification
        cls1 = pair_class("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1C(=O)[O-].[Na+]")
        self.assertEqual(cls1, "salts_and_ions")

        cls2 = pair_class("CC(=O)Oc1ccccc1C(=O)O", "C(C1C(C(C(C(O1)O)O)O)O)O")
        self.assertEqual(cls2, "sugars_polyols")

        cls3 = pair_class("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1")
        self.assertEqual(cls3, "clean_organic")


if __name__ == "__main__":
    unittest.main()

