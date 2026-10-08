"""Scaffold matching and retention regression tests."""
import random
import unittest
from rdkit import Chem, RDLogger
from scaffold_metrics import scaffold_checks, scaffold_core_from_scaffold, PROTOCOL

RDLogger.DisableLog('rdApp.*')


class ScaffoldMetricTests(unittest.TestCase):
    def test_bridge_examples_and_atom_order(self):
        core = 'C1=C2OCCCCOC(=C1)O2'
        for generated in ('COC1CCOC2=CC=C(OC1)O2', 'CC1CCOC2=CC=C(OC1)O2', 'CCOC1CCOC2=CC=C(OC1)O2'):
            mol = Chem.MolFromSmiles(generated)
            for seed in range(12):
                order = list(range(mol.GetNumAtoms()))
                random.Random(seed).shuffle(order)
                variant = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False, kekuleSmiles=bool(seed % 2))
                checked = scaffold_checks(variant, core)
                self.assertTrue(checked['core_match'], variant)
                self.assertTrue(checked['core_retention'], variant)

    def test_aromatic_n_dummy_is_capped(self):
        scaffold = '*n1cccc1'
        core = scaffold_core_from_scaffold(scaffold)
        self.assertIsNotNone(core)
        self.assertIn('[nH]', core)
        self.assertTrue(scaffold_checks('Cn1cccc1', core)['core_retention'])
        actual = 'O=c1cnn([1*:1])c(C#CC2CO2)c1'
        self.assertTrue(scaffold_checks('Nn1ncc(=O)cc1C#CC1CO1', scaffold_core_from_scaffold(actual))['core_retention'])

    def test_bond_order_and_element_changes_still_fail(self):
        for generated in ('C1CCCCC1', 'C1=CCCCC1', 'c1ccncc1'):
            self.assertFalse(scaffold_checks(generated, 'c1ccccc1')['core_match'])
        self.assertFalse(scaffold_checks('C1CC1', '[13CH2]1CC1')['core_retention'])
        self.assertFalse(scaffold_checks('[NH2+]1CCCCC1', 'N1CCCCC1')['core_retention'])

    def test_added_ring_matches_but_is_not_retained(self):
        r = scaffold_checks('c1ccc(-c2ccccc2)cc1', 'c1ccccc1')
        self.assertTrue(r['core_match'])
        self.assertFalse(r['core_retention'])

    def test_isomeric_retention_not_erased(self):
        a='C1CC[C@H]2CCCC[C@H]2C1'
        b='C1CC[C@H]2CCCC[C@@H]2C1'
        self.assertTrue(scaffold_checks(a,a)['core_retention'])
        r = scaffold_checks(a,b)
        self.assertFalse(r['core_retention'])

    def test_invalid_targets_are_explicit(self):
        for core in ('', None, '*c1ccccc1', 'not a smiles'):
            self.assertFalse(scaffold_checks('Cc1ccccc1', core)['eligible'])
        self.assertIsNone(scaffold_core_from_scaffold('C*C'))

    def test_cached_targets_are_not_mutated(self):
        core = 'c1ccccc1'
        original = scaffold_checks('Cc1ccccc1', core)
        scaffold_checks('C1CCCCC1', core)
        self.assertEqual(original, scaffold_checks('Cc1ccccc1', core))

    def test_retention_requires_match(self):
        molecules = ['c1ccccc1', 'C1CCCCC1', 'c1ccc2ccccc2c1', 'Cn1cccc1', 'C1=C2OCCCCOC(=C1)O2']
        for a in molecules:
            for b in molecules:
                r = scaffold_checks(a, b)
                self.assertFalse(r['core_retention'] and not r['core_match'])





if __name__ == '__main__':
    unittest.main()
