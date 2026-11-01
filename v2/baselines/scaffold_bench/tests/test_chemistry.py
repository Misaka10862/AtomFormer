import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import convert, join, score, reward, augmented_likelihood_score, derive_decoration


class ChemistryTests(unittest.TestCase):
    def test_repair_requires_fixed_scaffold_and_single_boundary(self):
        decoration, _ = derive_decoration('CCC', '*C')
        self.assertEqual(join('*C', decoration), 'CCC')
        with self.assertRaises(ValueError):
            derive_decoration('O', '*C')
        with self.assertRaises(ValueError):
            derive_decoration('C=C', '*C')

    def test_roundtrip_and_failure(self):
        row = convert((5, {'scaffold_smiles': '[*:1]c1ccccc1', 'rooted_decoration_smiles': 'C', 'canonical_smiles': 'Cc1ccccc1'}))
        self.assertEqual(row['record_id'], 5)
        self.assertEqual(join(row['scaffold'], row['decoration']), 'Cc1ccccc1')
        self.assertIsNone(join(row['scaffold'], '*C*'))
        self.assertIsNone(join(row['scaffold'], ''))
        sulfur = convert((6, {'scaffold_smiles': 'C1CC2OC23C1NCC3[1*:1]',
                              'rooted_decoration_smiles': '[SH](N)(=O)=O',
                              'canonical_smiles': 'NS(=O)(=O)C1CNC2CCC3OC231'}))
        self.assertEqual(join(sulfur['scaffold'], sulfur['decoration']), sulfur['source'])
        aromatic_n = convert((7, {'scaffold_smiles': '*CC', 'rooted_decoration_smiles': '[nH]1cccc1',
                                  'canonical_smiles': 'c1ccn(CC)c1'}))
        self.assertEqual(join(aromatic_n['scaffold'], aromatic_n['decoration']), aromatic_n['source'])
        original = convert((123, {'scaffold_smiles': '*CC', 'rooted_decoration_smiles': 'WRONG',
                                  'single_decoration_smiles': 'NS(=O)(=O)[1*:1]',
                                  'canonical_smiles': 'CCS(N)(=O)=O'}))
        self.assertNotIn('[1*]', original['decoration'])
        self.assertIn('*', original['decoration'])
        self.assertEqual(join(original['scaffold'], original['decoration']), original['source'])
        bridge = convert((15596, {'scaffold_smiles': 'O=C1NC2C3CC2([1*:1])CC3S1(=O)=O',
                                  'single_decoration_smiles': 'O[1*:1]',
                                  'canonical_smiles': 'O=C1NC2C3CC2(O)CC3S1(=O)=O'}))
        self.assertEqual(join(bridge['scaffold'], bridge['decoration']), bridge['source'])
        bad_validation = convert((15596, {'scaffold_smiles': 'C=C(C(O)CC)[1*:1]',
                                         'single_decoration_smiles': 'C=C(C(O)CC)[1*:1]',
                                         'rooted_decoration_smiles': 'C(=C)C(O)CC',
                                         'canonical_smiles': 'C=C(N=C1CNC=N1)C(O)CC'}))
        self.assertEqual(bad_validation['decoration_origin'], 'source_graph_complement')
        self.assertEqual(bad_validation['scaffold'], '*C(=C)C(O)CC')
        self.assertEqual(join(bad_validation['scaffold'], bad_validation['decoration']), bad_validation['source'])
        good = score((row, 'Cc1ccccc1', '*C', 0))
        self.assertTrue(good['valid'] and good['core_match'] and good['core_retention'])
        assembled = score((row, '__assemble__', '*C', 0))
        self.assertEqual(assembled['generated'], good['generated'])
        self.assertAlmostEqual(reward(good, set()) - reward(good, {'Cc1ccccc1'}), .2)
        invalid = score((row, None, '', 1))
        self.assertFalse(invalid['valid'])
        self.assertEqual(reward(invalid, set()), -2)
        self.assertEqual(augmented_likelihood_score(invalid, set()), 0)
        self.assertAlmostEqual(augmented_likelihood_score(good, set()), (reward(good, set()) + 2) / 3)
        mismatch = score((row, 'CCC', None, 2))
        self.assertTrue(mismatch['valid'])
        self.assertFalse(mismatch['core_match'])
        self.assertEqual(reward(mismatch, set()), -2)
        grown_ring = score((row, 'c1ccc(-c2ccccc2)cc1', None, 3))
        self.assertTrue(grown_ring['core_match'])
        self.assertFalse(grown_ring['core_retention'])

    def test_diversity_equivalence(self):
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from worker import diversity_partial
        smiles = ['Cc1ccccc1', 'Nc1ccccc1', 'Oc1ccccc1']
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles]
        expected = sum(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(3) for j in range(i+1, 3)) / 3
        partials = [diversity_partial((smiles, i, 2)) for i in range(2)]
        self.assertAlmostEqual(sum(x[0] for x in partials) / sum(x[1] for x in partials), expected, places=12)


if __name__ == '__main__': unittest.main()
