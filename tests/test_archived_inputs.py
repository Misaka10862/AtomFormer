import hashlib
import unittest

from preprocess import process_smiles
from study.restore_gdb13_inputs import restore_record


class ArchivedInputTests(unittest.TestCase):
    def test_reconstruction_reuses_coordinates_and_checks_all_fields(self):
        for smiles in ['CCc1ccccc1', 'CCOc1ccccc1']:
            reference = process_smiles(smiles)
            self.assertIsNotNone(reference)
            selector = {
                'source_smiles_sha256': hashlib.sha256(smiles.encode()).hexdigest(),
                'string_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in reference.items() if isinstance(v, str)},
                'numerical_fields': {k: v for k, v in reference.items() if not isinstance(v, str)},
            }
            self.assertEqual(restore_record(smiles, selector), reference)
            selector['source_smiles_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                restore_record(smiles, selector)


if __name__ == '__main__':
    unittest.main()
