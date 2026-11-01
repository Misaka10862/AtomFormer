import unittest
from re_atomformer.scheme_a.gdb13_repair import restore_record

class SourceRestorationTests(unittest.TestCase):
    def row(self):
        return dict(source_smiles='CC1CC1',canonical_smiles='CC1CC1',scaffold_smiles='C[1*:1]',cond_smiles='C[1*:1]',
                    cond_z=[6,6,6],cond_pos=[[0,0,0]]*3,rooted_decoration_smiles='C',anchor_z=6)
    def test_restores_parent_ring_and_preserves_all_other_fields(self):
        row=self.row();fixed,change=restore_record(row)
        self.assertEqual(change['unique_candidates'],1)
        for k in row:
            if k not in ('scaffold_smiles','cond_smiles'):self.assertEqual(row[k],fixed[k])
        self.assertEqual(row['scaffold_smiles'],'C[1*:1]')
        again,change=restore_record(fixed);self.assertEqual(again,fixed);self.assertIsNone(change)
    def test_wrong_target_or_composition_cannot_be_invented(self):
        row=self.row();row['rooted_decoration_smiles']='CC'
        with self.assertRaises(ValueError):restore_record(row)
        row=self.row();row['cond_z']=[7,7,7]
        with self.assertRaises(ValueError):restore_record(row)
    def test_original_dummy_target_proves_sulfur_identity_without_label_change(self):
        row=self.row();row.update(source_smiles='NS(=O)(=O)C1CC1',canonical_smiles='NS(=O)(=O)C1CC1',
            scaffold_smiles='C1CC1[1*:1]',cond_smiles='C1CC1[1*:1]',
            rooted_decoration_smiles='[SH](N)(=O)=O',single_decoration_smiles='NS(=O)(=O)[1*:1]',anchor_z=16)
        fixed,note=restore_record(row)
        self.assertEqual(fixed,row)
        self.assertTrue(note['known_rooted_assembly_failure'])
        self.assertTrue(note['upstream_dummy_assembly_matches'])

    def test_ambiguous_parent_identity_stops(self):
        row=self.row();row['source_smiles']='CCC'
        with self.assertRaises(ValueError):restore_record(row)

if __name__=='__main__':unittest.main()
