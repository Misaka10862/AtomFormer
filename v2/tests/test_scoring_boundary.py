"""CPU tests are server-only. Synthetic boundary-H and normal-path cases."""
import sys,unittest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import scaffold_metrics as legacy
from re_atomformer.scheme_a.kekule_repair import wrap_scaffold_checks,capped_framework

class KekuleRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform!='linux':raise RuntimeError('Run CPU tests on the approved server')
        cls.fixed=staticmethod(wrap_scaffold_checks(legacy.scaffold_checks))
    def test_synthetic_cut_carbanion_restores_H(self):
        product='C[c-]1c(=N)on[n+]1-c1ccccc1';core='N=c1[cH-][n+](-c2ccccc2)no1'
        with self.assertRaises(Chem.rdchem.KekulizeException):legacy.scaffold_checks(product,core)
        result=self.fixed(product,core)
        self.assertTrue(result['core_match']);self.assertTrue(result['core_retention'])
        self.assertEqual(result,legacy.scaffold_checks(core,core))
    def test_normal_outputs_exact(self):
        for product in (None,'invalid','CCO','Cc1ccccc1','c1ccncc1','N=c1[cH-][n+](-c2ccccc2)no1'):
            for core in ('c1ccccc1','c1ccncc1','CCO'):
                self.assertEqual(self.fixed(product,core),legacy.scaffold_checks(product,core))
    def test_no_target_guided_repair_or_parent_mutation(self):
        mol=Chem.MolFromSmiles('C[c-]1c(=N)on[n+]1-c1ccccc1');before=Chem.MolToSmiles(mol)
        f=capped_framework(mol,MurckoScaffold.GetScaffoldForMol)
        self.assertEqual(Chem.MolToSmiles(mol),before)
        self.assertEqual(Chem.MolToSmiles(f),'N=c1[cH-][n+](-c2ccccc2)no1')
        self.assertFalse(self.fixed(before,'c1ccccc1')['core_retention'])
    def test_no_unsupported_repairs(self):
        with self.assertRaises(ValueError):capped_framework(Chem.MolFromSmiles('Cc1ccccc1'),MurckoScaffold.GetScaffoldForMol)
    def test_atom_order_invariance_and_target_cache_unchanged(self):
        core='N=c1[cH-][n+](-c2ccccc2)no1';m=Chem.MolFromSmiles('C[c-]1c(=N)on[n+]1-c1ccccc1')
        target_before=Chem.MolToSmiles(legacy._target(core))
        expected=self.fixed(Chem.MolToSmiles(m),core)
        for shift in range(m.GetNumAtoms()):
            order=list(range(shift,m.GetNumAtoms()))+list(range(shift))
            product=Chem.MolToSmiles(Chem.RenumberAtoms(m,order),canonical=False)
            self.assertEqual(self.fixed(product,core),expected)
        self.assertEqual(Chem.MolToSmiles(legacy._target(core)),target_before)

if __name__=='__main__':unittest.main()
