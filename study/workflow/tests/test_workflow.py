import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'code'))
sys.path.insert(0, str(HERE))
import analyze
import run_stage
from multidata_campaign.evaluation import paired_bootstrap

class WorkflowTests(unittest.TestCase):
    def test_complete_parameter_matrix(self):
        rows = json.loads((HERE / 'run_parameters.json').read_text())['runs']
        expected = {(d, c, r) for d in ('gdb13', 'chembl37', 'moses') for c in run_stage.CONFIGS for r in (1, 2, 3)}
        self.assertEqual({(r['dataset'], r['configuration'], r['run']) for r in rows}, expected)
        for r in rows:
            if r['configuration'] in ('drugex', 'libinvent', 'promptsmiles'):
                self.assertTrue(r['supervised_parameter_blocks'])
                self.assertTrue(r['rl_parameter_blocks'])
            if r['configuration'] == 'plain_balanced_rl':
                self.assertEqual(r['rl_arguments']['representation'], 'plain')
                self.assertEqual(r['rl_arguments']['supervised_weight'], .2)
            if r['configuration'] == 'pure_rl':
                self.assertEqual(r['rl_arguments']['supervised_weight'], 0)

    def test_distinct_scoring_environment(self):
        x = json.loads((HERE / 'environment_versions.json').read_text())
        self.assertEqual(x['runtime_snapshots']['atomformer']['rdkit'], '2022.09.5')
        self.assertEqual(x['runtime_snapshots']['baseline']['rdkit'], '2026.03.5')
        self.assertEqual(x['common_final_scoring_rdkit'], '2022.09.5')

    def test_paired_interval_direction_and_grouping(self):
        l = {'0': {'core': 'A', 'statistics': {'validity': [20, 20]}},
             '1': {'core': 'B', 'statistics': {'validity': [0, 20]}}}
        r = {'0': {'core': 'A', 'statistics': {'validity': [0, 20]}},
             '1': {'core': 'B', 'statistics': {'validity': [0, 20]}}}
        a = paired_bootstrap([l]*3, [r]*3, replicates=200, cluster=True)
        b = paired_bootstrap([r]*3, [l]*3, replicates=200, cluster=True)
        self.assertEqual(a['metrics']['validity']['point'], .5)
        self.assertEqual(b['metrics']['validity']['point'], -.5)
        self.assertEqual(a['metrics']['validity']['ci95'], [0., 1.])
        self.assertFalse(a['training_seed_uncertainty'])
        with self.assertRaises(ValueError):
            paired_bootstrap([l]*3, [dict(r, extra=r['0'])]*3, replicates=2, cluster=True)

    def test_drugex_adapter_preserves_input_identity(self):
        sys.path.insert(0, str(HERE / 'code/baselines/drugex'))
        from drugex_baseline_utils import prepare_record
        record, status = prepare_record({'canonical_smiles':'CCc1ccccc1', 'scaffold_smiles':'c1ccccc1[*:1]'}, 'test', 7)
        self.assertEqual(status, 'ok')
        self.assertEqual(record['row_index'], 7)
        self.assertEqual(record['scaffold_core_smiles'], 'c1ccccc1')
        rejected, reason = prepare_record({'canonical_smiles':'CCO', 'scaffold_smiles':'c1ccccc1[*:1]'}, 'test', 8)
        self.assertIsNone(rejected)
        self.assertEqual(reason, 'core_not_substructure')

    def make_samples(self, supported=True):
        return [dict(record_id=0, sample_id=i, generated='CCO', valid=True,
                     assembly_success=True, core_match=True, core_retention=True,
                     decoration_valid=True, input_supported=supported, qed=.5, sas=2.) for i in range(20)]

    def save_samples(self, rows):
        directory = Path(tempfile.mkdtemp(prefix='atomformer-unit-'))
        path = directory / 'samples.jsonl.gz'
        with gzip.open(path, 'wt') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
        return path

    def test_native_and_requested_budget_are_distinct(self):
        rows = self.make_samples() + [dict(r, record_id=1, input_supported=False, valid=False,
                generated=None, core_match=False, core_retention=False, qed=None, sas=None) for r in self.make_samples()]
        got = analyze.summarize_native(self.save_samples(rows), {'CCO'}, 3, 1, 20)
        self.assertEqual(got['metrics']['validity'], 1.)
        self.assertAlmostEqual(got['metrics']['retained_yield'], 1/3)
        self.assertEqual(got['metrics']['novelty'], 0.)
        self.assertEqual(got['metrics']['num_generated'], 20)

    def test_component_records_keep_failed_attempts(self):
        rows = self.make_samples()
        rows[0].update(valid=False, generated=None, core_match=False, core_retention=False, qed=None, sas=None)
        inputs = [{'scaffold_smiles': 'c1ccccc1[*:1]'}]
        got = analyze.component_records(self.save_samples(rows), inputs, set())['0']
        self.assertEqual(got['statistics']['validity'], [19, 20])
        self.assertEqual(got['statistics']['qed_mean'], [9.5, 19])
        self.assertEqual(got['statistics']['unique_retained_per_intended_record'], [1, 1])
        self.assertEqual(got['core'], 'c1ccccc1')
        with self.assertRaises(ValueError):
            analyze.component_records(self.save_samples(rows[:-1]), inputs, set())

if __name__ == '__main__': unittest.main()
