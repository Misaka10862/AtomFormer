"""Small CPU integration checks for the documented command-line workflow."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import train
from condmolgen.checkpoint import load_checkpoint, save_checkpoint
from condmolgen.decoration_model import DecorationGenerator
from condmolgen.metrics import evaluate_generations
from condmolgen.model import ConditionalGeneratorConfig
from condmolgen.runtime import frozen_contract, source_signature
from condmolgen.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def command(self, *args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_entry_point_help(self):
        for name in ('preprocess.py', 'split_scaffold_disjoint.py', 'train.py',
                     'train_supervised.py', 'train_rl.py', 'evaluate.py'):
            with self.subTest(entry=name):
                self.assertIn('usage:', self.command(name, '--help').stdout)

    def test_wrapper_defaults_and_resume_forwarding(self):
        with patch.object(sys, 'argv', ['train.py', '--stage', 'rl', '--checkpoint', 'input.pt',
                                       '--train-path', 'train.pt', '--output-dir', 'run', '--resume',
                                       '--reward-workers', '2', '--checkpoint-every', '3']):
            args = train.parse_args()
        command = train.build_rl_command(args)
        self.assertEqual(Path(command[1]).name, 'train_rl.py')
        expected = {'--epochs': '1', '--batch-size': '64', '--learning-rate': '5e-06',
                    '--supervised-weight': '0.1', '--novelty-weight': '0.4', '--sas-weight': '0.05',
                    '--reward-workers': '2', '--checkpoint-every': '3'}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--resume', command)
        args.checkpoint = None
        with self.assertRaises(ValueError):
            train.build_rl_command(args)

    def test_checkpoint_roundtrip_preserves_weights_and_predictions(self):
        tokenizer = CharTokenizer(texts=['*c1ccccc1', 'CO'])
        config = ConditionalGeneratorConfig(vocab_size=tokenizer.vocab_size, embedding_dim=16,
                                            encoder_heads=2, decoder_heads=2, encoder_layers=1,
                                            decoder_layers=1, encoder_ffn_dim=32, decoder_ffn_dim=32,
                                            block_size=32, dropout=0.0)
        torch.manual_seed(7)
        model = DecorationGenerator(config).eval()
        z = torch.tensor([[6, 6]])
        pos = torch.tensor([[[0., 0., 0.], [1.4, 0., 0.]]])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'checkpoint.pt')
            save_checkpoint(path, model, tokenizer, config)
            base, loaded_tokenizer, extra = load_checkpoint(path)
            restored = DecorationGenerator(base.config).eval()
            restored.load_state_dict(base.state_dict(), strict=True)
            self.assertEqual(tokenizer.to_state(), loaded_tokenizer.to_state())
            self.assertNotIn('checkpoint_load_info', extra)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            kwargs = dict(z=z, pos=pos, tokenizer=tokenizer, scaffold_smiles=['*c1ccccc1'],
                          max_length=24, greedy=True)
            a, ids = model.generate_decorations(**kwargs)
            b, restored_ids = restored.generate_decorations(**kwargs)
            self.assertEqual(a, b)
            self.assertTrue(torch.equal(ids, restored_ids))

    def test_metric_denominators_and_length_validation(self):
        result = evaluate_generations(['Cn1cccc1', '__INVALID__', 'c1ccc(-c2ccccc2)cc1'],
                                      ['c1cc[nH]c1', 'c1cc[nH]c1', 'c1ccccc1'], [])
        self.assertEqual(result['num_generated'], 3)
        self.assertEqual(result['num_valid'], 2)
        self.assertEqual(result['validity'], 2 / 3)
        self.assertEqual(result['scaffold_core_metric_n'], 2)
        self.assertEqual(result['substructure_match_rate'], 1.)
        self.assertEqual(result['scaffold_retention_rate'], .5)
        with self.assertRaises(ValueError):
            evaluate_generations(['CC'], [], [])

    def test_contract_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.json'
            frozen_contract(path, {'learning_rate': .001})
            frozen_contract(path, {'learning_rate': .001})
            with self.assertRaises(RuntimeError):
                frozen_contract(path, {'learning_rate': .002})
        self.assertEqual(len(source_signature()), 64)

    def test_documented_workflow_on_synthetic_molecules(self):
        molecules = ['Cc1ccccc1', 'Cc1ccncc1', 'Cc1ccoc1', 'Cc1ccsc1', 'Cn1cccc1',
                     'CC1CCCCC1', 'CC1CCCC1', 'CC1CCC1', 'CC1CC1', 'CC1CCOCC1',
                     'CC1CCNCC1', 'CC1CO1']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'molecules.smi'
            source.write_text('\n'.join(molecules) + '\n')
            processed = root / 'processed.pt'
            splits = root / 'splits'
            supervised, rl = root / 'supervised', root / 'rl'
            self.command('preprocess.py', '--input-smi', source, '--output-path', processed)
            self.command('split_scaffold_disjoint.py', '--input-path', processed, '--output-dir', splits)
            manifest = json.loads((splits / 'split_manifest.json').read_text())
            self.assertTrue(all(n == 0 for n in manifest['scaffold_overlap_audit'].values()))
            self.assertTrue(all(s['records'] > 0 for s in manifest['splits'].values()))
            self.command('train.py', '--stage', 'supervised', '--train-path', splits / 'train.pt',
                         '--val-path', splits / 'validation.pt', '--output-dir', supervised,
                         '--device', 'cpu', '--epochs', '1', '--max-train-samples', '4',
                         '--max-val-samples', '2', '--batch-size', '2', '--embedding-dim', '16',
                         '--encoder-heads', '2', '--decoder-heads', '2', '--encoder-layers', '1',
                         '--decoder-layers', '1', '--encoder-ffn-dim', '32', '--decoder-ffn-dim', '32',
                         '--block-size', '64', '--dropout', '0', '--use-amp', '0')
            rl_args = ['train.py', '--stage', 'rl', '--checkpoint', supervised / 'last.pt',
                       '--train-path', splits / 'train.pt', '--output-dir', rl, '--device', 'cpu',
                       '--epochs', '1', '--max-train-samples', '4', '--val-ratio', '0',
                       '--batch-size', '2', '--max-length', '64', '--checkpoint-every', '1',
                       '--reward-workers', '2']
            self.command(*rl_args)
            before = torch.load(rl / 'last.pt', map_location='cpu')
            self.command(*rl_args, '--resume')
            after = torch.load(rl / 'last.pt', map_location='cpu')
            for name in before['model_state']:
                self.assertTrue(torch.equal(before['model_state'][name], after['model_state'][name]))
            summary, samples = root / 'summary.json', root / 'samples.jsonl'
            eval_args = ['evaluate.py', '--checkpoint', rl / 'last.pt', '--processed-path',
                         splits / 'test.pt', '--novelty-reference-path', splits / 'train.pt',
                         '--output-json', summary, '--output-samples-jsonl', samples,
                         '--greedy', '0', '--samples-per-scaffold', '2', '--max-samples', '1',
                         '--max-length', '64', '--require-full-coverage', '1', '--device', 'cpu']
            self.command(*eval_args)
            payload = json.loads(summary.read_text())
            self.assertEqual(payload['num_samples'], 2)
            self.assertEqual(len(samples.read_text().splitlines()), 2)
            self.assertTrue((root / 'run_manifest.json').exists())
            rejected = subprocess.run([sys.executable, *map(str, eval_args)], cwd=ROOT,
                                      capture_output=True, text=True, timeout=90)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Output files already exist', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
