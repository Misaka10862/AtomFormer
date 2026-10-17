"""Coordinate invariance and checkpoint architecture checks for the default model."""
import tempfile
from pathlib import Path
import unittest

import torch

from condmolgen.checkpoint import load_checkpoint, save_checkpoint
from condmolgen.decoration_model import DecorationGenerator
from condmolgen.model import AtomTypeEncoder, ConditionalGeneratorConfig
from condmolgen.tokenizer import CharTokenizer

torch.set_num_threads(1)


class No3DTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer(texts=['*CC', 'CO'])
        self.config = ConditionalGeneratorConfig(
            vocab_size=self.tokenizer.vocab_size, embedding_dim=16,
            encoder_heads=2, decoder_heads=2, encoder_layers=1, decoder_layers=1,
            encoder_ffn_dim=32, decoder_ffn_dim=32, block_size=32, dropout=0.)

    def test_coordinates_do_not_change_logits_gradients_or_samples(self):
        torch.manual_seed(7)
        model = DecorationGenerator(self.config).eval()
        self.assertIsInstance(model.encoder, AtomTypeEncoder)
        z = torch.tensor([[6, 8, 0], [6, 6, 6]])
        pos = torch.randn(2, 3, 3)
        ids = torch.tensor([self.tokenizer.encode_scaffold_target('*CC', 'CO')] * 2)
        baseline = None
        for coordinates in (pos, pos * 31 + 17, None):
            model.zero_grad(set_to_none=True)
            logits = model(z=z, pos=coordinates, input_ids=ids)['logits']
            logits.square().mean().backward()
            grads = {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}
            torch.manual_seed(13)
            _, samples = model.generate_decorations(z=z, pos=coordinates, tokenizer=self.tokenizer,
                scaffold_smiles=['*CC', '*CC'], max_length=20, greedy=False)
            if baseline is None:
                baseline = (logits.detach().clone(), grads, samples)
            else:
                self.assertTrue(torch.equal(logits, baseline[0]))
                self.assertEqual(grads.keys(), baseline[1].keys())
                for key in grads:
                    self.assertTrue(torch.equal(grads[key], baseline[1][key]), key)
                self.assertTrue(torch.equal(samples, baseline[2]))

    def test_explicit_encoder_checkpoints_and_pre_selector_geo(self):
        with tempfile.TemporaryDirectory() as folder:
            for encoder in ('atom', 'geo'):
                with self.subTest(encoder=encoder):
                    self.config.encoder_type = encoder
                    model = DecorationGenerator(self.config).eval()
                    path = Path(folder) / f'{encoder}.pt'
                    save_checkpoint(str(path), model, self.tokenizer, self.config)
                    restored, _, extra = load_checkpoint(str(path))
                    self.assertEqual(restored.config.encoder_type, encoder)
                    self.assertNotIn('checkpoint_load_info', extra)
                    for k, v in model.state_dict().items():
                        self.assertTrue(torch.equal(v, restored.state_dict()[k]), k)
                    if encoder == 'geo':
                        payload = torch.load(path, map_location='cpu')
                        payload['model_config'].pop('encoder_type')
                        torch.save(payload, path)
                        restored, _, extra = load_checkpoint(str(path))
                        self.assertEqual(restored.config.encoder_type, 'geo')
                        self.assertNotIn('checkpoint_load_info', extra)
                        for k, v in model.state_dict().items():
                            self.assertTrue(torch.equal(v, restored.state_dict()[k]), k)


if __name__ == '__main__':
    unittest.main()
