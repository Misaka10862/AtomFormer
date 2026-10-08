"""Small synthetic tests for deterministic RL recovery and ordered rewards."""
import argparse
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
from condmolgen.runtime import rng_state, restore_rng

torch.set_num_threads(1)

class RecoveryTests(unittest.TestCase):
    def test_rng_roundtrip(self):
        import numpy as np
        import torch
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        state=rng_state();expected=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(state);actual=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]))


    def test_saved_sampler_matches_original_dataloader_and_resume(self):
        import torch
        from torch.utils.data import DataLoader,RandomSampler,BatchSampler,TensorDataset
        from condmolgen.rl_training import SavedBatches
        dataset=TensorDataset(torch.arange(17))
        torch.manual_seed(7)
        original=[b[0].tolist() for b in DataLoader(dataset,batch_size=4,shuffle=True)]
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(7)
        sampler=SavedBatches(sampler=BatchSampler(RandomSampler(dataset),4,False))
        observed=[b[0].tolist() for b in DataLoader(dataset,batch_sampler=sampler)]
        self.assertEqual(original,observed);self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        resumed=SavedBatches(batches=sampler.batches,start=2)
        self.assertEqual([b[0].tolist() for b in DataLoader(dataset,batch_sampler=resumed)],original[2:])


    def test_parallel_rewards_match_serial_in_order(self):
        from condmolgen.rl_training import OrderedRewards
        args=argparse.Namespace(invalid_reward=-2.,core_failure_reward=-2.,qed_weight=1.,novelty_weight=.2,sas_weight=.2)
        scaffolds=['*c1ccccc1','*n1cccc1','*c1ccccc1','*c1ccccc1']
        decorations=['C','C','not a smiles','c1ccccc1']
        a=OrderedRewards(1,set(),args,'rooted');b=OrderedRewards(2,set(),args,'rooted')
        try:
            x,rows=a.score(decorations,scaffolds);y,parallel=b.score(decorations,scaffolds)
            self.assertEqual(rows,parallel);self.assertEqual(x.tolist(),y.tolist())
        finally:a.close();b.close()


    def _assert_resume_matches(self, stop_epoch, stop_batch):
        import torch
        from types import SimpleNamespace
        from condmolgen.rl_training import run
        from condmolgen.runtime import atomic_torch as real_save
        class Config:
            def to_dict(self):return {'fixture':True}
        class Tokenizer:
            def to_state(self):return {'fixture':True}
        class Tiny(torch.nn.Module):
            def __init__(self,config=None):
                super().__init__();self.weight=torch.nn.Parameter(torch.randn(2));self.config=config or Config()
            def sample_decorations_with_logprobs(self,**kw):
                n=len(kw['scaffold_smiles']);coin=torch.rand(n)
                return {'decorations':['C' if x>.5 else 'O' for x in coin],
                        'token_logprobs':self.weight[0]*torch.rand(n,2),'action_mask':torch.ones(n,2)}
            def supervised_loss(self,**kw):return {'loss':self.weight.square().mean()+self.weight[1]*kw['z'].sum()*.001}
        class Dataset:
            def __init__(self,records,*a,**k):self.records=records
            def __len__(self):return len(self.records)
            def __getitem__(self,i):return i
        def collate(indices):
            n=len(indices)
            return {'z':torch.tensor(indices).float().reshape(n,1),'pos':torch.zeros(n,1,3),
                'scaffold_smiles':['*c1ccccc1']*n,'input_ids':torch.zeros(n,1,dtype=torch.long),
                'labels':torch.zeros(n,1,dtype=torch.long),'decoder_padding_mask':torch.zeros(n,1,dtype=torch.bool),
                'anchor_z':torch.zeros(n,dtype=torch.long)}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'input.pt').write_text('fixture');(root/'supervised.pt').write_text('fixture')
            args=argparse.Namespace(checkpoint=str(root/'supervised.pt'),train_path=str(root/'input.pt'),
                output_dir=str(root/'continuous'),seed=7,device='cpu',representation='rooted',max_train_samples=0,
                val_ratio=0.,max_length=32,batch_size=2,learning_rate=.001,weight_decay=.01,reward_workers=1,
                epochs=2,temperature=1.,reward_baseline='batch_mean',supervised_weight=.2,max_grad_norm=1.,
                num_workers=0,pin_memory=0,persistent_workers=0,prefetch_factor=2,
                invalid_reward=-2.,core_failure_reward=-2.,qed_weight=1.,novelty_weight=.2,sas_weight=.2,
                checkpoint_every=1,resume=False)
            with patch('condmolgen.checkpoint.load_checkpoint',side_effect=lambda *a,**k:(Tiny(),Tokenizer(),{})),\
                 patch('condmolgen.data.load_records',return_value=[{'canonical_smiles':'Cc1ccccc1'}]*7),\
                 patch('condmolgen.train_utils.split_records',side_effect=lambda r,*a:(r,[])),\
                 patch('condmolgen.decoration_data.DecorationDataset',Dataset),\
                 patch('condmolgen.decoration_data.DecorationBatchCollator',side_effect=lambda _:collate),\
                 patch('condmolgen.decoration_model.DecorationGenerator',Tiny):
                run(args)
                args.output_dir=str(root/'resumed')
                def interrupted(path,payload):
                    real_save(path,payload)
                    if Path(path).name=='resume.pt' and payload['epoch']==stop_epoch and payload['next_batch']==stop_batch:
                        raise InterruptedError('injected test interruption')
                with patch('condmolgen.rl_training.atomic_torch',side_effect=interrupted):
                    with self.assertRaises(InterruptedError):run(args)
                args.resume=True;run(args)
            a=torch.load(root/'continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(root/'resumed/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(len(a['extra']['history']), 2)
            self.assertEqual(len(b['extra']['history']), 2)
            self.assertTrue(torch.equal(a['model_state']['weight'],b['model_state']['weight']))
            self.assertEqual((root/'continuous/rl_samples.jsonl').read_text(),(root/'resumed/rl_samples.jsonl').read_text())
            x=torch.load(root/'continuous/resume.pt',map_location='cpu',weights_only=False);y=torch.load(root/'resumed/resume.pt',map_location='cpu',weights_only=False)
            self.assertTrue(torch.equal(x['rng']['torch'],y['rng']['torch']))
            for key in ('exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(x['optimizer']['state'][0][key],y['optimizer']['state'][0][key]))

    def test_mid_epoch_resume_preserves_model_optimizer_and_sampling(self):
        self._assert_resume_matches(1, 2)

    def test_epoch_boundary_resume_preserves_model_optimizer_and_sampling(self):
        self._assert_resume_matches(1, 4)

    def test_final_batch_resume_preserves_model_optimizer_and_sampling(self):
        self._assert_resume_matches(2, 4)
