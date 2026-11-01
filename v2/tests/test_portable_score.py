import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from re_atomformer.io import write_rows,atomic_json,read_json
from re_atomformer.scheme_a.events import event_id

class PortableScore(unittest.TestCase):
    def test_twenty_identical_valid_products(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);events=[]
            for i in range(20):
                e={'generation_protocol':'supplement_A_v1','dataset_id':'synthetic','split':'test',
                   'record_id':'synthetic:0','method':'fixture','model_variant':'fixture',
                   'checkpoint_sha256':'0'*64,'sampling_seed':142,'attempt_id':i,
                   'input_supported':True,'assembled_output':'Cc1ccccc1','failure_reason':None}
                e['generation_event_id']=event_id(e);events.append(e)
            write_rows(root/'events.jsonl.gz',events)
            write_rows(root/'inputs.jsonl.gz',[{'record_id':'synthetic:0','core_smiles':'c1ccccc1'}])
            atomic_json(root/'reference.json',{'canonical_smiles':['Cc1ccccc1']})
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'atomformer.py'),'score',
                '--events',str(root/'events.jsonl.gz'),'--inputs',str(root/'inputs.jsonl.gz'),
                '--reference',str(root/'reference.json'),'--output',str(root/'out')],check=True)
            result=read_json(root/'out/summary.json')
            self.assertEqual(result['validity'],1.)
            self.assertEqual(result['retained_yield'],1.)
            self.assertEqual(result['mean_unique_retained'],1.)
            self.assertEqual(result['retained_diversity_all_records'],0.)
            self.assertEqual(result['novelty_vs_benchmark_train'],0.)
