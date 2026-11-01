#!/usr/bin/env python3
"""Rebuild AtomFormer GDB13 inputs from the licensed upstream/v1 reconstruction."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--legacy-inputs',required=True,help='Directory with train.pt, validation.pt, test.pt reconstructed using study/restore_gdb13_inputs.py')
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output)
if out.exists() and any(out.iterdir()):p.error('Use an empty output directory')
out.mkdir(parents=True,exist_ok=True)
from re_atomformer.checkpoints import load
from re_atomformer.supplement_data import import_record,validation_partition
from re_atomformer.scheme_a.gdb13_repair import restore_record
from re_atomformer.scheme_a.data import policy_ids
from re_atomformer.protocol import training_reference
from re_atomformer.io import write_rows,atomic_json,digest
from condmolgen.v4_data import build_v4_tokenizer
splits={}
for split in ('train','validation','test'):
    raw=load(Path(a.legacy_inputs)/(split+'.pt'))
    if split!='train':raw=[restore_record(r)[0] for r in raw]
    splits[split]=[import_record(r,'gdb13',split,i) for i,r in enumerate(raw)]
partition=validation_partition(splits['train'],splits['validation'],'gdb13')
for name,rows in dict(train=splits['train'],test=splits['test'],**partition).items():write_rows(out/(name+'.jsonl.gz'),rows)
ref,rhash=training_reference(splits['train']);atomic_json(out/'reference.json',{'canonical_smiles':sorted(ref),'sha256':rhash})
atomic_json(out/'tokenizer.json',build_v4_tokenizer(splits['train']).to_state())
for seed in (42,43,44):
    ids=policy_ids(splits['train'],seed);atomic_json(out/('policy_ids_'+str(seed)+'.json'),{'seed':seed,'ids':ids,'sha256':digest(ids)})
train_cores={r['score_core'] for r in splits['train']}
ids=[r['record_id'] for r in splits['test'] if r['score_core'] not in train_cores]
atomic_json(out/'restoration_counts.json',{'train':len(splits['train']),'test':len(splits['test']),**{k:len(v) for k,v in partition.items()},'test_nonoverlap':len(ids)})
atomic_json(out/'nonoverlap_record_ids.json',ids)
print(json.dumps({'output':str(out),'test_nonoverlap':len(ids)}))
