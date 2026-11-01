"""Scheme A imports historical pairs; external checkpoints do not constrain training inputs."""
from pathlib import Path
from ..supplement_data import import_record,validation_partition,append_test
from ..splits import stratified_order
from ..io import freeze,atomic_json,read_json,read_rows,write_rows,file_sha,digest
from ..protocol import training_reference
from . import PROTOCOL
from .contracts import DATASETS


def _import_payload(args):
    return import_record(*args)


def policy_ids(rows, seed, count=29491):
    chosen=stratified_order(rows,seed)[:count]
    if len(chosen)!=count:raise ValueError('Insufficient A/B/C-supported training records')
    return [r['record_id'] for r in chosen]


def prepare(config, output, stage=1):
    from ..checkpoints import load
    from condmolgen.v4_data import build_v4_tokenizer
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    freeze(output/'import_contract.json',{'protocol':PROTOCOL,'config':config,'implementation_sha256':file_sha(__file__)})
    if (output/'prepared.json').exists():
        old=read_json(output/'prepared.json')
        for name,sha in old['files'].items():
            if file_sha(output/name)!=sha:raise ValueError('Prepared file drift')
        return append_test(config,output,old) if stage==2 and 'test.jsonl.gz' not in old['files'] else old
    splits={};dataset=config['dataset']
    for split in ('train','validation'):
        entry=config['sources'][split]
        if config.get('verify_source_hashes',True) and file_sha(entry['path'])!=entry['sha256']:raise ValueError('Historical input hash differs')
        raw=load(entry['path'])
        if not isinstance(raw,list) or len(raw)!=entry['records']:raise ValueError('Historical count/schema differs')
        workers=config.get('import_workers',1)
        if workers>1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                splits[split]=list(pool.map(_import_payload,((r,dataset,split,i) for i,r in enumerate(raw)),chunksize=256))
        else:splits[split]=[import_record(r,dataset,split,i) for i,r in enumerate(raw)]
        del raw
    train=splits['train']
    if len(train)!=DATASETS[dataset]:raise ValueError('Paper training size changed')
    tok=build_v4_tokenizer(train);length=config['max_length']
    for row in train:
        tokens=tok.encode_smiles(row['scaffold_smiles']+row['rooted_decoration_smiles'])
        if len(tokens)+3>length or tok.unk_token_id in tokens:raise ValueError('Unexpected A/B/C input coverage change')
    partition=validation_partition(train,splits['validation'],dataset)
    files={}
    for name,values in dict(train=train,**partition).items():
        filename=name+'.jsonl.gz';write_rows(output/filename,values);files[filename]=file_sha(output/filename)
    ref,rhash=training_reference(train)
    metadata={'reference.json':{'canonical_smiles':sorted(ref),'sha256':rhash},'tokenizer.json':tok.to_state(),
              'support.json':{v:[r['record_id'] for r in train] for v in ('A','B','C')}}
    for seed in (42,43,44):
        ids=policy_ids(train,seed);metadata['policy_ids_'+str(seed)+'.json']={'seed':seed,'ids':ids,'sha256':digest(ids)}
    for name,value in metadata.items():freeze(output/name,value);files[name]=file_sha(output/name)
    report={'protocol':PROTOCOL,'dataset':dataset,'files':files,'train_records':len(train),
            'counts':{k:len(v) for k,v in partition.items()},'external_training_intersection':False,'formal_test_read':False,'source_hashes_recomputed':config.get('verify_source_hashes',True)}
    atomic_json(output/'prepared.json',report)
    return append_test(config,output,report) if stage==2 else report
