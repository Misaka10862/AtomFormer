"""Import hash-bound historical pairs without repartitioning or conformer generation."""
from collections import defaultdict
from pathlib import Path
from .io import digest, file_sha, freeze, atomic_json, write_rows, read_json, read_rows
from .protocol import canonical, training_reference
from .splits import key, stratified_order
from .supplement import PROTOCOL, DATASETS


def import_record(old, dataset, split, index):
    from rdkit import Chem
    from .data import molecule_features, Condition
    from scaffold_metrics import scaffold_core_from_scaffold
    scaffold=old['scaffold_smiles'];mol=Chem.MolFromSmiles(scaffold)
    if mol is None:raise ValueError('Historical scaffold cannot be parsed')
    dummy=[a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(dummy)!=1 or dummy[0].GetDegree()!=1:raise ValueError('Historical attachment differs')
    anchor=dummy[0].GetNeighbors()[0].GetIdx()
    for atom in mol.GetAtoms():atom.SetIntProp('_historical_index',atom.GetIdx())
    dummy[0].SetAtomicNum(1);dummy[0].SetIsotope(0);dummy[0].SetAtomMapNum(0)
    mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
    anchor_index=next(a.GetIdx() for a in mol.GetAtoms() if a.GetIntProp('_historical_index')==anchor)
    features=molecule_features(mol)
    legacy_z=old['cond_z'].tolist() if hasattr(old['cond_z'],'tolist') else list(old['cond_z'])
    if sorted(legacy_z)!=sorted(features['z']):raise ValueError('Historical atom composition changed')
    source=canonical(old.get('canonical_smiles') or old['source_smiles'])
    if not source:raise ValueError('Invalid historical source')
    row={'record_id':dataset+':'+split+':'+str(index), 'historical_record_id':old.get('stable_id',old.get('record_id',index)),
        'source_index':index,'dataset_id':dataset,'split':split,'canonical_smiles':source,
        'source_smiles':old.get('source_smiles',source),'scaffold_smiles':scaffold,
        'rooted_decoration_smiles':old['rooted_decoration_smiles'],'anchor_z':int(old['anchor_z']),
        'core_smiles':scaffold_core_from_scaffold(scaffold),'score_core':scaffold_core_from_scaffold(scaffold),
        'scaffold_group':old.get('scaffold_core',scaffold),'anchor_index':anchor_index,
        'legacy_atom_z':legacy_z,'geometry_source':'none','geometry_status':'not_requested',**features}
    for name in ('single_decoration_smiles','decoration_smiles','plain_decoration_smiles'):
        if name in old:row[name]=old[name]
    row['decoration_size']=Chem.MolFromSmiles(row['rooted_decoration_smiles']).GetNumHeavyAtoms()
    Condition.from_record(row)
    return row


def validation_partition(train, validation, dataset, seed=1729, quick_count=2048):
    train_mols={r['canonical_smiles'] for r in train};train_cores={r['score_core'] for r in train}
    if any(r['canonical_smiles'] in train_mols for r in validation):
        raise ValueError('Unexpected molecular overlap; user review required')
    overlap=[r for r in validation if r['score_core'] in train_cores]
    if overlap and dataset!='gdb13':raise ValueError('Unexpected scoring-core overlap; user review required')
    groups=defaultdict(list)
    for row in validation:
        if row['score_core'] not in train_cores:groups[row['score_core']].append(row)
    ordered=sorted(groups,key=lambda c:key(seed,c))
    target=round(.8*sum(map(len,groups.values())));selected=[];checked=[]
    for core in ordered:
        (selected if len(selected)<target else checked).extend(groups[core])
    if len(selected)<quick_count or not checked:raise ValueError('Insufficient independent validation coverage; user review required')
    selected=[dict(r,validation_role='v_select') for r in selected]
    checked=[dict(r,validation_role='v_check') for r in checked]
    quick=sorted(selected,key=lambda r:key(seed+1,r['record_id']))[:quick_count]
    quick=[dict(r,validation_role='v_select_quick') for r in quick]
    return {'v_select':selected,'v_check':checked,'v_select_quick':quick,'excluded_validation':overlap}


def common_policy_ids(train, supported, seed, count=29491):
    expected={'A','B','C','libinvent','promptsmiles'}
    if set(supported)!=expected:raise ValueError('All five training support ledgers are required')
    common=set.intersection(*(set(supported[m]) for m in expected))
    chosen=stratified_order([r for r in train if r['record_id'] in common],seed)[:count]
    if len(chosen)!=count:raise ValueError('Insufficient common RL inputs; user review required')
    return [r['record_id'] for r in chosen]


def build_support(train, config, tokenizer):
    from .external import SequenceBaseline,adapt_training_rows
    from .data import Condition
    limit=config['max_length'];ids=[]
    for r in train:
        tokens=tokenizer.encode_smiles(r['scaffold_smiles']+r['rooted_decoration_smiles'])
        if len(tokens)+3>limit or tokenizer.unk_token_id in tokens:
            raise ValueError('A/B/C historical training coverage changed; user review required')
        ids.append(r['record_id'])
    supports={m:list(ids) for m in ('A','B','C')}
    for name in ('libinvent','promptsmiles'):
        adapter=SequenceBaseline(name,train,{'max_length':limit},config['upstreams'],config['upstream_lock'],'cpu')
        native=adapt_training_rows(train)
        supports[name]=[r['record_id'] for r,n in zip(train,native)
                        if adapter.supports(Condition.from_record(r)) and adapter.backend.encodable(n)]
    return supports


def prepare(config, output, stage=1):
    from .checkpoints import load
    from condmolgen.v4_data import build_v4_tokenizer
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=config['dataset'];sources=config['sources']
    contract={'protocol':PROTOCOL,'config':config,'implementation_sha256':file_sha(__file__)}
    freeze(output/'import_contract.json',contract)
    if (output/'prepared.json').exists():
        existing=read_json(output/'prepared.json')
        for name,sha in existing['files'].items():
            if file_sha(output/name)!=sha:raise ValueError('Prepared data hash drift')
        if stage==1 or 'test.jsonl.gz' in existing['files']:return existing
        return append_test(config,output,existing)
    splits={}
    for split in ('train','validation'):
        source=sources[split]
        if file_sha(source['path'])!=source['sha256']:raise ValueError('Historical source digest differs: '+split)
        old=load(source['path'])
        if not isinstance(old,list) or len(old)!=source['records']:raise ValueError('Historical record count/schema differs')
        splits[split]=[import_record(r,dataset,split,i) for i,r in enumerate(old)]
        del old
    if len(splits['train'])!=DATASETS[dataset]:raise ValueError('Paper training population differs')
    partition=validation_partition(splits['train'],splits['validation'],dataset)
    # Exactly the historical character tokenizer algorithm; no heldout characters or extra grammar.
    tok=build_v4_tokenizer(splits['train'])
    support=build_support(splits['train'],config,tok)
    files={}
    def rows(name,values):
        target=output/(name+'.jsonl.gz');write_rows(target,values);files[target.name]=file_sha(target)
    rows('train',splits['train'])
    for name,values in partition.items():rows(name,values)
    ref,rhash=training_reference(splits['train'])
    for name,value in [('tokenizer.json',tok.to_state()),('reference.json',{'canonical_smiles':sorted(ref),'sha256':rhash}),
                       ('support.json',support)]:
        freeze(output/name,value);files[name]=file_sha(output/name)
    for seed in (42,43,44):
        ids=common_policy_ids(splits['train'],support,seed)
        name='policy_ids_'+str(seed)+'.json';freeze(output/name,{'ids':ids,'seed':seed,'sha256':digest(ids)})
        files[name]=file_sha(output/name)
    report={'protocol':PROTOCOL,'dataset':dataset,'files':files,'source_hashes':{k:v['sha256'] for k,v in sources.items()},
        'counts':{k:len(v) for k,v in partition.items()},'train_records':len(splits['train']),
        'support_counts':{k:len(v) for k,v in support.items()},'formal_test_read':stage==2}
    atomic_json(output/'prepared.json',report)
    return append_test(config,output,report) if stage==2 else report


def append_test(config, output, existing):
    """Add the test split without rewriting any training/selection gzip or frozen manifest."""
    from .checkpoints import load
    source=config['sources']['test'];dataset=config['dataset']
    if file_sha(source['path'])!=source['sha256']:raise ValueError('Historical test source digest differs')
    old=load(source['path'])
    if not isinstance(old,list) or len(old)!=source['records']:raise ValueError('Test count differs')
    test=[import_record(r,dataset,'test',i) for i,r in enumerate(old)]
    training=list(read_rows(output/'train.jsonl.gz'))
    traincores={r['score_core'] for r in training};trainmols={r['canonical_smiles'] for r in training}
    if any(r['canonical_smiles'] in trainmols for r in test):raise ValueError('Unexpected test molecular overlap')
    if dataset!='gdb13' and any(r['score_core'] in traincores for r in test):raise ValueError('Unexpected test core overlap')
    write_rows(output/'test.jsonl.gz',test)
    freeze(output/'test_nonoverlap_ids.json',[r['record_id'] for r in test if r['score_core'] not in traincores])
    report=dict(existing,files=dict(existing['files']),formal_test_read=True)
    for name in ('test.jsonl.gz','test_nonoverlap_ids.json'):report['files'][name]=file_sha(output/name)
    atomic_json(output/'prepared.json',report)
    return report
