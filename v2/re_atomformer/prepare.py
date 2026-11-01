"""Streaming source conversion and split preparation, invoked only on the server."""
import csv
import gzip
from collections import Counter
from pathlib import Path
import sqlite3
import json

from .data import extract_record,Condition
from .io import atomic_json,file_sha,write_rows,read_rows,digest,freeze
from .splits import grouped_split,audit_splits
from .protocol import training_reference,training_thresholds


def source_smiles(path,smiles_field='canonical_smiles'):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    if '.csv' in path.name or '.tsv' in path.name or 'chemreps' in path.name:
        with opener(path,'rt') as stream:
            reader=csv.DictReader(stream,delimiter='\t' if '.tsv' in path.name or 'chemreps' in path.name else ',')
            if smiles_field not in reader.fieldnames:
                raise ValueError('Missing source SMILES field '+smiles_field)
            for i,row in enumerate(reader):yield str(i),row[smiles_field]
    else:
        with opener(path,'rt') as stream:
            for i,line in enumerate(stream):
                if line.strip():yield str(i),line.split()[0]


def normalize_source(smiles,dataset):
    from rdkit import Chem
    if dataset!='chembl37':return smiles
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:return smiles
    frags=[m for m in Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True) if any(a.GetAtomicNum()==6 for a in m.GetAtoms())]
    if not frags:raise ValueError('no_organic_component')
    return Chem.MolToSmiles(sorted(frags,key=lambda m:(-m.GetNumHeavyAtoms(),Chem.MolToSmiles(m)))[0])


def prepare(config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sources=config['sources']
    identities=[]
    for source in sources:
        actual=file_sha(source['path'])
        if source.get('sha256') and source['sha256']!=actual:
            raise ValueError('Source hash mismatch')
        identities.append({'path':source['path'],'sha256':actual,'source_split':source.get('source_split','corpus')})
    implementation={name:file_sha(Path(__file__).parent/name) for name in ('prepare.py','data.py','splits.py','protocol.py')}
    freeze(out/'source_contract.json',{'config':config,'inputs':identities,'implementation':implementation})
    if (out/'manifest.json').exists():
        existing=json.loads((out/'manifest.json').read_text())
        if any(file_sha(out/(split+'.jsonl.gz'))!=sha for split,sha in existing['split_files'].items()):
            raise ValueError('Prepared dataset content changed')
        return existing
    db=sqlite3.connect(str(out/'records.sqlite'))
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, row TEXT NOT NULL)')
    counts=Counter(); seen_source_ids=set()
    for source in sources:
        if source.get('format')=='pocket_manifest':
            iterator=pocket_records(source['path'])
        else:
            iterator=((sid,smiles) for sid,smiles in source_smiles(source['path'],source.get('smiles_field','canonical_smiles')))
        for index,item in enumerate(iterator):
            try:
                if source.get('format')=='pocket_manifest':
                    row=item
                else:
                    sid,smiles=item
                    row=extract_record(normalize_source(smiles,config['dataset']),config['dataset'],config.get('conformer_seed',1729),config.get('geometry','parent'),config.get('conformer_count',1))
                    row['source_record_id']=sid
                if len(row['scaffold_smiles'])+len(row['rooted_decoration_smiles'])+3>config.get('max_length',256):
                    raise ValueError('sequence_length')
                row['source_split']=source.get('source_split',row.get('source_split','corpus'))
                counts['accepted_2d']+=1
                counts['geometry_'+row.get('geometry_status','unknown')]+=1
                if row['record_id'] in seen_source_ids:
                    counts['duplicate_molecule_condition']+=1
                seen_source_ids.add(row['record_id'])
                previous=db.execute('SELECT row FROM records WHERE id=?',(row['record_id'],)).fetchone()
                if previous:
                    prior=json.loads(previous[0])
                    if prior.get('source_split')!=row['source_split'] and ('test' in (prior.get('source_split'),row['source_split'])):
                        row['source_split']='test'
                        db.execute('UPDATE records SET row=? WHERE id=?',(json.dumps(row),row['record_id']))
                else:
                    db.execute('INSERT INTO records VALUES (?,?)',(row['record_id'],json.dumps(row)))
            except (ValueError,RuntimeError) as error:
                counts['rejected:'+str(error).split(':')[0]]+=1
            if (index+1)%1000==0:
                db.commit();atomic_json(out/'prepare_progress.json',dict(counts))
        db.commit()
    rows=[json.loads(v[0]) for v in db.execute('SELECT row FROM records ORDER BY id')]
    db.close()
    split,quarantine=grouped_split(rows,config.get('split_seed',1729))
    stats=audit_splits(split)
    if any(stats.get(s,0)==0 for s in ('train','validation','test')):
        raise ValueError('A split is empty; no automatic relaxing of grouping constraints')
    for s in ('train','validation','test'):
        write_rows(out/(s+'.jsonl.gz'),(r for r in split if r['split']==s))
    write_rows(out/'quarantine.jsonl.gz',quarantine)
    train=[r for r in split if r['split']=='train']
    reference,rhash=training_reference(train)
    atomic_json(out/'reference.json',{'canonical_smiles':sorted(reference),'sha256':rhash,'source':'complete_train_split'})
    atomic_json(out/'thresholds.json',training_thresholds(train))
    report={'counts':dict(counts),'splits':stats,'quarantined':len(quarantine),'unique_molecules':len({r['canonical_smiles'] for r in split}),
            'split_files':{s:file_sha(out/(s+'.jsonl.gz')) for s in ('train','validation','test')},'inputs':identities,
            'geometry_not_required_for_2d':True,'schema_version':1}
    atomic_json(out/'manifest.json',report)
    return report


def pocket_records(manifest):
    """Read source SDF/PDB identities; tensor-only exports cannot prove protein splits.

    Each row supplies full ligand SDF, scaffold SDF, scaffold-to-ligand map,
    scaffold anchor, pocket PDB, protein_group and official source_split.
    Atom mappings and coordinates are verified, never inferred from array order.
    """
    import numpy as np
    from rdkit import Chem
    from .data import molecule_features
    from condmolgen.v2_assembly import strip_single_attachment_dummy_with_root,assemble_single_attachment_rooted
    from .protocol import canonical
    for raw in read_rows(manifest):
        required=('ligand_sdf','scaffold_sdf','scaffold_to_ligand','anchor_index','pocket_pdb','protein_group','pocket_id')
        if any(not raw.get(k) and raw.get(k)!=0 for k in required):
            raise ValueError('Pocket manifest lacks auditable identities/mapping')
        mol=next(iter(Chem.SDMolSupplier(raw['ligand_sdf'],removeHs=True)))
        core=next(iter(Chem.SDMolSupplier(raw['scaffold_sdf'],removeHs=True)))
        if mol is None or core is None:raise ValueError('Invalid ligand/scaffold SDF')
        mapping=raw['scaffold_to_ligand'];anchor=raw['anchor_index']
        if len(mapping)!=core.GetNumAtoms() or len(set(mapping))!=len(mapping):raise ValueError('Invalid pocket atom map')
        if any(mol.GetAtomWithIdx(j).GetAtomicNum()!=core.GetAtomWithIdx(i).GetAtomicNum() for i,j in enumerate(mapping)):
            raise ValueError('Pocket atom map elements differ')
        pos=mol.GetConformer().GetPositions();core_pos=core.GetConformer().GetPositions()
        if not np.allclose(pos[mapping],core_pos,atol=1e-3):raise ValueError('Pocket scaffold coordinate map differs')
        inside=set(mapping)
        boundaries=[b for b in mol.GetBonds() if (b.GetBeginAtomIdx() in inside)!=(b.GetEndAtomIdx() in inside)]
        if len(boundaries)!=1 or boundaries[0].GetBondType()!=Chem.BondType.SINGLE:raise ValueError('Not a single R-group')
        edge=boundaries[0]
        if mapping[anchor] not in (edge.GetBeginAtomIdx(),edge.GetEndAtomIdx()):raise ValueError('Anchor differs from boundary')
        new_anchor=edge.GetOtherAtomIdx(mapping[anchor])
        fragmented=Chem.FragmentOnBonds(mol,[edge.GetIdx()],dummyLabels=[(1,1)])
        maps=[];frags=Chem.GetMolFrags(fragmented,asMols=True,fragsMolAtomMapping=maps)
        si=next(i for i,m in enumerate(maps) if mapping[anchor] in m)
        scaf=Chem.MolToSmiles(frags[si]);deco=Chem.MolToSmiles(frags[1-si])
        rooted=strip_single_attachment_dummy_with_root(deco)
        if rooted is None or canonical(assemble_single_attachment_rooted(scaf,rooted[0]))!=canonical(Chem.MolToSmiles(mol)):
            raise ValueError('Pocket stereo roundtrip failed')
        mapped=Chem.Mol(core)
        for i,a in enumerate(mapped.GetAtoms()):a.SetAtomMapNum(i+1)
        pocket=Chem.MolFromPDBFile(raw['pocket_pdb'],sanitize=False,removeHs=True)
        if pocket is None:raise ValueError('Invalid pocket PDB')
        outside=[i for i in range(mol.GetNumAtoms()) if i not in inside]
        outside.remove(new_anchor);outside.insert(0,new_anchor)
        order=mapping+outside;full=molecule_features(mol)
        core_smiles=Chem.MolToSmiles(core)
        row={'schema_version':1,'dataset_id':'crossdock_single','record_id':digest({'source':raw,'molecule':Chem.MolToSmiles(mol)}),
             'canonical_smiles':canonical(Chem.MolToSmiles(mol)),'scaffold_smiles':scaf,'core_smiles':core_smiles,
             'mapped_core_smiles':Chem.MolToSmiles(mapped),'scaffold_group':Chem.MolToSmiles(core,isomericSmiles=False),'score_core':core_smiles,
             'anchor_index':anchor,'anchor_z':rooted[1],'rooted_decoration_smiles':rooted[0],'decoration_size':len(outside),
             'pos':core_pos.tolist(),'conformers':[core_pos.tolist()],'geometry_source':'pocket_complex','geometry_status':'success',
             'pocket_z':[a.GetAtomicNum() for a in pocket.GetAtoms()], 'pocket_pos':pocket.GetConformer().GetPositions().tolist(),
             'pocket_pdb':raw['pocket_pdb'],'pocket_id':raw['pocket_id'],'protein_group':raw['protein_group'], 'source_split':raw.get('source_split','corpus'),
             'target_3d':{'z':[full['z'][i] for i in outside],'pos':pos[outside].tolist(),'bonds':[[full['bonds'][i][j] for j in order] for i in order]},
             **molecule_features(core)}
        Condition.from_record(row)
        yield row
