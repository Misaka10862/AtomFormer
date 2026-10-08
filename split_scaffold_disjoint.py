"""Create canonical molecule/scaffold-disjoint train/validation/test splits."""
import argparse, hashlib, json, random
from functools import lru_cache
from pathlib import Path
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

@lru_cache(maxsize=200000)
def canon(s):
    m = Chem.MolFromSmiles(s or "")
    return Chem.MolToSmiles(m, canonical=True) if m else None

@lru_cache(maxsize=200000)
def core(s):
    m = Chem.MolFromSmiles(s or "")
    if m is None: return None
    # Remove dummy atoms before canonicalizing the scaffold core.
    rw = Chem.RWMol(m)
    for a in list(rw.GetAtoms()):
        if a.GetAtomicNum() == 0: rw.RemoveAtom(a.GetIdx())
    m = rw.GetMol()
    Chem.SanitizeMol(m, catchErrors=True)
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc, canonical=True) if sc is not None else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-path', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args(); records=torch.load(a.input_path, map_location='cpu')
    groups={}; seen_mols=set()
    for r in records:
        cm=canon(r.get('canonical_smiles', r.get('source_smiles','')))
        if not cm or cm in seen_mols: continue
        seen_mols.add(cm); sc=core(r.get('scaffold_smiles','')) or ''
        rr=dict(r); rr['canonical_smiles']=cm; rr['_scaffold_core']=sc; groups.setdefault(sc,[]).append(rr)
    keys=list(groups); random.Random(a.seed).shuffle(keys)
    n=len(keys); cuts=(int(n*.8), int(n*.9)); buckets=[keys[:cuts[0]], keys[cuts[0]:cuts[1]], keys[cuts[1]:]]
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    names=['train','validation','test']; paths={}; manifest={'seed':a.seed,'input_path':str(Path(a.input_path).resolve()),'input_sha256':hashlib.sha256(Path(a.input_path).read_bytes()).hexdigest(),'num_input_records':len(records),'num_unique_records':len(seen_mols),'scaffold_groups':n,'splits':{}}
    for name, ks in zip(names,buckets):
        rows=[r for k in ks for r in groups[k]]
        for r in rows: r.pop('_scaffold_core',None)
        path=out/f'{name}.pt'; torch.save(rows,path); paths[name]=str(path)
        manifest['splits'][name]={'path':str(path.resolve()),'records':len(rows),'scaffolds':len(ks),'scaffold_cores':sorted(ks),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['scaffold_overlap_audit']={f'{x}_{y}':len(set(manifest['splits'][x]['scaffold_cores']) & set(manifest['splits'][y]['scaffold_cores'])) for x,y in [('train','validation'),('train','test'),('validation','test')]}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
