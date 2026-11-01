"""One fixed evaluator, independent of all model-native scoring and training rewards."""
import math
from collections import Counter
from statistics import mean
from . import SCORER_VERSION
from .events import uniform_events,validate_events
from ..io import digest


class Evaluator:
    def __init__(self, reference, enforce_version=True):
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import QED
        from rdkit.Contrib.SA_Score import sascorer
        if enforce_version and rdkit.__version__!='2022.09.5':raise RuntimeError('Scoring environment must be RDKit 2022.09.5')
        self.reference=set(reference);self.qed=QED.qed;self.sas=sascorer.calculateScore
        # Missing scores/resources or systematic setup failures must stop before processing attempts.
        for smi in ('CCO','c1ccccc1'):
            mol=Chem.MolFromSmiles(smi)
            if not all(math.isfinite(fn(mol)) for fn in (self.qed,self.sas)):
                raise RuntimeError('Scoring preflight returned nonfinite values')
        self.environment={'scorer_version':SCORER_VERSION,'rdkit':rdkit.__version__,
                          'novelty_reference_sha256':digest(sorted(self.reference))}

    def score(self, product, core):
        from rdkit import Chem
        from scaffold_metrics import scaffold_checks
        from ..protocol import canonical
        row={'canonical_smiles':None,'valid':False,'core_match':False,'core_retention':False,
             'qed':None,'sas':None,'novelty_vs_benchmark_train':None,'score_errors':{}}
        # A malformed condition is an engineering error, not a failed generated molecule.
        if not core or Chem.MolFromSmiles(core) is None:raise ValueError('Invalid scoring target')
        canon=canonical(product)
        if canon is None:
            row['failure_reason']='invalid_or_assembly_failure';return row
        mol=Chem.MolFromSmiles(canon);checks=scaffold_checks(canon,core)
        row.update(canonical_smiles=canon,valid=True,core_match=checks['core_match'],core_retention=checks['core_retention'],
                   novelty_vs_benchmark_train=float(canon not in self.reference),failure_reason=None)
        for name,fn in (('qed',self.qed),('sas',self.sas)):
            try:
                value=float(fn(mol))
                if not math.isfinite(value):raise ValueError('Nonfinite molecular property')
                row[name]=value
            except (ValueError,ArithmeticError) as error:
                row['score_errors'][name]=type(error).__name__
        return row

    def reward(self, product, condition):
        core=condition.get('core_smiles') or condition.get('scoring_target_core')
        row=self.score(product,core)
        gate=row['valid'] and row['core_match'] and row['core_retention'] and row['qed'] is not None and row['sas'] is not None
        if not gate and not row.get('failure_reason'):
            row['failure_reason']='core_match_failure' if not row['core_match'] else ('exact_core_failure' if not row['core_retention'] else 'molecular_property_failure')
        return dict(row,gate_passed=gate,novelty=row['novelty_vs_benchmark_train'] or 0.,
                    reward=row['qed']+.2*row['novelty_vs_benchmark_train']-.2*row['sas'] if gate else -2.)


def diversity(smiles):
    """Exact duplicate-preserving pair mean; multiplicities only avoid redundant fingerprint work."""
    from rdkit import Chem,DataStructs
    from rdkit.Chem import AllChem
    if len(smiles)<2:return 0.
    counts=Counter(smiles);names=sorted(counts)
    fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=2048) for s in names]
    values=[]
    for i,fp in enumerate(fps[:-1]):
        values.extend((1.-sim)*counts[names[i]]*counts[names[j]]
                      for j,sim in enumerate(DataStructs.BulkTanimotoSimilarity(fp,fps[i+1:]),i+1))
    return math.fsum(values)/(len(smiles)*(len(smiles)-1)/2)


def summarize(rows, requested_ids, report_ids=None, population='full_requests'):
    ids=list(requested_ids);support=validate_events(rows,ids)
    selected=ids if report_ids is None else list(report_ids)
    if len(selected)!=len(set(selected)) or not set(selected)<=set(ids):raise ValueError('Invalid reporting population')
    if not selected:return {'population':population,'records':0,'status':'EMPTY_POPULATION'}
    selected_set=set(selected);events=[r for r in rows if r['record_id'] in selected_set]
    groups={rid:[] for rid in selected}
    for r in events:groups[r['record_id']].append(r)
    valid=[r for r in events if r['valid']];retained=[r for r in valid if r['core_retention']]
    supported=sum(support[rid] for rid in selected);total=20*len(selected)
    def prop(rs,name):
        values=[r[name] for r in rs if r[name] is not None]
        return (mean(values) if values else None,len(values))
    qed,nq=prop(valid,'qed');sas,ns=prop(valid,'sas');qr,nqr=prop(retained,'qed');sr,nsr=prop(retained,'sas')
    per=[]
    for rid,group in groups.items():
        v=[r for r in group if r['valid']];r=[r for r in v if r['core_retention']]
        smi=[x['canonical_smiles'] for x in r]
        per.append({'record_id':rid,'score_core':group[0]['scoring_target_core'],'attempts':20,'input_supported':support[rid],
            'valid_outputs':len(v),'retained_outputs':len(r),'unique_retained':len(set(smi)),
            'retained_diversity_all_records':diversity(smi),
            'within_record_diversity':diversity([x['canonical_smiles'] for x in v]) if v else None,
            'unique_retained_diversity':diversity(sorted(set(smi))),
            'qed_sum':math.fsum(x['qed'] for x in v if x['qed'] is not None),'qed_n':sum(x['qed'] is not None for x in v),
            'sas_sum':math.fsum(x['sas'] for x in v if x['sas'] is not None),'sas_n':sum(x['sas'] is not None for x in v),
            'retained_qed_sum':math.fsum(x['qed'] for x in r if x['qed'] is not None),'retained_qed_n':sum(x['qed'] is not None for x in r),
            'retained_sas_sum':math.fsum(x['sas'] for x in r if x['sas'] is not None),'retained_sas_n':sum(x['sas'] is not None for x in r),
            'novel_count':sum(x['novelty_vs_benchmark_train'] for x in v),'core_match_count':sum(x['core_match'] for x in v)})
    sampled,manifest=uniform_events(events)
    return {'scorer_version':SCORER_VERSION,'population':population,'population_ids_sha256':digest(sorted(selected)),
        'records':len(selected),'requested_attempts':total,'supported_records':supported,'supported_attempts':20*supported,
        'denominators':{'qed':nq,'sas':ns,'novelty':len(valid),'core_match':len(valid),'core_retention':len(valid),'validity':20*supported,'full_request_slots':total},
        'coverage':supported/len(selected),'valid_count':len(valid),'validity':len(valid)/(20*supported) if supported else None,
        'valid_yield_all_requests':len(valid)/total,'qed':qed,'qed_n':nq,'sas':sas,'sas_n':ns,
        'novelty_vs_benchmark_train':mean(r['novelty_vs_benchmark_train'] for r in valid) if valid else None,
        'core_match':sum(r['core_match'] for r in valid)/len(valid) if valid else None,
        'core_retention':len(retained)/len(valid) if valid else None,
        'aggregate_diversity_uniform':diversity([r['canonical_smiles'] for r in sampled]),'diversity_sample':manifest,
        'retained_yield':len(retained)/total,'mean_unique_retained':mean(r['unique_retained'] for r in per),
        'retained_diversity_all_records':mean(r['retained_diversity_all_records'] for r in per),
        'retained_properties':{'qed':qr,'sas':sr,'qed_n':nqr,'sas_n':nsr},
        'mean_within_record_diversity':mean(r['within_record_diversity'] for r in per if r['valid_outputs']) if valid else None,
        'within_record_n':sum(bool(r['valid_outputs']) for r in per),
        'unique_retained_diversity_all_records':mean(r['unique_retained_diversity'] for r in per),
        'unsupported_slots':20*(len(selected)-supported),
        'failure_counts':dict(Counter(r.get('failure_reason') for r in events if r.get('failure_reason'))),'per_record':per}


def three_populations(method_rows, requested_ids):
    support={name:validate_events(rows,requested_ids) for name,rows in method_rows.items()}
    common=sorted(rid for rid in requested_ids if all(mask[rid] for mask in support.values()))
    return {'common_ids':common,'common_ids_sha256':digest(common),'methods':{
        name:{'full_requests':summarize(rows,requested_ids),
              'native_support':summarize(rows,requested_ids,[rid for rid in requested_ids if support[name][rid]],'native_support'),
              'common_support':summarize(rows,requested_ids,common,'common_support')}
        for name,rows in method_rows.items()}}
