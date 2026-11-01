"""Frozen official checkpoints with their native vocabularies and raw generation APIs."""
import inspect
import sys
from pathlib import Path
import torch
from .conditions import atomformer_condition
from .inference import cpu_cuda_allocations,assert_plain_output
from ..checkpoints import load,rng_state,restore_rng
from ..io import digest,read_json,file_sha


class OfficialCheckpointBlocked(RuntimeError):pass


def object_state(value):
    """Canonical semantic tokenizer/vocabulary state, without pickle addresses."""
    from collections.abc import Mapping
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,Mapping):
        return [[object_state(k),object_state(v)] for k,v in sorted(value.items(),key=lambda kv:(type(kv[0]).__name__,str(kv[0])))]
    if isinstance(value,(list,tuple)):return [object_state(v) for v in value]
    if isinstance(value,set):return sorted((object_state(v) for v in value),key=str)
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'pattern') and hasattr(value,'flags'):return {'regex':value.pattern,'flags':value.flags}
    if hasattr(value,'__dict__'):return {'class':type(value).__module__+'.'+type(value).__name__,'fields':object_state(vars(value))}
    raise TypeError('Unserializable tokenizer component: '+type(value).__name__)


def activate_sources(name,directory,lock):
    directory=Path(directory).resolve();receipt=read_json(directory/'upstream_source.json')
    if receipt['commit']!=lock[name]['commit']:raise OfficialCheckpointBlocked('Pinned upstream commit differs')
    for entry in receipt['files']:
        if file_sha(directory/entry['path'])!=entry['sha256']:raise OfficialCheckpointBlocked('Upstream file changed: '+entry['path'])
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    if name=='libinvent':
        # Upstream models/__init__ eagerly imports training/scoring frontends. The
        # frozen loader uses the unchanged model modules through a namespace only.
        import types,importlib.machinery
        package_path=str(directory/'models')
        if 'models' in sys.modules:
            if list(getattr(sys.modules['models'],'__path__',[]))!=[package_path]:
                raise OfficialCheckpointBlocked('Conflicting models package; use a fresh worker')
        else:
            package=types.ModuleType('models');package.__path__=[package_path]
            package.__spec__=importlib.machinery.ModuleSpec('models',loader=None,is_package=True)
            sys.modules['models']=package
    return directory


def support_result(supported,native,reason=None):
    return {'supported':bool(supported),'native_input':native,'reason':reason}


class FrozenAdapter:
    def __init__(self):self._supports={}

    def generate(self,request,count=20):
        if not isinstance(count,int) or not 1<=count<=20:raise ValueError('Invalid raw generation count')
        # Small CPU diagnostics may use fewer trajectories; the formal ledger always requires 20.
        if not self.support(request)['supported']:raise ValueError('Unsupported input must be ledgered before sampling')
        result=self.native_generate(request,count)
        if len(result)!=count:raise RuntimeError('Native filtering/overgeneration: repair interface before evaluation')
        assert_plain_output(result)
        return result


class AtomFormerAdapter(FrozenAdapter):
    def __init__(self,model,tokenizer,max_length):
        super().__init__();self.network=model;self.tokenizer=tokenizer;self.max_length=max_length
    def support(self,request):
        native={'scaffold_prefix':request.scaffold_smiles_with_attachment}
        tokens=self.tokenizer.encode_smiles(request.scaffold_smiles_with_attachment)
        if self.tokenizer.unk_token_id in tokens:return support_result(False,native,'input_vocabulary')
        if len(tokens)+2>=self.max_length:return support_result(False,native,'prefix_length')
        if self.network.encoder_kind=='smiles' and len(tokens)>self.network.config.scaffold_block_size:
            return support_result(False,native,'encoder_length')
        condition=atomformer_condition(request)
        if max(condition.z)>=self.network.config.max_z:return support_result(False,native,'input_element')
        return support_result(True,dict(native,scaffold_tokens=tokens,z=condition.z))
    def native_generate(self,request,count):
        from ..data import collate_conditions
        from condmolgen.v2_assembly import assemble_single_attachment_rooted
        condition=atomformer_condition(request)
        batch=collate_conditions([condition]*count,next(self.network.parameters()).device)
        out=self.network.sample(batch,[condition.scaffold_smiles]*count,self.tokenizer,self.max_length,temperature=1.,with_grad=False)
        result=[]
        for i,decoration in enumerate(out['decorations']):
            assembled=assemble_single_attachment_rooted(condition.scaffold_smiles,decoration) if decoration else None
            result.append({'raw_generated_output':decoration,'assembled_output':assembled,
                'failure_reason':None if assembled else 'assembly_or_eos_failure','native_trajectory':out['generated'][i].detach().cpu().tolist(),
                'complete_candidate_attempts':1})
        return result


class LibInventOfficial(FrozenAdapter):
    def __init__(self,checkpoint,device,settings):
        super().__init__();self.device=device;self.settings=settings
        from models.model import DecoratorModel
        from models.decorator import Decorator
        from running_modes.enums import GenerativeModelRegimeEnum
        from reinvent_chemistry import Conversions
        from reinvent_chemistry.library_design import BondMaker,AttachmentPoints
        saved=load(checkpoint,device)
        network=Decorator(**saved['decorator']['params']);network.load_state_dict(saved['decorator']['state'],strict=True);network.to(device)
        self.model=DecoratorModel(decorator=network,no_cuda=str(device).startswith('cpu'),mode=GenerativeModelRegimeEnum().INFERENCE,**saved['model'])
        self.network=self.model.network;self.vocab=self.model.vocabulary
        self.conversions=Conversions();self.bond_maker=BondMaker();self.points=AttachmentPoints()
        self.tokenizer_sha256=digest(object_state(self.vocab))
        self.loaded_configuration={'decorator_params':saved['decorator']['params'],'max_sequence_length':self.model.max_sequence_length,
                                   'randomize':False,'sample_uniquely':False,'cpu_allocation_shim':str(device).startswith('cpu')}
    def support(self,request):
        from baselines.scaffold_bench.common import normalize_dummy
        from scaffold_metrics import scaffold_core_from_scaffold
        # RDKit emits bare *; the frozen official vocabulary uses the equivalent [*] token.
        scaffold=self.points.remove_attachment_point_numbers(normalize_dummy(request.scaffold_smiles_with_attachment).replace('*','[*]'))
        native={'scaffold':scaffold}
        try:
            encoded=self.vocab.encode_scaffold(scaffold)
            if encoded is None:return support_result(False,native,'input_vocabulary')
            decoded=self.vocab.decode_scaffold(encoded)
            if scaffold_core_from_scaffold(decoded)!=request.scoring_target_core:
                return support_result(False,native,'native_core_roundtrip')
            return support_result(True,dict(native,encoded_scaffold=encoded.tolist()))
        except (KeyError,ValueError,IndexError):return support_result(False,native,'input_encoding')
    def native_generate(self,request,count):
        from models.dataset import Dataset
        native=self.support(request)['native_input']
        data=Dataset([native['scaffold']]*count,self.vocab.scaffold_vocabulary,self.vocab.scaffold_tokenizer)
        with cpu_cuda_allocations(self.device):
            batch=Dataset.collate_fn([data[i] for i in range(count)])
            # pack_padded_sequence requires CPU int64 lengths, even for CUDA data.
            batch=(batch[0].to(self.device),batch[1].cpu())
            samples=list(self.model.sample_decorations(*batch))
        result=[]
        for scaffold,decoration,nll in samples:
            labelled=self.points.add_attachment_point_numbers(scaffold,canonicalize=False)
            mol=self.bond_maker.join_scaffolds_and_decorations(labelled,decoration)
            product=self.conversions.mol_to_smiles(mol,isomericSmiles=False,canonical=False) if mol else None
            result.append({'raw_generated_output':decoration,'assembled_output':product,'native_scaffold':scaffold,
                           'native_nll':float(nll),'complete_candidate_attempts':1,
                           'failure_reason':None if product else 'native_assembly_failure'})
        return result


class PromptSmilesOfficial(FrozenAdapter):
    def __init__(self,checkpoint,device,settings):
        super().__init__();self.device=device;self.settings=settings
        from smilesrnn.rnn import Model
        self.model=Model.load_from_file(str(checkpoint),sampling_mode=True,device=torch.device(device))
        self.network=self.model.network
        self.tokenizer_sha256=digest(object_state([self.model.vocabulary,self.model.tokenizer]))
        self.loaded_configuration={'max_sequence_length':self.model.max_sequence_length,'network_params':self.network.get_params(),**settings}
    def wrapper(self,request,count):
        from promptsmiles import ScaffoldDecorator
        from baselines.scaffold_bench.common import normalize_dummy
        return ScaffoldDecorator(scaffold=normalize_dummy(request.scaffold_smiles_with_attachment),batch_size=count,
            sample_fn=self.model._pSMILES_sample,evaluate_fn=self.model._pSMILES_evaluate,
            sample_fn_kwargs={'temperature':1.},batch_prompts=True,optimize_prompts=True,shuffle=True,
            random_seed=self.settings.get('prompt_random_seed',123),return_all=False)
    def support(self,request):
        if request.identity() in self._supports:return self._supports[request.identity()]
        from scaffold_metrics import scaffold_core_from_scaffold
        from baselines.scaffold_bench.common import normalize_dummy
        state=rng_state();native={'scaffold':normalize_dummy(request.scaffold_smiles_with_attachment)}
        try:
            with torch.no_grad():
                self.network.eval();wrapper=self.wrapper(request,20)
            variants=wrapper.scaffolds[0]['variants'];encoded=[]
            for v in variants:
                if scaffold_core_from_scaffold(v.orig_smiles)!=request.scoring_target_core:
                    return support_result(False,native,'native_core_roundtrip')
                tokens=self.model.tokenizer.tokenize(v.strip_smiles.replace('[SH]','S'),with_begin_and_end=False)
                values=self.model.vocabulary.encode(tokens)
                if len(values)>=self.model.max_sequence_length:return support_result(False,native,'native_prompt_length')
                encoded.append(values.tolist())
            result=support_result(True,dict(native,prompt_variants=[dict(v._asdict(),nll=float(v.nll) if v.nll is not None else None) for v in variants],encoded_prompts=encoded))
        except (KeyError,ValueError,IndexError):result=support_result(False,native,'input_encoding')
        finally:restore_rng(state)
        self._supports[request.identity()]=result;return result
    def native_generate(self,request,count):
        with torch.no_grad():
            wrapper=self.wrapper(request,count)
            products=wrapper.sample(batch_size=count,return_all=False)
        variants=[dict(v._asdict(),nll=float(v.nll) if v.nll is not None else None) for v in wrapper.scaffolds[0]['variants']]
        return [{'raw_generated_output':product,'assembled_output':product,'prompt_variants':variants,
                 'complete_candidate_attempts':1,'failure_reason':None} for product in products]


class DrugExOfficial(FrozenAdapter):
    def __init__(self,checkpoint,vocabulary,device,settings):
        super().__init__();self.device=torch.device(device)
        from drugex.data.corpus.vocabulary import VocGraph
        from drugex.training.generators import GraphTransformer
        self.vocab=VocGraph.fromFile(str(vocabulary))
        state=load(checkpoint,device)
        if not isinstance(state,dict) or 'emb_word.weight' not in state:
            raise OfficialCheckpointBlocked('Official DrugEx state format needs verified loader adaptation')
        import re
        layers={int(re.search(r'attn.blocks.(\d+)\.',k).group(1)) for k in state if re.search(r'attn.blocks.(\d+)\.',k)}
        args={'d_emb':state['emb_word.weight'].shape[1],'d_model':state['rnn.weight_hh'].shape[1],
              'd_inner':state['attn.blocks.0.pffn.w_1.weight'].shape[0],'n_layer':max(layers)+1,'n_head':settings['n_head']}
        self.network=GraphTransformer(self.vocab,device=self.device,use_gpus=(0,) if self.device.type=='cuda' else (-1,),**args)
        self.network.load_state_dict(state,strict=True)
        self.network.optim=None  # Native constructor creates an optimizer; frozen inference never uses it.
        self.tokenizer_sha256=digest(object_state(self.vocab))
        self.loaded_configuration={'model_args':args,'max_len':self.vocab.max_len,'n_frags':self.vocab.n_frags,
                                   'legal_action_masks':'native unchanged','optimizer_disabled':True}
    def encode(self,request):
        # Use the same condition-only dummy-template pipeline as native loaderFromFrags.
        from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
        from drugex.data.fragments import GraphFragmentEncoder
        pairs=dummyMolsFromFragments()(request.scoring_target_core)
        if not pairs or len(pairs)!=1 or not pairs[0][1]:raise ValueError('Native dummy-template conversion failed')
        fragment,dummy=pairs[0];encoder=GraphFragmentEncoder(self.vocab)
        tokens,_=encoder.encodeMol(dummy);encoded=encoder.encodeFrag(dummy,tokens,fragment)
        if encoded is None:raise ValueError('Native fragment encoding failed')
        tensor=torch.as_tensor(encoded,dtype=torch.long,device=self.device).reshape(1,self.vocab.max_len,5)
        return tensor,dummy
    def support(self,request):
        from baselines.scaffold_bench.common import same_molecule
        native={'dummy_free_core':request.scoring_target_core}
        try:
            encoded,dummy=self.encode(request)
            fragments,products=self.vocab.decode(encoded.detach().cpu().numpy())
            if not same_molecule(fragments[0],request.scoring_target_core) or not same_molecule(products[0],dummy):
                return support_result(False,native,'native_graph_roundtrip')
            return support_result(True,dict(native,native_dummy_template=dummy,encoded_graph=encoded[0].detach().cpu().tolist()))
        except (KeyError,ValueError,IndexError):return support_result(False,native,'input_encoding')
    def check_native_input(self,request,count=2):
        loader=self.network.loaderFromFrags([request.scoring_target_core],batch_size=count,n_proc=1)
        direct=next(iter(loader)).to(self.device)
        encoded,_=self.encode(request)
        expected=encoded.repeat(count,1,1)
        torch.testing.assert_close(direct,expected,rtol=0,atol=0)
        return {'passed':True,'native_loader':'GraphTransformer.loaderFromFrags','condition_only_dummy_template':True}
    def native_generate(self,request,count):
        from torch.utils.data import DataLoader
        encoded,_=self.encode(request)
        tensor=encoded.repeat(count,1,1)
        loader=DataLoader(tensor,batch_size=count,shuffle=False)
        smiles,fragments=self.network.sample(loader)
        return [{'raw_generated_output':s,'assembled_output':s,'native_fragment':f,'complete_candidate_attempts':1,
                 'failure_reason':None} for s,f in zip(smiles,fragments)]


def load_official(entry,asset,source_root,lock,device='cpu'):
    if asset.get('status') not in ('DOWNLOADED_AWAITING_LOAD','CPU_VERIFIED','READY'):
        raise OfficialCheckpointBlocked(asset.get('reason','Official checkpoint unavailable'))
    for name in entry['upstream_names']:activate_sources(name,Path(source_root)/name,lock)
    method=entry['method']
    if method=='drugex':
        packages=[f for f in asset['files'] if f['path'].endswith('.pkg')]
        vocabularies=[f for f in asset['files'] if f['path'].endswith('.vocab')]
        if len(packages)!=1 or len(vocabularies)!=1:raise OfficialCheckpointBlocked('Ambiguous official DrugEx package/vocabulary')
        for f in packages+vocabularies:
            if file_sha(f['path'])!=f['sha256']:raise OfficialCheckpointBlocked('Official artifact digest changed')
        adapter=DrugExOfficial(packages[0]['path'],vocabularies[0]['path'],device,entry['native_decoding_config'])
        adapter.checkpoint_sha256=packages[0]['sha256']
    else:
        if file_sha(asset['path'])!=asset['checkpoint_sha256']:raise OfficialCheckpointBlocked('Official weight digest changed')
        factory={'libinvent':LibInventOfficial,'promptsmiles':PromptSmilesOfficial}[method]
        adapter=factory(asset['path'],device,entry['native_decoding_config']);adapter.checkpoint_sha256=asset['checkpoint_sha256']
    for key in ('checkpoint_sha256','tokenizer_sha256'):
        if entry.get(key) and getattr(adapter,key)!=entry[key]:raise OfficialCheckpointBlocked('Frozen registry mismatch: '+key)
    adapter.network.eval()
    for parameter in adapter.network.parameters():parameter.requires_grad_(False)
    return adapter
