"""Formal eval/no-grad/stochastic sessions and native-versus-adapter differential checks."""
import hashlib
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import torch
from ..checkpoints import rng_state,restore_rng,seed_all
from ..io import digest


def state_sha256(network):
    h=hashlib.sha256()
    for name,tensor in sorted(network.state_dict().items()):
        t=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def assert_plain_output(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:raise RuntimeError('Evaluation output retains gradients')
    elif isinstance(value,dict):
        for item in value.values():assert_plain_output(item)
    elif isinstance(value,(list,tuple)):
        for item in value:assert_plain_output(item)


@contextmanager
def frozen_session(network):
    network.eval()
    for p in network.parameters():
        p.requires_grad_(False);p.grad=None
    before=state_sha256(network);handles=[]
    def guard(module,args):
        if module.training or torch.is_grad_enabled():raise RuntimeError('Native forward escaped eval/no_grad')
    def forbidden(*args,**kwargs):raise RuntimeError('Training/backward forbidden during formal inference')
    for module in network.modules():
        if not list(module.children()):handles.append(module.register_forward_pre_hook(guard))
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(patch.object(torch.Tensor,'backward',forbidden))
            stack.enter_context(patch.object(torch.autograd,'backward',forbidden))
            stack.enter_context(patch.object(torch.autograd,'grad',forbidden))
            for value in vars(torch.optim).values():
                if isinstance(value,type) and issubclass(value,torch.optim.Optimizer) and 'step' in value.__dict__:
                    stack.enter_context(patch.object(value,'step',forbidden))
            yield {'state_before_sha256':before,'model_mode':'eval','grad_enabled':False,'decoding':'stochastic'}
    finally:
        for handle in handles:handle.remove()
        if any(m.training for m in network.modules()):raise RuntimeError('Native code changed model training mode')
        if any(p.requires_grad for p in network.parameters()):raise RuntimeError('Native inference unfroze model parameters')
        if any(p.grad is not None for p in network.parameters()):raise RuntimeError('Inference populated gradients')
        if before!=state_sha256(network):raise RuntimeError('Frozen model persistent state changed')


@contextmanager
def cpu_cuda_allocations(device):
    """Pinned LibINVENT hardcodes .cuda for tensors; CPU shim changes allocation only."""
    if str(device).startswith('cpu'):
        with patch.object(torch.Tensor,'cuda',lambda value,*a,**kw:value.to('cpu')):yield
    else:yield


@contextmanager
def native_trace(network):
    trace={'draws':[],'linear_probes':[],'embedding_inputs':[],'attention_masks':[]};original=torch.multinomial;original_method=torch.Tensor.multinomial;handles=[]
    def multinomial(probabilities,*args,**kwargs):
        if torch.is_grad_enabled():raise RuntimeError('Native random sampler enabled gradients')
        result=original(probabilities,*args,**kwargs)
        trace['draws'].append({'probabilities':probabilities.detach().cpu().clone(),'actions':result.detach().cpu().clone()})
        return result
    def tensor_multinomial(probabilities,*args,**kwargs):
        if torch.is_grad_enabled():raise RuntimeError('Native random sampler enabled gradients')
        result=original_method(probabilities,*args,**kwargs)
        trace['draws'].append({'probabilities':probabilities.detach().cpu().clone(),'actions':result.detach().cpu().clone()})
        return result
    def probe(module,args,out):
        if len(trace['linear_probes'])<64 and isinstance(out,torch.Tensor):
            trace['linear_probes'].append({'shape':list(out.shape),'values':out.detach().flatten()[:256].cpu().clone()})
    def embedding_probe(module,args):
        if len(trace['embedding_inputs'])<64:
            trace['embedding_inputs'].append([x.detach().cpu().clone() for x in args if isinstance(x,torch.Tensor)])
    for module in network.modules():
        if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(probe))
        if isinstance(module,torch.nn.Embedding):handles.append(module.register_forward_pre_hook(embedding_probe))
    def attention_wrapper(fn):
        def call(*args,**kwargs):
            if len(trace['attention_masks'])<64:
                trace['attention_masks'].append({k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items() if k in ('attn_mask','key_padding_mask','is_causal')})
            return fn(*args,**kwargs)
        return call
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch,'multinomial',multinomial))
            stack.enter_context(patch.object(torch.Tensor,'multinomial',tensor_multinomial))
            for module in network.modules():
                if isinstance(module,torch.nn.MultiheadAttention):stack.enter_context(patch.object(module,'forward',attention_wrapper(module.forward)))
            yield trace
    finally:
        for handle in handles:handle.remove()


def _compare(a,b):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor):raise AssertionError('Native trace type differs')
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError('Native output fields differ')
        for k in a:_compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('Native output length differs')
        for x,y in zip(a,b):_compare(x,y)
    elif isinstance(a,float):
        import math
        if not math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-6):raise AssertionError('Native scalar differs')
    elif a!=b:raise AssertionError('Native output/encoding differs')


def check_native_equivalence(adapter,request,seed=142,batch_size=20):
    """Run the official callable directly, then the logged adapter under identical RNG."""
    state=rng_state()
    try:
        with frozen_session(adapter.network) as receipt:
            support=adapter.support(request)
            if not support['supported']:raise ValueError('Consistency request must be natively supported')
            input_check=adapter.check_native_input(request) if hasattr(adapter,'check_native_input') else {'passed':True,'native_input':'official tokenization used directly'}
            seed_all(seed)
            with native_trace(adapter.network) as direct_trace:
                direct=adapter.native_generate(request,batch_size)
            seed_all(seed)
            with native_trace(adapter.network) as adapter_trace:
                actual=adapter.generate(request,batch_size)
            _compare(direct,actual);_compare(direct_trace,adapter_trace)
            assert_plain_output(actual)
            if not direct_trace['draws']:raise AssertionError('No native stochastic sampling operation observed')
        return {'passed':True,**receipt,'native_input_check':input_check,'raw_outputs_sha256':digest(actual),'native_input_sha256':digest(support['native_input']),
                'sampling_calls':len(direct_trace['draws']),'linear_probe_count':len(direct_trace['linear_probes']),
                'embedding_input_probes':len(direct_trace['embedding_inputs']),'attention_mask_probes':len(direct_trace['attention_masks']),
                'raw_candidates':len(actual),'state_after_sha256':state_sha256(adapter.network),
                'atol':1e-6,'rtol':1e-5,'scope':'same backend native actions, probabilities, bounded linear probes, prompt/NLL/output'}
    finally:restore_rng(state)
