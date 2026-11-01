"""Explicit RL trajectory weighting, baselines and diagnostic gradient geometry."""
import torch
from .protocol import augmented_score


def advantages(rewards,groups=None,baseline='batch_mean'):
    if baseline=='none':
        return rewards
    if baseline=='batch_mean':
        return rewards-rewards.mean()
    if baseline!='leave_one_out' or groups is None or len(groups)!=len(rewards):
        raise ValueError('Invalid advantage baseline')
    result=torch.empty_like(rewards)
    for group in sorted(set(groups)):
        ids=[i for i,g in enumerate(groups) if g==group]
        if len(ids)<2:
            raise ValueError('Leave-one-out needs at least two samples per condition')
        selected=rewards[ids]
        result[ids]=selected-(selected.sum()-selected)/(len(ids)-1)
    return result


def policy_loss(logprobs,mask,rewards,groups=None,baseline='batch_mean',normalization='length_mean',fixed_scale=32.):
    trajectory=(logprobs*mask).sum(1)
    if normalization=='length_mean':
        trajectory=trajectory/mask.sum(1).clamp(min=1)
    elif normalization=='fixed_scale' and fixed_scale>0:
        trajectory=trajectory/fixed_scale
    else:
        raise ValueError('Invalid trajectory normalization')
    adv=advantages(rewards,groups,baseline).detach()
    return -(trajectory*adv).mean()


def augmented_likelihood_loss(agent_logprob,prior_logprob,rewards,sigma=120.):
    scores=torch.tensor([augmented_score(float(v)) for v in rewards.detach()],device=rewards.device)
    target=prior_logprob.detach()+sigma*scores
    return ((agent_logprob-target)**2).mean()


def gradient_diagnostics(rl_loss,supervised_loss,parameters,weight):
    params=[p for p in parameters if p.requires_grad]
    gr=torch.autograd.grad(rl_loss,params,retain_graph=True,allow_unused=True)
    gs=torch.autograd.grad(supervised_loss,params,retain_graph=True,allow_unused=True)
    zero=rl_loss.new_tensor(0.)
    nr=sum((g.detach().square().sum() for g in gr if g is not None),zero).sqrt()
    ns=sum(((weight*g.detach()).square().sum() for g in gs if g is not None),zero).sqrt()
    dot=sum(((a.detach()*b.detach()*weight).sum() for a,b in zip(gr,gs) if a is not None and b is not None),zero)
    return {'rl_gradient_norm':float(nr),'weighted_supervised_gradient_norm':float(ns),
            'gradient_ratio':float(ns/(nr+1e-12)),'gradient_cosine':float(dot/(nr*ns+1e-12))}


def contrastive_geometry_loss(graph_embeddings,geometry_embeddings,temperature=.1,group_ids=None):
    """Aligned 2D/3D training views; same-scaffold negatives are excluded."""
    if len(graph_embeddings)<2 or temperature<=0:
        raise ValueError('Contrastive training requires multiple examples')
    a=torch.nn.functional.normalize(graph_embeddings,dim=-1)
    b=torch.nn.functional.normalize(geometry_embeddings,dim=-1)
    logits=a@b.T/temperature
    labels=torch.arange(len(a),device=a.device)
    if group_ids is not None:
        same=torch.tensor([[x==y for y in group_ids] for x in group_ids],device=a.device)
        same.fill_diagonal_(False)
        logits=logits.masked_fill(same,-torch.inf)
    return (torch.nn.functional.cross_entropy(logits,labels)+torch.nn.functional.cross_entropy(logits.T,labels))/2
