import json
from pathlib import Path
from rerun_campaign.runtime import atomic_torch, atomic_json, frozen_contract, rng_state, restore_rng, sha256
from .training import durable_epochs

class ScheduledState:

    def __init__(self, optim):
        self.optim = optim
        self.state = optim._optimizer.state

    def state_dict(self):
        return {'optimizer': self.optim._optimizer.state_dict(), 'n_steps': self.optim.n_steps}

    def load_state_dict(self, state):
        self.optim._optimizer.load_state_dict(state['optimizer'])
        self.optim.n_steps = state['n_steps']
        self.state = self.optim._optimizer.state

    def zero_grad(self, *args, **kw):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

def model_for(data, device='cuda'):
    from drugex.training.generators import GraphTransformer
    from drugex_baseline_utils import load_vocab
    args = {'d_emb': 512, 'd_model': 512, 'd_inner': 1024, 'n_head': 8, 'n_layer': 12, 'pad_idx': 0}
    m = GraphTransformer(load_vocab(data / 'vocab.json'), device=__import__('torch').device(device), use_gpus=(0,) if device == 'cuda' else (-1,), **args)
    return (m, args)

def supervised(plan, unit, data, profile=False):
    import torch
    from condmolgen.train_utils import seed_everything
    seed_everything(unit['seed'])
    model, args = model_for(data)
    train = torch.load(data / 'train.pt', map_location='cpu')
    val = torch.load(data / 'validation.pt', map_location='cpu')
    if profile:
        train = train[:1024]
        val = train[:100]
    root = Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'supervised'
    optim = ScheduledState(model.optim)
    contract = {'unit': unit, 'policy': plan['policy_sha256'], 'profile': profile, 'inputs': {n: sha256(data / f'{n}.pt') for n in ('train', 'validation')}, 'objective': 'native graph NLL'}

    def loss(batch):
        return sum((-x.mean() for x in model(batch.to(model.device), is_train=True)))

    def step(indices):
        optim.zero_grad()
        value = loss(train[indices])
        value.backward()
        optim.step()
        return float(value.detach())

    def validate():
        torch.empty((), dtype=torch.int64).random_()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(val), 64):
                total += float(loss(val[offset:offset + 64])) * len(val[offset:offset + 64])
        return total / len(val)

    def save(path, epoch, history):
        atomic_torch(path, {'model_state': model.state_dict(), 'model_args': args, 'vocab': json.loads((data / 'vocab.json').read_text()), 'epoch': epoch, 'seed': unit['seed'], 'contract': contract, 'history': history, 'optimizer_steps': model.optim.n_steps})
    result = durable_epochs(model, optim, len(train), step, validate, save, root, contract, 12, 32, max_updates=50 if profile else None)
    atomic_json(root / 'run_manifest.json', {'contract': contract, 'result': result})
    return root / ('last.pt' if profile else 'best.pt')

def reinforcement(plan, unit, data, checkpoint, profile=False):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, RandomSampler, BatchSampler
    from drugex.training.environment import DrugExEnvironment
    from drugex.training.explorers import FragGraphExplorer
    from drugex.training.rewards import ParetoCrowdingDistance
    from drugex.training.scorers.properties import Property
    from drugex.training.scorers.modifiers import ClippedScore
    from drugex.training.monitors import NullMonitor
    from condmolgen.train_utils import seed_everything
    from rerun_campaign.af_training import SavedBatches
    seed_everything(unit['seed'])
    agent, args = model_for(data)
    prior, _ = model_for(data)
    checkpoint_payload = torch.load(checkpoint, map_location='cuda')
    agent.load_state_dict(checkpoint_payload['model_state'])
    prior.load_state_dict(checkpoint_payload['model_state'])
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    env = DrugExEnvironment(scorers=[Property('QED', modifier=ClippedScore(lower_x=0.0, upper_x=1.0)), Property('SA', modifier=ClippedScore(lower_x=10.0, upper_x=1.0))], thresholds=[0.5, 0.5], reward_scheme=ParetoCrowdingDistance())
    explorer = FragGraphExplorer(agent, env, mutate=prior, batch_size=64, epsilon=0.1, beta=0.0, n_samples=1024 if profile else 32768, device=agent.device, use_gpus=(0,))
    explorer.monitor = NullMonitor()
    optim = ScheduledState(explorer.optim)
    net = torch.nn.DataParallel(explorer, device_ids=explorer.gpus)
    train = torch.load(data / 'train.pt', map_location='cpu')
    validation = torch.load(data / 'validation.pt', map_location='cpu')
    if profile:
        train = train[:1024]
        validation = train[:100]
    root = Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'rl'
    root.mkdir(parents=True, exist_ok=True)
    contract = {'unit': unit, 'profile': profile, 'policy': plan['policy_sha256'], 'prior': sha256(checkpoint), 'data': {n: sha256(data / f'{n}.pt') for n in ('train', 'validation')}, 'objective': 'native Pareto QED/SA'}
    frozen_contract(root / 'rl_contract.json', contract)
    resume = root / 'resume.pt'
    state = torch.load(resume, map_location='cpu') if resume.exists() else None
    epoch = 0
    phase = 'epoch_start'
    pairs = None
    valid = None
    valid_batch = 64
    valid_shuffle = False
    batches = None
    next_batch = 0
    history = []
    updates = 0
    if state:
        if state['contract'] != contract:
            raise RuntimeError('DrugEx resume provenance differs')
        agent.load_state_dict(state['agent'])
        optim.load_state_dict(state['optimizer'])
        for v in optim.state.values():
            for k, t in v.items():
                if isinstance(t, torch.Tensor):
                    v[k] = t.to(agent.device)
        epoch, phase, pairs, valid, valid_batch, valid_shuffle, batches, next_batch, history, updates = (state[k] for k in ('epoch', 'phase', 'pairs', 'valid', 'valid_batch', 'valid_shuffle', 'batches', 'next_batch', 'history', 'updates'))
        agent.train(state['agent_training'])
        prior.train(state['prior_training'])
        restore_rng(state['rng'])

    def save():
        atomic_torch(resume, {'contract': contract, 'agent': agent.state_dict(), 'optimizer': optim.state_dict(), 'rng': rng_state(), 'epoch': epoch, 'phase': phase, 'pairs': pairs, 'valid': valid, 'valid_batch': valid_batch, 'valid_shuffle': valid_shuffle, 'batches': batches, 'next_batch': next_batch, 'history': history, 'updates': updates, 'agent_training': agent.training, 'prior_training': prior.training})
    if not state:
        save()
    while epoch < (1 if profile else 4):
        if phase == 'epoch_start':
            original = DataLoader(train, batch_size=64, shuffle=True, drop_last=False)
            vl = DataLoader(validation, batch_size=64, shuffle=False, drop_last=False)
            selected = explorer.sample_input(original)
            vl = explorer.sample_input(vl, is_test=True)
            generated = explorer.sampleEncodedPairsToLoader(net, selected)
            pairs = generated.dataset
            valid = vl.dataset
            valid_batch = vl.batch_size
            valid_shuffle = isinstance(vl.sampler, RandomSampler)
            batches = SavedBatches(sampler=BatchSampler(RandomSampler(pairs), 64, False))
            loader = DataLoader(pairs, batch_sampler=batches)
            iterator = iter(loader)
            first = next(iterator, None)
            order = batches.batches
            batches = order
            next_batch = 0
            phase = 'policy'
            atomic_torch(root / f'epoch_{epoch + 1:03d}_generated_pairs.pt', {'pairs': pairs, 'validation': valid, 'valid_batch': valid_batch, 'valid_shuffle': valid_shuffle, 'contract': contract})
            save()
        for bi in range(next_batch, len(batches)):
            src = pairs[batches[bi]]
            frags, smiles, loss = explorer.getBatchOutputs(torch.nn.DataParallel(agent, device_ids=explorer.gpus), src)
            reward = env.getRewards(smiles, frags=frags)
            if explorer.no_multifrag_smiles:
                reward = np.array([r if s.count('.') == 0 else [0] for s, r in zip(smiles, reward)])
            value = -(loss * (torch.tensor(reward, device=agent.device, dtype=torch.float32) - explorer.beta)).mean()
            if not torch.isfinite(value):
                raise ValueError('Non-finite DrugEx RL')
            optim.zero_grad()
            value.backward()
            optim.step()
            updates += 1
            next_batch = bi + 1
            trace = root / 'rl_trace' / f'{epoch + 1:03d}_{bi:06d}.json'
            payload = {'epoch': epoch + 1, 'batch': bi, 'pair_indices': batches[bi], 'smiles': smiles, 'fragments': frags, 'reward': reward.tolist(), 'loss': float(value.detach()), 'contract': contract}
            if trace.exists() and json.loads(trace.read_text()) != payload:
                raise RuntimeError('DrugEx deterministic replay differs')
            atomic_json(trace, payload)
            atomic_json(root / 'progress.json', {'epoch': epoch + 1, 'batch': next_batch, 'batches': len(batches), 'updates': updates, 'loss': float(value.detach()), 'time': __import__('time').time()})
            if updates % 50 == 0 or next_batch == len(batches) or (profile and updates == 20):
                save()
            if profile and updates >= 20:
                break
        if profile:
            break
        smiles, frags = agent.sample(DataLoader(valid, batch_size=valid_batch, shuffle=valid_shuffle))
        scores = agent.evaluate(smiles, frags, evaluator=env, no_multifrag_smiles=explorer.no_multifrag_smiles)
        scores['SMILES'], scores['Frags'] = (smiles, frags)
        metrics = explorer.getNovelMoleculeMetrics(scores)
        history.append({'epoch': epoch + 1, **{k: float(v) for k, v in metrics.items()}})
        epoch += 1
        phase = 'epoch_start'
        pairs = None
        valid = None
        batches = None
        next_batch = 0
        save()
    atomic_torch(root / 'last.pt', {'model_state': agent.state_dict(), 'model_args': args, 'vocab': checkpoint_payload['vocab'], 'contract': contract, 'seed': unit['seed'], 'history': history})
    atomic_json(root / 'run_manifest.json', {'contract': contract, 'history': history, 'updates': updates, 'checkpoint_sha256': sha256(root / 'last.pt'), 'trace_hashes': {str(p.relative_to(root)): sha256(p) for p in (root / 'rl_trace').glob('*.json')}})
    return root / 'last.pt'
