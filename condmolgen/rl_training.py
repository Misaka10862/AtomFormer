"""Ordered CPU rewards and durable, exact-position AtomFormer RL continuation."""
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
import time

from .runtime import atomic_json, atomic_torch, frozen_contract, rng_state, restore_rng, sha256, write_jsonl

_REFERENCE = None
_ARGS = None
_REPRESENTATION = None


def initialize_reward_worker(reference, args, representation):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    global _REFERENCE, _ARGS, _REPRESENTATION
    _REFERENCE, _ARGS, _REPRESENTATION = reference, args, representation


def reward_chunk(items):
    from train_rl import compute_rewards
    _, rows = compute_rewards([x[0] for x in items], [x[1] for x in items],
                              _REFERENCE, _ARGS, _REPRESENTATION)
    return rows


class OrderedRewards:
    def __init__(self, workers, reference, args, representation):
        self.reference, self.args, self.representation = reference, args, representation
        self.pool = None
        self.workers = max(1, workers)
        if self.workers > 1:
            self.pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize_reward_worker, initargs=(reference, args, representation))

    def score(self, decorations, scaffolds):
        import torch
        if self.pool is None:
            from train_rl import compute_rewards
            return compute_rewards(decorations, scaffolds, self.reference, self.args, self.representation)
        items = list(zip(decorations, scaffolds))
        size = max(1, (len(items) + self.workers - 1) // self.workers)
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        rows = [row for group in self.pool.map(reward_chunk, chunks) for row in group]
        return torch.tensor([r['reward'] for r in rows], dtype=torch.float32), rows

    def close(self):
        if self.pool: self.pool.shutdown(wait=True)


class SavedBatches:
    """Materialize RandomSampler lazily, after DataLoader consumes its base seed."""
    def __init__(self, sampler=None, batches=None, start=0):
        self.sampler, self.batches, self.start = sampler, batches, start

    def __iter__(self):
        if self.batches is None:
            self.batches = list(iter(self.sampler))
        yield from self.batches[self.start:]

    def __len__(self):
        return len(self.sampler) if self.batches is None else len(self.batches) - self.start


def run(args):
    import torch
    from torch.utils.data import DataLoader, RandomSampler, BatchSampler
    from torch.optim import AdamW
    from condmolgen.checkpoint import load_checkpoint
    from condmolgen.data import load_records
    from condmolgen.representation import get_representation_spec
    from condmolgen.train_utils import seed_everything, split_records
    from condmolgen.decoration_data import DecorationDataset, DecorationBatchCollator
    from condmolgen.decoration_model import DecorationGenerator
    from scaffold_metrics import protocol_metadata
    from train_rl import build_train_canonical, loader_kwargs, summarize_reward_rows
    from .runtime import source_signature
    out = Path(args.output_dir)
    contract = {'args': {k:v for k,v in vars(args).items() if k not in ('resume', 'reward_workers', 'output_dir')},
                **protocol_metadata(), 'input_sha256': sha256(args.train_path),
                'supervised_sha256': sha256(args.checkpoint), 'code_sha256': source_signature()}
    if out.exists() and not (out / 'rl_contract.json').exists() and any(out.iterdir()):
        raise RuntimeError('Output directory contains unrelated files')
    frozen_contract(out / 'rl_contract.json', contract)
    resume_path = out / 'resume.pt'
    if resume_path.exists() and not args.resume:
        raise RuntimeError('An explicit --resume is required')
    if args.resume and not resume_path.exists():
        raise RuntimeError('No resume checkpoint exists in the output directory')
    seed_everything(args.seed)
    representation = get_representation_spec(args.representation)
    loaded, tokenizer, extra = load_checkpoint(args.checkpoint, map_location=args.device)
    model = DecorationGenerator(loaded.config).to(args.device)
    model.load_state_dict(loaded.state_dict(), strict=True)
    del loaded
    records = load_records(args.train_path)
    if args.max_train_samples: records = records[:args.max_train_samples]
    train_records, _ = split_records(records, args.val_ratio, args.seed)
    reference = build_train_canonical(train_records)
    dataset = DecorationDataset(train_records, tokenizer, args.max_length, target_key=representation.target_key)
    if not len(dataset) or len(dataset) != len(train_records):
        raise RuntimeError('RL input coverage changed')
    collator = DecorationBatchCollator(tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    state = torch.load(resume_path, map_location=args.device, weights_only=False) if resume_path.exists() else None
    if state:
        if state['contract'] != contract: raise RuntimeError('RL checkpoint contract differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
    history = state['history'] if state else []
    start_epoch = state['epoch'] if state else 1
    rewards = OrderedRewards(args.reward_workers, reference, args, representation.name)
    trace = out / 'rl_trace'
    trace.mkdir(exist_ok=True)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            continuing = state is not None and epoch == state['epoch']
            start = state['next_batch'] if continuing else 0
            batches = SavedBatches(batches=state['batches'], start=start) if continuing else SavedBatches(
                sampler=BatchSampler(RandomSampler(dataset), args.batch_size, drop_last=False))
            loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collator, **loader_kwargs(args))
            iterator = iter(loader)
            if continuing:
                # Iterator construction consumes a base seed. It is not an
                # extra policy RNG event in the resumed trajectory.
                restore_rng(state['rng'])
            totals = list(state['totals']) if continuing else [0.0, 0.0, 0.0, 0]
            started = time.monotonic()
            model.train(True)
            for batch_index, batch in enumerate(iterator, start=start):
                z, pos = batch['z'].to(args.device), batch['pos'].to(args.device)
                sampled = model.sample_decorations_with_logprobs(z=z, pos=pos, tokenizer=tokenizer,
                    scaffold_smiles=batch['scaffold_smiles'], max_length=args.max_length, temperature=args.temperature)
                reward_cpu, rows = rewards.score(sampled['decorations'], batch['scaffold_smiles'])
                reward = reward_cpu.to(args.device)
                advantages = reward - reward.mean() if args.reward_baseline == 'batch_mean' else reward
                logprob = (sampled['token_logprobs'] * sampled['action_mask']).sum(dim=1)
                action_count = sampled['action_mask'].sum(dim=1).clamp(min=1.0)
                rl_loss = -(logprob / action_count * advantages.detach()).mean()
                supervised = model.supervised_loss(z=z, pos=pos, input_ids=batch['input_ids'].to(args.device),
                    labels=batch['labels'].to(args.device), decoder_padding_mask=batch['decoder_padding_mask'].to(args.device),
                    anchor_z=batch['anchor_z'].to(args.device))['loss']
                loss = rl_loss + args.supervised_weight * supervised
                if not torch.isfinite(loss): raise RuntimeError('Non-finite RL loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if args.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                totals = [totals[0]+loss.item(), totals[1]+rl_loss.item(), totals[2]+supervised.item(), totals[3]+1]
                ids = batches.batches[batch_index]
                trace_rows = [{**r, 'epoch':epoch, 'batch_index':batch_index, 'policy_record_id':idx,
                               'attempt_id':(epoch-1)*len(dataset)+batch_index*args.batch_size+k,
                               **protocol_metadata()} for k,(r,idx) in enumerate(zip(rows,ids))]
                path = trace / f'{epoch:03d}_{batch_index:06d}.json'
                if path.exists() and json.loads(path.read_text()) != trace_rows:
                    raise RuntimeError('RL replay differs from its previously saved trace')
                atomic_json(path, trace_rows)
                if (batch_index + 1) % getattr(args,'checkpoint_every',50) == 0 or batch_index + 1 == len(batches.batches):
                    atomic_torch(resume_path, {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'rng':rng_state(), 'epoch':epoch, 'next_batch':batch_index+1, 'batches':batches.batches,
                        'totals':totals, 'history':history, 'contract':contract})
                atomic_json(out / 'progress.json', {'epoch':epoch, 'batch':batch_index+1,
                    'batches':len(batches.batches), 'time':time.time(), 'checkpoint_exists':resume_path.exists()})
            epoch_rows = [r for p in sorted(trace.glob(f'{epoch:03d}_*.json')) for r in json.loads(p.read_text())]
            if len(epoch_rows) != len(dataset): raise RuntimeError('RL epoch trace coverage mismatch')
            entry = {'epoch':epoch,'loss':totals[0]/totals[3],'rl_loss':totals[1]/totals[3],
                'supervised_loss':totals[2]/totals[3],'steps':totals[3], 'elapsed_sec':time.monotonic()-started,
                **summarize_reward_rows(epoch_rows)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            state = None
            # Store the finished epoch separately; a restart at this boundary
            # can reconstruct its summary and the next epoch's original RNG.
            atomic_json(out / 'history.json', {'history':history,'args':vars(args),**protocol_metadata()})
        atomic_torch(out / 'last.pt', {'model_state':model.state_dict(),'model_config':model.config.to_dict(),
            'tokenizer_state':tokenizer.to_state(),'extra':{'history':history,'args':vars(args),
                'task':'rl_finetuning', 'source_checkpoint':args.checkpoint, **protocol_metadata()}})
        write_jsonl(out / 'rl_samples.jsonl', (r for p in sorted(trace.glob('*.json')) for r in json.loads(p.read_text())))
        atomic_json(out / 'run_manifest.json', {'contract':contract,'history':history,
            'train_dataset_records':len(dataset),'attempts':len(dataset)*args.epochs,
            'checkpoint_sha256':sha256(out / 'last.pt'),'trace_sha256':sha256(out / 'rl_samples.jsonl')})
    finally:
        rewards.close()
