import json
import time
from pathlib import Path
from rerun_campaign.runtime import atomic_json, atomic_torch, frozen_contract, rng_state, restore_rng, sha256

def durable_epochs(model, optimizer, train_size, train_step, evaluate, save_model, root, contract, epochs, batch_size, scaler=None, patience=None, checkpoint_every=50, stop_after=None, max_updates=None, resume_origin=None):
    import torch
    from torch.utils.data import RandomSampler
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    frozen_contract(root / 'training_contract.json', contract)
    resume = root / 'resume.pt'
    state = torch.load(resume, map_location='cpu') if resume.exists() else None
    history = []
    best = float('inf')
    bad = 0
    epoch = 0
    order = None
    cursor = 0
    totals = [0.0, 0]
    updates = 0
    if state:
        if state['contract'] != contract:
            import copy

            def scientific_contract(value):
                value = copy.deepcopy(value)
                value['unit'].pop('origin_campaign_id', None)
                return value
            if not resume_origin or sha256(resume) != resume_origin['checkpoint_sha256']:
                raise ValueError('Training resume contract differs without verified lineage')
            if state['contract'] != resume_origin['contract'] or scientific_contract(state['contract']) != scientific_contract(contract):
                raise ValueError('Training resume scientific contract differs')
            if any((not torch.isfinite(v).all() for v in state['model'].values())):
                raise ValueError('Imported checkpoint has non-finite parameters')
            if any((isinstance(v, torch.Tensor) and (not torch.isfinite(v).all()) for d in state['optimizer']['state'].values() for v in d.values())):
                raise ValueError('Imported checkpoint has non-finite optimizer state')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        device = next(model.parameters()).device
        for v in optimizer.state.values():
            for k, t in v.items():
                if isinstance(t, torch.Tensor):
                    v[k] = t.to(device)
        if scaler is not None:
            scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        history, best, bad, epoch, order, cursor, totals, updates = (state[k] for k in ('history', 'best', 'bad', 'epoch', 'order', 'cursor', 'totals', 'updates'))

    def checkpoint():
        atomic_torch(resume, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if scaler is not None else {}, 'rng': rng_state(), 'contract': contract, 'history': history, 'best': best, 'bad': bad, 'epoch': epoch, 'order': order, 'cursor': cursor, 'totals': totals, 'updates': updates})
    while epoch < epochs and (patience is None or bad < patience):
        if order is None:
            torch.empty((), dtype=torch.int64).random_()
            order = list(RandomSampler(range(train_size)))
            cursor = 0
            totals = [0.0, 0]
            checkpoint()
        model.train()
        for offset in range(cursor, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            value = train_step(indices)
            if not isinstance(value, (int, float)) or not __import__('math').isfinite(value):
                raise ValueError('Non-finite training step')
            totals = [totals[0] + value * len(indices), totals[1] + len(indices)]
            cursor = offset + len(indices)
            updates += 1
            atomic_json(root / 'progress.json', {'epoch': epoch + 1, 'records': cursor, 'total': train_size, 'updates': updates, 'time': time.time()})
            if updates % checkpoint_every == 0 or cursor == len(order) or stop_after == updates or (max_updates == updates):
                checkpoint()
            if stop_after == updates:
                raise InterruptedError('Controlled test interruption')
            if max_updates == updates:
                save_model(root / 'last.pt', epoch + 1, history)
                return {'updates': updates, 'profile_only': True}
        model.eval()
        validation = float(evaluate())
        if not __import__('math').isfinite(validation):
            raise ValueError('Non-finite validation')
        improved = validation < best
        bad = 0 if improved else bad + 1
        best = min(best, validation)
        history.append({'epoch': epoch + 1, 'train_loss': totals[0] / totals[1], 'validation_loss': validation})
        if improved:
            save_model(root / 'best.pt', epoch + 1, history)
        save_model(root / 'last.pt', epoch + 1, history)
        epoch += 1
        order = None
        cursor = 0
        totals = [0.0, 0]
        checkpoint()
        atomic_json(root / 'history.json', {'history': history, 'epochs': epoch, 'updates': updates})
    return {'history': history, 'best': best, 'updates': updates, 'epochs': epoch}

def supervised(plan, unit, profile=False):
    import torch
    from torch.optim import AdamW
    from condmolgen.data import load_records
    from condmolgen.tokenizer import CharTokenizer
    from condmolgen.model import ConditionalGeneratorConfig
    from condmolgen.v4_model import V4ConditionalAutoregGenerator
    from condmolgen.v4_data import V4DecorationAutoregDataset, V4BatchCollator, build_v4_tokenizer
    from condmolgen.train_utils import seed_everything
    from scaffold_metrics import protocol_metadata
    from .worker import data_dir, resource_for
    seed_everything(unit['seed'])
    dataset = unit['dataset_id']
    resources = resource_for(plan, unit, profile)
    data = data_dir(plan, dataset)
    rows = load_records(str(data / 'train.pt'))
    val = load_records(str(data / 'validation.pt'))
    if profile:
        rows = rows[:1024]
        val = rows[:100]
    target = 'plain_decoration_smiles' if unit['representation'] == 'plain' else 'rooted_decoration_smiles'
    tokenizer = CharTokenizer(tokenizer_path=str(data / 'tokenizer.json')) if (data / 'tokenizer.json').exists() else build_v4_tokenizer(rows, target_key=target)
    ds = V4DecorationAutoregDataset(rows, tokenizer, unit['max_length'], target_key=target)
    vd = V4DecorationAutoregDataset(val, tokenizer, unit['max_length'], target_key=target)
    if len(ds) != len(rows) or len(vd) != len(val):
        raise ValueError('Silent supervised filtering prohibited')
    config = ConditionalGeneratorConfig(vocab_size=tokenizer.vocab_size, max_z=64, embedding_dim=256, encoder_ffn_dim=512, encoder_layers=4, encoder_heads=8, decoder_ffn_dim=1024, decoder_layers=6, decoder_heads=8, block_size=unit['max_length'], cutoff=5.0, num_rbf=32, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, activation_function='silu', anchor_aux_weight=0.1, decoder_pad_token_id=tokenizer.pad_token_id, encoder_type=unit['encoder_type'])
    model = V4ConditionalAutoregGenerator(config).to('cuda')
    prior = unit.get('profile_prior') if profile else None
    if prior:
        from rerun_campaign.runtime import verify_files
        verify_files(prior['files'])
        payload = torch.load(prior['checkpoint'], map_location='cpu')
        old = payload['extra']['contract']
        if payload['model_config'] != config.to_dict() or payload['tokenizer_state'] != tokenizer.to_state():
            raise RuntimeError('Profile prior architecture/tokenizer mismatch')
        if old['policy'] != plan['policy_sha256'] or old['data'] != {n: sha256(data / f'{n}.pt') for n in ('train', 'validation')}:
            raise RuntimeError('Profile prior data/protocol mismatch')
        for key in ('seed', 'dataset_id', 'representation', 'encoder_type', 'max_length'):
            if old['unit'][key] != unit[key]:
                raise RuntimeError('Profile prior unit mismatch: ' + key)
        if not old['profile'] or payload['extra']['epoch'] != 12:
            raise RuntimeError('Unqualified profile supervision')
        model.load_state_dict(payload['model_state'], strict=True)
    collate = V4BatchCollator(tokenizer)
    optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    out = Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'supervised'
    out.mkdir(parents=True, exist_ok=True)
    contract = {'unit': unit, 'resource': resources, 'policy': plan['policy_sha256'], 'profile': profile, 'data': {n: sha256(data / f'{n}.pt') for n in ('train', 'validation')}, **protocol_metadata()}

    def forward(batch):
        return model.supervised_loss(**{k: batch[k].to('cuda') for k in ('z', 'pos', 'input_ids', 'labels', 'decoder_padding_mask', 'anchor_z')})

    def step(indices):
        batch = collate([ds[i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        token_n = int(batch['labels'][:, 1:].ne(-100).sum())
        loss_total = 0.0
        physical = resources['physical_batch']
        for start in range(0, len(indices), physical):
            stop = min(start + physical, len(indices))
            part = {k: v[start:stop] if isinstance(v, torch.Tensor) else v[start:stop] for k, v in batch.items()}
            count = int(part['labels'][:, 1:].ne(-100).sum())
            with torch.cuda.amp.autocast(enabled=True):
                output = forward(part)
                loss = output['lm_loss'] * (count / token_n) + 0.1 * output['anchor_aux_loss'] * ((stop - start) / len(indices))
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        scaler.step(optimizer)
        scaler.update()
        return loss_total

    def evaluate():
        torch.empty((), dtype=torch.int64).random_()
        values = []
        with torch.no_grad():
            for start in range(0, len(vd), resources['physical_batch']):
                batch = collate([vd[i] for i in range(start, min(start + resources['physical_batch'], len(vd)))])
                with torch.cuda.amp.autocast(enabled=True):
                    values.append(float(forward(batch)['loss']))
        return sum(values) / len(values)

    def save(path, epoch, history):
        atomic_torch(path, {'model_state': model.state_dict(), 'model_config': config.to_dict(), 'tokenizer_state': tokenizer.to_state(), 'extra': {'epoch': epoch, 'history': history, 'representation': unit['representation'], 'contract': contract, **protocol_metadata()}})
    result = durable_epochs(model, optimizer, len(ds), step, evaluate, save, out, contract, 12, 256, scaler=scaler, max_updates=(4 if prior else 50) if profile else None, resume_origin=plan.get('supervised_resume_reuse', {}).get(unit['unit_id']) if not profile else None)
    atomic_json(out / 'run_manifest.json', {'contract': contract, 'result': result, 'checkpoint_sha256': sha256(out / 'last.pt'), 'resume_lineage': plan.get('supervised_resume_reuse', {}).get(unit['unit_id']) if not profile else None, 'precision_strategy': 'amp-geometric-attention-finite-fp32-recompute-v1'})
    return out / 'last.pt'
