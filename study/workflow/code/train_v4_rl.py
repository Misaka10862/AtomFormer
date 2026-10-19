import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from rdkit import RDLogger
from condmolgen.checkpoint import load_checkpoint, save_checkpoint
from condmolgen.data import load_records
from condmolgen.metrics import canonicalize_smiles
from condmolgen.train_utils import seed_everything, split_records, write_json
from condmolgen.v4_data import V4BatchCollator, V4DecorationAutoregDataset
from condmolgen.v4_model import V4ConditionalAutoregGenerator
from condmolgen.representation import assemble_with_representation, get_representation_spec
from condmolgen.run_manifest import write_run_manifest
from scaffold_metrics import scaffold_checks, scaffold_core_from_scaffold, protocol_metadata
RDLogger.DisableLog('rdApp.*')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Lightweight REINFORCE fine-tuning for v4 conditional autoregressive decoder.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-samples', type=int, default=0)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--supervised-weight', type=float, default=0.2)
    parser.add_argument('--qed-weight', type=float, default=1.0)
    parser.add_argument('--novelty-weight', type=float, default=0.2)
    parser.add_argument('--sas-weight', type=float, default=0.05)
    parser.add_argument('--invalid-reward', type=float, default=-2.0)
    parser.add_argument('--core-failure-reward', type=float, default=-2.0)
    parser.add_argument('--reward-baseline', default='batch_mean', choices=['none', 'batch_mean'])
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', type=int, default=0)
    parser.add_argument('--persistent-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--representation', choices=['rooted', 'plain'], default='rooted')
    parser.add_argument('--expected-rdkit', default=None)
    parser.add_argument('--campaign-resume', action='store_true', help='Use the recorded resumable RL implementation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reward-workers', type=int, default=1)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--require-full-coverage', type=int, default=0, help='Fail if max-length filtering drops any RL input record.')
    return parser.parse_args()

def _import_rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import QED, RDConfig
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import os
    import sys
    RDLogger.DisableLog('rdApp.*')
    contrib_dir = getattr(RDConfig, 'RDContribDir', None)
    if contrib_dir:
        sa_dir = os.path.join(contrib_dir, 'SA_Score')
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer
    return (Chem, QED, MurckoScaffold, sascorer)

def loader_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {'num_workers': args.num_workers, 'pin_memory': bool(args.pin_memory)}
    if args.num_workers > 0:
        kwargs['persistent_workers'] = bool(args.persistent_workers)
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs

def build_train_canonical(records: Iterable[Dict]) -> set:
    canonical = set()
    for record in records:
        smiles = record.get('canonical_smiles')
        if smiles:
            canon = canonicalize_smiles(smiles)
            if canon:
                canonical.add(canon)
    return canonical

def scaffold_core_ok(assembled_smiles: str, scaffold_smiles: str) -> bool:
    core = scaffold_core_from_scaffold(scaffold_smiles)
    if not core:
        raise ValueError('Cannot construct target scaffold core')
    return scaffold_checks(assembled_smiles, core)['core_retention']

def compute_rewards(decorations: List[str], scaffolds: List[str], train_canonical: set, args: argparse.Namespace, representation: str='rooted') -> Tuple[torch.Tensor, List[Dict]]:
    Chem, QED, _, sascorer = _import_rdkit()
    rewards: List[float] = []
    rows: List[Dict] = []
    for decoration, scaffold in zip(decorations, scaffolds):
        assembled = assemble_with_representation(representation, scaffold, decoration)
        row = {'scaffold_smiles': scaffold, 'generated_decoration': decoration, 'assembled_smiles': assembled, 'assembly_success': assembled is not None, 'validity': False, 'scaffold_core_ok': False, 'qed': None, 'sas': None, 'novelty': 0.0, 'reward': args.invalid_reward}
        if assembled is None:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue
        mol = Chem.MolFromSmiles(assembled)
        if mol is None:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue
        row['validity'] = True
        if not scaffold_core_ok(assembled, scaffold):
            row['reward'] = args.core_failure_reward
            rewards.append(args.core_failure_reward)
            rows.append(row)
            continue
        row['scaffold_core_ok'] = True
        try:
            qed = float(QED.qed(mol))
            sas = float(sascorer.calculateScore(mol))
        except Exception:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue
        canon = canonicalize_smiles(assembled)
        novelty = 1.0 if canon and canon not in train_canonical else 0.0
        reward = args.qed_weight * qed + args.novelty_weight * novelty - args.sas_weight * sas
        row.update({'qed': qed, 'sas': sas, 'novelty': novelty, 'reward': reward})
        rewards.append(reward)
        rows.append(row)
    return (torch.tensor(rewards, dtype=torch.float32), rows)

def summarize_reward_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    valid = [row for row in rows if row['validity']]
    assembled = [row for row in rows if row['assembly_success']]
    core_ok = [row for row in rows if row['scaffold_core_ok']]
    qeds = [row['qed'] for row in rows if row['qed'] is not None]
    sas = [row['sas'] for row in rows if row['sas'] is not None]
    rewards = [row['reward'] for row in rows]
    return {'reward_mean': float(sum(rewards) / max(len(rewards), 1)), 'reward_min': float(min(rewards)), 'reward_max': float(max(rewards)), 'assembly_success_rate': len(assembled) / max(len(rows), 1), 'validity': len(valid) / max(len(rows), 1), 'scaffold_core_ok_rate': len(core_ok) / max(len(rows), 1), 'qed_mean': float(sum(qeds) / max(len(qeds), 1)) if qeds else 0.0, 'sas_mean': float(sum(sas) / max(len(sas), 1)) if sas else 0.0}

def main() -> None:
    args = parse_args()
    if args.validate_only:
        print('CLI_VALIDATED')
        return
    if args.expected_rdkit and args.expected_rdkit != protocol_metadata()['rdkit_version']:
        raise RuntimeError('RDKit version differs from the prepared RL plan')
    if args.campaign_resume:
        from rerun_campaign.af_training import run
        return run(args)
    representation = get_representation_spec(args.representation)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(((output_dir / name).exists() for name in ('last.pt', 'history.json', 'rl_samples.jsonl'))):
        raise RuntimeError('Use a new output directory for the corrected scaffold reward protocol')
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_model, tokenizer, extra = load_checkpoint(args.checkpoint, map_location=args.device)
    model = V4ConditionalAutoregGenerator(loaded_model.config)
    model.load_state_dict(loaded_model.state_dict(), strict=True)
    model = model.to(torch.device(args.device))
    records = load_records(args.train_path)
    if args.max_train_samples:
        records = records[:args.max_train_samples]
    train_records, _ = split_records(records, args.val_ratio, args.seed)
    train_canonical = build_train_canonical(train_records)
    train_dataset = V4DecorationAutoregDataset(train_records, tokenizer, max_length=args.max_length, target_key=representation.target_key)
    if len(train_dataset) == 0:
        raise ValueError('No RL train samples remain after v4 filtering.')
    if args.require_full_coverage and len(train_dataset) != len(train_records):
        raise ValueError(f'{representation.name} representation filtered RL records: expected={len(train_records)} actual={len(train_dataset)}')
    collator = V4BatchCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, **loader_kwargs(args))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    all_reward_rows: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train(True)
        epoch_rows: List[Dict] = []
        total_loss = 0.0
        total_rl_loss = 0.0
        total_sup_loss = 0.0
        total_steps = 0
        start = time.time()
        for batch in train_loader:
            z = batch['z'].to(args.device)
            pos = batch['pos'].to(args.device)
            sampled = model.sample_decorations_with_logprobs(z=z, pos=pos, tokenizer=tokenizer, scaffold_smiles=batch['scaffold_smiles'], max_length=args.max_length, temperature=args.temperature)
            rewards_cpu, reward_rows = compute_rewards(decorations=sampled['decorations'], scaffolds=batch['scaffold_smiles'], train_canonical=train_canonical, args=args, representation=representation.name)
            rewards = rewards_cpu.to(args.device)
            if args.reward_baseline == 'batch_mean':
                advantages = rewards - rewards.mean()
            else:
                advantages = rewards
            action_logprob_sum = (sampled['token_logprobs'] * sampled['action_mask']).sum(dim=1)
            action_count = sampled['action_mask'].sum(dim=1).clamp(min=1.0)
            rl_loss = -(action_logprob_sum / action_count * advantages.detach()).mean()
            sup_out = model.supervised_loss(z=z, pos=pos, input_ids=batch['input_ids'].to(args.device), labels=batch['labels'].to(args.device), decoder_padding_mask=batch['decoder_padding_mask'].to(args.device), anchor_z=batch['anchor_z'].to(args.device))
            supervised_loss = sup_out['loss']
            loss = rl_loss + args.supervised_weight * supervised_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            total_rl_loss += float(rl_loss.item())
            total_sup_loss += float(supervised_loss.item())
            total_steps += 1
            epoch_rows.extend(reward_rows)
        elapsed = time.time() - start
        summary = summarize_reward_rows(epoch_rows)
        row = {'epoch': epoch, 'loss': total_loss / max(total_steps, 1), 'rl_loss': total_rl_loss / max(total_steps, 1), 'supervised_loss': total_sup_loss / max(total_steps, 1), 'steps': total_steps, 'elapsed_sec': elapsed, **summary}
        history.append(row)
        all_reward_rows.extend(epoch_rows)
        print(row)
        save_checkpoint(path=str(output_dir / 'last.pt'), model=model, tokenizer=tokenizer, config=model.config, extra={**protocol_metadata(), 'history': history, 'epoch': epoch, 'args': vars(args), 'task': 'v4_cond_autoreg_rl_finetune', 'source_checkpoint_extra': extra})
    write_json(str(output_dir / 'history.json'), {**protocol_metadata(), 'history': history, 'args': vars(args), 'task': 'v4_cond_autoreg_rl_finetune', 'source_checkpoint': args.checkpoint})
    write_run_manifest(output_dir / 'run_manifest.json', data_paths=[args.train_path, args.checkpoint], split={'train': args.train_path}, model_config=model.config.to_dict(), seed=args.seed, checkpoint_lineage=args.checkpoint, outputs=[output_dir], experiment_metadata={**protocol_metadata(), 'representation': representation.name, 'target_key': representation.target_key, 'assembly_mode': representation.assembly_mode, 'train_records_after_split': len(train_records), 'train_dataset_records': len(train_dataset)})
    with (output_dir / 'rl_samples.jsonl').open('w', encoding='utf-8') as handle:
        for row in all_reward_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()
