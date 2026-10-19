from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex_baseline_utils import load_vocab, save_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-samples', type=int, default=32768)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=100)
    return parser.parse_args()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    data = torch.load(path, map_location='cpu')
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    device = torch.device(args.device)
    gpus = (args.gpu,) if device.type == 'cuda' else (-1,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location=device)
    vocab = load_vocab(args.dataset_dir / 'vocab.json')
    model_args = dict(payload['model_args'])
    agent = GraphTransformer(vocab, device=device, use_gpus=gpus, **model_args)
    prior = GraphTransformer(vocab, device=device, use_gpus=gpus, **model_args)
    agent.load_state_dict(payload['model_state'], strict=True)
    prior.load_state_dict(payload['model_state'], strict=True)
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    train_loader = load_loader(args.dataset_dir / 'train.pt', args.batch_size, shuffle=True)
    valid_loader = load_loader(args.dataset_dir / 'validation.pt', args.batch_size, shuffle=False)
    environment = DrugExEnvironment(scorers=[Property('QED', modifier=ClippedScore(lower_x=0.0, upper_x=1.0)), Property('SA', modifier=ClippedScore(lower_x=10.0, upper_x=1.0))], thresholds=[0.5, 0.5], reward_scheme=ParetoCrowdingDistance())
    explorer = FragGraphExplorer(agent, environment, mutate=prior, batch_size=args.batch_size, epsilon=args.epsilon, beta=args.beta, n_samples=args.n_samples, device=device, use_gpus=gpus)
    run_config = {'task': 'drugex_graph_transformer_pareto_rl', 'args': vars(args), 'model_args': model_args, 'dataset_dir': str(args.dataset_dir.resolve()), 'checkpoint': str(args.checkpoint.resolve()), 'reward': {'scheme': 'ParetoCrowdingDistance', 'objectives': ['QED', 'SA'], 'thresholds': [0.5, 0.5], 'epsilon': args.epsilon, 'beta': args.beta}, 'torch_version': torch.__version__, 'cuda_build': torch.version.cuda}
    save_json(args.output_dir / 'run_config.json', run_config)
    monitor_prefix = args.output_dir / 'drugex_graph_rl'
    monitor = FileMonitor(str(monitor_prefix), save_smiles=True, save_model_option='best')
    explorer.fit(train_loader, valid_loader, epochs=args.epochs, patience=args.epochs, monitor=monitor)
    torch.save({'model_state': agent.state_dict(), 'model_args': model_args, 'vocab': json.loads((args.dataset_dir / 'vocab.json').read_text(encoding='utf-8')), 'epoch': args.epochs, 'seed': args.seed, 'checkpoint': str(args.checkpoint.resolve()), 'rl_args': vars(args)}, args.output_dir / 'last.pt')
    print(json.dumps({'output_dir': str(args.output_dir), 'checkpoint': str(args.output_dir / 'last.pt')}, default=str, indent=2))
if __name__ == '__main__':
    main()
