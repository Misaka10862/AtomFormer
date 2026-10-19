from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from drugex.training.generators import GraphTransformer
from drugex_baseline_utils import load_vocab, save_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--valid-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--d-emb', type=int, default=512)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--d-inner', type=int, default=1024)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-layer', type=int, default=12)
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

def batch_loss(model: GraphTransformer, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = model(batch.to(device), is_train=True)
    return sum((-item.mean() for item in outputs))

def run_epoch(model: GraphTransformer, loader: DataLoader, device: torch.device, train: bool, progress_every: int) -> float:
    model.train(train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            if train:
                model.optim.zero_grad()
            loss = batch_loss(model, batch, device)
            if train:
                loss.backward()
                model.optim.step()
            batch_count = int(batch.shape[0])
            total += float(loss.detach().cpu()) * batch_count
            count += batch_count
            if progress_every and step % progress_every == 0:
                print(f"{('train' if train else 'valid')} step={step} loss={float(loss):.6f}", flush=True)
    return total / max(count, 1)

def save_checkpoint(path: Path, model: GraphTransformer, vocab_payload: Dict, model_args: Dict, epoch: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'model_args': model_args, 'vocab': vocab_payload, 'epoch': epoch, 'seed': seed, 'optimizer_steps': int(model.optim.n_steps)}, path)

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    device = torch.device(args.device)
    gpus = (args.gpu,) if args.device == 'cuda' else (-1,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.dataset_dir / 'vocab.json'
    vocab = load_vocab(vocab_path)
    vocab_payload = json.loads(vocab_path.read_text(encoding='utf-8'))
    train_loader = load_loader(args.dataset_dir / 'train.pt', args.batch_size, shuffle=True)
    valid_loader = load_loader(args.dataset_dir / 'validation.pt', args.valid_batch_size, shuffle=False)
    model_args = {'d_emb': args.d_emb, 'd_model': args.d_model, 'd_inner': args.d_inner, 'n_head': args.n_head, 'n_layer': args.n_layer, 'pad_idx': 0}
    model = GraphTransformer(vocab, device=device, use_gpus=gpus, **model_args)
    run_config = {'task': 'drugex_graph_transformer_supervised', 'args': vars(args), 'model_args': model_args, 'dataset_dir': str(args.dataset_dir.resolve()), 'device': str(device), 'torch_version': torch.__version__, 'cuda_build': torch.version.cuda}
    save_json(args.output_dir / 'run_config.json', run_config)
    best_loss = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, True, args.progress_every)
        valid_loss = run_epoch(model, valid_loader, device, False, args.progress_every)
        row = {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': valid_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_checkpoint(args.output_dir / 'last.pt', model, vocab_payload, model_args, epoch, args.seed)
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(args.output_dir / 'best.pt', model, vocab_payload, model_args, epoch, args.seed)
    save_json(args.output_dir / 'history.json', {'history': history, 'best_validation_loss': best_loss})
    print(json.dumps({'output_dir': str(args.output_dir), 'best_validation_loss': best_loss}, indent=2))
if __name__ == '__main__':
    main()
