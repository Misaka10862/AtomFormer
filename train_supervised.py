import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from condmolgen.checkpoint import save_checkpoint
from condmolgen.data import load_records
from condmolgen.model import ConditionalGeneratorConfig
from condmolgen.train_utils import seed_everything, split_records, write_json
from condmolgen.decoration_data import DecorationBatchCollator, DecorationDataset, build_decoration_tokenizer
from condmolgen.decoration_model import build_decoration_model
from condmolgen.run_manifest import write_run_manifest
from condmolgen.representation import get_representation_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AtomFormer scaffold-conditioned autoregressive decoration decoder.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-z", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-ffn-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--decoder-ffn-dim", type=int, default=1024)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num-rbf", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--activation-dropout", type=float, default=0.0)
    parser.add_argument("--activation-function", default="silu")
    parser.add_argument("--anchor-aux-weight", type=float, default=0.1)
    parser.add_argument("--encoder-type", choices=["geo", "atom"], default="geo")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--use-amp", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--representation", choices=["rooted", "plain"], default="rooted")
    parser.add_argument(
        "--require-full-coverage",
        type=int,
        default=0,
        help="Fail if max-length filtering drops any input record.",
    )
    return parser.parse_args()


def loader_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def run_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    train: bool,
    use_amp: bool,
    grad_accum_steps: int,
) -> Dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_steps = 0
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    if train:
        optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for batch in dataloader:
        z = batch["z"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        decoder_padding_mask = batch["decoder_padding_mask"].to(device, non_blocking=True)
        anchor_z = batch["anchor_z"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                output = model.supervised_loss(
                    z=z,
                    pos=pos,
                    input_ids=input_ids,
                    labels=labels,
                    decoder_padding_mask=decoder_padding_mask,
                    anchor_z=anchor_z,
                )
                loss = output["loss"] / max(grad_accum_steps, 1)
            if train:
                scaler.scale(loss).backward()
                if (total_steps + 1) % max(grad_accum_steps, 1) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            shift_logits = output["logits"][:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = shift_labels.ne(-100)
            if bool(mask.any()):
                pred = torch.argmax(shift_logits, dim=-1)
                total_correct += int((pred[mask] == shift_labels[mask]).sum().item())
                total_tokens += int(mask.sum().item())

        total_loss += float(loss.item()) * max(grad_accum_steps, 1)
        total_steps += 1

    if train and total_steps % max(grad_accum_steps, 1) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - start
    samples = len(dataloader.dataset)
    return {
        "loss": total_loss / max(total_steps, 1),
        "token_acc": total_correct / max(total_tokens, 1),
        "tokens": total_tokens,
        "steps": total_steps,
        "samples": samples,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples / max(elapsed, 1e-8),
    }


def main() -> None:
    args = parse_args()
    representation = get_representation_spec(args.representation)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.train_path)
    if args.max_train_samples:
        records = records[: args.max_train_samples]

    if args.val_path:
        train_records = records
        val_records = load_records(args.val_path)
        if args.max_val_samples:
            val_records = val_records[: args.max_val_samples]
    else:
        train_records, val_records = split_records(records, args.val_ratio, args.seed)

    tokenizer = build_decoration_tokenizer(
        train_records,
        tokenizer_path=args.tokenizer_path,
        target_key=representation.target_key,
    )
    tokenizer.save(str(output_dir / "tokenizer.json"))

    train_dataset = DecorationDataset(
        train_records,
        tokenizer,
        max_length=args.block_size,
        target_key=representation.target_key,
    )
    val_dataset = (
        DecorationDataset(
            val_records,
            tokenizer,
            max_length=args.block_size,
            target_key=representation.target_key,
        )
        if val_records
        else None
    )
    if len(train_dataset) == 0:
        raise ValueError("No train samples remain after AtomFormer filtering.")
    if args.require_full_coverage:
        coverage = [("train", len(train_records), len(train_dataset))]
        if val_records:
            coverage.append(("validation", len(val_records), len(val_dataset)))
        failures = [row for row in coverage if row[1] != row[2]]
        if failures:
            raise ValueError(f"{representation.name} representation filtered records: {failures}")

    collator = DecorationBatchCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs(args),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            **loader_kwargs(args),
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    config = ConditionalGeneratorConfig(
        vocab_size=tokenizer.vocab_size,
        max_z=args.max_z,
        embedding_dim=args.embedding_dim,
        encoder_ffn_dim=args.encoder_ffn_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        block_size=args.block_size,
        cutoff=args.cutoff,
        num_rbf=args.num_rbf,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function=args.activation_function,
        encoder_pad_token_id=0,
        decoder_pad_token_id=tokenizer.pad_token_id,
        anchor_aux_weight=args.anchor_aux_weight,
        encoder_type=args.encoder_type,
    )
    model = build_decoration_model(config).to(torch.device(args.device))
    write_run_manifest(
        output_dir / "run_manifest.json",
        data_paths=[args.train_path] + ([args.val_path] if args.val_path else []),
        split={"train": args.train_path, "validation": args.val_path},
        model_config=config.to_dict(),
        seed=args.seed,
        outputs=[output_dir],
        experiment_metadata={
            "representation": representation.name,
            "target_key": representation.target_key,
            "assembly_mode": representation.assembly_mode,
            "train_records": len(train_records),
            "validation_records": len(val_records),
            "train_dataset_records": len(train_dataset),
            "validation_dataset_records": len(val_dataset) if val_dataset is not None else 0,
        },
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            torch.device(args.device),
            train=True,
            use_amp=bool(args.use_amp),
            grad_accum_steps=args.grad_accum_steps,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_token_acc": train_metrics["token_acc"],
            "train_samples_per_sec": train_metrics["samples_per_sec"],
            "train_effective_batch_size": args.batch_size * max(args.grad_accum_steps, 1),
        }
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                torch.device(args.device),
                train=False,
                use_amp=bool(args.use_amp),
                grad_accum_steps=1,
            )
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_token_acc": val_metrics["token_acc"],
                    "val_samples_per_sec": val_metrics["samples_per_sec"],
                }
            )
        history.append(row)
        print(row)
        save_checkpoint(
            path=str(output_dir / "last.pt"),
            model=model,
            tokenizer=tokenizer,
            config=config,
            extra={
                "history": history,
                "epoch": epoch,
                "args": vars(args),
                "task": "supervised_training",
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset) if val_dataset is not None else 0,
            },
        )

    write_json(
        str(output_dir / "history.json"),
        {
            "history": history,
            "args": vars(args),
            "task": "supervised_training",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "vocab_size": tokenizer.vocab_size,
        },
    )


if __name__ == "__main__":
    main()
