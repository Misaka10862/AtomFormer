import argparse
import subprocess
import sys
from typing import List
from pathlib import Path





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scaffold-conditioned decoration generator."
    )
    parser.add_argument("--stage", required=True, choices=["supervised", "rl"])
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--representation", choices=["rooted", "plain"], default="rooted")
    parser.add_argument("--require-full-coverage", type=int, default=None)

    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", type=int, default=None)
    parser.add_argument("--persistent-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)

    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-z", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--encoder-ffn-dim", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--encoder-heads", type=int, default=None)
    parser.add_argument("--decoder-ffn-dim", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--num-rbf", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-dropout", type=float, default=None)
    parser.add_argument("--activation-dropout", type=float, default=None)
    parser.add_argument("--activation-function", default=None)
    parser.add_argument("--anchor-aux-weight", type=float, default=None)
    parser.add_argument("--use-amp", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)

    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--supervised-weight", type=float, default=None)
    parser.add_argument("--qed-weight", type=float, default=None)
    parser.add_argument("--novelty-weight", type=float, default=None)
    parser.add_argument("--sas-weight", type=float, default=None)
    parser.add_argument("--invalid-reward", type=float, default=None)
    parser.add_argument("--core-failure-reward", type=float, default=None)
    parser.add_argument("--reward-baseline", choices=["none", "batch_mean"], default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reward-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    return parser.parse_args()


def _append_optional(cmd, flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_supervised_command(args: argparse.Namespace) -> List[str]:
    representation = getattr(args, "representation", "rooted")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_supervised.py")),
        "--train-path",
        args.train_path,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--representation",
        representation,
        "--epochs",
        str(args.epochs if args.epochs is not None else 12),
        "--batch-size",
        str(args.batch_size if args.batch_size is not None else 128),
        "--learning-rate",
        str(args.learning_rate if args.learning_rate is not None else 3e-4),
        "--use-amp",
        str(args.use_amp if args.use_amp is not None else 1),
        "--anchor-aux-weight",
        str(args.anchor_aux_weight if args.anchor_aux_weight is not None else 0.1),
    ]
    _append_optional(cmd, "--weight-decay", args.weight_decay)
    _append_optional(cmd, "--val-ratio", args.val_ratio)
    _append_optional(cmd, "--seed", args.seed)
    _append_optional(cmd, "--max-train-samples", args.max_train_samples)
    _append_optional(cmd, "--num-workers", args.num_workers)
    _append_optional(cmd, "--pin-memory", args.pin_memory)
    _append_optional(cmd, "--persistent-workers", args.persistent_workers)
    _append_optional(cmd, "--prefetch-factor", args.prefetch_factor)
    _append_optional(cmd, "--tokenizer-path", args.tokenizer_path)
    _append_optional(cmd, "--val-path", args.val_path)
    _append_optional(cmd, "--max-val-samples", args.max_val_samples)
    _append_optional(cmd, "--max-z", args.max_z)
    _append_optional(cmd, "--embedding-dim", args.embedding_dim)
    _append_optional(cmd, "--encoder-ffn-dim", args.encoder_ffn_dim)
    _append_optional(cmd, "--encoder-layers", args.encoder_layers)
    _append_optional(cmd, "--encoder-heads", args.encoder_heads)
    _append_optional(cmd, "--decoder-ffn-dim", args.decoder_ffn_dim)
    _append_optional(cmd, "--decoder-layers", args.decoder_layers)
    _append_optional(cmd, "--decoder-heads", args.decoder_heads)
    _append_optional(cmd, "--block-size", args.block_size)
    _append_optional(cmd, "--cutoff", args.cutoff)
    _append_optional(cmd, "--num-rbf", args.num_rbf)
    _append_optional(cmd, "--dropout", args.dropout)
    _append_optional(cmd, "--attention-dropout", args.attention_dropout)
    _append_optional(cmd, "--activation-dropout", args.activation_dropout)
    _append_optional(cmd, "--activation-function", args.activation_function)
    _append_optional(cmd, "--grad-accum-steps", args.grad_accum_steps)
    _append_optional(cmd, "--require-full-coverage", getattr(args, "require_full_coverage", None))
    return cmd


def build_rl_command(args: argparse.Namespace) -> List[str]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage rl.")
    representation = getattr(args, "representation", "rooted")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_rl.py")),
        "--checkpoint",
        args.checkpoint,
        "--train-path",
        args.train_path,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--representation",
        representation,
        "--epochs",
        str(args.epochs if args.epochs is not None else 1),
        "--batch-size",
        str(args.batch_size if args.batch_size is not None else 64),
        "--learning-rate",
        str(args.learning_rate if args.learning_rate is not None else 5e-6),
        "--supervised-weight",
        str(args.supervised_weight if args.supervised_weight is not None else 0.1),
        "--novelty-weight",
        str(args.novelty_weight if args.novelty_weight is not None else 0.4),
        "--qed-weight",
        str(args.qed_weight if args.qed_weight is not None else 1.0),
        "--sas-weight",
        str(args.sas_weight if args.sas_weight is not None else 0.05),
        "--reward-baseline",
        str(args.reward_baseline if args.reward_baseline is not None else "batch_mean"),
    ]
    _append_optional(cmd, "--weight-decay", args.weight_decay)
    _append_optional(cmd, "--val-ratio", args.val_ratio)
    _append_optional(cmd, "--seed", args.seed)
    _append_optional(cmd, "--max-train-samples", args.max_train_samples)
    _append_optional(cmd, "--max-length", args.max_length)
    _append_optional(cmd, "--temperature", args.temperature)
    _append_optional(cmd, "--invalid-reward", args.invalid_reward)
    _append_optional(cmd, "--core-failure-reward", args.core_failure_reward)
    _append_optional(cmd, "--max-grad-norm", args.max_grad_norm)
    _append_optional(cmd, "--num-workers", args.num_workers)
    _append_optional(cmd, "--pin-memory", args.pin_memory)
    _append_optional(cmd, "--persistent-workers", args.persistent_workers)
    _append_optional(cmd, "--prefetch-factor", args.prefetch_factor)
    _append_optional(cmd, "--require-full-coverage", getattr(args, "require_full_coverage", None))
    _append_optional(cmd, "--reward-workers", getattr(args, "reward_workers", 1))
    _append_optional(cmd, "--checkpoint-every", getattr(args, "checkpoint_every", 50))
    if getattr(args, "resume", False):
        cmd.append("--resume")
    return cmd


def main() -> None:
    args = parse_args()
    if args.stage == "supervised":
        cmd = build_supervised_command(args)
    else:
        cmd = build_rl_command(args)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
