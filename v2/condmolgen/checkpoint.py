from pathlib import Path
from typing import Dict, Tuple

import torch

from .model import ConditionalGeneratorConfig, ConditionalMolGenerator
from .tokenizer import CharTokenizer


def save_checkpoint(
    path: str,
    model: ConditionalMolGenerator,
    tokenizer: CharTokenizer,
    config: ConditionalGeneratorConfig,
    extra: Dict = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "tokenizer_state": tokenizer.to_state(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[ConditionalMolGenerator, CharTokenizer, Dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    tokenizer = CharTokenizer.from_state(payload["tokenizer_state"])
    saved_config = dict(payload["model_config"])
    # Missing fields identify historical behavior, never silently upgrade weights.
    saved_config.setdefault("auxiliary_pooling", "legacy_mean")
    saved_config.setdefault("encoder_norm_first", False)
    saved_config.setdefault("atom_attention_dropout", saved_config.get("dropout", 0.1))
    saved_config.setdefault("encoder_type", "geo")
    config = ConditionalGeneratorConfig(**saved_config)
    model = ConditionalMolGenerator(config)
    load_info = model.load_state_dict(payload["model_state"], strict=True)
    extra = payload.get("extra", {})
    if load_info.missing_keys or load_info.unexpected_keys:
        extra = {
            **extra,
            "checkpoint_load_info": {
                "missing_keys": list(load_info.missing_keys),
                "unexpected_keys": list(load_info.unexpected_keys),
            },
        }
    return model, tokenizer, extra
