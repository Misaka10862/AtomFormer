import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[UNK]"]


@dataclass
class TokenizerState:
    id2token: Dict[int, str]


class CharTokenizer:
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        texts: Optional[Iterable[str]] = None,
    ) -> None:
        if tokenizer_path and Path(tokenizer_path).exists():
            state = self.load_state(tokenizer_path)
            self.id2token = state.id2token
        elif texts is not None:
            self.id2token = self.build_vocab(texts)
            if tokenizer_path:
                self.save(tokenizer_path)
        else:
            raise ValueError("Need an existing tokenizer_path or texts to build a tokenizer.")

        self.token2id = {token: idx for idx, token in self.id2token.items()}

    @staticmethod
    def build_vocab(texts: Iterable[str]) -> Dict[int, str]:
        chars = sorted(set("".join(texts)))
        id2token: Dict[int, str] = {}
        cursor = 0
        for token in SPECIAL_TOKENS:
            id2token[cursor] = token
            cursor += 1
        for char in chars:
            if char not in SPECIAL_TOKENS:
                id2token[cursor] = char
                cursor += 1
        return id2token

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    @property
    def pad_token_id(self) -> int:
        return self.token2id["[PAD]"]

    @property
    def bos_token_id(self) -> int:
        return self.token2id["[BOS]"]

    @property
    def eos_token_id(self) -> int:
        return self.token2id["[EOS]"]

    @property
    def unk_token_id(self) -> int:
        return self.token2id["[UNK]"]

    @property
    def sep_token_id(self) -> int:
        return self.token2id["[SEP]"]

    def encode_smiles(self, smiles: str) -> List[int]:
        tokens = ["[BOS]"] + list(smiles) + ["[EOS]"]
        return [self.token2id.get(token, self.unk_token_id) for token in tokens]

    def encode_scaffold_target(self, scaffold_smiles: str, full_smiles: str) -> List[int]:
        tokens = ["[BOS]"] + list(scaffold_smiles) + ["[SEP]"] + list(full_smiles) + ["[EOS]"]
        return [self.token2id.get(token, self.unk_token_id) for token in tokens]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id2token[int(idx)]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_state(self) -> Dict[str, Dict[str, str]]:
        return {"id2token": {str(k): v for k, v in self.id2token.items()}}

    @staticmethod
    def from_state(state: Dict[str, Dict[str, str]]) -> "CharTokenizer":
        tokenizer = CharTokenizer.__new__(CharTokenizer)
        tokenizer.id2token = {int(k): v for k, v in state["id2token"].items()}
        tokenizer.token2id = {token: idx for idx, token in tokenizer.id2token.items()}
        return tokenizer

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_state(), handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_state(path: str) -> TokenizerState:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return TokenizerState(id2token={int(k): v for k, v in raw["id2token"].items()})
