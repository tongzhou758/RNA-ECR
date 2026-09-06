#!/usr/bin/env python3
"""Predict IRES activity with the IRES classifier.

Example:
    python predict/predict_IRES.py \
      --input data/IRES_prediction/input.csv \
      --model model/IRES/IRES_best.pt \
      --pretrained checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt \
      --output result/IRES_prediction/predicted_ires.csv
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from esm.modules import ConvTransformerLayer
from fairseq import checkpoint_utils
from src.ernie_rna.tasks import ernie_rna as _ernie_rna_task  # noqa: F401
from src.ernie_rna.models import ernie_rna as _ernie_rna_model  # noqa: F401
from src.ernie_rna.criterions import ernie_rna as _ernie_rna_criterion  # noqa: F401


def paired(x: int, y: int, lamda: float = 0.8) -> float:
    return {(5, 6): 2, (6, 5): 2, (4, 7): 3, (7, 4): 3,
            (4, 6): lamda, (6, 4): lamda}.get((x, y), 0)


def creatmat(tokens: np.ndarray, base_range: int = 1, lamda: float = 0.8) -> np.ndarray:
    """Build the 2D pairing feature identically to IRES training."""
    pair_map = np.array([[paired(i, j, lamda) for i in range(30)] for j in range(30)])
    indices = np.arange(len(tokens))
    coefficient = np.zeros((len(tokens), len(tokens)))
    score_mask = np.full((len(tokens), len(tokens)), True)
    for offset in range(base_range):
        x, y = indices - offset, indices + offset
        score_mask &= ((x >= 0)[:, None] & (y < len(tokens))[None, :])
        x, y = np.meshgrid(x.clip(0, len(tokens) - 1), y.clip(0, len(tokens) - 1), indexing="ij")
        score = pair_map[tokens[x], tokens[y]]
        score_mask &= score != 0
        coefficient += score * score_mask * math.exp(-0.5 * offset * offset)
        if not score_mask.any():
            break
    score_mask = coefficient > 0
    for offset in range(1, base_range):
        x, y = indices + offset, indices - offset
        score_mask &= ((x < len(tokens))[:, None] & (y >= 0)[None, :])
        x, y = np.meshgrid(x.clip(0, len(tokens) - 1), y.clip(0, len(tokens) - 1), indexing="ij")
        score = pair_map[tokens[x], tokens[y]]
        score_mask &= score != 0
        coefficient += score * score_mask * math.exp(-0.5 * offset * offset)
        if not score_mask.any():
            break
    return coefficient


class IRESClassificationHead(nn.Module):
    def __init__(self, dropout: float, embed_dim: int = 128, nodes: int = 40, heads: int = 16):
        super().__init__()
        self.convtransformer = nn.ModuleList([
            ConvTransformerLayer(embed_dim, embed_dim * 4, heads, 7 - 2 * i,
                                 dropout=dropout, use_esm1b_layer_norm=True)
            for i in range(3)
        ])
        self.mlp_head = nn.Sequential(
            nn.Linear(6 * embed_dim, nodes * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nodes * 4, nodes), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nodes, 2),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.convtransformer:
            x, _ = layer(x, self_attn_padding_mask=pad_mask)
        x = torch.flip(x, dims=[1])
        mask = ~torch.flip(pad_mask, dims=[1]).unsqueeze(2)
        pooled = []
        for frame, frame_mask in zip((x[:, 0::3], x[:, 1::3], x[:, 2::3]),
                                     (mask[:, 0::3], mask[:, 1::3], mask[:, 2::3])):
            pooled.extend((torch.max(frame, dim=1)[0],
                           torch.sum(frame * frame_mask, dim=1) /
                           (torch.sum(frame_mask, dim=1) + 1e-8)))
        return self.mlp_head(torch.cat(pooled, dim=1))


class IRESPredictor(nn.Module):
    def __init__(self, dropout: float = 0.3, embed_dim: int = 128,
                 embedding_dim: int = 768, nodes: int = 40):
        super().__init__()
        self.reductio_module = nn.Linear(embedding_dim, embed_dim)
        self.predictor = IRESClassificationHead(dropout, embed_dim, nodes)

    def forward(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        pad_mask, eos_mask = tokens.eq(1), tokens.eq(2)
        embeddings = self.reductio_module(embeddings)
        embeddings[pad_mask | eos_mask, :] = 0
        return self.predictor(embeddings, pad_mask)


class ERNIEConvFormerIRES(nn.Module):
    def __init__(self, ernie_encoder: nn.Module, dropout: float = 0.3):
        super().__init__()
        self.ernie_encoder = ernie_encoder
        self.predictor = IRESPredictor(dropout=dropout)

    def forward(self, tokens: torch.Tensor, twod_input: torch.Tensor) -> torch.Tensor:
        _, _, out_dict = self.ernie_encoder(tokens, twod_tokens=twod_input, is_twod=True,
                                            extra_only=False, masked_only=False)
        embeddings = out_dict["inner_states"][-1][1:-1].transpose(0, 1)
        return self.predictor(tokens[:, 1:-1], embeddings)


class IRESDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str]], dictionary, sequence_length: int):
        self.records, self.dictionary, self.sequence_length = records, dictionary, sequence_length
        self.tokens = [self._encode(sequence) for _, sequence in records]

    def _encode(self, sequence: str) -> torch.Tensor:
        sequence = sequence.upper().replace("T", "U")
        if not sequence:
            raise ValueError("RNA sequence cannot be empty.")
        if len(sequence) > self.sequence_length:
            raise ValueError(f"Sequence length {len(sequence)} exceeds --sequence-length {self.sequence_length}.")
        invalid = set(sequence) - set("ACGUN")
        if invalid:
            raise ValueError(f"Unsupported bases {sorted(invalid)} in sequence {sequence!r}")
        ids = np.full(self.sequence_length + 2, self.dictionary.pad(), dtype=np.int64)
        ids[0] = self.dictionary.bos()
        ids[1:len(sequence) + 1] = [self.dictionary.pad() if base == "N" else self.dictionary.index(base)
                                    for base in sequence]
        ids[len(sequence) + 1] = self.dictionary.eos()
        return torch.from_numpy(ids).long()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record_id, sequence = self.records[index]
        tokens = self.tokens[index]
        twod = creatmat(tokens.numpy(), base_range=1, lamda=0.8)[:, :, None]
        return record_id, sequence, tokens, torch.from_numpy(twod).float()


def read_records(path: Path, sequence_column: str) -> List[Tuple[str, str]]:
    if path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        records, name, parts = [], None, []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name, parts = line[1:] or f"sequence_{len(records) + 1}", []
            else:
                parts.append(line)
        if name is not None:
            records.append((name, "".join(parts)))
        if not records:
            raise ValueError("No FASTA records found.")
        return records
    frame = pd.read_csv(path)
    if sequence_column not in frame:
        raise ValueError(f"Input CSV must contain a {sequence_column!r} column.")
    ids = frame["id"] if "id" in frame else pd.Series(range(1, len(frame) + 1))
    return [(str(record_id), str(sequence)) for record_id, sequence in zip(ids, frame[sequence_column])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IRES activity with ERNIE-RNA Conv-Former.")
    parser.add_argument("--input", required=True, type=Path, help="CSV (default sequence column: utr) or FASTA input.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model" / "IRES" / "IRES_best.pt")
    parser.add_argument("--pretrained", type=Path,
                        default=PROJECT_ROOT / "checkpoint" / "ERNIE-RNA_checkpoint" / "ERNIE-RNA_pretrain.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result" / "IRES_prediction" / "predicted_ires.csv")
    parser.add_argument("--sequence-column", default="utr")
    parser.add_argument("--dictionary-dir", default="src/dict")
    parser.add_argument("--sequence-length", type=int, default=450,
                        help="Maximum input length used during IRES model training.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.pretrained)], arg_overrides={"data": str(Path(args.dictionary_dir).resolve())})
    model = ERNIEConvFormerIRES(models[0].encoder)
    checkpoint = torch.load(args.model, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.to(device).eval()

    dataset = IRESDataset(read_records(args.input, args.sequence_column), task.dictionary, args.sequence_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.inference_mode():
        for ids, sequences, tokens, twod in loader:
            probabilities = torch.softmax(model(tokens.to(device), twod.to(device)), dim=1)[:, 1].cpu().numpy()
            labels = (probabilities >= 0.5).astype(int)
            rows.extend(zip(ids, sequences, probabilities, labels))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "utr", "ires_probability", "predicted_ires"]).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
