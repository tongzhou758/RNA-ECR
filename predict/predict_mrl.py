#!/usr/bin/env python3
'''
cd /Users/tongtong/Desktop/UTR/初稿/RNA-ECR-main
python predict/predict_mrl.py \
  --input data/MRL_prediction/input.csv \
  --model model/MRL/MRL_best_model_50.pt \
  --output result/MRL_prediction/predicted_mrl.csv \
  --scaler model/MRL/scaler.save \
  --pretrained checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt
'''

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
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


SEQ_LEN = 50


def paired(x: int, y: int, lamda: float = 0.8) -> float:
    """Return the base-pair score used by the training pipeline."""
    scores = {(5, 6): 2, (6, 5): 2, (4, 7): 3, (7, 4): 3,
              (4, 6): lamda, (6, 4): lamda}
    return scores.get((x, y), 0)


def creatmat(tokens: np.ndarray, base_range: int = 1, lamda: float = 0.8) -> np.ndarray:
    """Build the 2D pairing feature exactly as in the 50-nt training code."""
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


class ConvTransformerPredictor(nn.Module):
    def __init__(self, dropout: float, embed_dim: int = 128, nodes: int = 40, heads: int = 16):
        super().__init__()
        self.convtransformer_decoder = nn.ModuleList([
            ConvTransformerLayer(embed_dim, embed_dim * 4, heads, 7 - 2 * i,
                                 dropout=dropout, use_esm1b_layer_norm=True)
            for i in range(3)
        ])
        self.mlp_head = nn.Sequential(
            nn.Linear(6 * embed_dim, nodes * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nodes * 4, nodes), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nodes, 1),
        )

    def forward(self, seqs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = seqs
        for layer in self.convtransformer_decoder:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)

        x = torch.flip(x, dims=[1])
        mask = ~torch.flip(padding_mask, dims=[1]).unsqueeze(2)
        frames = [x[:, i::3, :] for i in range(3)]
        frame_masks = [mask[:, i::3, :] for i in range(3)]
        pooled = []
        for frame, frame_mask in zip(frames, frame_masks):
            pooled.append(torch.max(frame, dim=1)[0])
            pooled.append(torch.sum(frame * frame_mask, dim=1) /
                          (torch.sum(frame_mask, dim=1) + 1e-8))
        return self.mlp_head(torch.cat(pooled, dim=1))


class UTRRegression(nn.Module):
    def __init__(self, dropout: float = 0.3, embed_dim: int = 128, embedding_dim: int = 768):
        super().__init__()
        self.reductio_module = nn.Linear(embedding_dim, embed_dim)
        self.predictor = ConvTransformerPredictor(dropout=dropout, embed_dim=embed_dim)

    def forward(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        padding_mask, eos_mask = tokens.eq(1), tokens.eq(2)
        embeddings = self.reductio_module(embeddings)
        embeddings[padding_mask | eos_mask, :] = 0
        return self.predictor(embeddings, padding_mask)


class MRL50Model(nn.Module):
    def __init__(self, sentence_encoder: nn.Module):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.head = UTRRegression()

    def forward(self, tokens: torch.Tensor, twod_tokens: torch.Tensor) -> torch.Tensor:
        inner_tokens = tokens[:, 1:-1]
        _, _, output = self.sentence_encoder(tokens, twod_tokens=twod_tokens, is_twod=True,
                                             extra_only=True, masked_only=False)
        embeddings = output["inner_states"][-1][1:-1].transpose(0, 1)
        return self.head(inner_tokens, embeddings)


class PredictionDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str]], dictionary):
        self.records, self.dictionary = records, dictionary
        self.tokens = [self._encode(sequence) for _, sequence in records]

    def _encode(self, sequence: str) -> torch.Tensor:
        sequence = sequence.upper().replace("T", "U")
        if len(sequence) != SEQ_LEN:
            raise ValueError(f"Expected exactly {SEQ_LEN} nt, got {len(sequence)} nt: {sequence!r}")
        allowed = set("ACGUN")
        invalid = set(sequence) - allowed
        if invalid:
            raise ValueError(f"Unsupported bases {sorted(invalid)} in sequence {sequence!r}")
        token_ids = [self.dictionary.bos()]
        token_ids.extend(self.dictionary.pad() if base == "N" else self.dictionary.index(base)
                         for base in sequence)
        token_ids.append(self.dictionary.eos())
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        twod = creatmat(tokens.numpy(), base_range=1, lamda=0.8).T[..., None]
        record_id, sequence = self.records[index]
        return record_id, sequence, tokens, torch.from_numpy(twod).float()


def read_records(input_path: Path, sequence_column: str) -> List[Tuple[str, str]]:
    if input_path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        records, name, pieces = [], None, []
        for raw_line in input_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(pieces)))
                name, pieces = line[1:] or f"sequence_{len(records) + 1}", []
            else:
                pieces.append(line)
        if name is not None:
            records.append((name, "".join(pieces)))
        if not records:
            raise ValueError("No FASTA records found.")
        return records

    frame = pd.read_csv(input_path)
    if sequence_column not in frame:
        raise ValueError(f"Input CSV must have a {sequence_column!r} column.")
    ids = frame["id"] if "id" in frame else pd.Series(range(1, len(frame) + 1))
    return [(str(record_id), str(sequence)) for record_id, sequence in zip(ids, frame[sequence_column])]


def load_finetuned_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict MRL from 50-nt UTR sequences with ERNIE Conv-Former.")
    parser.add_argument("--input", required=True, type=Path, help="CSV (with utr column) or FASTA input.")
    parser.add_argument("--model", required=True, type=Path, help="Fine-tuned 50-nt model .pt file.")
    parser.add_argument("--scaler", required=True, type=Path, help="joblib StandardScaler fitted during the matching training fold.")
    parser.add_argument("--pretrained", required=True, type=Path, help="ERNIE-RNA_pretrain.pt checkpoint.")
    parser.add_argument("--output", default=PROJECT_ROOT / "result" / "mrl_predictions.csv", type=Path,
                        help="Output CSV path (default: result/mrl_predictions.csv).")
    parser.add_argument("--sequence-column", default="utr")
    parser.add_argument("--dictionary-dir", default="src/dict",
                        help="ERNIE-RNA vocabulary directory (normally RNA-ECR-main/src/dict).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    overrides = {"data": str(Path(args.dictionary_dir).resolve())}
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([str(args.pretrained)], arg_overrides=overrides)
    model = MRL50Model(models[0].encoder)
    load_finetuned_weights(model, args.model, device)
    model.to(device).eval()

    dataset = PredictionDataset(read_records(args.input, args.sequence_column), task.dictionary)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    scaler = joblib.load(args.scaler)
    output_rows = []
    with torch.inference_mode():
        for ids, sequences, tokens, twod in loader:
            standardized = model(tokens.to(device), twod.to(device)).cpu().numpy()
            predictions = scaler.inverse_transform(standardized).reshape(-1)
            output_rows.extend(zip(ids, sequences, predictions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=["id", "utr", "predicted_mrl"]).to_csv(args.output, index=False)
    print(f"Saved {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
