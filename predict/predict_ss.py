#!/usr/bin/env python3
"""Predict RNA secondary structure with the ERNIE Conv-Former ResNet model."""
'''python predict/predict_ss.py \
  --input data/ss_prediction/input.csv \
  --model model/SS/SS_best_model.pt \
  --output result/SS_prediction \
  --pretrained checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt 
'''

import argparse
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from esm.modules import ConvTransformerLayer
from esm.postprocess_ss import postprocess_new
from fairseq import checkpoint_utils
from src.ernie_rna.tasks import ernie_rna as _ernie_rna_task
from src.ernie_rna.models import ernie_rna as _ernie_rna_model
from src.ernie_rna.criterions import ernie_rna as _ernie_rna_criterion


def weights_init_kaiming(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ConvFormerEncoder(nn.Module):
    def __init__(self, dropout: float, n_layers: int = 3, kmer: int = 7,
                 embed_dim: int = 128, heads: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvTransformerLayer(embed_dim, embed_dim * 4, heads, kmer - 2 * i,
                                 dropout=dropout, use_esm1b_layer_norm=True)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=pad_mask)
        return x


class MyBasicResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 groups: int = 1, base_width: int = 64, dilation: int = 1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.dropout(out))
        return self.conv2(out) + identity


class ContactMapDecoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.compress = nn.Conv2d(embed_dim * 2, 1, kernel_size=1)
        self.conv1 = nn.Conv2d(1, 8, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(8, 63, 7, 1, 3)
        blocks = [MyBasicResBlock(64, 64, dilation=2 ** (i % 3)) for i in range(8)]
        self.proj = nn.Sequential(OrderedDict([
            ("resnet", nn.Sequential(*blocks)),
            ("final", nn.Conv2d(64, 1, kernel_size=3, padding=1)),
        ]))
        self.proj.apply(weights_init_kaiming)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, channels = h.shape
        left = h.unsqueeze(2).expand(batch, length, length, channels)
        right = h.unsqueeze(1).expand(batch, length, length, channels)
        pair = torch.cat([left, right], dim=-1).permute(0, 3, 1, 2)
        pair_1ch = self.compress(pair)
        out = self.conv2(self.relu(self.dropout(self.conv1(pair_1ch))))
        logits = self.proj(torch.cat((out, pair_1ch), dim=1))
        return (logits + logits.transpose(-1, -2)) / 2.0


class UTR_ECR_SS(nn.Module):
    def __init__(self, sentence_encoder: nn.Module, dropout: float = 0.2,
                 embed_dim: int = 128, ernie_dim: int = 768):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.reduction = nn.Linear(ernie_dim, embed_dim)
        self.reduction.apply(weights_init_kaiming)
        self.encoder = ConvFormerEncoder(dropout=dropout, embed_dim=embed_dim)
        self.decoder = ContactMapDecoder(embed_dim=embed_dim)

    def forward(self, tokens: torch.Tensor, twod_input: torch.Tensor) -> torch.Tensor:
        pad_mask, eos_mask = tokens.eq(1), tokens.eq(2)
        _, _, out_dict = self.sentence_encoder(
            tokens, twod_tokens=twod_input, is_twod=True, extra_only=True, masked_only=False
        )
        x = out_dict["inner_states"][-1][1:-1].transpose(0, 1)
        x = self.reduction(x)
        inner_pad = pad_mask[:, 1:-1]
        x[inner_pad | eos_mask[:, 1:-1]] = 0.0
        return self.decoder(self.encoder(x, inner_pad))


def get_cut_len(length: int, minimum: int = 80) -> int:
    return minimum if length <= minimum else ((length - 1) // 16 + 1) * 16


def pairing_matrix(tokens: np.ndarray, lamda: float = 0.8) -> np.ndarray:
    pairs = {(5, 6): 2, (6, 5): 2, (4, 7): 3, (7, 4): 3,
             (4, 6): lamda, (6, 4): lamda}
    table = np.array([[pairs.get((i, j), 0) for i in range(30)] for j in range(30)])
    index = np.arange(len(tokens))
    coeff = np.zeros((len(tokens), len(tokens)))
    mask = np.ones((len(tokens), len(tokens)), dtype=bool)
    for offset in range(1):
        x, y = index - offset, index + offset
        mask &= (x >= 0)[:, None] & (y < len(tokens))[None, :]
        grid_x, grid_y = np.meshgrid(x.clip(0, len(tokens) - 1), y.clip(0, len(tokens) - 1), indexing="ij")
        score = table[tokens[grid_x], tokens[grid_y]]
        mask &= score != 0
        coeff += score * mask * math.exp(-0.5 * offset * offset)
        if not mask.any():
            break
    return coeff


def read_records(path: Path, sequence_column: str) -> List[Tuple[str, str]]:
    if path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        records, name, pieces = [], None, []
        for raw in path.read_text().splitlines():
            line = raw.strip()
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
    frame = pd.read_csv(path)
    if sequence_column not in frame:
        raise ValueError(f"Input CSV must contain a {sequence_column!r} column.")
    ids = frame["id"] if "id" in frame else pd.Series(range(1, len(frame) + 1))
    return [(str(record_id), str(sequence)) for record_id, sequence in zip(ids, frame[sequence_column])]


def encode(sequence: str, dictionary, padded_length: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
    normalized = sequence.upper().replace("T", "U")
    if not normalized:
        raise ValueError("RNA sequence cannot be empty.")
    invalid = set(normalized) - set("ACGUN")
    if invalid:
        raise ValueError(f"Unsupported bases {sorted(invalid)} in sequence {sequence!r}")
    token_ids = [dictionary.bos()]
    token_ids += [dictionary.pad() if base == "N" else dictionary.index(base) for base in normalized]
    token_ids += [dictionary.eos()] + [dictionary.pad()] * (padded_length - len(normalized))
    tokens = torch.tensor(token_ids, dtype=torch.long)
    onehot = np.zeros((len(normalized), 4), dtype=np.float32)
    for i, base in enumerate(normalized):
        if base in "AUCG":
            onehot[i, "AUCG".index(base)] = 1.0
    return normalized, tokens, torch.from_numpy(onehot)


def select_pairs(contact_map: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    candidates = [(float(contact_map[i, j]), i, j)
                  for i in range(len(contact_map)) for j in range(i + 1, len(contact_map))
                  if contact_map[i, j] > threshold]
    used, selected = set(), []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used and j not in used:
            selected.append((i, j))
            used.update((i, j))
    return selected


def write_ct(path: Path, name: str, sequence: str, pairs: List[Tuple[int, int]]) -> None:
    partner = [0] * len(sequence)
    for i, j in pairs:
        partner[i], partner[j] = j + 1, i + 1
    lines = [f"{len(sequence)} {name}"]
    for i, base in enumerate(sequence, start=1):
        lines.append(f"{i} {base} {i - 1 if i > 1 else 0} {i + 1 if i < len(sequence) else 0} {partner[i - 1]} {i}")
    path.write_text("\n".join(lines) + "\n")


def safe_name(name: str, index: int) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name).strip("_")
    return cleaned or f"sequence_{index}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RNA secondary structure with UTR-ECR SS ResNet.")
    parser.add_argument("--input", required=True, type=Path, help="FASTA or CSV input (CSV needs an utr column).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result" / "ss_predictions")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model" / "SS" / "SS_best_model.pt")
    parser.add_argument("--pretrained", type=Path,
                        default=PROJECT_ROOT / "checkpoint" / "ERNIE-RNA_checkpoint" / "ERNIE-RNA_pretrain.pt")
    parser.add_argument("--dictionary-dir", type=Path, default=PROJECT_ROOT / "src" / "dict")
    parser.add_argument("--sequence-column", default="utr")
    parser.add_argument("--max-length", type=int, default=600, help="Maximum supported RNA length.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Postprocessed contact-map threshold.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.pretrained)], arg_overrides={"data": str(args.dictionary_dir.resolve())}
    )
    model = UTR_ECR_SS(models[0].encoder)
    state = torch.load(args.model, map_location=device)
    state = state.get("state_dict", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    with torch.inference_mode():
        for index, (record_id, raw_sequence) in enumerate(read_records(args.input, args.sequence_column), start=1):
            if len(raw_sequence) > args.max_length:
                raise ValueError(f"{record_id}: length {len(raw_sequence)} exceeds --max-length {args.max_length}.")
            padded_length = get_cut_len(len(raw_sequence))
            normalized, tokens, onehot = encode(raw_sequence, task.dictionary, padded_length)
            twod = pairing_matrix(tokens.numpy())[:, :, None]
            logits = model(tokens.unsqueeze(0).to(device), torch.from_numpy(twod).float().unsqueeze(0).to(device))
            logits = logits[:, 0, :len(normalized), :len(normalized)]
            contact = postprocess_new(logits, onehot.unsqueeze(0).to(device), 0.01, 0.1, 100, 1.6, True, math.log(9.0))
            contact_np = contact[0].cpu().numpy()
            pairs = select_pairs(contact_np, args.threshold)
            stem = safe_name(record_id, index)
            write_ct(args.output / f"{stem}_predicted.ct", record_id, normalized, pairs)
            summary.append({"id": record_id, "rna": normalized, "length": len(normalized),
                            "num_base_pairs": len(pairs), "ct_file": f"{stem}_predicted.ct"})

    pd.DataFrame(summary).to_csv(args.output / "ss_prediction_summary.csv", index=False)
    print(f"Saved {len(summary)} predictions to {args.output}")


if __name__ == "__main__":
    main()
