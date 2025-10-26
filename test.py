"""
Evaluate a trained MFFNet checkpoint on the DAIC-WOZ test split and report
both the custom binary metrics and the scikit-learn classification report.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataloader.tradMFFNet import make_loader
from model.tradMFFNet import MFFNetCore
from train import TrainConfig, binary_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the .ckpt file produced by train.py (best or last).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory (defaults to cfg.dataset_dir saved in the checkpoint).",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Override labels directory (defaults to cfg.labels_dir or <dataset_dir>/label).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation dataloader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. cuda:0). Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to sigmoid probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Optional path to dump results (loss, metrics, classification report) as JSON.",
    )
    return parser.parse_args()


def build_model(cfg: TrainConfig, d_audio: int, d_text: int, device: torch.device) -> MFFNetCore:
    model = MFFNetCore(
        d_audio=d_audio,
        d_text=d_text,
        d_model=cfg.d_model,
        ms_layers=cfg.ms_layers,
        rpm_channels=cfg.rpm_channels,
        dropout=cfg.dropout,
        ffn_mult=cfg.ffn_mult,
        num_classes=cfg.num_classes,
    )
    return model.to(device)


def determine_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path.as_posix(), map_location="cpu")
    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        raise KeyError("Checkpoint is missing the saved TrainConfig under 'config'.")
    cfg = TrainConfig(**cfg_dict)

    dataset_dir = Path(args.dataset_dir or cfg.dataset_dir).expanduser().resolve()
    labels_dir = args.labels_dir or cfg.labels_dir
    if labels_dir is None:
        labels_dir = os.path.join(dataset_dir.as_posix(), "label")
    else:
        labels_dir = Path(labels_dir).expanduser().resolve().as_posix()

    device = determine_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Labels dir:  {labels_dir}")

    _, test_loader = make_loader(
        root_dir=dataset_dir.as_posix(),
        split="test",
        labels_dir=labels_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Peek to infer modality dimensions.
    peek_audio, peek_text, _, _, _ = next(iter(test_loader))
    d_audio = peek_audio.size(-1)
    d_text = peek_text.size(-1)
    print(f"Inferred dims -> d_audio={d_audio}, d_text={d_text}")

    model = build_model(cfg, d_audio, d_text, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pos_weight = checkpoint.get("pos_weight")
    if pos_weight is None:
        pos_weight = torch.tensor([1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    total_loss = 0.0
    probs = []
    trues = []

    with torch.no_grad():
        for audio, text, mask, labels, _ in tqdm(test_loader, desc="Test", leave=False):
            audio = audio.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = labels.float().unsqueeze(1).to(device, non_blocking=True)

            logits = model(audio, text, mask)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * audio.size(0)
            probs.append(torch.sigmoid(logits).squeeze(1).cpu())
            trues.append(labels.cpu())

    avg_loss = total_loss / len(test_loader.dataset)
    y_prob = torch.cat(probs)
    y_true = torch.cat(trues).long()
    metrics = binary_metrics(y_true, y_prob)

    y_pred = (y_prob >= args.threshold).long()
    report_text = classification_report(
        y_true.numpy(),
        y_pred.numpy(),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true.numpy(),
        y_pred.numpy(),
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\n===== Test Set Metrics =====")
    print(f"Loss: {avg_loss:.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print("\n===== Classification Report =====")
    print(report_text)

    if args.save_report is not None:
        payload: Dict[str, Any] = {
            "checkpoint": ckpt_path.as_posix(),
            "loss": avg_loss,
            "metrics": metrics,
            "classification_report": report_dict,
            "threshold": args.threshold,
        }
        out_path = Path(args.save_report).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
