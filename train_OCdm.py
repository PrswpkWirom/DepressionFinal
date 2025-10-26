import os, json, time, shutil, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# === your modules (updated to support one-class + MFFNetOneClass) ===
from dataloader.OCdm import make_loader           # uses one_class_label
from model.OCdm import MFFNetOneClass             # backbone + DeepSVDD head

# ---------------------------
# Paths (outputs fixed to /home/popsatorn/depressionFinalProject)
# ---------------------------
THIS_FILE = Path(__file__).resolve()
REPO_DIR  = THIS_FILE.parent                                  # .../DepressionFinal
OUTPUT_ROOT = REPO_DIR.parent               # /home/popsatorn/depressionFinalProject
OUTPUT_ROOT = OUTPUT_ROOT / "output"


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------
# Metrics (binary, weighted like your original)
# ---------------------------
def _metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true: (N,) 0/1
    y_pred: (N,) 0/1
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    N = max(len(y_true), 1)
    acc = (tp + tn) / N

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return p, r, f1

    p_pos, r_pos, f1_pos = prf(tp, fp, fn)
    p_neg, r_neg, f1_neg = prf(tn, fn, fp)

    n_pos = tp + fn
    n_neg = tn + fp
    tot   = max(n_pos + n_neg, 1)

    return {
        "acc": acc,
        "precision_w": (p_pos * n_pos + p_neg * n_neg) / tot,
        "recall_w":    (r_pos * n_pos + r_neg * n_neg) / tot,
        "f1_w":        (f1_pos * n_pos + f1_neg * n_neg) / tot,
        "pos_support": float(n_pos),
        "neg_support": float(n_neg),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def _best_threshold_by_f1(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Grid-search threshold over unique scores (distance^2). Predict 1 if score >= thr.
    Returns (best_thr, metrics_dict)
    """
    scores = scores.astype(float)
    y_true = y_true.astype(int)

    # Handle degenerate case
    uniq = np.unique(scores)
    if uniq.size == 1:
        thr = float(uniq[0])
        y_pred = (scores >= thr).astype(int)
        return thr, _metrics_from_pred(y_true, y_pred)

    best = {"f1_w": -1.0}
    best_thr = float(uniq[0])
    # Evaluate at unique score cutpoints
    for thr in uniq:
        y_pred = (scores >= thr).astype(int)
        m = _metrics_from_pred(y_true, y_pred)
        if m["f1_w"] > best["f1_w"]:
            best = m
            best_thr = float(thr)
    return best_thr, best


# ---------------------------
# Config
# ---------------------------
@dataclass
class TrainConfig:
    # data
    dataset_dir: str = "DAIC-WOZ/DAIC_embeddings"
    labels_dir: str  = None  # default: <dataset_dir>/label

    # model
    d_model: int = 200
    ms_layers: int = 1
    rpm_channels: int = 64
    dropout: float = 0.3
    ffn_mult: float = 2.0

    # Deep-SVDD head
    nu: float = 0.1
    soft_boundary: bool = False   # start with hard boundary for stability

    # training
    batch_size: int = 16
    max_epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    early_stop_patience: int = 20

    # fragments (optional)
    use_fragments: bool = False
    seg_len: int = 64
    pad_threshold: float = 0.2

    # io
    exp_name: str = "mffnet_svdd"
    device: str = "cuda:0"


# ---------------------------
# Epoch loops (Deep-SVDD)
# ---------------------------
def train_epoch_svdd(model, loader, opt, scaler, device, cfg: TrainConfig):
    """
    Train on NORMAL-only batches (label 0).
    Optimizes Deep-SVDD loss over embeddings.
    """
    model.train()
    total_loss = 0.0
    total_count = 0
    mean_dist2 = 0.0

    for audio, text, mask, labels, _ in tqdm(loader, desc="Train", leave=False):
        # (defense) ensure normals-only loader
        if (labels != 0).any():
            # If misconfigured, filter in-batch
            keep = (labels == 0)
            audio, text, mask = audio[keep], text[keep], mask[keep]
            if audio.numel() == 0:
                continue

        audio = audio.to(device, non_blocking=True)
        text  = text.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if cfg.amp:
            with amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                z = model.backbone.encode(audio, text, mask)  # (B,D)
                loss, dist2 = model.head.loss(z)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if cfg.grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
        else:
            z = model.backbone.encode(audio, text, mask)
            loss, dist2 = model.head.loss(z)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()

        bs = audio.size(0)
        total_loss += float(loss.item()) * bs
        mean_dist2 += float(dist2.mean().item()) * bs
        total_count += bs

    loss_epoch = total_loss / max(total_count, 1)
    mean_d2 = mean_dist2 / max(total_count, 1)
    return loss_epoch, {"mean_dist2": mean_d2}


@torch.no_grad()
def eval_epoch_scores(model, loader, device):
    """
    Compute anomaly scores (distance^2) on a (mixed) loader.
    Returns scores (N,), labels (N,)
    """
    model.eval()
    all_scores = []
    all_labels = []

    for audio, text, mask, labels, _ in tqdm(loader, desc="Eval", leave=False):
        audio = audio.to(device, non_blocking=True)
        text  = text.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)

        scores, _ = model(audio, text, mask)  # (B,)
        all_scores.append(scores.detach().cpu().float())
        all_labels.append(labels.cpu().long())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return scores, labels


# ---------------------------
# Train entry
# ---------------------------
def main(cfg: TrainConfig):
    set_seed(1337)

    # ---- device ----
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device(cfg.device)
    print(device)
    torch.cuda.set_device(int(cfg.device.split(":")[-1]))

    # ---- experiment directory under /home/popsatorn/depressionFinalProject ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = OUTPUT_ROOT / f"{cfg.exp_name}_{ts}"
    print("Experiment dir:", exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir.as_posix())

    # ---- data ----
    labels_dir = cfg.labels_dir or os.path.join(cfg.dataset_dir, "label")

    # TRAIN on normals only (label 0)
    _, train_dl_normals = make_loader(
        root_dir=cfg.dataset_dir,
        split="train",
        labels_dir=labels_dir,
        batch_size=cfg.batch_size,
        shuffle=True,
        one_class_label=0,                 # <<< keep only normal
        use_fragments=cfg.use_fragments,
        seg_len=cfg.seg_len,
        pad_threshold=cfg.pad_threshold,
    )

    # VALIDATE on full set (0 + 1) to pick threshold
    _, val_dl_all = make_loader(
        root_dir=cfg.dataset_dir,
        split="validate",
        labels_dir=labels_dir,
        batch_size=cfg.batch_size,
        shuffle=False,
        one_class_label=None,              # <<< keep all labels
        use_fragments=cfg.use_fragments,
        seg_len=cfg.seg_len,
        pad_threshold=cfg.pad_threshold,
    )

    # infer dims
    sample_audio, sample_text, _, _, _ = next(iter(train_dl_normals))
    d_audio = sample_audio.size(-1)
    d_text  = sample_text.size(-1)
    print(f"Data dims: d_audio={d_audio}, d_text={d_text}")

    # ---- model ----
    model = MFFNetOneClass(
        d_audio=d_audio, d_text=d_text, d_model=cfg.d_model,
        ms_layers=cfg.ms_layers, rpm_channels=cfg.rpm_channels,
        dropout=cfg.dropout, ffn_mult=cfg.ffn_mult,
        nu=cfg.nu, soft_boundary=cfg.soft_boundary
    ).to(device)
    print("number of model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---- optimizer & scaler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = amp.GradScaler(enabled=cfg.amp and not torch.cuda.is_bf16_supported())

    # ---- initialize center c with normal-only loader ----
    model.head.init_center(model=model.backbone, loader=train_dl_normals, device=device)

    # ---- save config (for exact reuse) ----
    with open(exp_dir / "config.json", "w") as f:
        json.dump({**asdict(cfg),
                   "dataset_dir": cfg.dataset_dir,
                   "labels_dir": labels_dir,
                   "d_audio": d_audio, "d_text": d_text},
                  f, indent=2)

    best_f1 = -1.0
    best_thr = None
    best_ckpt = exp_dir / "best.ckpt"
    last_ckpt = exp_dir / "last.ckpt"
    patience = cfg.early_stop_patience
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # ---- train one epoch on normals
        train_loss, train_stats = train_epoch_svdd(model, train_dl_normals, optimizer, scaler, device, cfg)

        # optional: update soft-boundary radius R by (1 - nu) quantile of train dist^2
        if cfg.soft_boundary:
            model.eval()
            all_d2 = []
            with torch.no_grad():
                for audio, text, mask, labels, _ in train_dl_normals:
                    audio = audio.to(device); text = text.to(device); mask = mask.to(device)
                    z = model.backbone.encode(audio, text, mask)
                    _, d2 = model.head.loss(z)
                    all_d2.append(d2)
            d2_cat = torch.cat(all_d2)
            R_new = torch.sqrt(torch.quantile(d2_cat, q=1.0 - model.head.nu))
            model.head.R.data = R_new

        # ---- evaluate on full validation set (search best threshold)
        val_scores, val_labels = eval_epoch_scores(model, val_dl_all, device)
        thr, val_m = _best_threshold_by_f1(val_scores, val_labels)

        scheduler.step(train_loss)  # (keep "minimize loss" scheduling)

        # ---- logs
        writer.add_scalar("loss/train_svdd", train_loss, epoch)
        writer.add_scalar("train/mean_dist2", train_stats["mean_dist2"], epoch)
        writer.add_scalar("metric/val_f1_w",   val_m["f1_w"], epoch)
        writer.add_scalar("metric/val_acc",    val_m["acc"],  epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("threshold/val_best", thr, epoch)
        writer.flush()

        print(f"[{epoch:03d}] SVDD Loss {train_loss:.4f} | mean_d2 {train_stats['mean_dist2']:.4f} | "
              f"Val F1_w {val_m['f1_w']:.4f} | Val Acc {val_m['acc']:.4f} | thr {thr:.6f}")

        # ---- save last (resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if cfg.amp else None,
            "config": asdict(cfg),
            "val_metrics": val_m,
            "best_threshold": thr,
        }, last_ckpt.as_posix())

        # ---- save best (by weighted F1)
        if val_m["f1_w"] > best_f1:
            best_f1 = val_m["f1_w"]; best_thr = float(thr); no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if cfg.amp else None,
                "config": asdict(cfg),
                "val_metrics": val_m,
                "best_threshold": best_thr,
            }, best_ckpt.as_posix())

            # export lightweight for reuse
            export_dir = exp_dir / "export"
            export_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), (export_dir / "best_model_state.pt").as_posix())
            with open(export_dir / "inference_meta.json", "w") as f:
                json.dump({
                    "d_audio": d_audio, "d_text": d_text, "d_model": cfg.d_model,
                    "nu": cfg.nu, "soft_boundary": cfg.soft_boundary,
                    "best_threshold": best_thr,
                    "epoch_best": epoch
                }, f, indent=2)
            print(f"  ✓ Saved BEST to {best_ckpt}")
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f"Early stop (patience={patience}). Best F1_w={best_f1:.4f} at thr={best_thr}")
            break

    writer.close()
    print("Experiment dir:", exp_dir)
    print("Best checkpoint:", best_ckpt)
    if best_thr is not None:
        print("Best validation threshold (score ≥ thr => anomalous):", best_thr)


if __name__ == "__main__":
    cfg = TrainConfig(
        dataset_dir="DAIC-WOZ/DAIC_embeddings",
        labels_dir="DepressionFinal/label",
        exp_name="trad_mffnet_svdd",
        device="cuda:0",
        batch_size=32,
        max_epochs=300,
        lr=2e-4,
        weight_decay=1e-4,
        dropout=0.3,
        d_model=128,
        ms_layers=1,
        rpm_channels=128,
        ffn_mult=2.0,
        early_stop_patience=40,
        nu=0.1,
        soft_boundary=True,
        use_fragments=False,   # set True to explode data with segments
        seg_len=64,
        pad_threshold=0.2,
    )
    main(cfg)
