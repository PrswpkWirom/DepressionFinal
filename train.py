import os, json, time, shutil, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict


import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataloader.tradMFFNet import make_loader, collate_mffnet
#from dataloader.tsaug_tradMFFNet import make_loader, collate_mffnet, RepeatAugmentDataset
from model.tradMFFNet import MFFNetCore  # your model file

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
# Metrics (binary)
# ---------------------------
@torch.no_grad()
def binary_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).long()
    N = y_true.numel()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(N, 1)

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
        "acc": (tp + tn) / tot,
        "precision_w": (p_pos * n_pos + p_neg * n_neg) / tot,
        "recall_w":    (r_pos * n_pos + r_neg * n_neg) / tot,
        "f1_w":        (f1_pos * n_pos + f1_neg * n_neg) / tot,
        "pos_support": float(n_pos),
        "neg_support": float(n_neg),
    }


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
    num_classes: int = 1  # single logit -> BCEWithLogits

    # training
    batch_size: int = 16
    max_epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    amp: bool = True
    early_stop_patience: int = 20

    # io
    exp_name: str = "mffnet_daicwoz_bce"
    device: str = "cuda:0"


# ---------------------------
# Loss weight from train set
# ---------------------------
def compute_pos_weight(train_loader) -> torch.Tensor:
    pos = 0
    neg = 0
    for _, _, _, labels, _ in train_loader:
        labels = labels.view(-1)
        pos += int((labels == 1).sum())
        neg += int((labels == 0).sum())
    pos = max(pos, 1)
    return torch.tensor([neg / pos], dtype=torch.float32)


# ---------------------------
# Epoch loops
# ---------------------------
def train_epoch(model, loader, opt, scaler, loss_fn, device, cfg: TrainConfig):
    model.train()
    total = 0.0
    probs, trues = [], []

    for audio, text, mask, labels, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device, non_blocking=True)
        text  = text.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)
        y     = labels.float().unsqueeze(1).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if cfg.amp:
            with amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(audio, text, mask)  # (B,1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if cfg.grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(audio, text, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()

        total += loss.item() * audio.size(0)
        probs.append(torch.sigmoid(logits).squeeze(1).detach().cpu())
        trues.append(labels.cpu())

    loss_epoch = total / len(loader.dataset)
    y_prob = torch.cat(probs)
    y_true = torch.cat(trues).long()
    return loss_epoch, binary_metrics(y_true, y_prob)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    probs, trues = [], []

    for audio, text, mask, labels, _ in tqdm(loader, desc="Eval", leave=False):
        audio = audio.to(device, non_blocking=True)
        text  = text.to(device, non_blocking=True)
        mask  = mask.to(device, non_blocking=True)
        y     = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(audio, text, mask)
        loss = loss_fn(logits, y)
        total += loss.item() * audio.size(0)
        probs.append(torch.sigmoid(logits).squeeze(1).cpu())
        trues.append(labels.cpu())

    loss_epoch = total / len(loader.dataset)
    y_prob = torch.cat(probs)
    y_true = torch.cat(trues).long()
    return loss_epoch, binary_metrics(y_true, y_prob)


# ---------------------------
# Train entry
# ---------------------------
def main(cfg: TrainConfig):
    set_seed(1337)

    # ---- device (force CUDA:0) ----
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    print(device)
    torch.cuda.set_device(0)

    # ---- experiment directory under /home/popsatorn/depressionFinalProject ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = OUTPUT_ROOT / f"{cfg.exp_name}_{ts}"
    print("Experiment dir:", exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir.as_posix())

    # ---- data ----
    labels_dir = cfg.labels_dir or os.path.join(cfg.dataset_dir, "label")
    train_ds, train_dl = make_loader(cfg.dataset_dir, "train", labels_dir=labels_dir,
                                     batch_size=cfg.batch_size, shuffle=True, augment_train=False)
    base_train_ds, _ = make_loader(
        cfg.dataset_dir, "train",
        labels_dir=labels_dir,
        batch_size=cfg.batch_size, shuffle=True,
        augment_train=True,      # ensure base has an augmenter
    )

    # # Expand: keep original + 2 augmented views -> 3× dataset
    # train_ds = RepeatAugmentDataset(base_train_ds, k=2, keep_original=True)

    # # Now make a loader from the expanded dataset
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=collate_mffnet,
    #     drop_last=False,
    #     worker_init_fn=lambda _: (np.random.seed(torch.initial_seed() % 2**32), random.seed(torch.initial_seed() % 2**32)),
    # )

    val_ds,   val_dl   = make_loader(cfg.dataset_dir, "validate", labels_dir=labels_dir,
                                     batch_size=cfg.batch_size, shuffle=False, augment_train=False)

    # infer dims
    sample_audio, sample_text, _, _, _ = next(iter(train_dl))
    d_audio = sample_audio.size(-1)
    d_text  = sample_text.size(-1)
    print(f"Data dims: d_audio={d_audio}, d_text={d_text}")
    # ---- model ----
    model = MFFNetCore(
        d_audio=d_audio, d_text=d_text, d_model=cfg.d_model,
        ms_layers=cfg.ms_layers, rpm_channels=cfg.rpm_channels,
        dropout=cfg.dropout, ffn_mult=cfg.ffn_mult, num_classes=cfg.num_classes
    ).to(device)
    print("number of model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---- loss (weighted BCE) ----
    pos_weight = compute_pos_weight(train_dl).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = amp.GradScaler(enabled=cfg.amp and not torch.cuda.is_bf16_supported())

    # ---- save config (for exact reuse) ----
    with open(exp_dir / "config.json", "w") as f:
        json.dump({**asdict(cfg),
                   "dataset_dir": cfg.dataset_dir,
                   "labels_dir": labels_dir,
                   "d_audio": d_audio, "d_text": d_text,
                   "pos_weight": float(pos_weight.item()),
                   "train_size": len(train_ds), "val_size": len(val_ds)},
                  f, indent=2)

    best_f1 = -1.0
    best_ckpt = exp_dir / "best.ckpt"
    last_ckpt = exp_dir / "last.ckpt"
    patience = cfg.early_stop_patience
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_m = train_epoch(model, train_dl, optimizer, scaler, loss_fn, device, cfg)
        val_loss,   val_m   = eval_epoch(model,   val_dl,   loss_fn, device)
        scheduler.step(val_loss)

        # logs
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("metric/train_f1_w", train_m["f1_w"], epoch)
        writer.add_scalar("metric/val_f1_w",   val_m["f1_w"],   epoch)
        writer.add_scalar("metric/val_acc",    val_m["acc"],    epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        print(f"[{epoch:03d}] Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"F1_w {val_m['f1_w']:.4f} | Val Acc {val_m['acc']:.4f} | Train Acc {train_m['acc']:.4f}")

        # save last (resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if cfg.amp else None,
            "pos_weight": pos_weight,
            "config": asdict(cfg),
            "val_metrics": val_m,
        }, last_ckpt.as_posix())

        # save best (by weighted F1)
        if val_m["f1_w"] > best_f1:
            best_f1 = val_m["f1_w"]; no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if cfg.amp else None,
                "pos_weight": pos_weight,
                "config": asdict(cfg),
                "val_metrics": val_m,
            }, best_ckpt.as_posix())

            # export lightweight for reuse
            export_dir = exp_dir / "export"
            export_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), (export_dir / "best_model_state.pt").as_posix())
            with open(export_dir / "inference_meta.json", "w") as f:
                json.dump({
                    "d_audio": d_audio, "d_text": d_text, "d_model": cfg.d_model,
                    "num_classes": cfg.num_classes, "epoch_best": epoch,
                    "pos_weight": float(pos_weight.item())
                }, f, indent=2)
            print(f"  ✓ Saved BEST to {best_ckpt}")
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f"Early stop (patience={patience}). Best F1_w={best_f1:.4f}")
            break

    writer.close()
    print("Experiment dir:", exp_dir)
    print("Best checkpoint:", best_ckpt)


if __name__ == "__main__":
    cfg = TrainConfig(
        dataset_dir="DAIC-WOZ/DAIC_embeddings",
        labels_dir="DepressionFinal/label",                      # uses <dataset_dir>/label
        exp_name="trad_mffnet_tsuag",       # will be prefixed to output folder
        device="cuda:0",
        batch_size=32,
        max_epochs=600,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        num_classes=1,           # 2 classes use 1 logit -> BCEWithLogits
        d_model = 64,
        ms_layers = 1,
        rpm_channels = 64,
        ffn_mult = 2.0,
        early_stop_patience=50,
    )
    main(cfg)