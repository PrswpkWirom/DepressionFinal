import torch
import torch.nn as nn
from .tradMFFNet import MSFastformerEncoder, GatedFusionModule, RPM, AFM, ensure_at_least_one, masked_mean_time
class MFFNetCore(nn.Module):
    """
    (unchanged ... EXCEPT we add `encode()` and make `forward()` call it)
    """
    def __init__(self, d_audio, d_text, d_model=200, ms_layers=3,
                 rpm_channels=256, dropout=0.1, ffn_mult=2.0, num_classes=2):
        super().__init__()
        # Project to common token dim
        self.pa = nn.Linear(d_audio, d_model)
        self.pt = nn.Linear(d_text,  d_model)

        self.ms_audio = MSFastformerEncoder(d_model=d_model, num_layers=ms_layers,
                                            dropout=dropout, ffn_mult=ffn_mult)
        self.ms_text  = MSFastformerEncoder(d_model=d_model, num_layers=ms_layers,
                                            dropout=dropout, ffn_mult=ffn_mult)

        self.gfm = GatedFusionModule(d_model=d_model, hidden_ratio=2.0, dropout=dropout)
        self.rpm = RPM(d_model=d_model, d_pyr=rpm_channels)
        self.afm = AFM(d_model=d_model)

        self.pre_pool_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)  # kept for your binary mode

    def encode(self, audio_embed, text_embed, mask):  # -> (B, D)
        """Return the pooled embedding before the classifier."""
        mask = ensure_at_least_one(mask)
        a = self.pa(audio_embed)
        t = self.pt(text_embed)

        a = self.ms_audio(a, mask)  # (B,T,D)
        t = self.ms_text(t, mask)   # (B,T,D)

        fused = self.gfm(a, t)      # (B,T,D)

        feats6, masks6 = self.rpm(fused, mask)     # 6×(B,T,D), 6×(B,T)
        fused_multi = self.afm(feats6, masks6)     # (B,T,D)

        fused_multi = self.pre_pool_norm(fused_multi)
        pooled = masked_mean_time(fused_multi, mask)  # (B,D)
        return pooled

    def forward(self, audio_embed, text_embed, mask):  # binary path (unchanged behavior)
        pooled = self.encode(audio_embed, text_embed, mask)
        logits = self.fc(pooled)  # (B,C)
        return logits


class DeepSVDDHead(nn.Module):
    """
    Deep-SVDD one-class head.
    - Hard-boundary (default): minimize mean ||z - c||^2 over normal data.
    - Soft-boundary (optional): also learns radius R with a nu parameter.
    """
    def __init__(self, dim, nu=0.1, soft_boundary=False, init_eps=0.1):
        super().__init__()
        self.nu = float(nu)
        self.soft_boundary = bool(soft_boundary)
        self.init_eps = float(init_eps)

        # Center c (not trained by SGD; we initialize once)
        self.register_buffer("c", torch.zeros(dim))
        self.register_buffer("c_initialized", torch.tensor(0, dtype=torch.uint8))

        # Soft-boundary radius (optional)
        if self.soft_boundary:
            self.R = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("R", None)

    @torch.no_grad()
    def init_center(self, model, loader, device, max_batches=None, embed_fn=None):
        """
        Compute the mean embedding over NORMAL-only training data.
        `loader` should yield batches of normal samples.
        If your dataset returns labels, just ignore them or prefilter outside.
        """
        model.eval()
        s = torch.zeros_like(self.c, device=device)
        n = 0
        it = 0
        for batch in loader:
            it += 1
            if max_batches is not None and it > max_batches:
                break
            # --- Expect either (audio, text, mask) or (audio, text, mask, *rest)
            if len(batch) >= 3:
                audio, text, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                raise RuntimeError("Loader must yield (audio, text, mask, ...).")

            z = embed_fn(audio, text, mask) if embed_fn is not None else model.encode(audio, text, mask)
            s += z.sum(dim=0)
            n += z.size(0)

        c = s / max(n, 1)
        # Avoid zero dims (per Deep-SVDD)
        c[(c.abs() < self.init_eps)] = self.init_eps
        self.c.copy_(c.detach())
        self.c_initialized.fill_(1)

    def loss(self, z):
        """
        z: (B, D) embeddings of NORMAL samples
        returns: loss scalar, distances^2 (B,)
        """
        dist2 = torch.sum((z - self.c) ** 2, dim=1)
        if self.soft_boundary:
            # Soft-boundary objective (R learned, nu in (0,1])
            hinge = torch.clamp(dist2 - self.R.clamp_min(0.0) ** 2, min=0.0)
            loss = self.R.clamp_min(0.0) ** 2 + (1.0 / self.nu) * hinge.mean()
        else:
            loss = dist2.mean()
        return loss, dist2

    def score(self, z):
        """Anomaly score (higher = more anomalous)."""
        return torch.sum((z - self.c) ** 2, dim=1)  # (B,)


class MFFNetOneClass(nn.Module):
    """
    Wraps your backbone and the DeepSVDD head.
    Forward returns (scores, embeddings)
    """
    def __init__(self, d_audio, d_text, d_model=200, ms_layers=3, rpm_channels=256,
                 dropout=0.1, ffn_mult=2.0, nu=0.1, soft_boundary=False):
        super().__init__()
        self.backbone = MFFNetCore(d_audio, d_text, d_model, ms_layers,
                                   rpm_channels, dropout, ffn_mult, num_classes=2)
        dim = d_model
        self.head = DeepSVDDHead(dim=dim, nu=nu, soft_boundary=soft_boundary)

    def forward(self, audio_embed, text_embed, mask):
        z = self.backbone.encode(audio_embed, text_embed, mask)  # (B,D)
        scores = self.head.score(z)  # (B,)
        return scores, z
