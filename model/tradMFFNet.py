# MFFNet core (PyTorch) — MSfastformer → GFM → RPM → AFM → classifier
# Variable-length SAFE: pass a boolean mask (B, T) where True = valid token.
#
# Inputs:
#   audio_embed: FloatTensor (B, T, d_audio)
#   text_embed : FloatTensor (B, T, d_text)
#   mask       : BoolTensor  (B, T)  True for real positions, False for pad
#
# Output:
#   logits     : FloatTensor (B, num_classes)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Helpers & Basic Modules
# =========================

def ensure_at_least_one(mask: torch.Tensor) -> torch.Tensor:
    """
    Ensure each sequence has at least one True entry. If a row is all False,
    flip its first position to True to avoid NaNs in masked softmax/means.
    """
    assert mask.dim() == 2, "mask must be (B,T)"
    out = mask.clone()
    bad = ~out.any(dim=1)
    if bad.any():
        out[bad, 0] = True
    return out

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    scores: (B,T,1) or (B,T,D) ; mask: (B,T) with True = keep.
    Applies softmax along 'dim', ignoring masked positions.
    """
    # widen mask to broadcast
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(-1)
    scores = scores.masked_fill(~mask, -1e30)
    return torch.softmax(scores, dim=dim)

def masked_mean_time(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B,T,D), mask: (B,T). Returns (B,D) mean over valid time steps.
    """
    m = mask.unsqueeze(-1).float()
    num = (x * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1e-6)
    return num / den

def _conv1d_same(in_ch, out_ch, k, stride=1, groups=1):
    """1D conv with 'same' padding for odd kernels."""
    pad = (k - 1) // 2
    return nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, groups=groups, bias=True)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, mult=2.0, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(mult*d_model)),
            nn.GELU(),
            nn.Linear(int(mult*d_model), d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):  # (B,T,D)
        return self.net(x)


class CrossFastformer(nn.Module):
    """
    Additive-attention Fastformer across Q, K, V with masking.
    (B,T,D) -> (B,T,D)
    """
    def __init__(self, d_model, dropout=0.0, share_qv=False):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model, bias=True)
        self.proj_k = nn.Linear(d_model, d_model, bias=True)
        # Optional query-value sharing as suggested in the paper
        self.proj_v = self.proj_q if share_qv else nn.Linear(d_model, d_model, bias=True)

        # Two separate additive-attention scorers (w_q, w_k)
        self.scalar_q = nn.Linear(d_model, 1, bias=False)
        self.scalar_k = nn.Linear(d_model, 1, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, Q, K, V, mask):  # mask: (B,T) bool
        mask = ensure_at_least_one(mask)

        Qp, Kp, Vp = self.proj_q(Q), self.proj_k(K), self.proj_v(V)

        # 1) additive attention over Q -> global q
        o = masked_softmax(self.scalar_q(Qp) / self.scale, mask, dim=1)   # (B,T,1)
        q_global = (o * Qp).sum(dim=1)                                    # (B,D)

        # 2) interact with K via ⊙ and additive attention -> global k
        M = Kp * q_global.unsqueeze(1)                                    # (B,T,D)
        a = masked_softmax(self.scalar_k(M) / self.scale, mask, dim=1)    # (B,T,1)
        k_global = (a * M).sum(dim=1)                                     # (B,D)

        # 3) interact with V via ⊙, linear out; 4) residual add with Q
        R = self.out(Vp * k_global.unsqueeze(1))                           # (B,T,D)
        return self.dropout(R) + Q



# =========================
# MSfastformer (stack)
# =========================

class MSFastformerBlock(nn.Module):
    """
    One residual block:
      - LN → multi-scale convs (k=1,3,5) → U1,U3,U5
      - Three CrossFastformer interactions sum → P
      - Residual FFN: FFN(P) + x
    """
    def __init__(self, d_model, dropout=0.0, ffn_mult=2.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.conv1 = _conv1d_same(d_model, d_model, k=1)
        self.conv3 = _conv1d_same(d_model, d_model, k=3)
        self.conv5 = _conv1d_same(d_model, d_model, k=5)

        self.fast_53_3 = CrossFastformer(d_model, dropout=dropout)  # (U5, U3, U3)
        self.fast_31_1 = CrossFastformer(d_model, dropout=dropout)  # (U3, U1, U1)
        self.fast_15_5 = CrossFastformer(d_model, dropout=dropout)  # (U1, U5, U5)

        self.ffn = PositionwiseFFN(d_model, mult=ffn_mult, dropout=dropout)

    def _conv_feats(self, x):  # x: (B,T,D)
        x_n = self.norm_in(x)
        xt = x_n.transpose(1, 2)          # (B,D,T)
        U1 = self.conv1(xt).transpose(1, 2)
        U3 = self.conv3(xt).transpose(1, 2)
        U5 = self.conv5(xt).transpose(1, 2)
        return U1, U3, U5

    def forward(self, x, mask):  # (B,T,D), (B,T)
        U1, U3, U5 = self._conv_feats(x)
        P = (
            self.fast_53_3(U5, U3, U3, mask) +
            self.fast_31_1(U3, U1, U1, mask) +
            self.fast_15_5(U1, U5, U5, mask)
        )
        return self.ffn(P) + x

class MSFastformerEncoder(nn.Module):
    """
    Stack of MSfastformer residual blocks; final LN+Linear.
    """
    def __init__(self, d_model=300, num_layers=3, dropout=0.0, ffn_mult=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MSFastformerBlock(d_model, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])
        self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def forward(self, x, mask):  # (B,T,D), (B,T)
        mask = ensure_at_least_one(mask)
        for blk in self.layers:
            x = blk(x, mask)
        return self.post(x)


# =========================
# Gated Fusion Module (GFM)
# =========================

class GatedFusionModule(nn.Module):
    """
    g = sigmoid(MLP([EA, ET]));  EF = g ⊙ EA + (1-g) ⊙ ET
    """
    def __init__(self, d_model, hidden_ratio=2.0, dropout=0.0):
        super().__init__()
        h = int(hidden_ratio * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, EA, ET):  # (B,T,D) each
        z = torch.cat([EA, ET], dim=-1)
        g = self.mlp(z)
        EF = g * EA + (1.0 - g) * ET
        return self.dropout(EF)


# =========================
# Recurrent Pyramid Model (RPM)
# =========================

class Fuse(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mix = nn.Sequential(
            nn.Linear(2*d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
    def forward(self, a, b):  # (B,T,D) + (B,T,D) -> (B,T,D)
        a = self.ln1(a); b = self.ln2(b)
        return self.mix(torch.cat([a, b], dim=-1))

def upsample_to(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """x: (B,T,D) -> up/down to (B,target_T,D) via linear interpolation."""
    return F.interpolate(x.transpose(1,2), size=target_T, mode='linear', align_corners=False).transpose(1,2)

def up_mask(mask: torch.Tensor, target_T: int) -> torch.Tensor:
    """mask: (B,T) -> (B,target_T)"""
    m = mask.float().unsqueeze(1)  # (B,1,T)
    m = F.interpolate(m, size=target_T, mode='linear', align_corners=False)
    return (m.squeeze(1) > 0.5)

def down_mask(mask: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    Approximate downsample: average-pool then (if needed) interpolate to exact length.
    """
    B, T = mask.shape
    m = mask.float().unsqueeze(1)  # (B,1,T)
    if target_T == T:
        return mask
    if target_T < T:
        # pool by integer factor then interpolate if necessary
        scale = max(1, T // target_T)
        m = F.avg_pool1d(m, kernel_size=scale, stride=scale, ceil_mode=True)
    if m.size(-1) != target_T:
        m = F.interpolate(m, size=target_T, mode='linear', align_corners=False)
    return (m.squeeze(1) > 0.5)

class RPM(nn.Module):
    """
    3-level temporal pyramid with top-down and bottom-up fusions.
    Returns 6 aligned sequences and their masks, all at the finest resolution T.
    """
    def __init__(self, d_model, d_pyr=256):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_pyr)
        # Base & downsampling stages
        self.c1 = nn.Conv1d(d_pyr, d_pyr, kernel_size=1, stride=1, padding=0)  # same T
        self.c2 = nn.Conv1d(d_pyr, d_pyr, kernel_size=3, stride=2, padding=1)  # ~ceil(T/2)
        self.c3 = nn.Conv1d(d_pyr, d_pyr, kernel_size=3, stride=2, padding=1)  # ~ceil(T/4)

        self.fuse12 = Fuse(d_pyr)  # L2 ⊕ up(L3)
        self.fuse01 = Fuse(d_pyr)  # L1 ⊕ up(L2')
        self.fuse21 = Fuse(d_pyr)  # L2' ⊕ down(L1')
        self.fuse32 = Fuse(d_pyr)  # L3  ⊕ down(L2'')

        self.out_proj = nn.Linear(d_pyr, d_model)

    @staticmethod
    def _to_BDT(x):  # (B,T,D) -> (B,D,T)
        return x.transpose(1, 2)

    @staticmethod
    def _to_BTD(x):  # (B,D,T) -> (B,T,D)
        return x.transpose(1, 2)

    def forward(self, x, mask):  # x: (B,T,D); mask: (B,T)
        mask = ensure_at_least_one(mask)
        B, T, _ = x.shape
        h = self.in_proj(x)  # (B,T,d_pyr)

        # Build pyramid (feature)
        l1 = self._to_BTD(self.c1(self._to_BDT(h)))            # (B,T,d_pyr)
        l2 = self._to_BTD(self.c2(self._to_BDT(l1)))           # (B,T2,d_pyr)
        l3 = self._to_BTD(self.c3(self._to_BDT(l2)))           # (B,T3,d_pyr)
        T1, T2, T3 = l1.size(1), l2.size(1), l3.size(1)

        # Build pyramid (mask)
        m1 = mask
        m2 = down_mask(m1, T2)
        m3 = down_mask(m2, T3)

        # --- Top-down path ---
        td3, m_td3 = l3, m3
        td2, m_td2 = self.fuse12(l2, upsample_to(l3, T2)), up_mask(m3, T2)
        td1, m_td1 = self.fuse01(l1, upsample_to(td2, T1)), up_mask(m_td2, T1)

        # --- Bottom-up path ---
        def down_to(y, target_T):
            return upsample_to(y, target_T) if y.size(1) != target_T else y

        bu1, m_bu1 = td1, m_td1
        bu2, m_bu2 = self.fuse21(td2, down_to(td1, T2)), down_mask(m_td1, T2)
        bu3, m_bu3 = self.fuse32(td3, down_to(bu2, T3)), down_mask(m_bu2, T3)

        # Align all 6 to finest T for AFM and project back to d_model
        feats = [td1, td2, td3, bu1, bu2, bu3]
        masks = [m_td1, m_td2, m_td3, m_bu1, m_bu2, m_bu3]

        feats = [upsample_to(z, T) for z in feats]     # each (B,T,d_pyr)
        masks = [up_mask(m, T) for m in masks]         # each (B,T)

        feats = [self.out_proj(z) for z in feats]      # each (B,T,D)
        return feats, masks


# =========================
# Adaptive Fusion Module (AFM)
# =========================

class AFM(nn.Module):
    """
    Stack six branches → (B,6,T,D). Build 6-way attention from compressed
    (mean) summaries, apply softmax weights across branches, and add a residual.
    Masks ensure padding doesn’t skew the summaries.
    """
    def __init__(self, d_model):
        super().__init__()
        self.fc_attn = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats6, masks6):
        B = feats6[0].size(0)
        x = torch.stack(feats6, dim=1)                 # (B,6,T,D)
        m = torch.stack(masks6, dim=1).unsqueeze(-1)   # (B,6,T,1)
        x_n = self.norm(x)

        m_f = m.float().clamp_min(0.0)
        # Two distinct compressed views per branch:
        # (a) temporal-first (mean over T, masked) -> (B,6,D) -> then mean over D -> (B,6)
        t_mean = (x_n * m_f).sum(dim=2) / m_f.sum(dim=2).clamp_min(1e-6)   # (B,6,D)
        s_t = t_mean.mean(dim=2)                                            # (B,6)

        # (b) channel-first (mean over D) -> (B,6,T), then masked mean over T -> (B,6)
        c_mean = x_n.mean(dim=3)                                            # (B,6,T)
        s_d = (c_mean * m.squeeze(-1).float()).sum(dim=2) / m.squeeze(-1).float().sum(dim=2).clamp_min(1e-6)  # (B,6)

        joint = torch.cat([s_t, s_d], dim=-1)                               # (B,12)
        w = torch.softmax(self.fc_attn(joint), dim=-1).view(B, 6, 1, 1)     # (B,6,1,1)

        # Apply attention and add residual on the stacked tensor (closer to paper)
        x_att = w * x_n                                                     # (B,6,T,D)
        x_out = (x_att + x_n).sum(dim=1)                                    # (B,T,D)
        return x_out



# =========================
# Full Core: MFFNet
# =========================

class MFFNetCore(nn.Module):
    """
    End-to-end core:
      audio/text → project to D
      → MSfastformer encoders
      → GFM
      → RPM (6 branches) + AFM
      → masked global mean pool → FC
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
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, audio_embed, text_embed, mask):  # mask: (B,T) bool
        mask = ensure_at_least_one(mask)
        a = audio_embed
        t = text_embed
        a = self.pa(a)
        t = self.pt(t)

        a = self.ms_audio(a, mask)  # (B,T,D)
        t = self.ms_text(t, mask)   # (B,T,D)

        fused = self.gfm(a, t)      # (B,T,D)

        feats6, masks6 = self.rpm(fused, mask)     # 6×(B,T,D), 6×(B,T)
        fused_multi = self.afm(feats6, masks6)     # (B,T,D)

        fused_multi = self.pre_pool_norm(fused_multi)
        pooled = masked_mean_time(fused_multi, mask)  # (B,D)

        logits = self.fc(pooled)  # (B,C)
        return logits

