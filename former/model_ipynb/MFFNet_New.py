import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_att = nn.Linear(self.head_dim, 1, bias=False)
        self.m_att     = nn.Linear(self.head_dim, 1, bias=False)
        self.out_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.merge_heads = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        init_range = getattr(self.config, "initializer_range", 0.02)

        nn.init.normal_(self.query.weight, mean=0.0, std=init_range)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)

        nn.init.normal_(self.key.weight, mean=0.0, std=init_range)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)

        nn.init.normal_(self.value.weight, mean=0.0, std=init_range)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)

        nn.init.normal_(self.query_att.weight, mean=0.0, std=init_range)
        
        nn.init.normal_(self.m_att.weight, mean=0.0, std=init_range)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_range)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        nn.init.normal_(self.merge_heads.weight, mean=0.0, std=init_range)
        if self.merge_heads.bias is not None:
            nn.init.zeros_(self.merge_heads.bias)


    def split_heads(self, x: torch.Tensor):
        """
        [B, S, D] -> [B, H, S, d]
        """
        B, S, D = x.size()
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def merge_heads_fn(self, x: torch.Tensor):
        """
        [B, H, S, d] -> [B, S, D]
        """
        B, H, S, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, S, H, d]
        return x.view(B, S, H * d)  # [B, S, D]

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        B, Sq, _ = query_states.shape
        Sk = key_states.size(1)
        Q = self.split_heads(self.query(query_states))   # [B, H, Sq, d]
        K = self.split_heads(self.key(key_states))       # [B, H, Sk, d]
        V = self.split_heads(self.value(value_states))   # [B, H, Sk, d]

        # O_i = softmax( W_q^T q_i / sqrt(d) ),  q = sum_i O_i * q_i
        aggregator_logits_q = self.query_att(Q) / math.sqrt(self.head_dim)
        aggregator_logits_q = aggregator_logits_q.squeeze(-1)  # [B, H, Sq]

        if attention_mask is not None:
            aggregator_logits_q = aggregator_logits_q + attention_mask.view(B, 1, -1)

        att_weights_q = self.softmax(aggregator_logits_q)   # [B, H, Sq]
        # print("Q shape:", Q.shape)
        # print("att_weights_q shape:", att_weights_q.shape)

        q_global = torch.einsum('bhsd,bhs->bhd', Q, att_weights_q)  # [B, H, d]

        # M = q_global * K (element-wise for each position i)
        # M_i = q_global \odot K_i
        qg = q_global.unsqueeze(2)             # [B, H, 1, d]
        M  = qg * K                            # [B, H, Sk, d]

        # a_i = softmax( W_q^T m_i / sqrt(d) ) over i=1..Sk
        aggregator_logits_m = self.m_att(M) / math.sqrt(self.head_dim)  # [B, H, Sk, 1]
        aggregator_logits_m = aggregator_logits_m.squeeze(-1)           # [B, H, Sk]

        if attention_mask is not None:
            aggregator_logits_m = aggregator_logits_m + attention_mask.view(B, 1, -1)


        att_weights_m = self.softmax(aggregator_logits_m)  # [B, H, Sk]

        # 5) k_global = sum_i a_i * m_i  => [B, H, d]
        k_global = torch.einsum('bhsd,bhs->bhd', M, att_weights_m)
 
        kg = k_global.unsqueeze(2)                  # [B, H, 1, d]
        KV_interaction = kg * V                     # [B, H, Sk, d]
        E = self.out_proj(KV_interaction)           # [B, H, Sk, d]

        if Sq != Sk:
            raise ValueError("Fastformer aggregator: mismatch in seq_len (Sq vs Sk).")

        out_heads = E + Q  # shape [B, H, Sq, d]
        out = self.merge_heads_fn(out_heads)  # [B, Sq, all_head_size]
        #print("out after aggregator:", out.shape)

        return out

class FastAttention(nn.Module):
    """
    Wraps FastSelfAttention in a minimal 'attention + output-projection' block,
    optionally adding a residual connection.
    """
    def __init__(self, config):
        super().__init__()
        self.self = FastSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None):
        self_output = self.self(query_tensor, key_tensor, value_tensor, attention_mask)
        # typical residual:
        attention_output = self.output(self_output) + query_tensor
        attention_output = self.layernorm(attention_output)
        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FastAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, key_states, value_states, attention_mask=None):
        # 1) Fastformer-based attention
        attention_output = self.attention(
            hidden_states, key_states, value_states, attention_mask
        )
        # 2) Feed-forward + residual
        inter = self.intermediate(attention_output)
        inter = self.activation(inter)
        ff_out = self.output(inter)
        ff_out = ff_out + attention_output
        layer_output = self.layernorm(ff_out)
        return layer_output

# MSFastformer
class MSFastformer(nn.Module):
    """
    Multi-Scale Fastformer block:
      - LayerNorm + three different 1D Convs (kernel_size=1,3,5)
      - Each conv output goes into a FastformerLayer call (paired as in eq. (2))
      - Sum the resulting feature maps
      - Then FC->GELU->FC + residual, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Single definitions of LayerNorm
        self.layernorm_in = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Convolutions
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)

        # Fastformer layers
        self.fastformer_1 = FastformerLayer(config)
        self.fastformer_3 = FastformerLayer(config)
        self.fastformer_5 = FastformerLayer(config)
    #     print("fastformer_1 out_proj weight:", 
    # self.fastformer_1.attention.self.out_proj.weight.shape)

    #     print("fastformer_3 out_proj weight:", 
    # self.fastformer_3.attention.self.out_proj.weight.shape)

    #     print("fastformer_5 out_proj weight:", 
    # self.fastformer_5.attention.self.out_proj.weight.shape)


        # Feed-forward
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.output_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, attention_mask=None):
        """
        x: [B, seq_len, hidden_size]
        """
        B, S, D = x.shape

        # 1) LN, then conv
        x_norm = self.layernorm_in(x)                 # [B, S, D]
        x_t = x_norm.transpose(1, 2)                  # [B, D, S]

        U1 = self.conv1(x_t).transpose(1, 2)          # [B, S, D]
        U3 = self.conv3(x_t).transpose(1, 2)          # [B, S, D]
        U5 = self.conv5(x_t).transpose(1, 2)          # [B, S, D]

        P1 = self.fastformer_1(U5, U3, U3, attention_mask)
        #print("P1 shape:", P1.shape)
        P2 = self.fastformer_3(U3, U1, U1, attention_mask)
        P3 = self.fastformer_5(U1, U5, U5, attention_mask)
        P  = P1 + P2 + P3  # [B, S, D]
        #print("P shape after sum:", P.shape)
        

        P_norm = self.layernorm_out(P)
        I = self.fc2(self.gelu(self.fc1(P_norm)))
        I = self.dropout(I)
        J = I + x  # residual with original x
        J_norm = self.layernorm_out(J)
        y = self.output_fc(J_norm)
        y = self.dropout(y)
        return y

# Config
class Config:
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=16,
                 intermediate_size=3072,
                 num_labels=2,
                 num_hidden_layers=12,
                 hidden_dropout_prob=0.1,       # Use this for general dropout
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 layer_norm_eps=1e-12,
                 initializer_range=0.02,
                 hidden_act="gelu",
                 pooler_type='weightpooler',
                 num_attention_layers=12):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_labels = num_labels
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.pooler_type = pooler_type
        self.num_attention_layers = num_attention_layers

# Gated Fusion
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.1):
        super(GatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Et, Ea):
        assert Et.shape == Ea.shape, "Et and Ea must have the same shape."
        concat = torch.cat((Et, Ea), dim=-1)  # [B, T, 2*hidden_dim]
        out = self.fc1(concat)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        g = self.sigmoid(out)
        Ef = g * Ea + (1 - g) * Et
        return Ef

#Recurrent Pyramid Network
class Conv1x1(nn.Module):
    """
    A simple 1D conv block (kernel_size=1) for channel adjustment/fusion in 1D feature maps.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
    def forward(self, x):
        return self.conv(x)

class FuseModule(nn.Module):
    """
    A lightweight 'fusion' that combines two feature maps (same resolution) by:
      1) elementwise add
      2) optional 1x1 conv for channel mixing
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv1x1(channels, channels)
    
    def forward(self, f_top, f_bottom):
        # f_top, f_bottom shapes: [B, C, length]
        fused = f_top + f_bottom
        out = self.conv(fused)
        return out

class RecurrentPyramidModel(nn.Module):
    """
    A 'Recurrent Pyramid Model' (RPM) as per Fig. 6 in the paper:
     - Input: three feature maps at different 1D scales: f1 (x1), f2 (x1/2), f3 (x1/4)
     - Step 1 (Top-Down): upsample from f3->f2->f1, fusing at each step
     - Step 2 (Bottom-Up): downsample from td1->td2->td3, fusing at each step
     - Final: concatenate the outputs of these two paths along channels
    """
    def __init__(self, in_channels):
        super().__init__()

        # For top-down and bottom-up, we'll define small modules:
        self.fuse12 = FuseModule(in_channels)  # fuse f1<->f2 or td1<->td2
        self.fuse23 = FuseModule(in_channels)  # fuse f2<->f3 or td2<->td3
        
        # If we want them separate for top-down vs bottom-up, we can do that:
        self.fuse_td2 = FuseModule(in_channels)
        self.fuse_td1 = FuseModule(in_channels)
        self.fuse_bu2 = FuseModule(in_channels)
        self.fuse_bu3 = FuseModule(in_channels)
        
        # 1D conv (stride=2) for downsampling,
        # or we could use average pool for a simpler approach.
        self.downsample = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=3, stride=2, padding=1, bias=True
        )
        
        self.upsample = lambda x, scale=2: F.interpolate(
            x, scale_factor=scale, mode='nearest'
        )
        
    def forward(self, f1, f2, f3):
        """
        f1 -> shape [B, C, T1]
        f2 -> shape [B, C, T2]
        f3 -> shape [B, C, T3]
        
        Where T2 ~ T1/2, T3 ~ T1/4.
        """

        td3 = f3
        # Upsample f3 -> same size as f2
        up3 = F.interpolate(td3, size=f2.shape[-1], mode='nearest')
        td2 = self.fuse_td2(up3, f2)
        
        # Upsample td2 -> same size as f1
        up2 = F.interpolate(td2, size=f1.shape[-1], mode='nearest')
        td1 = self.fuse_td1(up2, f1)
        bu1 = td1
        
        # Downsample td1 -> size of td2
        down1 = F.interpolate(bu1, size=td2.shape[-1], mode='nearest')  # T1->T2
        bu2 = self.fuse_bu2(td2, down1)
        
        # Downsample bu2 -> size of td3
        down2 = F.interpolate(bu2, size=td3.shape[-1], mode='nearest')
        bu3 = self.fuse_bu3(td3, down2)
        
        td3_up = F.interpolate(td3, size=f1.shape[-1], mode='nearest')
        td2_up = F.interpolate(td2, size=f1.shape[-1], mode='nearest')
        bu2_up = F.interpolate(bu2, size=f1.shape[-1], mode='nearest')
        bu3_up = F.interpolate(bu3, size=f1.shape[-1], mode='nearest')

        final = torch.cat([td1, td2_up, td3_up, bu1, bu2_up, bu3_up], dim=1)

        final = torch.cat([td1, td2_up, td3_up, bu1, bu2_up, bu3_up], dim=1)
        # final shape: [B, 6*C, T1]
        
        return final, (td1, td2, td3, bu1, bu2, bu3)

# Adaptive Fusion Module
class FeatureAlignment(nn.Module):
    """
    1) Align each of the 6 multi-resolution features in time dimension 
       by applying a 1D conv and (if needed) interpolate to the max length.
    2) Return a single 4D tensor [B, 6, T, D], 
       matching the 'Feature Dimension Alignment' block in Figure 7.
    """
    def __init__(self, feature_dim):
        super(FeatureAlignment, self).__init__()
        self.align_convs = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            for _ in range(6)
        ])

    def forward(self, features):
        """
        features: list/tuple of 6 tensors, each shape [B, T_x, D], 
                  possibly different T_x lengths
        Returns:  H of shape [B, 6, T_max, D]
        """
        max_len = max(f.shape[1] for f in features)

        aligned_list = []
        for i, f in enumerate(features):
            f_1d = f.transpose(1, 2)  # => [B, D, T_i]
            # apply the 1x1 conv
            f_aligned = self.align_convs[i](f_1d)  # still [B, D, T_i]

            # if needed, interpolate to max_len
            if f_aligned.shape[2] != max_len:
                # use linear interpolation along the temporal axis
                f_aligned = F.interpolate(
                    f_aligned, 
                    size=max_len, 
                    mode='linear', 
                    align_corners=False
                )
            # now shape is [B, D, max_len]
            # next we want to store it as [B, max_len, D]
            f_aligned = f_aligned.transpose(1, 2)  # => [B, max_len, D]
            aligned_list.append(f_aligned)

        # Step 2: stack along a new dimension => [B, 6, max_len, D]
        H = torch.stack(aligned_list, dim=1)
        return H
    
class MultiFeatureFusion(nn.Module):
    """
    1) Compute T-dim mean and D-dim mean over the stacked features => 
       [B, 6] + [B, 6] => [B, 12].
    2) Generate an attention vector for the 6 channels via FC->ReLU->Softmax.
    3) Apply the attention to H, sum across dimension=1 (the '6' dimension).
    4) Add a residual connection from H (averaged or direct).
    5) Output final fused feature => shape [B, T, D].
    """
    def __init__(self, feature_dim):
        super(MultiFeatureFusion, self).__init__()
        # We produce a single attention weight per each of the 6 features
        # from a 12-dimensional input (concatenated means).
        self.attention_fc = nn.Linear(12, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # normalize across the 6 dimension

    def forward(self, H):
        """
        H: [B, 6, T, D], from FeatureAlignment
        Return fused: [B, T, D]
        """
        B, N, T, D = H.shape  # N=6

        # 1) Means along T and D:
        # mean_T => [B, N, D], i.e. average over T dimension
        mean_T = H.mean(dim=2)  
        # mean_D => [B, N, T], i.e. average over D dimension
        mean_D = H.mean(dim=3)

        # 2) Then average again across the leftover dimension to get [B, N]
        #    We want a single scalar per each of the N=6 slices, from T or D perspective
        mean_TD = mean_T.mean(dim=2)  # => [B, N]
        mean_DT = mean_D.mean(dim=2)  # => [B, N]

        # 3) Concatenate them => [B, 2N] = [B, 12]
        mean_features = torch.cat([mean_TD, mean_DT], dim=1)  # [B, 12]

        # 4) FC -> ReLU -> Softmax => attention vector of shape [B, 6]
        att_scores = self.attention_fc(mean_features)  # [B, 6]
        att_scores = self.relu(att_scores)
        att_weights = self.softmax(att_scores)  # [B, 6]
        # expand to broadcast over T,D
        att_weights = att_weights.view(B, N, 1, 1)  # [B, 6, 1, 1]

        # 5) Weighted sum across N=6 dimension => [B, T, D]
        weighted = H * att_weights  # [B, 6, T, D]
        fused = weighted.sum(dim=1)  # [B, T, D]

        # 6) Residual connection
        #    We can simply average H across the 6 dimension to get a 'baseline'
        #    shape => [B, T, D]
        residual = H.mean(dim=1)  # [B, T, D]
        fused = fused + residual

        return fused

class AdaptiveFusionModule(nn.Module):
    """
    Overall AFM from Figure 7:
      - Feature dimension alignment via 1D conv + optional interpolation
      - Concat into shape [B, 6, T, D]
      - Multi-feature fusion with channel-wise attention + residual
    """
    def __init__(self, feature_dim):
        super(AdaptiveFusionModule, self).__init__()
        self.feature_alignment = FeatureAlignment(feature_dim)
        self.feature_fusion = MultiFeatureFusion(feature_dim)

    def forward(self, features):
        """
        features: a list/tuple of 6 multi-resolution features, 
                  each shaped [B, T_i, D]
        Returns:
          fused_output: shape [B, T_max, D]
        """
        # Step 1: Align to [B, 6, T_max, D]
        H = self.feature_alignment(features)

        # Step 2: Fuse => [B, T_max, D]
        fused_output = self.feature_fusion(H)
        return fused_output

class MFFNet(nn.Module):
    """
    The overall MFFNet architecture:
      1) Two MSFastformer encoders (one for text, one for speech)
      2) Gated Fusion of text + speech -> fused rep
      3) Recurrent Pyramid Model on that fused rep, generating multi-res outputs
      4) Adaptive Fusion Module (AFM) on those 6 multi-res outputs
      5) Final FC to produce binary depression classification
    """
    def __init__(self, config, num_classes=2):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.text_encoder = MSFastformer(config)
        self.speech_encoder = MSFastformer(config)
        self.gated_fusion = GatedFusion(hidden_dim=config.hidden_size,
                                        dropout_prob=config.hidden_dropout_prob)

        self.rpm = RecurrentPyramidModel(in_channels=config.hidden_size)
        self.afm = AdaptiveFusionModule(feature_dim=config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

    def forward(self, text_in, speech_in, attention_mask_text=None, attention_mask_speech=None):
        """
        text_in, speech_in: [B, T, D] embeddings for text and speech
        attention_mask_*: optional [B, T] for each
        Returns: logits => [B, num_classes]
        """
        text_rep = self.text_encoder(text_in, attention_mask_text)      # [B, T, D]
        speech_rep = self.speech_encoder(speech_in, attention_mask_speech)  # [B, T, D]

        fused_rep = self.gated_fusion(text_rep, speech_rep)

        fused_rep_t = fused_rep.transpose(1, 2)  # => [B, D, T], the 'C' dimension is D

        f1 = fused_rep_t  # full scale => shape [B, D, T]
        f2 = F.avg_pool1d(f1, kernel_size=2, stride=2)  # => [B, D, T//2]
        f3 = F.avg_pool1d(f2, kernel_size=2, stride=2)  # => [B, D, T//4]

        rpm_out, (td1, td2, td3, bu1, bu2, bu3) = self.rpm(f1, f2, f3)

        td1_ = td1.transpose(1, 2)  # => [B, T1, D]
        td2_ = td2.transpose(1, 2)  # => [B, T2, D]
        td3_ = td3.transpose(1, 2)  # => [B, T3, D]
        bu1_ = bu1.transpose(1, 2)
        bu2_ = bu2.transpose(1, 2)
        bu3_ = bu3.transpose(1, 2)

        fused_afm = self.afm([td1_, td2_, td3_, bu1_, bu2_, bu3_])  # => [B, T_max, D]

        pooled = fused_afm.mean(dim=1)  # [B, D]

        logits = self.classifier(pooled)  # => [B, num_classes]
        return logits
