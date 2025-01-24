import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Implementation of Fastformer (modified from  wuch15/Fastformer repository)
class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # W_q^T q_i in eq. (3), W_q^T m_i in eq. (6)
        self.query_att = nn.Linear(self.head_dim, 1, bias=False)
        self.m_att     = nn.Linear(self.head_dim, 1, bias=False)

        # last linear transform after element-wise product of k and V.
        self.out_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)

        # Combine all heads back to hidden_size
        self.merge_heads = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Simple initialization scheme."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def split_heads(self, x: torch.Tensor):
        """
        Split [B, seq_len, hidden_size] -> [B, num_heads, seq_len, head_dim].
        """
        B, S, D = x.size()
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, S, d]

    def merge_heads_fn(self, x: torch.Tensor):
        """
        Merge [B, num_heads, seq_len, head_dim] -> [B, seq_len, hidden_size].
        """
        B, H, S, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, S, H * d)

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
        q_global = torch.einsum('bhsi,bhs->bhd', Q, att_weights_q)  # [B, H, d]

        # M = q_global * K (element-wise for each position i)
        # M_i = q_global \odot K_i
        qg = q_global.unsqueeze(2)             # [B, H, 1, d]
        M  = qg * K                            # [B, H, Sk, d]

        # a_i = softmax( W_q^T m_i / sqrt(d) ) over i=1..Sk
        aggregator_logits_m = self.m_att(M) / math.sqrt(self.head_dim)  # [B, H, Sk, 1]
        aggregator_logits_m = aggregator_logits_m.squeeze(-1)           # [B, H, Sk]

        if attention_mask is not None:
            pass

        att_weights_m = self.softmax(aggregator_logits_m)  # [B, H, Sk]

        # 5) k_global = sum_i a_i * m_i  => [B, H, d]
        k_global = torch.einsum('bhsi,bhs->bhd', M, att_weights_m)

        # 6) For each position i in V, do elementwise multiply with k_global
        #    => e_i = out_proj( k_global \odot v_i )
        # so E has shape [B, H, Sk, d]. Then we'll combine E with the "original Q" somehow
        kg = k_global.unsqueeze(2)                  # [B, H, 1, d]
        KV_interaction = kg * V                     # [B, H, Sk, d]
        E = self.out_proj(KV_interaction)           # [B, H, Sk, d]

        if Sq != Sk:
            raise ValueError("Fastformer aggregator: mismatch in seq_len (Sq vs Sk).")

        out_heads = E + Q  # shape [B, H, Sq, d]
        out = self.merge_heads(out_heads)  # [B, Sq, all_head_size]

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

        self.layernorm_in = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        self.fastformer_1 = FastformerLayer(config)
        self.fastformer_3 = FastformerLayer(config)
        self.fastformer_5 = FastformerLayer(config)

        self.layernorm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        P2 = self.fastformer_3(U3, U1, U1, attention_mask)
        P3 = self.fastformer_5(U1, U5, U5, attention_mask)
        P  = P1 + P2 + P3  # [B, S, D]

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
        # in_channels is the channel dimension for f1, f2, f3
        # We assume all 3 features have the same # of channels 
        # (the difference is in their time resolution).
        
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
        
        # For upsampling, we can use interpolation
        # (or a conv-transpose if you prefer).
        # We do nearest neighbor here.
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
        # ============= STEP 1: Top-Down Pass =============
        # We'll keep the original names:
        #   td3 = f3
        #   td2 = fuse( f2, upsample(f3) )
        #   td1 = fuse( f1, upsample(td2) )
        
        td3 = f3
        # Upsample f3 -> same size as f2
        up3 = self.upsample(td3, scale=2)  # T3->T2
        td2 = self.fuse_td2(up3, f2)
        
        # Upsample td2 -> same size as f1
        up2 = self.upsample(td2, scale=2)  # T2->T1
        td1 = self.fuse_td1(up2, f1)
        
        # ============= STEP 2: Bottom-Up Pass =============
        # The paper says: "the bottom fused features are combined with the top features
        # through downsampling, and again top-down fusion is performed..."
        #
        # A straightforward way: 
        #   bu1 = td1
        #   bu2 = fuse( td2, downsample(td1) )
        #   bu3 = fuse( td3, downsample(bu2) )
        
        bu1 = td1
        
        # Downsample td1 -> size of td2
        down1 = self.downsample(bu1)  # T1->T2
        bu2 = self.fuse_bu2(td2, down1)
        
        # Downsample bu2 -> size of td3
        down2 = self.downsample(bu2)  # T2->T3
        bu3 = self.fuse_bu3(td3, down2)
        
        # ============= STEP 3: Concatenate results =============
        #
        # "Finally, the outputs of these two fusion paths are concatenated to obtain
        #  the final feature representation of the RPM."
        #
        # We have two "paths":
        #   - top-down path: td1, td2, td3
        #   - bottom-up path: bu1, bu2, bu3
        #
        # We'll just concat them channel-wise. 
        # Often you'll then feed them into another 1x1 conv or your next network block.
        
        # Each of these has a different temporal length (td1 is T1, td2 is T2, td3 is T3),
        # so typically you'd either keep them separate or upsample/downsample to unify dimension
        # before final concatenation.
        #
        # If you want one big tensor, you must unify or flatten them. 
        # For the sake of demonstration, let's unify them to T1 by upsampling the smaller ones:
        
        td2_up = self.upsample(td2, scale=2)  # T2->T1
        td3_up = self.upsample(td3, scale=4)  # T3->T1
        bu2_up = self.upsample(bu2, scale=2)
        bu3_up = self.upsample(bu3, scale=4)
        
        # Now all are [B, C, T1]. We can safely concat along channel dimension:
        final = torch.cat([td1, td2_up, td3_up, bu1, bu2_up, bu3_up], dim=1)
        # final shape: [B, 6*C, T1]
        
        return final, (td1, td2, td3, bu1, bu2, bu3)
