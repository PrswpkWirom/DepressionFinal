import torch
import torch.nn as nn
import math
import torch.nn.functional as F

############################################################################
# Implementation of Fastformer (modified from  wuch15/Fastformer repository)
############################################################################

class FastSelfAttention(nn.Module):
    """
    Implements the Fastformer additive-attention mechanism:
      1) Project input into Q, K, V (and also aggregator projections).
      2) Aggregate queries into a single 'global' query vector q (per head).
      3) Multiply q element-wise with each K to get M, then aggregate M into
         a single 'global' key vector k (per head).
      4) Element-wise multiply k with each V, transform that to get E,
         and then combine E with the original Q for the output.
    """
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

############################################################################
# MSFastformer
############################################################################

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
