import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Fastformer and Related Components
###############################################################################
class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim = config.hidden_size

        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context_states, attention_mask=None):
        batch_size, seq_len_hidden, _ = hidden_states.size()
        seq_len_context = context_states.size(1)

        # Compute queries and keys
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context_states)

        # Compute attention scores
        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2) 
            / self.attention_head_size ** 0.5
        )
        key_for_score = (
            self.key_att(mixed_key_layer).transpose(1, 2) 
            / self.attention_head_size ** 0.5
        )

        attention_scores = torch.matmul(
            query_for_score.transpose(1, 2), key_for_score
        )  # [B, seq_len_hidden, seq_len_context]

        if attention_mask is not None:
            attention_scores += attention_mask

        # Compute attention weights
        attention_weights = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_weights, mixed_key_layer)  # [B, seq_len_hidden, hidden_size]

        # Combine with value projection
        output = self.transform(context_layer) + mixed_query_layer
        return output

class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_tensor, context_tensor, attention_mask=None):
        self_output = self.self(input_tensor, context_tensor, attention_mask)
        attention_output = self.output(self_output) + input_tensor
        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, context_states, attention_mask=None):
        attention_output = self.attention(hidden_states, context_states, attention_mask)
        attention_output = self.layernorm(attention_output)
        intermediate_output = self.activation(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output) + attention_output
        layer_output = self.layernorm(layer_output)
        return layer_output

class MSFastformer(nn.Module):
    def __init__(self, config):
        super(MSFastformer, self).__init__()
        self.config = config
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        self.fastformer = FastformerLayer(config)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.apply(self.init_weights)
        
        # Use config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, attention_mask):
        batch_size, seq_len, hidden_size = x.size()
        x = self.layernorm(x)
        x_transposed = x.transpose(1, 2)  # [B, hidden_size, seq_len]

        U1 = self.conv1(x_transposed)  # [B, hidden_size, seq_len]
        U3 = self.conv3(x_transposed)
        U5 = self.conv5(x_transposed)

        U1 = U1.transpose(1, 2)
        U3 = U3.transpose(1, 2)
        U5 = U5.transpose(1, 2)

        P1 = self.fastformer(U5, U3, attention_mask=None)
        P2 = self.fastformer(U3, U1, attention_mask=None)
        P3 = self.fastformer(U1, U5, attention_mask=None)
        P = P1 + P2 + P3
        P_norm = self.layernorm(P)

        I = self.fc2(self.gelu(self.fc1(P_norm)))
        I = self.dropout(I)
        J = I + x
        J_norm = self.layernorm(J)
        y = self.output_fc(J_norm)
        y = self.dropout(y)
        return y

###############################################################################
# Config
###############################################################################
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

###############################################################################
# Gated Fusion
###############################################################################
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.1):
        super(GatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Use passed-in dropout probability
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

###############################################################################
# Multimodal Feature Enhancer (MFE) with BiFPN
###############################################################################
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm=True, activation=True):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels) if norm else None
        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class BiFPNLayer(nn.Module):
    def __init__(self, feature_size):
        super(BiFPNLayer, self).__init__()
        self.conv_p2_td = SeparableConvBlock(feature_size, feature_size)
        self.conv_p1_td = SeparableConvBlock(feature_size, feature_size)
        self.conv_p0_td = SeparableConvBlock(feature_size, feature_size)
        self.conv_p0_out = SeparableConvBlock(feature_size, feature_size)
        self.conv_p1_out = SeparableConvBlock(feature_size, feature_size)
        self.conv_p2_out = SeparableConvBlock(feature_size, feature_size)

    def forward(self, p0, p1, p2):
        p2_td = self.conv_p2_td(p2)
        p2_upsampled = F.interpolate(p2_td, size=p1.size(-1), mode='linear', align_corners=False)
        p1_td = self.conv_p1_td(p1 + p2_upsampled)

        p1_upsampled = F.interpolate(p1_td, size=p0.size(-1), mode='linear', align_corners=False)
        p0_td = self.conv_p0_td(p0 + p1_upsampled)

        p0_downsampled = F.interpolate(p0_td, size=p1_td.size(-1), mode='linear', align_corners=False)
        p0_out = self.conv_p0_out(p0_downsampled + p1_td)

        p0_out_downsampled = F.interpolate(p0_out, size=p2_td.size(-1), mode='linear', align_corners=False)
        p1_out = self.conv_p1_out(p0_out_downsampled + p2_td)

        p1_out_downsampled = F.interpolate(p1_out, size=p2_td.size(-1), mode='linear', align_corners=False)
        p2_out = self.conv_p2_out(p1_out_downsampled + p2_td)

        return p0_td, p1_td, p2_td, p0_out, p1_out, p2_out

class BiFPN(nn.Module):
    def __init__(self, feature_size=256, num_layers=1):
        super(BiFPN, self).__init__()
        self.num_layers = num_layers
        self.bifpn_layers = nn.ModuleList([BiFPNLayer(feature_size) for _ in range(num_layers)])

    def forward(self, features):
        p0, p1, p2 = features
        for bifpn_layer in self.bifpn_layers:
            p0_td, p1_td, p2_td, p0_out, p1_out, p2_out = bifpn_layer(p0, p1, p2)
            p0, p1, p2 = p0_td, p1_td, p2_td
        return [p0_td, p1_td, p2_td, p0_out, p1_out, p2_out]

class MultimodalFeatureEnhancer(nn.Module):
    def __init__(self, input_dim, feature_size=256, num_bifpn_layers=2):
        super(MultimodalFeatureEnhancer, self).__init__()
        self.input_dim = input_dim
        self.feature_size = feature_size
        self.conv = nn.Conv1d(input_dim, feature_size, kernel_size=1)
        self.conv_p1 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.conv_p2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.bifpn = BiFPN(feature_size=feature_size, num_layers=num_bifpn_layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        p0 = self.conv(x)
        p1 = self.conv_p1(p0)
        p2 = self.conv_p2(p1)
        outputs = self.bifpn([p0, p1, p2])
        # Transpose each back to [B, T, feature_size]
        out_list = []
        for out in outputs:
            out_list.append(out.transpose(1, 2))
        return out_list

###############################################################################
# AdaptiveFusionModule (AFM)
###############################################################################
class FeatureAlignment(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAlignment, self).__init__()
        self.align_convs = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            for _ in range(6)
        ])

    def forward(self, features):
        max_len = max(f.shape[1] for f in features)
        aligned_features = []
        for i, f in enumerate(features):
            f = f.transpose(1, 2)  # [B, feature_dim, T]
            f_aligned = self.align_convs[i](f)
            if f_aligned.shape[2] != max_len:
                f_aligned = F.interpolate(f_aligned, size=max_len, mode='linear', align_corners=False)
            aligned_features.append(f_aligned)
        H = torch.stack(aligned_features, dim=1)  # [B, 6, feature_dim, T]
        return H

class MultiFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super(MultiFeatureFusion, self).__init__()
        self.attention_fc = nn.Linear(12, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        B, N, D, T = H.shape
        mean_T = H.mean(dim=3)        # [B, 6, D]
        mean_D = H.mean(dim=2)        # [B, 6, T]
        mean_TD = mean_T.mean(dim=2)  # [B, 6]
        mean_DT = mean_D.mean(dim=2)  # [B, 6]
        mean_features = torch.cat([mean_TD, mean_DT], dim=1)  # [B, 12]

        attention_scores = self.attention_fc(mean_features)   # [B, 6]
        attention_scores = self.relu(attention_scores)
        attention_weights = self.softmax(attention_scores).unsqueeze(-1).unsqueeze(-1)  # [B, 6, 1, 1]

        weighted_H = H * attention_weights
        fused_features = weighted_H.sum(dim=1)  # [B, D, T]

        # Residual-like skip
        residual = H.mean(dim=1)  # [B, D, T]
        fused_features = fused_features + residual
        fused_features = fused_features.transpose(1, 2)  # [B, T, D]
        return fused_features

class AdaptiveFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AdaptiveFusionModule, self).__init__()
        self.feature_alignment = FeatureAlignment(feature_dim)
        self.feature_fusion = MultiFeatureFusion(feature_dim)

    def forward(self, features):
        H = self.feature_alignment(features)   # [B, 6, D, T]
        fused_output = self.feature_fusion(H)  # [B, T, D]
        return fused_output

###############################################################################
# Final MFFNet Model
###############################################################################
class MFFNet(nn.Module):
    def __init__(
        self, 
        text_config, 
        audio_config, 
        hidden_dim=768, 
        feature_size=256, 
        num_bifpn_layers=2, 
        num_labels=2
    ):
        super(MFFNet, self).__init__()
        
        # Two MSFastformer models for text and audio
        self.text_model = MSFastformer(text_config)
        self.audio_model = MSFastformer(audio_config)

        # Gated Fusion (use the same dropout as text_config or a separate constant)
        self.gated_fusion = GatedFusion(
            hidden_dim=hidden_dim, 
            dropout_prob=text_config.hidden_dropout_prob
        )

        # Multimodal Feature Enhancer
        self.mfe = MultimodalFeatureEnhancer(
            input_dim=hidden_dim, 
            feature_size=feature_size, 
            num_bifpn_layers=num_bifpn_layers
        )

        # Adaptive Fusion Module
        self.afm = AdaptiveFusionModule(feature_dim=feature_size)

        # Final classification layer
        self.classifier = nn.Linear(feature_size, num_labels)

    def forward(self, input_text, input_audio, attention_mask_text=None, attention_mask_audio=None):
        # [B, T, hidden_dim] from each MSFastformer
        output_text = self.text_model(input_text, attention_mask_text)
        output_audio = self.audio_model(input_audio, attention_mask_audio)

        # Fuse text and audio representations
        Ef = self.gated_fusion(output_text, output_audio)  # [B, T, hidden_dim]

        # Apply MultimodalFeatureEnhancer (MFE)
        features = self.mfe(Ef)  # List of 6 feature maps: each [B, T_i, feature_size]

        # Apply AdaptiveFusionModule (AFM)
        fused_output = self.afm(features)  # [B, T, feature_size]

        # Pool the fused output (mean pooling)
        fused_pooled = fused_output.mean(dim=1)  # [B, feature_size]

        # Classification
        logits = self.classifier(fused_pooled)  # [B, num_labels]

        return logits
