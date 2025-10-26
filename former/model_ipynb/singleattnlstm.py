import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
      f_t = sigmoid( V_f * tanh(W_f * C_{t-1}) )
      C_t = f_t * C_{t-1} + (1 - f_t) * C_tilde
      i_t = 1 - f_t
      o_t = sigmoid(W_o [C_t, h_{t-1}, x_t])
      h_t = o_t * tanh(C_t)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # f_t = σ(V_f [tanh(W_f * C_{t-1})])
        self.W_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_f = nn.Linear(hidden_size, hidden_size, bias=False)

        # C_tilde = tanh(W_C [h_{t-1}, x_t])
        self.W_C = nn.Linear(hidden_size + input_size, hidden_size, bias=True)

        # o_t = σ(W_o [C_t, h_{t-1}, x_t])
        self.W_o = nn.Linear(hidden_size + hidden_size + input_size, hidden_size, bias=True)

    def forward(self, x_t, hidden):
        """
        x_t: (batch, input_size)
        hidden: tuple (h_{t-1}, C_{t-1}), each (batch, hidden_size)
        Returns: (h_t, C_t)
        """
        h_prev, C_prev = hidden

        # Attention-based forget gate
        f_t = torch.sigmoid(self.V_f(torch.tanh(self.W_f(C_prev))))

        # Candidate cell
        combined = torch.cat([h_prev, x_t], dim=1) #[batch, hidden+input]
        C_tilde = torch.tanh(self.W_C(combined))

        # Coupled cell update:
        C_t = f_t*C_prev + ((1.0-f_t)*C_tilde)

        # Output gate
        # o_t = sigmoid(W_o [C_t, h_{t-1}, x_t])
        out_in = torch.cat([C_t, h_prev, x_t], dim=1) #[batch, hidden+hidden+input]
        o_t = torch.sigmoid(self.W_o(out_in)) #[batch, hidden]

        # h_t = o_t * tanh(C_t)
        h_t = o_t * torch.tanh(C_t) #[batch, hidden]

        return h_t, C_t
    
class CoupledAttentionGateLSTM(nn.Module):
    """
    stacks 2 of the above cells in series (as in the diagram).
      - bottom layer: (h1_{t}, C1_{t}) from (h1_{t-1}, C1_{t-1}) and x_t
      - top layer:    (h2_{t}, C2_{t}) from (h2_{t-1}, C2_{t-1}) and h1_{t}
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell1 = AttentionGate(input_size, hidden_size)
        self.cell2 = AttentionGate(hidden_size, hidden_size)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns o_all_time: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        #initialize layer 1
        h1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        C1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        #initialize layer 2
        h2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        C2 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []  # store top layer (h2) at each step

        for t in range(seq_len):
            #bottom
            x_t = x[:, t, :]
            h1, C1 = self.cell1(x_t, (h1, C1))
            #top
            h2, C2 = self.cell2(h1, (h2, C2))
            outputs.append(h2.unsqueeze(1))  # (batch, 1, hidden_size)
        o_all_time = torch.cat(outputs, dim=1)
        return o_all_time
    
class DualAttention(nn.Module):
    """
      time attention       output_T = s_T * o_all_time                      where s_T = softmax(o_max_time * (o_all_time*W_t)^T)
      feature attention:   output_F = sum(s_F {hadamard-prod} o_all)time    where s_F = softmax( tanh(o_all_time * W_F) * v_F )
      concat [output_T, output_F].
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # time attention:
        self.W_t = nn.Linear(hidden_size, hidden_size, bias=False)

        # feature attention:
        self.W_F = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_F = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, o_all_time):
        """
        o_all_time: (batch, seq_len, hidden_size)
        Returns: concatenated_out: (batch, 2*hidden_size)
        """
        batch_size, seq_len, hidden_size = o_all_time.shape

        #time attention
        o_max_time = o_all_time[:, -1, :]  #last output: [batch, hidden_size]

        M_t = self.W_t(o_all_time) # (batch, seq_len, hidden_size)
        scores_time = torch.bmm(o_max_time.unsqueeze(1), M_t.transpose(1, 2)) # (batch, 1, seq_len)
        alpha_t = F.softmax(scores_time, dim=2)  # (batch, 1, seq_len
        output_T = torch.bmm(alpha_t, o_all_time)  # => (batch, 1, hidden_size)

        #feature attention
        M_f = torch.tanh(self.W_F(o_all_time)) #(batch, seq_len, hidden_size)
        scores_feature = self.v_F(M_f) #(batch, seq_len, 1)
        alpha_f = F.softmax(scores_feature, dim=1)  #(batch, seq_len, 1)
        output_F = torch.sum(alpha_f * o_all_time, dim=1, keepdim=True) #(batch, 1, hidden_size)

        output_T = output_T.squeeze(1)  #(batch, hidden_size)
        output_F = output_F.squeeze(1)  #(batch, hidden_size)
        concatenated_out = torch.cat([output_T, output_F], dim=1)  #(batch, 2*hidden_size)
        return concatenated_out

class SingleHeadAttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = CoupledAttentionGateLSTM(input_size, hidden_size)
        self.dual_attn = DualAttention(hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: logits (batch, num_classes)
        """
        o_all_time = self.lstm(x)  # (batch, seq_len, hidden_size)
        features = self.dual_attn(o_all_time)  # (batch, 2*hidden_size)
        logits = self.fc(features)  # (batch, num_classes)
        return logits
    
class MultiHeadTimeDimensionAttention(nn.Module):
    """
    for each head i in [1..n]:
      K_i = W_{i,k}*o_all + b_{i,k}
      V_i = W_{i,v}*o_all + b_{i,v}
      Q_i = W_{i,q}*o_last + b_{i,q}

      then,
      s_i = softmax(Q_i x K_i^T),  (B, 1, T)
      context_i = s_i x V_i,       (B, 1, Z_head)

      finally,
      (8) CV = Concat([context_1, ..., context_n]) => (B, 1, Z)

    - B: batch size
    - T: number of time steps
    - Z: total feature dimension (embedding dimension)
    - n: number of heads
    - Z_head: Z // n, dimension per head
    """

    def __init__(self,
                 d_model: int,    # total feature dimension Z
                 num_heads: int,  # n
                 bias: bool = True):
        super().__init__()
        assert d_model%num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads 

        # create separate (W_{i,k}, b_{i,k}), (W_{i,v}, b_{i,v}), (W_{i,q}, b_{i,q})
        # for each head i, they'll be stored in ModuleLists for clarity.
        # each W_{i,k} is a linear mapping from R^Z -> R^Z_head (from d_model to d_k). --> same for W_{i,v}, W_{i,q}.

        self.linear_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=bias)
                                       for _ in range(num_heads)])
        self.linear_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=bias)
                                       for _ in range(num_heads)])
        self.linear_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=bias)
                                       for _ in range(num_heads)])

    def forward(self, o_all: torch.Tensor, o_last: torch.Tensor) -> torch.Tensor:
        """
        o_all:  (B, T, Z) 
        o_last: (B, 1, Z) 
        returns: CV: (B, 1, Z)  
        """
        B, T, Z = o_all.shape
        head_contexts = []

        # loop over each head i
        for i in range(self.num_heads):
            K_i = self.linear_k[i](o_all) #(B, T, d_k) 
            V_i = self.linear_v[i](o_all) #(B, T, d_k)
            Q_i = self.linear_q[i](o_last) #(B, 1, d_k)
            scores = torch.bmm(Q_i, K_i.transpose(1, 2))  #(B, 1, T)
            s_i = F.softmax(scores, dim=-1)  #(B, 1, T)
            context_i = torch.bmm(s_i, V_i)  # (B, 1, d_k)
            head_contexts.append(context_i)

        # (8) CV = Concat([context_1, ..., context_n]) => shape (B, 1, n * d_k) = (B, 1, Z).
        CV = torch.cat(head_contexts, dim=2)  # cat along feature dim => (B, 1, num_heads * d_k) = (B, 1, d_model)
        return CV