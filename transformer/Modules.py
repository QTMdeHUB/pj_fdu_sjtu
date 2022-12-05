import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # 输入：q.shape = k.shape = v.shape = bs * n_head * max_len * d_model
        #   mask.shape = bs * 1 * len * len
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # attn.shape: bs * n_head * len * len

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
