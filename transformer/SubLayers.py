''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        # pdb.set_trace()
        # (Pdb) a
        # self = ScaledDotProductAttention(
        #   (dropout): Dropout(p=0.1)
        #   (softmax): Softmax()
        # )
        # temperature = 8.0
        # attn_dropout = 0.1

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        # pdb.set_trace()
        # (Pdb) print(q.size())
        # torch.Size([512, 29, 64])

        # (Pdb) print(k.size())
        # torch.Size([512, 29, 64])
        # (Pdb) print(k.transpose(1, 2).size())
        # torch.Size([512, 64, 29])

        # (Pdb) print(v.size())
        # torch.Size([512, 29, 64])

        # (Pdb) print(mask.size())
        # torch.Size([512, 29, 29])

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        # pdb.set_trace()
        # (Pdb) a
        # self = MultiHeadAttention(
        #   (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #   (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #   (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #   (attention): ScaledDotProductAttention(
        #     (dropout): Dropout(p=0.1)
        #     (softmax): Softmax()
        #   )
        #   (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #   (fc): Linear(in_features=512, out_features=512, bias=True)
        #   (dropout): Dropout(p=0.1)
        # )
        # n_head = 8
        # d_model = 512
        # d_k = 64
        # d_v = 64
        # dropout = 0.1


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # pdb.set_trace()
        # (Pdb) print(mask.size())
        # torch.Size([64, 25, 25])

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        # pdb.set_trace()
        # (Pdb) print(mask.size())
        # torch.Size([512, 29, 29])

        # (Pdb) print(q.size(), k.size(), v.size())
        # torch.Size([512, 29, 64]) torch.Size([512, 29, 64]) torch.Size([512, 29, 64])

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

        # pdb.set_trace()
        # self = PositionwiseFeedForward(
        #   (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #   (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #   (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #   (dropout): Dropout(p=0.1)
        # )
        # d_in = 512
        # d_hid = 2048
        # dropout = 0.1


    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        # pdb.set_trace()
        # (Pdb) print(x.size(), output.size())
        # torch.Size([64, 25, 512]) torch.Size([64, 25, 512])

        return output
