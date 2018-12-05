''' Define the Layers '''
import pdb

import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        # pdb.set_trace()
        # (Pdb) a
        # self = EncoderLayer(
        #   (slf_attn): MultiHeadAttention(
        #     (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #     (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #     (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #     (attention): ScaledDotProductAttention(
        #       (dropout): Dropout(p=0.1)
        #       (softmax): Softmax()
        #     )
        #     (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #     (fc): Linear(in_features=512, out_features=512, bias=True)
        #     (dropout): Dropout(p=0.1)
        #   )
        #   (pos_ffn): PositionwiseFeedForward(
        #     (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #     (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #     (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #     (dropout): Dropout(p=0.1)
        #   )
        # )
        # d_model = 512
        # d_inner = 2048
        # n_head = 8
        # d_k = 64
        # d_v = 64
        # dropout = 0.1


    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        # pdb.set_trace()
        # (Pdb) print(enc_input.size(), non_pad_mask.size(), slf_attn_mask.size())
        # torch.Size([64, 29, 512]) torch.Size([64, 29, 1]) torch.Size([64, 29, 29])
        # (Pdb) print(enc_output.size(), enc_slf_attn.size())
        # torch.Size([64, 29, 512]) torch.Size([512, 29, 29])

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        # pdb.set_trace()
        # (Pdb) a
        # self = DecoderLayer(
        #   (slf_attn): MultiHeadAttention(
        #     (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #     (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #     (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #     (attention): ScaledDotProductAttention(
        #       (dropout): Dropout(p=0.1)
        #       (softmax): Softmax()
        #     )
        #     (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #     (fc): Linear(in_features=512, out_features=512, bias=True)
        #     (dropout): Dropout(p=0.1)
        #   )
        #   (enc_attn): MultiHeadAttention(
        #     (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #     (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #     (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #     (attention): ScaledDotProductAttention(
        #       (dropout): Dropout(p=0.1)
        #       (softmax): Softmax()
        #     )
        #     (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #     (fc): Linear(in_features=512, out_features=512, bias=True)
        #     (dropout): Dropout(p=0.1)
        #   )
        #   (pos_ffn): PositionwiseFeedForward(
        #     (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #     (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #     (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #     (dropout): Dropout(p=0.1)
        #   )
        # )
        # d_model = 512
        # d_inner = 2048
        # n_head = 8
        # d_k = 64
        # d_v = 64
        # dropout = 0.1


    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        # pdb.set_trace()
        # (Pdb) print(dec_input.size(), enc_output.size(), non_pad_mask.size(), slf_attn_mask.size(), dec_enc_attn_mask.size())
        # torch.Size([64, 27, 512]) torch.Size([64, 29, 512]) torch.Size([64, 27, 1]) torch.Size([64, 27, 27]) torch.Size([64, 27, 29])

        # (Pdb) print(dec_output.size(), dec_slf_attn.size(), dec_enc_attn.size())
        # torch.Size([64, 27, 512]) torch.Size([512, 27, 27]) torch.Size([512, 27, 29])

        return dec_output, dec_slf_attn, dec_enc_attn
