''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
import pdb

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    # pdb.set_trace()
    # (Pdb) seq.size()
    # torch.Size([64, 25])
    # (Pdb) seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1).size()
    # torch.Size([64, 25, 1])

    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    # pdb.set_trace()
    # (Pdb) a
    # n_position = 53
    # d_hid = 512
    # (Pdb) sinusoid_table.shape
    # (53, 512)

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    # pdb.set_trace()
    # (Pdb) seq_q.size()
    # torch.Size([64, 27])
    # (Pdb) padding_mask = seq_k.eq(Constants.PAD)
    # (Pdb) padding_mask.size()
    # torch.Size([64, 27])
    # (Pdb) padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    # (Pdb) padding_mask.size()
    # torch.Size([64, 27, 27])

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    # pdb.set_trace()
    # (Pdb) seq.size()
    # torch.Size([64, 27])
    # (Pdb) subsequent_mask.size()
    # torch.Size([27, 27])

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    # pdb.set_trace()
    # (Pdb) subsequent_mask.size()
    # torch.Size([64, 27, 27])

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # pdb.set_trace()
        # (Pdb) a
        # self = Encoder(
        #   (src_word_emb): Embedding(2911, 512, padding_idx=0)
        #   (position_enc): Embedding(53, 512)
        #   (layer_stack): ModuleList(
        #     (0): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (1): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (2): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (3): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (4): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (5): EncoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #   )
        # )
        # n_src_vocab = 2911
        # len_max_seq = 52
        # d_word_vec = 512
        # n_layers = 6
        # n_head = 8
        # d_k = 64
        # d_v = 64
        # d_model = 512
        # d_inner = 2048
        # dropout = 0.1


    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        # pdb.set_trace()
        # (Pdb) print(return_attns)
        # False
        # (Pdb) print(src_seq.size(), src_pos.size())
        # torch.Size([64, 29]) torch.Size([64, 29])
        # (Pdb) print(enc_output.size())
        # torch.Size([64, 29, 512])

        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # pdb.set_trace()
        # (Pdb) a
        # self = Decoder(
        #   (tgt_word_emb): Embedding(3149, 512, padding_idx=0)
        #   (position_enc): Embedding(53, 512)
        #   (layer_stack): ModuleList(
        #     (0): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (1): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (2): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (3): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (4): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #     (5): DecoderLayer(
        #       (slf_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (enc_attn): MultiHeadAttention(
        #         (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #         (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #         (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #         (attention): ScaledDotProductAttention(
        #           (dropout): Dropout(p=0.1)
        #           (softmax): Softmax()
        #         )
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (fc): Linear(in_features=512, out_features=512, bias=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #       (pos_ffn): PositionwiseFeedForward(
        #         (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #         (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #         (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #         (dropout): Dropout(p=0.1)
        #       )
        #     )
        #   )
        # )
        # n_tgt_vocab = 3149
        # len_max_seq = 52
        # d_word_vec = 512
        # n_layers = 6
        # n_head = 8
        # d_k = 64
        # d_v = 64
        # d_model = 512
        # d_inner = 2048
        # dropout = 0.1


    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        # pdb.set_trace()
        # (Pdb) print(tgt_seq.size(), tgt_pos.size(), src_seq.size(), enc_output.size(), return_attns)
        # torch.Size([64, 29]) torch.Size([64, 29]) torch.Size([64, 28]) torch.Size([64, 28, 512]) False

        # (Pdb) print(dec_output.size())
        # torch.Size([64, 29, 512])

        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # tgt_emb_prj_weight_sharing == True 
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        # emb_src_tgt_weight_sharing == False
        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

        # pdb.set_trace()
        # (Pdb) a
        # self = Transformer(
        #   (encoder): Encoder(
        #     (src_word_emb): Embedding(2911, 512, padding_idx=0)
        #     (position_enc): Embedding(53, 512)
        #     (layer_stack): ModuleList(
        #       (0): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (1): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (2): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (3): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (4): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (5): EncoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #     )
        #   )
        #   (decoder): Decoder(
        #     (tgt_word_emb): Embedding(3149, 512, padding_idx=0)
        #     (position_enc): Embedding(53, 512)
        #     (layer_stack): ModuleList(
        #       (0): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (1): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (2): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (3): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (4): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #       (5): DecoderLayer(
        #         (slf_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (enc_attn): MultiHeadAttention(
        #           (w_qs): Linear(in_features=512, out_features=512, bias=True)
        #           (w_ks): Linear(in_features=512, out_features=512, bias=True)
        #           (w_vs): Linear(in_features=512, out_features=512, bias=True)
        #           (attention): ScaledDotProductAttention(
        #             (dropout): Dropout(p=0.1)
        #             (softmax): Softmax()
        #           )
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (fc): Linear(in_features=512, out_features=512, bias=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #         (pos_ffn): PositionwiseFeedForward(
        #           (w_1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        #           (w_2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        #           (layer_norm): LayerNorm(torch.Size([512]), eps=1e-05, elementwise_affine=True)
        #           (dropout): Dropout(p=0.1)
        #         )
        #       )
        #     )
        #   )
        #   (tgt_word_prj): Linear(in_features=512, out_features=3149, bias=False)
        # )
        # n_src_vocab = 2911
        # n_tgt_vocab = 3149
        # len_max_seq = 52
        # d_word_vec = 512
        # d_model = 512
        # d_inner = 2048
        # n_layers = 6
        # n_head = 8
        # d_k = 64
        # d_v = 64
        # dropout = 0.1
        # tgt_emb_prj_weight_sharing = True
        # emb_src_tgt_weight_sharing = False


    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        # pdb.set_trace()
        # (Pdb) print(src_seq.size(), src_pos.size(), tgt_seq.size(), tgt_pos.size())
        # torch.Size([64, 28]) torch.Size([64, 28]) torch.Size([64, 25]) torch.Size([64, 25])

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        # torch.Size([64, 24]) torch.Size([64, 24]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        # pdb.set_trace()
        # (Pdb) print(seq_logit.size())
        # torch.Size([64, 22, 3149])

        # pdb.set_trace()
        # (Pdb) print(enc_output.size())
        # torch.Size([64, 26, 512])
        # (Pdb) print(dec_output.size())
        # torch.Size([64, 27, 512])

        # (Pdb) seq_logit.view(-1, seq_logit.size(2)).size()
        # torch.Size([1728, 3149])

        return seq_logit.view(-1, seq_logit.size(2))
