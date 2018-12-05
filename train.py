'''
This script handling the training process.
'''

import os
import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
import dataset
import vocab
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

import pdb

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    # pdb.set_trace()
    # (Pdb) a
    # pred = tensor([2136,  956,  996,  ...,    0,    0,    0], device='cuda:0')
    # gold = tensor([2136,  956,  996,  ...,    0,    0,    0], device='cuda:0')
    # smoothing = True
    # (Pdb) print(pred.size(), gold.size())
    # torch.Size([1920]) torch.Size([1920])

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # pdb.set_trace()
        # (Pdb) gold
        # tensor([1419, 2911,  397,  ...,    0,    0,    0], device='cuda:0')
        # (Pdb) gold.size()
        # torch.Size([1920])
        # (Pdb) one_hot.size()
        # torch.Size([1920, 3149])

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # if one_hot == 0:
        #   one_hot == eps/(n_class - 1)
        # else:
        #   one_hot = (1 - eps)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # pdb.set_trace()
        # smoothing = True
        # (Pdb) print(type(src_seq), src_seq.size(), src_seq)
        # <class 'torch.Tensor'> torch.Size([64, 26]) tensor([[   2, 2434, 1736,  ...,    0,    0,    0],
        #         [   2, 2434,   71,  ...,    0,    0,    0],
        #         [   2, 1557, 1071,  ...,    0,    0,    0],
        #         ...,
        #         [   2, 2434,  729,  ...,    0,    0,    0],
        #         [   2, 1557, 2010,  ...,    0,    0,    0],
        #         [   2, 1252,    1,  ...,    0,    0,    0]], device='cuda:0')
        # (Pdb) print(type(src_pos), src_pos.size(), src_pos)
        # <class 'torch.Tensor'> torch.Size([64, 26]) tensor([[1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         ...,
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0]], device='cuda:0')
        # (Pdb) print(type(tgt_seq), tgt_seq.size(), tgt_seq)
        # <class 'torch.Tensor'> torch.Size([64, 25]) tensor([[   2, 2136,  645,  ...,    0,    0,    0],
        #         [   2, 2136, 2296,  ...,    0,    0,    0],
        #         [   2,  251, 1146,  ...,    0,    0,    0],
        #         ...,
        #         [   2, 2136, 1914,  ...,    0,    0,    0],
        #         [   2,  251, 1484,  ...,    0,    0,    0],
        #         [   2, 2136,  164,  ...,    0,    0,    0]], device='cuda:0')
        # (Pdb) print(type(tgt_pos), tgt_pos.size(), tgt_pos)
        # <class 'torch.Tensor'> torch.Size([64, 25]) tensor([[1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         ...,
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0]], device='cuda:0')


        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
        # (Pdb) print(type(pred), pred.size(), pred)
        # <class 'torch.Tensor'> torch.Size([1536, 3149]) tensor([[-0.6007,  2.4810, -1.2152,  ..., -1.2766, -1.5151, -0.7077],
        #         [-0.1354,  8.6667, -3.4856,  ..., -1.4034, -0.7370, -0.5578],
        #         [-0.6275,  4.8268, -0.1633,  ..., -2.6749, -1.8535, -0.6473],
        #         ...,
        #         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        #         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        #         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
        #        device='cuda:0', grad_fn=<ViewBackward>)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        # (Pdb) print(type(loss), loss.size(), loss)
        # <class 'torch.Tensor'> torch.Size([]) tensor(1821.7072, device='cuda:0', grad_fn=<SumBackward0>)
        # (Pdb) print(type(n_correct), n_correct)
        # <class 'int'> 667

        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    # (Pdb) print(opt.log)
    # None
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    # pdb.set_trace()
    # (Pdb) a
    # model = Transformer(
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
    # training_data = <torch.utils.data.dataloader.DataLoader object at 0x7fb9c3cef0f0>
    # validation_data = <torch.utils.data.dataloader.DataLoader object at 0x7fb9c3cef320>
    # optimizer = <transformer.Optim.ScheduledOptim object at 0x7fb960e680f0>
    # device = device(type='cuda')
    # opt = Namespace(batch_size=64, cuda=True, d_inner_hid=2048, d_k=64, d_model=512, d_v=64, d_word_vec=512, data='data/multi30k.atok.low.pt', dropout=0.1, embs_share_weight=False, epoch=200, label_smoothing=True, log=None, max_token_seq_len=52, n_head=8, n_layers=6, n_warmup_steps=4000, no_cuda=False, proj_share_weight=True, save_mode='best', save_model='trained', src_vocab_size=2911, tgt_vocab_size=3149)



    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', required=True)

    parser.add_argument('-train_atok', required=True)
    parser.add_argument('-valid_atok', required=True)

    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=8)

    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    train_atok = torch.load(opt.train_atok)
    valid_atok = torch.load(opt.valid_atok)

    train_vocab = vocab.Vocab(train_atok['settings'].vocab)

    training_data = dataset.translation_dataloader(train_atok, opt.batch_size, shuffle=True)
    validation_data = dataset.translation_dataloader(valid_atok, opt.batch_size, shuffle=False)


    # data = torch.load(opt.data)
    opt.max_token_seq_len = train_atok['settings'].max_seq_len

    # training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = train_vocab.size()
    opt.tgt_vocab_size =  train_vocab.size()

    #========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
    #         'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    if os.path.exists("trained.chkpt"):
        x = torch.load("trained.chkpt")
        # print(type(x["model"]))
        transformer.load_state_dict(x["model"])

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)



if __name__ == '__main__':
    main()
