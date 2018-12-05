import numpy as np
import torch
import torch.utils.data
import pdb

from transformer import Constants


def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array(
        [inst + [Constants.PAD] * (max_len - len(inst)) for inst in insts])

    batch_pos = np.array([[
        pos_i + 1 if w_i != Constants.PAD else 0
        for pos_i, w_i in enumerate(inst)
    ] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    # print("batch_seq:", batch_seq.size(), batch_seq)
    # print("batch_pos:", batch_pos.size(), batch_pos)

    return batch_seq, batch_pos


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, atok):
        self.atok = atok
        self.n_examples = len(atok['data']['src'])

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.atok['data']['src'][idx], self.atok['data']['tgt'][idx]


def translation_dataloader(atok, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        TranslationDataset(atok),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=shuffle)

    return loader


def Test():
    atok = torch.load("data/valid/valid.en-zh.atok")
    print("Max Seq Len: ", atok['settings'].max_seq_len)
    valid_dataloader = translation_dataloader(atok, 1)

    for batch in valid_dataloader:
        src_seq, src_pos, tgt_seq, tgt_pos = batch
        print("src_seq: ", src_seq)
        print("src_pos: ", src_pos)
        print("tgt_seq: ", tgt_seq)
        print("tgt_pos: ", tgt_pos)
        break


# Test()