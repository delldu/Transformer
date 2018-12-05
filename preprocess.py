"""Handling the data io"""

import argparse
import torch
import transformer.Constants as Constants
import vocab
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def check_examples(filename, max_words, vocab):
    ''' Convert file into word seq lists via vocab '''

    src_tgt_sep = "----"

    with open(filename) as f:
        for line in f:
            s = line.split(src_tgt_sep)
            if len(s) != 2:
                logger.Info('Line {} miss SEP: {}'.format(line, src_tgt_sep))
                continue

            src_tokens = vocab.split(s[0])
            if (len(src_tokens) > max_words):
                logger.Info('Source {} has {} tokens, more than {}.'.format(s[0], len(src_tokens), max_words))
            vocab.checkoov(s[0])

            tgt_tokens = vocab.split(s[1])
            if (len(tgt_tokens) > max_words):
                logger.Info('Target {} has {} tokens, more than {}.'.format(s[1], len(tgt_tokens), max_words))
            vocab.checkoov(s[1])



def read_examples(filename, max_words, vocab):
    ''' Convert file into word seq lists via vocab '''

    src_tgt_sep = "----"
    trimmed_count = 0
    src_lists = []
    tgt_lists = []
    count = 0
    with open(filename) as f:
        for line in f:
            s = line.split(src_tgt_sep)
            if len(s) != 2:
                continue

            src_tokens = vocab.split(s[0])
            if (len(src_tokens) > max_words):
                trimmed_count += 1
                src_tokens = src_tokens[:max_words]
            src_indexs = vocab.indexs(src_tokens)
            src_indexs = [Constants.BOS] + src_indexs + [Constants.EOS]

            tgt_tokens = vocab.split(s[1])
            if (len(tgt_tokens) > max_words):
                trimmed_count += 1
                tgt_tokens = tgt_tokens[:max_words]
            tgt_indexs = vocab.indexs(tgt_tokens)
            tgt_indexs = [Constants.BOS] + tgt_indexs + [Constants.EOS]

            src_lists.append(src_indexs)
            tgt_lists.append(tgt_indexs)

            count += 1
            if count % 1000 == 0:
              print("Preprocess examples lines {} ... ".format(count))

    if trimmed_count > 0:
        logger.info('{} examples are trimmed for the max sentence length {}.'
              .format(trimmed_count, max_words))

    return src_lists, tgt_lists


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file', required=True)
    parser.add_argument('-output_file', required=True)
    parser.add_argument('-max_len', '--max_seq_len', type=int, default=64)
    parser.add_argument('-vocab', type=str, default="model/multilingual/vocab.txt")
    parser.add_argument('-checkoov', default=False)

    opt = parser.parse_args()

    multi_language_vocab = vocab.Vocab(opt.vocab)

    if (opt.checkoov):
        check_examples(opt.input_file, opt.max_seq_len, multi_language_vocab)

    src_lists, tgt_lists = read_examples(opt.input_file, opt.max_seq_len, multi_language_vocab)

    data = {
        'settings': opt,
        'data': {
            'src': src_lists,
            'tgt': tgt_lists}
        }

    logger.info('Dumping the processed data to file {}'.format(opt.output_file))
    torch.save(data, opt.output_file)
    logger.info('Finish.')

if __name__ == '__main__':
    main()
