# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018-11-28 14:04:41
# ***
# ************************************************************************************/

import torch


class Config(object):
    """Base configuration class."""

    name = "Transformer"

    cuda = True

    attention_probs_dropout_prob = 0.1
    directionality = "bidi"
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1
    hidden_size = 768
    initializer_range = 0.02
    intermediate_size = 3072
    max_position_embeddings = 512
    num_attention_heads = 12
    num_hidden_layers = 12
    pooler_fc_size = 768
    pooler_num_attention_heads = 12
    pooler_num_fc_layers = 3
    pooler_size_per_head = 128
    pooler_type = "first_token_transform"
    type_vocab_size = 2
    vocab_size = 21128

    def __init__(self):
        if self.cuda:
            self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")


    def dump(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()

def Test():
    c = Config()
    c.dump()

# Test()