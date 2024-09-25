import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, init_w=None):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        if init_w is not None:
            self.weight.data.uniform_(0, init_w)