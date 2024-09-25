import torch.nn as nn

from .position import PositionalEmbedding
from .token import TokenEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vectors, vocab_size, embed_size, dropout=0.1, init_w=0.02):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        if vectors == None:
            self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, init_w=init_w)
        else:
            self.token = nn.Embedding.from_pretrained(vectors)
            self.token.weight.requires_grad = True

        # self.relation_emb = RelationEmbedding(vocab_size=3, embed_size=embed_size)

        # RelationEmbedding(torch.zeros())
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, pos=True):
        if pos:
            x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)# + self.position(sequence)
        else :
            x = self.token(sequence)
        return self.dropout(x)
