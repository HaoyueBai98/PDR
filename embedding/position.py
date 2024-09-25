import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.pe[:, :x.size(1)]


class ComplexNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.label_embedding = torch.nn.Embedding(opt['n_token'], opt['hdim'])
        self.frequency_embedding = torch.nn.Embedding(opt['n_token'], opt['hdim'])
        self.initial_phase_emb = torch.nn.Embedding(opt['n_token'], opt['hdim'])

    def get_embedding(self, x):
        amplitude = self.label_embedding(x)
        frequency = self.frequency_embedding(x)
        self.initial_phase_emb.weight = torch.nn.Parameter(self.initial_phase_emb.weight % (2 * math.pi))

        sentence_len = x.size(-1)
        pos_seq = torch.arange(1, sentence_len + 1, 1.0, device=amplitude.device)

        pos_seq = pos_seq.unsqueeze(0).unsqueeze(-1)
        pos_seq = pos_seq.repeat([x.size(0), 1, amplitude.size(-1)])

        dimension_bias = self.initial_phase_emb(x)

        enc_output_phase = torch.mul(pos_seq, frequency) + dimension_bias
        enc_output_real = amplitude * torch.cos(enc_output_phase)
        enc_output_image = amplitude * torch.sin(enc_output_phase)
        return enc_output_real, enc_output_image

    def forward(self, x):
        return self.get_embedding(x)


class OneHotPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=9):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        # print(self.pe.shape)
        return self.pe[pos]


if __name__ == '__main__':
    cpx = ComplexNN({'n_token': 10, 'hdim': 5})
    opx = OneHotPositionalEmbedding(128)
    lang = torch.LongTensor([1, 2, 3, 4])
    # print(cpx.forward(lang))
    print(opx.forward(lang))
