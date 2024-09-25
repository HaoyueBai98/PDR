import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, FloatTensor
from transformers import BartTokenizer

from feature_env import FeatureEvaluator
from logger import info

SOS_ID = -1
EOS_ID = -1


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1,
                                                                                                        self.output_dim)

        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length  # total length to decode
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.eos_id = 2
        self.sos_id = 0
        self.pad_id = 1
        self.gpu = gpu

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):

        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:  # if not given x, then we are inferring!
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden,
                                                    encoder_outputs,
                                                    self.gpu)  # hidden is layer-wise out, out is final output
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def decode(step_, step_output_, step_attn_):
            decoder_outputs.append(step_output_)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn_)
            symbols_ = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols_)
            eos_batches = symbols_.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step_) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols_

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    """
    beam search version of lstm decoder
    """
    def beam_forward(self, x, encoder_hidden=None, encoder_outputs=None, beams=5):
        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:  # if not given x, then we are inferring!
            inference = True
        else:
            inference = False
        # 1. construct encoder_hidden as nBatch x Seq x Hidden,
        encoder_hidden = self._prepare_encoder_hidden_for_beam_search(encoder_hidden, beams)
        encoder_outputs = self._prepare_encoder_output_for_beam_search(encoder_outputs, beams)
        x, batch_size, length = self._validate_args(x, encoder_hidden,
                                                    encoder_outputs,
                                                    self.gpu)  # hidden is layer-wise out, out is final output

        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = [x]
        lengths = np.array([length] * batch_size)
        previous_p = [torch.zeros(batch_size, 1, device=encoder_outputs.device)]
        def decode(step_, step_output_, step_attn_):
            decoder_outputs.append(step_output_) # step out is the log p
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn_)
            prob_val, symbols_ = decoder_outputs[-1].topk(beams)
            current_accumulated_prob = prob_val + previous_p[-1]
            current_accumulated_prob = current_accumulated_prob.reshape(int(batch_size / beams), -1)
            current_symbols_ = symbols_.reshape(int(batch_size / beams), -1)
            # get previous symbols
            previous_symbols_ = sequence_symbols[-1]
            # now we can obtain the current beams value and current selected beam indice
            current_beams_value_, current_beams_indice = current_accumulated_prob.topk(beams, dim=-1)
            # we can obtain the previous beams id:
            previous_selected_beams = torch.div(current_beams_indice, 5, rounding_mode='floor')
            previous_symbols_ = torch.stack([prev_s[prev] for prev, prev_s in zip(previous_selected_beams, previous_symbols_.reshape(int(batch_size / beams), -1))]).reshape(batch_size, -1)
            sequence_symbols.pop(-1)
            sequence_symbols.append(previous_symbols_)
            # make the current:
            current_symbols_ = torch.stack([prev_s[prev] for prev, prev_s in zip(current_beams_indice, current_symbols_)]).reshape(batch_size, -1)
            sequence_symbols.append(current_symbols_)
            previous_p.append(current_beams_value_.reshape(-1, 1))
            eos_batches = current_symbols_.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step_) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return current_symbols_

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            """
            decoder_output (beam x batch) x 1 x token_size
            """
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden


    # encoder hidden B x S x H => (beams x B) x (S) x H
    def _prepare_encoder_hidden_for_beam_search(self, encoder_hidden, beams:int):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h.repeat(1, beams, 1) for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden.repeat(1, beams, 1)
        return encoder_hidden

    def _prepare_encoder_output_for_beam_search(self, encoder_output, beams:int):
        return encoder_output.repeat(beams, 1, 1)

    def _validate_args(self, x, encoder_hidden, encoder_outputs, gpu=0):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)

        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda(gpu)
            max_length = self.length
        else:
            max_length = x.size(1)

        return x, batch_size, max_length

    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self.forward(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, x: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        pass


class RNNDecoder(Decoder):

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu
                 ):
        super(RNNDecoder, self).__init__(
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length, gpu)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)  # class num + 1 eos

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)  # attention from decoder_output and encoder_output
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn


def construct_decoder(fe: FeatureEvaluator, args, tokenizer:BartTokenizer) -> Decoder:
    name = args.method_name
    size = fe.ds_size + 2
    info(f'Construct Decoder with method {name}...')
    if name == 'rnn':
        return RNNDecoder(
            layers=args.decoder_layers,
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.decoder_hidden_size,
            dropout=args.decoder_dropout,
            length=size,
            gpu=args.gpu
        )
    elif name == 'transformer':
        assert False
    else:
        assert False
