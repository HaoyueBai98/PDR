from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer

from feature_env import FeatureEvaluator
from utils.datacollection.logger import info
import pdb
import time
class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.sens_bound = []
        self.sens_bound2 = []

    def infer(self, x, privacy_boundary, predict_lambda, direction='-'):
        start_time = time.time()
        current_privacy_boundary = privacy_boundary.cuda()
        current_privacy_boundary_item =  privacy_boundary.cuda().mean().item()
        out_flag = 1
        encoder_outputs, encoder_hidden, seq_emb, predict_value, _ = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        for step in range(predict_lambda):
            if direction == '+':
                # new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
                new_encoder_outputs = encoder_outputs + grads_on_outputs

            elif direction == '-':
                new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
            else:
                raise ValueError('Direction must be + or -, got {} instead'.format(direction))
            
            for _ in range(100):
                new_predict_snes = self.get_predict_snes(new_encoder_outputs)
                pgd_mask = new_predict_snes > current_privacy_boundary # 0.01
                
                self.sens_bound2.append(new_predict_snes.mean().item())
                
                # current_privacy_boundary = self.get_predict_snes(new_encoder_outputs)
                new_privacy_boundary = self.get_predict_snes(new_encoder_outputs)
                if not pgd_mask.any():
                # if new_privacy_boundary.mean().item() < current_privacy_boundary_item:
                    current_privacy_boundary = new_privacy_boundary
                    current_privacy_boundary_item = new_privacy_boundary.mean().item()
                    self.sens_bound.append(current_privacy_boundary.mean().item())
                    # out_flag = 0
                    break
                # out_flag = 1
                pgd_mask = pgd_mask.unsqueeze(-1).expand_as(new_encoder_outputs)
                new_encoder_outputs.requires_grad_(True)            
                pdb.set_trace()
                info('------------In PGD step---------------')
                grads_on_outputs = torch.autograd.grad(new_predict_snes, new_encoder_outputs, torch.ones_like(new_predict_snes))[0]
                with torch.no_grad():
                    new_encoder_outputs[pgd_mask] -=  grads_on_outputs[pgd_mask]
            # new_predict_snes = self.get_predict_snes(new_encoder_outputs)            
            # grads_on_outputs = torch.autograd.grad(new_predict_snes, new_encoder_outputs, torch.ones_like(new_predict_snes))[0]
            # with torch.no_grad():
            #     new_encoder_outputs -=  grads_on_outputs               
                    # new_encoder_outputs -=  100 * grads_on_outputs

                # new_encoder_outputs.requires_grad_(False)
            # if not out_flag:
            #     current_privacy_boundary = self.get_predict_snes(new_encoder_outputs)
            #     info('----------------Narrowing the boundaries of privacy-----------------------')
        # info('----------------------------------------------------------------------')
        # info(f'------------------{time.time()-start_time}---------------------------')
        # info('----------------------------------------------------------------------')
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass


class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout
                 ):
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        self.regressor_sens = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)
        # TODO add length constrain in here as DiffER
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out_value = self.regressor(out)
        predict_value = torch.sigmoid(out_value)

        out_sens = self.regressor_sens(out)
        predict_sens = torch.sigmoid(out_sens)

        return encoder_outputs, encoder_hidden, seq_emb, predict_value, predict_sens

    def get_predict_snes(self, encoder_outputs):
        out = torch.mean(encoder_outputs, dim=1)
        out = F.normalize(out, 2, dim=-1)
        out = self.mlp(out)
        out_sens = self.regressor_sens(out)
        predict_sens = torch.sigmoid(out_sens)
        return predict_sens


def construct_encoder(fe: FeatureEvaluator, args, tokenizer:BartTokenizer) -> Encoder:
    name = args.method_name
    info(f'Construct Encoder with method {name}...')
    if name == 'rnn':
        return RNNEncoder(
            layers=args.encoder_layers,
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.encoder_hidden_size,
            dropout=args.encoder_dropout,
            mlp_layers=args.mlp_layers,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_dropout=args.encoder_dropout
        )
    elif name == 'transformer':
        assert False
    else:
        assert False
