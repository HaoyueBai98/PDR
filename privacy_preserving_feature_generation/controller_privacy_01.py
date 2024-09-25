import torch
import torch.nn as nn
from transformers import BartTokenizer

from feature_env_privacy_01 import FeatureEvaluator
from decoder_privacy_01 import construct_decoder
from encoder_privacy_01 import construct_encoder

import os
SOS_ID = 0
EOS_ID = 0


# gradient based automatic feature selection
class GAFS(nn.Module):
    def __init__(self,
                 fe:FeatureEvaluator,
                 args,
                 tokenizer,
                 ):
        super(GAFS, self).__init__()
        self.params = args
        self.style = args.method_name
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args, tokenizer)
        self.decoder = construct_decoder(fe, args, tokenizer)
        if self.style == 'rnn':
            self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, feat_emb, predict_value, predict_sens = self.encoder.forward(input_variable)
        decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, feat, predict_sens

    def generate_new_feature(self, input_variable, privacy_boundary, predict_lambda=1, direction='-', beams=None):

        encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
            self.encoder.infer(input_variable, privacy_boundary, predict_lambda, direction=direction)
        new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
        if beams is None or beams == 0:
            decoder_outputs, decoder_hidden, ret = self.decoder.forward(x=None, encoder_hidden=new_encoder_hidden,
                                                                             encoder_outputs=new_encoder_outputs)
        else:
            decoder_outputs, decoder_hidden, ret = self.decoder.beam_forward(x=None, encoder_hidden=new_encoder_hidden,
                                                                         encoder_outputs=new_encoder_outputs, beams=beams)
        new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_feat_seq

    @staticmethod
    def from_pretrain(path, fe, params, tokenizer: BartTokenizer, epoch, keyword=''):
        encoder_dict = torch.load(
            os.path.join(path, f'{params.task_name}', f'model_dmp{keyword}', f'{params.task_name}_{epoch}.encoder.pt'))
        decoder_dict = torch.load(
            os.path.join(path, f'{params.task_name}', f'model_dmp{keyword}', f'{params.task_name}_{epoch}.decoder.pt'))
        model = GAFS(fe, params, tokenizer)
        model.encoder.load_state_dict(encoder_dict)
        model.decoder.load_state_dict(decoder_dict)
        return model

    def save_to(self, path, epoch, keyword=''):
        base = os.path.join(path, f'{self.params.task_name}', f'model_dmp{keyword}')
        if not os.path.exists(base):
            os.mkdir(base)
        if keyword is None:
            keyword = ''
        torch.save(self.encoder.state_dict(),
                   os.path.join(path, f'{self.params.task_name}', f'model_dmp{keyword}', f'{self.params.task_name}_{epoch}.encoder.pt'))
        torch.save(self.decoder.state_dict(),
                   os.path.join(path, f'{self.params.task_name}', f'model_dmp{keyword}',
                                f'{self.params.task_name}_{epoch}.decoder.pt'))

