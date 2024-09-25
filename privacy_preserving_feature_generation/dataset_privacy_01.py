import math
from typing import List

import numpy
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

from Record_privacy_01 import TransformationRecordList
from feature_env_privacy_01 import TransformationFeatureEvaluator
from gen_config import init_config

pad_idx = 1
eos_idx = 2

base_path = './data'


# out = {
#             'id': item,
#             'encoder_inputs':source_seq, # source seq may contain noise
#             'decoder_inputs':decoder_seq, # decoder input is right-shift original seq
#             'encoder_target':original_p, # the performance
#             'decoder_target':original_deocer_target # original decoder target is the same as givens
#         }

# def dataset_collate(batch, train=True):
#
#
#     # labels = [i['target_seq'] for i in batch]
#     # inputs = {}
#     # inputs['input_ids'] = torch.LongTensor(input_sent)
#     # inputs['attention_mask'] = torch.cat(attention_mask)
#     # inputs['labels'] = torch.LongTensor(labels)
#     # inputs['performance'] = torch.tensor([i['target_p'] for i in batch])
#
#
#     return inputs


class DenoiseDataModule:
    def __init__(self, params):
        # tokenizer only to call the subsequent interface, itself useless, can be replaced by any relevant file.
        suppose_vocab = f'vocab.json'
        suppose_merge_file = f'merges.txt'
        self.tokenizer = BartTokenizer(suppose_vocab, suppose_merge_file)
        
        self.fe = TransformationFeatureEvaluator(params.task_name, rec_length_strategy='length', percentage=0.95,
                                                 from_scratch=False)


        r_list = list(self.fe.records.r_list)
        train_top_k = min(len(r_list), params.train_top_k)
        # train, test = train_test_split(r_list, train_size=0.9, test_size=0.1)
        self.train_dataset = DenoiseDataset(r_list=r_list, num=params.num, ds_size=self.fe.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=train_top_k)

        # self.val_dataset = DenoiseDataset(r_list=test, num=0, ds_size=self.fe.ds_size, tokenizer=self.tokenizer,
        #                                   mask_whole_op_p=0, mask_op_p=0, disorder_p=0, top=params.top)
        self.val_dataset = DenoiseDataset(r_list=r_list, num=params.num, ds_size=self.fe.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=params.top_k,
                                            train=False)

        self.infer_dataset = DenoiseDataset(r_list=r_list, num=0, ds_size=self.fe.ds_size, tokenizer=self.tokenizer,
                                            mask_whole_op_p=0, mask_op_p=0, disorder_p=0, minmax=False, top=params.top_k,
                                            train=False)

        self.batch_size = params.batch_size

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=128)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader

    def infer_dataloader(self):
        loader = DataLoader(self.infer_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader


class DenoiseDataset(Dataset):
    def __init__(self, r_list, ds_size, tokenizer: BartTokenizer, num=10
                 , mask_whole_op_p=0.1, mask_op_p=0.3, disorder_p=0.1, minmax=True, top=None, train=True):
        # step1 initialize the feature evaluator
        # - in here 0.95 will cover most case and providing most suitable lengthy seq
        self.num = num
        self.max_seq_length = ds_size + 2  # with <s> and <\s>
        self.vocab = tokenizer.get_vocab()
        self.tokenizer = tokenizer
        self.original_records = list(r_list)
        self.operation_list = []  # for normal mask
        self.ops_list = []  # store the split ops [for permuatation and specific mask]
        self.performance_list = []  # store the truth label
        self.sensitive_list = []
        self.length_op_list = []
        self.train = train
        performance_list = []
        sensitive_list = []
        for record in self.original_records:
            performance_list.append(record.performance)
            

        if top is not None:
            _, indices = torch.topk(torch.tensor(performance_list), top, dim=0)
        else:
            indices = range(len(performance_list))
        performance_list = []

        for indice in tqdm.tqdm(indices, desc='denoise data gen'):
            record = self.original_records[indice]
            self.ops_list.append(record.ops)
            performance_list.append(record.performance)
            sensitive_list.append(record.sensitive)
            self.operation_list.append(record.operation)
            tmp = []
            for op_ in record.ops:
                tmp.append(len(op_))
            self.length_op_list.append(tmp)
            if num > 1:
                all, p, s = record.get_permutated_ops(num)
                for ops_, p_, s_ in zip(all, p, s):
                    self.ops_list.append(ops_)
                    performance_list.append(p_)
                    sensitive_list.append(s_)
                    source_seq = []
                    for i in ops_:
                        source_seq.extend([str(j) for j in i])
                        source_seq.append('4')
                    source_seq = source_seq[:-1]
                    self.operation_list.append(source_seq)
                    tmp = []
                    for op_ in ops_[:-1]:
                        tmp.append(len(op_))
                    self.length_op_list.append(tmp)

        if minmax:
            min_val = min(performance_list)
            max_val = max(performance_list)
            self.original_performance = performance_list.copy()
            if min_val == max_val:
                self.performance_list = [[1] * len(performance_list)]
            self.performance_list = [[(i - min_val) / (max_val - min_val)] for i in performance_list]

            min_val = min(sensitive_list)
            max_val = max(sensitive_list)

            sensitive_list = [item.item() if isinstance(item, numpy.ndarray) else 1 for item in sensitive_list]

            
            
            self.original_sensitive = sensitive_list.copy()
            if min_val == max_val:
                self.sensitive_list = [[1] * len(sensitive_list)]
            self.sensitive_list = [[(i - min_val) / (max_val - min_val)] for i in sensitive_list]
        else:
            self.original_performance = [[i] for i in performance_list]
            self.performance_list = self.original_performance.copy()

            self.original_sensitive = [[i.item()] for i in sensitive_list]
            self.sensitive_list = self.original_sensitive.copy()
        '''
        special token keep for BART training
        '''
        # self.mask = '<mask>'
        '''
        some config for sample and gen
        '''
        self.mask_whole_op_p = mask_whole_op_p
        self.disorder_p = disorder_p
        self.mask_op_p = mask_op_p
        self.item_list = []
        self.count = 1
        for i in range(len(self)):
            processed_item = self.process_item(i)
            if len(self.item_list) >0 and len(processed_item['encoder_input']) != len(self.item_list[-1]['encoder_input']):
                self.item_list.append(self.item_list[-1])
                continue
            self.item_list.append(processed_item)
            self.count += 1

    def __getitem__(self, item):
        return self.item_list[item]
        # original_seq = [str(i) for i in self.operation_list[item]]
        # original_ops = self.ops_list[item]
        # original_p = self.performance_list[item]
        # ops = original_ops.copy()
        # if self.mask_whole_op_p > 0:
        #     ops = self.add_whole_op_mask(ops, self.mask_whole_op_p)
        # if self.disorder_p > 0:
        #     ops = self.disorder_ops(ops, self.disorder_p)
        # source_seq = []
        # for i in ops:
        #     source_seq.extend([str(j) for j in i])
        #     source_seq.append('4')
        # source_seq = source_seq[:-1]
        # if self.mask_op_p > 0:
        #     source_seq = self.add_op_mask(source_seq, self.mask_op_p)
        # # print(len(original_seq), len(source_seq))
        # assert len(original_seq) == len(source_seq)
        # # original_seq = [0] + [self.vocab[i] for i in original_seq] + [2]
        # # source_seq = [0] + [self.vocab[i] for i in source_seq] + [2]
        # # original_seq, original_mask = self.padding_seq(original_seq)
        # decoder_seq, decoder_mask = self.padding_seq(original_seq, for_encoder=False)
        # original_deocer_target, _ = self.padding_seq(original_seq)
        # source_seq, source_mask = self.padding_seq(source_seq)
        #
        # if self.train:
        #     sample = {
        #         'encoder_input': torch.LongTensor(source_seq),
        #         'decoder_input': torch.LongTensor(decoder_seq),
        #         'encoder_target': torch.FloatTensor(original_p),  # the performance
        #         'decoder_target': torch.LongTensor(original_deocer_target)
        #         # original decoder target is the same as givens
        #     }
        # else:
        #     sample = {
        #         'encoder_input': torch.LongTensor(source_seq),
        #         'decoder_target': torch.LongTensor(original_deocer_target)
        #     }
        #     if original_p is not None:
        #         sample['encoder_target'] = torch.FloatTensor(original_p)
        # return sample


    def process_item(self, item):
        original_seq = [str(i) for i in self.operation_list[item]]
        original_ops = self.ops_list[item]
        original_p = self.performance_list[item]
        original_s = self.sensitive_list[item]
        ops = original_ops.copy()
        if self.mask_whole_op_p > 0:
            ops = self.add_whole_op_mask(ops, self.mask_whole_op_p)
        if self.disorder_p > 0:
            ops = self.disorder_ops(ops, self.disorder_p)
        source_seq = []
        for i in ops:
            source_seq.extend([str(j) for j in i])
            source_seq.append('4')
        source_seq = source_seq[:-1]
        if self.mask_op_p > 0:
            source_seq = self.add_op_mask(source_seq, self.mask_op_p)
        # print(len(original_seq), len(source_seq))
        assert len(original_seq) == len(source_seq)
        # original_seq = [0] + [self.vocab[i] for i in original_seq] + [2]
        # source_seq = [0] + [self.vocab[i] for i in source_seq] + [2]
        # original_seq, original_mask = self.padding_seq(original_seq)
        decoder_seq, decoder_mask = self.padding_seq(original_seq, for_encoder=False)
        original_deocer_target, _ = self.padding_seq(original_seq)
        source_seq, source_mask = self.padding_seq(source_seq)

        if self.train:
            sample = {
                'encoder_input': torch.LongTensor(source_seq),
                'decoder_input': torch.LongTensor(decoder_seq),
                'encoder_target': torch.FloatTensor(original_p),  # the performance
                'decoder_target': torch.LongTensor(original_deocer_target),
                'privacy_boundary': torch.FloatTensor(original_s),

                # original decoder target is the same as givens
            }

        else:
            sample = {
                'encoder_input': torch.LongTensor(source_seq),
                'decoder_target': torch.LongTensor(original_deocer_target), 
                'privacy_boundary': torch.FloatTensor(original_s),
            }
            if original_p is not None:
                sample['encoder_target'] = torch.FloatTensor(original_p)
                
        # print(self.count, len(torch.LongTensor(source_seq)), len(torch.LongTensor(decoder_seq)), len(torch.LongTensor(original_deocer_target)))
        return sample


    def __len__(self):
        return len(self.operation_list)

    def padding_seq(self, seq: List[str], for_encoder=True) -> object:
        if for_encoder:
            code = self.encode_build(seq)
        else:
            code = self.decode_build(seq)
        t_padding = [self.tokenizer.pad_token_id for _ in range(self.max_seq_length - len(code))]
        padding_mask = torch.tensor([1 for _ in range(self.max_seq_length)])
        padding_mask[len(code):] = 0
        code.extend(t_padding)
        return code, padding_mask

    def encode_build(self, seq_list):
        return [self.vocab[i] if i in self.vocab.keys() else 3 for i in seq_list]

    def decode_build(self, seq_list):
        return [2] + [self.vocab[i] if i in self.vocab.keys() else 3 for i in seq_list]

    """
    for all $op$ in $self.ops$, mask them by a given prob $p$
    """

    def add_whole_op_mask(self, ops, p):
        if p <= 0:
            return ops
        ops_length = len(ops)
        num_to_mask_whole = math.ceil(numpy.sum(ops_length) * p)
        mask_whole_indice = numpy.random.choice(ops_length, num_to_mask_whole, replace=False)
        ops_ = []
        for indice, op in enumerate(ops):
            if indice in mask_whole_indice:
                op = ['<mask>'] * len(op)
            ops_.append(op)
        return ops_

    """
    given a seq of op, by given p as prob to disorder them
    """

    def disorder_ops(self, ops, p):
        if p <= 0:
            return ops
        ops_length = len(ops)
        num_to_disorder = math.ceil(ops_length * p)
        disorder_indice = numpy.random.choice(ops_length, num_to_disorder, replace=False)
        ops_ = []
        for indice, op in enumerate(ops):
            if indice in disorder_indice:
                op = numpy.random.permutation(op)
            ops_.append(op)
        return ops_

    """
    for each char in $ops_str$, mask them via a given prob $p$
    the input 
    """

    def add_op_mask(self, ops_seq, p):
        # if not isinstance(ops_seq[0], str):
        #     raise Exception('add op mask only use on tokenized seq, pls check your code')
        if p <= 0:
            return ops_seq
        ops_length = len(ops_seq)
        num_to_mask = math.ceil(ops_length * p)
        mask_indice = numpy.random.choice(ops_length, num_to_mask, replace=False)
        for i in mask_indice:
            ops_seq[i] = '<mask>'
        return ops_seq

    def select_top_k(self, k):
        p = torch.tensor(self.original_performance)
        values, indices = torch.topk(p, k, dim=0)
        op_list = []
        performance_list = values.tolist()
        for i in indices:
            op_list.append(self.operation_list[i.numpy()[0]])
        TRL = TransformationRecordList(self.max_seq_length - 2)
        for op, per in zip(op_list, performance_list):
            TRL.append(op, per)

        return DenoiseDataset(TRL.r_list, self.max_seq_length - 2, self.tokenizer, num=self.num
                              , mask_whole_op_p=0, mask_op_p=0, disorder_p=0)


if __name__ == '__main__':
    task_dict = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
        'cls', 'german_credit': 'cls', 'higgs': 'cls',
                 'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
                 'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
                 'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
                 'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
                 'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
                 'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
                 }
    task_names = task_dict.keys()
    for name in task_names:
        args = init_config()
        args.task_name = name
        dl = DenoiseDataModule(args)
        for i in dl.train_dataloader():
            print(i)
            break
        break
