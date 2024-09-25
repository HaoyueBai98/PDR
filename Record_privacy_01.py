import os
from typing import List

import pandas
import torch
import torch.nn.functional as F
import numpy

from utils.datacollection.Operation import eos_token, sep_token, op_post_seq, converge, show_ops, check_valid


class Record(object):
    def __init__(self, operation, performance, sensitive):
        if isinstance(operation, List):
            self.operation = numpy.array(operation)
        elif isinstance(operation, torch.Tensor):
            self.operation = operation.numpy()
        else:
            assert isinstance(operation, numpy.ndarray)
            self.operation = operation
        self.performance = performance
        self.sensitive = sensitive

    def get_permutated(self):
        pass

    def get_ordered(self):
        pass

    def repeat(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return str(self.operation).__hash__()


class TransformationRecord(Record):
    def __init__(self, operation, performance, sensitive, sep_token=sep_token, is_mid=False):
        self.sep_token = sep_token
        ops = []
        tmp = []
        unique = set()
        for i in operation:
            if isinstance(i, str):
                try:
                    i = int(i)
                except:
                    print(operation)
                    continue
            if i == self.sep_token:
                if is_mid:
                    ops.append(converge(show_ops(tmp)))
                else:
                    ops.append(tmp)
                tmp = []
            else:
                tmp.append(i)
        if len(tmp) != 0:
            if is_mid:
                ops.append(converge(show_ops(tmp)))
            else:
                ops.append(tmp)
        # print(ops)
        self.input_ops = ops  # store the input ops, includ invalid and duplicated ops
        ops = []
        for op in self.input_ops:
            str_op = str.join(' ', [str(i) for i in op])
            if not unique.__contains__(str_op):
                unique.add(str_op)
                ops.append(op)
        self.ops = []
        for op in ops:
            if check_valid(op):
                self.ops.append(op)

        # if is_mid: # all raw op is in mid from data collection
        #     operation = converge(show_ops(operation))
        operation_seq = []
        for op in self.ops:
            operation_seq.extend(op)
            operation_seq.append(sep_token)
        self.valid = len(operation_seq) > 0 and self.is_valid()
        if self.valid and operation_seq[-1] == sep_token:
            operation_seq.pop(-1)
        super(TransformationRecord, self).__init__(operation_seq, performance, sensitive)

    def get_permutated(self, max_size=0, num=25, padding=True, padding_token=eos_token):
        size = len(self.ops)
        all = []
        for i in range(num):
            round = []
            indice = numpy.random.permutation(range(size))
            for j in indice:
                round.extend(self.ops[j])
                round.append(self.sep_token)
            round.pop(-1)
            all.append(round)
        shuffled_indices = torch.Tensor(all)
        if padding and size < max_size:
            shuffled_indices = F.pad(shuffled_indices, (0, (max_size - size)), 'constant', padding_token)
        return shuffled_indices, torch.FloatTensor([self.performance]).unsqueeze(0).repeat(num + 1, 1)

    def get_permutated_ops(self, num=25):
        size = len(self.ops)
        all = []
        p = []
        s = []
        for i in range(num):
            indice = numpy.random.permutation(range(size))
            round = []
            for j in indice:
                round.append(self.ops[j])
            p.append(self.performance)
            s.append(self.sensitive)
            all.append(round)
        return all, p, s

    @staticmethod
    def load_from_padded(padded_seq):
        seq = []
        for i in padded_seq:
            if i == eos_token:
                break
            else:
                seq.append(i)
        return TransformationRecord(seq, -1)

    def op(self, df, with_original=False):
        df = df.reset_index(drop=True)
        if self.valid:
            ret_df = []
            col_names = []
            for op in self.ops:
                ret_df.append(pandas.DataFrame(op_post_seq(df, op).values,
                                               columns=[str.join(' ', show_ops(op))], index=range(len(df))))
                col_names.append(str.join(' ', show_ops(op)))
            ret_df.append(pandas.DataFrame(
                df.iloc[:, -1].values, columns=['label'], index=range(len(df))))
            col_names.append('label')
            new_dg = pandas.concat(ret_df, axis=1)
            pre_name = [str(i) for i in df.columns[:-1]]
            pre_name.extend(col_names)
            if with_original:
                out_dg = pandas.concat(
                    [pandas.DataFrame(df.iloc[:, :-1].values, columns=[df.columns[:-1]], index=range(len(df))),
                     new_dg], axis=1)
                out_dg.columns = pre_name
            else:
                out_dg = new_dg
                out_dg.columns = col_names
            return out_dg
        else:
            return None

    def is_valid(self):
        for op in self.ops:
            if not check_valid(op):
                return False
        return True

    @staticmethod
    def from_tensor(inputs, tokenizer):
        # inputs: List
        # [[]
        # ['6']
        # ['6']
        # []]
        # print("\033[96m {}\033[00m".format(inputs))
        toks = []
        for i in inputs:
            toks.append(tokenizer.convert_ids_to_tokens(i.unsqueeze(0), skip_special_tokens=True))
        op_seq = []
        for i in toks:
            if len(i) < 1:
                continue
            if isinstance(i[0], str):
                try:
                    int(i[0])
                except:
                    print(i)
                    continue
            if len(i) == 0:  # skip the special token
                continue
            else:
                op_seq.append(i[0])  # append the token to ret
        return TransformationRecord(op_seq, -1, None)


class SelectionRecord(Record):
    def __init__(self, operation, performance):
        super().__init__(operation, performance)
        self.max_size = operation.shape[0]

    def _get_ordered(self):
        indice_select = torch.arange(0, self.max_size)[self.operation == 1]
        return indice_select, torch.FloatTensor([self.performance])

    def get_permutated(self, num=25, padding=True, padding_value=-1):
        ordered, performance = self._get_ordered()
        size = ordered.shape[0]
        shuffled_indices = torch.empty(num + 1, size)
        shuffled_indices[0] = ordered
        label = performance.unsqueeze(0).repeat(num + 1, 1)
        for i in range(num):
            shuffled_indices[i + 1] = ordered[torch.randperm(size)]
        if padding and size < self.max_size:
            shuffled_indices = F.pad(shuffled_indices, (0, (self.max_size - size)), 'constant', padding_value)
        return shuffled_indices, label

    def repeat(self, num=25, padding=True, padding_value=-1):
        ordered, performance = self._get_ordered()
        size = ordered.shape[0]
        label = performance.unsqueeze(0).repeat(num + 1, 1)
        indices = ordered.unsqueeze(0).repeat(num + 1, 1)
        if padding and size < self.max_size:
            indices = F.pad(indices, (0, (self.max_size - size)), 'constant', padding_value)
        return indices, label


class RecordList(object):
    def __init__(self):
        self.r_list = set()

    def append(self, op, val):
        self.r_list.add(SelectionRecord(op, val))

    def __len__(self):
        return len(self.r_list)

    def generate(self, num=25, padding=True, padding_value=-1):
        results = []
        labels = []
        for record in self.r_list:
            result, label = record.get_permutated(num, padding, padding_value)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)


class TransformationRecordList(RecordList):
    def __init__(self, max_size):
        super(TransformationRecordList, self).__init__()
        self.max_size = max_size

    def append(self, op, val, sens):
        if len(op) < self.max_size:
            self.r_list.add(TransformationRecord(op, val, sens, is_mid=False))

    def append_record(self, r:TransformationRecord):
        self.r_list.add(r)

    def generate(self, num=25, padding=True, padding_value=eos_token):
        results = []
        labels = []
        for record in self.r_list:
            result, label = record.get_permutated(max_size=self.max_size, num=25, padding=padding,
                                                  padding_value=padding_value)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

