"""
feature env
interactive with the actor critic for the state and state after action
"""
import os
import sys

sys.path.append('./')
import pickle
from collections import namedtuple
from typing import List, Optional

import numpy as np
import pandas
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split

from Record_privacy_01 import RecordList, TransformationRecordList, Record, TransformationRecord
from utils.datacollection.Operation import show_ops_r, show_ops, converge, sep_token
from utils.datacollection.logger import error, info
from utils.rlac_tools import test_task_new, downstream_task_new, downstream_task_by_method, \
    downstream_task_by_method_std, sensitive_test
from utils.datacollection.tools import hsic
import warnings
import wandb

warnings.filterwarnings('ignore')

base_path = '/data'

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity': 'mcls', 'mice_protein': 'mcls', 'coil-20': 'mcls',
             'isolet': 'mcls', 'minist': 'mcls',
             'minist_fashion': 'mcls'
             }

SENS_TASK_DICT = {'uci_credit_card':'reg','mice_protein':'reg', 'activity':'reg', 'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':'reg', 
             'german_credit': 'cls', 'housing_boston': 'reg', 'ionosphere': 'cls', 
             'lymphography': 'cls', 'spam_base': 'reg', 'spectf': 'cls', 
             'uci_credit_card': 'reg', 'wine_red': 'reg', 'wine_white': 'reg',
            'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             }

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score', 'roc_auc'],
    'reg': ['mae', 'mse', 'rae', 'rmse'],
    'det': ['map', 'f1_score', 'ras', 'recall'],
    'mcls': ['precision', 'recall', 'mif1', 'maf1']
}

model_performance = {
    'mcls': namedtuple('ModelPerformance', MEASUREMENT['mcls']),
    'cls': namedtuple('ModelPerformance', MEASUREMENT['cls']),
    'reg': namedtuple('ModelPerformance', MEASUREMENT['reg']),
    'det': namedtuple('ModelPerformance', MEASUREMENT['det'])
}


class Evaluator(object):
    def __init__(self, task, task_type=None, dataset=None):
        self.original_report = None
        self.records = RecordList()
        self.task_name = task
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type

        if dataset is None:
            data_path = os.path.join(base_path, self.task_name + '.hdf')
            original = pd.read_hdf(data_path)
        else:
            original = dataset
        col = np.arange(original.shape[1])
        self.col_names = original.columns
        original.columns = col
        self.original = original.fillna(value=0)
        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]
        
        info('initialize the train and test dataset')
        self._check_path()
        if not os.path.exists(os.path.join(base_path, 'history', self.task_name + '.hdf')):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                                random_state=0, shuffle=True)
            self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
            self.train.reset_index(drop=True, inplace=True)
            self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
            self.test.reset_index(drop=True, inplace=True)
            self.train.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='train')
            self.test.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='test')
            self.original.to_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='raw')
        else:
            self.train = pd.read_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='train')
            self.test = pd.read_hdf(os.path.join(base_path, 'history', self.task_name + '.hdf'), key='test')

    def __len__(self):
        return len(self.records)

    def generate_data(self, operation, flag):
        pass

    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def report_ds(self):
        pass

    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/{self.task_name}/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/{self.task_name}/choice.pt')
        torch.save(performances, f'{base_path}/history/{self.task_name}/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/{self.task_name}/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/history/'):
            os.mkdir(f'{base_path}/history/')
        if not os.path.exists(f'{base_path}/history/{self.task_name}'):
            os.mkdir(f'{base_path}/history/{self.task_name}')

    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_records(self, num=0, eos=-1):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def get_triple_record(self, num=0, eos=-1, mode='ht'):
        h_results = []
        labels = []
        t_results = []
        h_seed = []
        labels_seed = []
        for record in self.records.r_list:
            if mode.__contains__('h'):
                h, label = record.get_permutated(num, True, eos)
            else:
                h, label = record.repeat(num, True, eos)
            if mode.__contains__('t'):
                t, _ = record.get_permutated(num, True, eos)
            else:
                t, _ = record.repeat(num, True, eos)
            h_results.append(h)
            t_results.append(t)
            labels.append(label)
            h_seed.append(h_results[0])
            labels_seed.append(labels[0])
        return torch.cat(h_results, 0), torch.cat(labels, 0), torch.cat(t_results), \
               torch.cat(h_seed), torch.cat(labels_seed),

    def report_performance(self, choice, store=True, rp=True, flag=''):
        opt_ds = self.generate_data(choice, flag)
        a, b, c, d = test_task_new(opt_ds, task=self.task_type)
        report = model_performance[self.task_type](a, b, c, d)
        if flag == 'test':
            store = False
        if self.original_report is None:
            a, b, c, d = test_task_new(self.test, task=self.task_type)
            self.original_report = (a, b, c, d)
        else:
            a, b, c, d = self.original_report
        original_report = model_performance[self.task_type](a, b, c, d)
        if self.task_type == 'reg':
            final_result = report.rae
            if rp:
                info('1-MAE on original is: {:.4f}, 1-MAE on generated is: {:.4f}'.
                     format(original_report.mae, report.mae))
                info('1-MSE on original is: {:.4f}, 1-MSE on generated is: {:.4f}'.
                     format(original_report.mse, report.mse))
                info('1-RAE on original is: {:.4f}, 1-RAE on generated is: {:.4f}'.
                     format(original_report.rae, report.rae))
                info('1-RMSE on original is: {:.4f}, 1-RMSE on generated is: {:.4f}'.
                     format(original_report.rmse, report.rmse))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('F-1 on original is: {:.4f}, F-1 on generated is: {:.4f}'.
                     format(original_report.f1_score, report.f1_score))
                info('ROC/AUC on original is: {:.4f}, ROC/AUC on generated is: {:.4f}'.
                     format(original_report.roc_auc, report.roc_auc))
        elif self.task_type == 'det':
            final_result = report.ras
            if rp:
                info(
                    'Average Precision Score on original is: {:.4f}, Average Precision Score on generated is: {:.4f}'
                    .format(original_report.map, report.map))
                info(
                    'F1 Score on original is: {:.4f}, F1 Score on generated is: {:.4f}'
                    .format(original_report.f1_score, report.f1_score))
                info(
                    'ROC AUC Score on original is: {:.4f}, ROC AUC Score on generated is: {:.4f}'
                    .format(original_report.ras, report.ras))
                info(
                    'Recall on original is: {:.4f}, Recall Score on generated is: {:.4f}'
                    .format(original_report.recall, report.recall))
        elif self.task_type == 'mcls':
            final_result = report.mif1
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('Micro-F1 on original is: {:.4f}, Micro-F1 on generated is: {:.4f}'.
                     format(original_report.mif1, report.mif1))
                info('Macro-F1 on original is: {:.4f}, Macro-F1 on generated is: {:.4f}'.
                     format(original_report.maf1, report.maf1))
        else:
            error('wrong task name!!!!!')
            assert False
        if store:
            self._store_history(choice, final_result)
        return final_result


class FeatureEvaluator(Evaluator):
    def __init__(self, task, task_type=None, dataset=None):
        super().__init__(task, task_type, dataset)
        self.ds_size = self.original.shape[1] - 1

    def generate_data(self, choice, flag=''):
        if choice.shape[0] != self.ds_size:
            error('wrong shape of choice')
            assert False
        if flag == 'test':
            ds = self.test
        elif flag == 'train':
            ds = self.train
        else:
            ds = self.original
        X = ds.iloc[:, :-1]
        indice = torch.arange(0, self.ds_size)[choice == 1]
        X = X.iloc[:, indice].astype(np.float64)
        y = ds.iloc[:, -1].astype(np.float64)
        Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        return Dg

    def _full_mask(self):
        return torch.FloatTensor([1] * self.ds_size)

    def report_ds(self):
        per = self.get_performance()
        info(f'current dataset : {self.task_name}')
        info(f'the size of shape is : {self.original.shape[1]}')
        info(f'original performance is : {per}')
        self._store_history(self._full_mask(), per)


'''
rec_length_strategy: 
(1) all: simply return all records
(2) sigma: using n-sigma to choose the records, you need to provide $sigma$ as input
(3) length: using statistical length to choose the records, you need to provide $percentage$ as input
'''


class TransformationFeatureEvaluator(Evaluator):
    def __init__(self, task: str, from_scratch: bool = False, all: bool = True,
                 rec_length_strategy: Optional[str] = 'sigma', sigma: Optional[int] = 4,
                 percentage: Optional[float] = 0.99, force: Optional[bool] = False, limit=1000):
        super().__init__(task)
        if rec_length_strategy == 'sigma':
            appendix = f'{sigma}-sigma'
        elif rec_length_strategy == 'length':
            appendix = f'length[{percentage}]'
        else:
            appendix = 'all'
        suppose_file_cache_name = f'{base_path}/history/{self.task_name}/fe-{appendix}.{limit}.pkl'
        self.overall_processed_seq, self.overall_performance, self.overall_hsic, ds_size, self.contain_best = \
            self._get_data_from_local(all=all,
                                      rec_length_strategy=rec_length_strategy,
                                      sigma=sigma,
                                      percentage=percentage,
                                      force=force)

        self.ds_size = int(ds_size) + 1
        if os.path.exists(suppose_file_cache_name) and not from_scratch:
            info('load records from disk...')
            with open(suppose_file_cache_name, 'rb') as f:
                self.best_grfg, self.records = pickle.load(f)
        else:
            info('load records from scratch...')
            self.records = TransformationRecordList(self.ds_size)

            # the -1 is the y, and -2 is the [sep], we drop those in training data
            # top_performance, indices = torch.topk(torch.tensor(self.overall_performance), 2 * limit, dim=0)
            top_performance, indices = torch.topk(torch.tensor(self.overall_performance), len(self.overall_performance) if len(self.overall_performance) < 2 * limit else 2 * limit, dim=0)
            top_seq = []
            hsic_when_top_p = []
            for i in indices:
                index = i[0].item()
                top_seq.append(self.overall_processed_seq[index])
                hsic_when_top_p.append(self.overall_hsic[index])
            flag = 0
            for seq, per, _hsic in tqdm.tqdm(zip(top_seq, top_performance, hsic_when_top_p), desc='processed seq processing'):
                if len(self.records) == limit:
                    break
                try:
                    tmp_record = TransformationRecord(seq[:-1], per, _hsic)
                    tmp_df = tmp_record.op(self.test, with_original=True)
                    test_per = self.get_performance(tmp_df.iloc[:, 1:])
                    test_hsic = hsic(torch.tensor(tmp_df.iloc[:, 1:-1].values, dtype=torch.float32), torch.tensor(tmp_df.iloc[:,0].values, dtype=torch.float32).view(-1,1), 0.25).numpy()
                    # try: 
                    #     int(test_hsic)
                    # except:
                    #     test_hsic = 1

                    if np.isnan(test_hsic):
                        test_hsic = np.array(1, dtype=np.float32)
                    if flag == 0:
                        self.best_grfg = test_per
                        wandb.log({'GRFG/performance':test_per, 'GRFG/hsic_when_top_p':test_hsic})

                        
                    self.records.append(seq[:-1], test_per, test_hsic)
                    flag += 1
                except:
                    continue
            with open(suppose_file_cache_name, 'wb') as f:
                pickle.dump((self.best_grfg, self.records), f)


    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def get_sensitive(self, data=None):
        task_type = SENS_TASK_DICT[self.task_name]
        if data is None:
            data = self.original
        return sensitive_test(data, task_type)    
    
    def get_performance_via_method(self, data=None, method='RF'):
        if data is None:
            data = self.original
        return downstream_task_by_method(data, self.task_type, method)

    def _generate_data_via_seq(self, seq: List):
        return self._generate_data_via_record(TransformationRecord(seq, performance=-1, sensitive=-1))

    def _generate_data_via_record(self, record: TransformationRecord):
        if record.is_valid():
            gen = record.op(self.original)
            return pandas.concat([self.original, gen], axis=1)
        else:
            return None

    def generate_data(self, record, flag=''):
        if isinstance(record, List):
            return self._generate_data_via_seq(record)
        else:
            return self._generate_data_via_record(record)

    def _store_history(self, choice, performance):
        pass

    def get_records(self, num=0):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def report_ds(self):
        per = self.get_performance()
        info(f'current dataset : {self.task_name}')
        info(f'the size of shape is : {self.original.shape[1]}')
        info(f'original performance is : {per}')
        # self._store_history(self._full_mask(), per)

    def _store_history(self, choice, performance):
        pass

    '''
    load recordlist from local
    '''

    def _get_data_from_local(self, file_base_path='tmp/', all=True,
                             rec_length_strategy='sigma', sigma=4, percentage=0.75, force=False):
        name = self.task_name
        opt_path = f'{base_path}/history/{self.task_name}'
        opt_file = f'operation.list.{all}.pkl'
        performance_file = f'performance.list.{all}.pkl'
        hsic_file = f'hsic.list.{all}.pkl'
        size = []
        overall_processed_seq = []
        overall_performance = []
        overall_hsic = []
        max_length = -1
        if os.path.exists(os.path.join(opt_path, opt_file)) and \
                os.path.exists(os.path.join(opt_path, performance_file)) and \
                os.path.exists(os.path.join(opt_path, hsic_file)) and not force:
            info(f'have processed the data, load cache from local in :'
                 f'{opt_path}/performance.list.{all}.pkl and {opt_path}/operation.list.{all}.pkl')
            with open(os.path.join(opt_path, performance_file), 'rb') as f:
                overall_performance = pickle.load(f)
            with open(os.path.join(opt_path, opt_file), 'rb') as f:
                overall_processed_seq = pickle.load(f)
            with open(os.path.join(opt_path, hsic_file), 'rb') as f:
                overall_hsic = pickle.load(f)
            for i in tqdm.tqdm(overall_processed_seq, desc='calculate the max length'):
                size.append(len(i))
                if len(i) > max_length:
                    max_length = len(i)
        else:
            info('initial the data from local file')
            if not os.path.exists(os.path.join(file_base_path, name)):
                os.mkdir(os.path.join(file_base_path, name))
            db_list = os.listdir(os.path.join(file_base_path, name))
            filtered_name = []
            if all:
                suffix = '.adata'
            else:
                suffix = '.bdata'
            for f in db_list:
                if f.__contains__(suffix):
                    filtered_name.append(os.path.join(file_base_path, name, f))
            if not os.path.exists(opt_path):
                os.mkdir(opt_path)
            for i in filtered_name:
                with open(i) as f:
                    lines = f.readlines()
                    for line in tqdm.tqdm(lines, desc=f'processing local data'):
                        seq = line.strip().split(',')
                        processed_seq = []
                        performance = []
                        hsic_list = []
                        tmp = []
                        # Converts an operation sequence to a postorder sequence
                        for i in seq[:-4]:
                            if i == str(sep_token):  # split: setting to '4'
                                if len(tmp) == 1:
                                    # processed_seq.append(show_ops_r(converge(show_ops(tmp))))
                                    processed_seq.append(tmp.pop(0))
                                    processed_seq.append(int(i))
                                else:
                                    for tok in show_ops_r(converge(show_ops(tmp))):
                                        processed_seq.append(tok)
                                    processed_seq.append(int(i))
                                    tmp = []
                            else:
                                tmp.append(int(i))
                        if len(tmp) != 0:
                            for tok in show_ops_r(converge(show_ops(tmp))):
                                processed_seq.append(tok)
                        else:
                            processed_seq.pop(-1)
                        if max_length < len(processed_seq):
                            max_length = len(processed_seq)
                        performance.append(float(seq[-3]))
                        hsic_list.append(float(seq[-4]))
                        overall_processed_seq.append(processed_seq)
                        size.append(len(processed_seq))
                        overall_performance.append(performance)
                        overall_hsic.append(hsic_list)
            with open(os.path.join(opt_path, opt_file), 'wb') as f:
                pickle.dump(overall_processed_seq, f)
            with open(os.path.join(opt_path, performance_file), 'wb') as f:
                pickle.dump(overall_performance, f)
            with open(os.path.join(opt_path, hsic_file), 'wb') as f:
                pickle.dump(overall_hsic, f)
        outlier = []  # 将异常值保存
        outlier_x = []
        okay = True
        choosed_seq = []
        choosed_performance = []
        choosed_hsic = []
        if rec_length_strategy == 'all':
            choosed_performance = overall_performance
            choosed_seq = overall_performance
            choosed_hsic = overall_hsic
        elif rec_length_strategy == 'sigma':

            ymean = np.mean(size)
            ystd = np.std(size)
            threshold1 = ymean - sigma * ystd
            threshold2 = ymean + sigma * ystd
            max_size = -1
            for i in tqdm.tqdm(range(0, len(size)), desc=f'{sigma}-sigma processing'):
                if (size[i] < threshold1) | (size[i] > threshold2):
                    outlier.append(size[i])
                    outlier_x.append(overall_performance[i])
                else:
                    if len(overall_processed_seq[i]) > max_size:
                        max_size = len(overall_processed_seq[i])
                    choosed_seq.append(overall_processed_seq[i])
                    choosed_performance.append(overall_performance[i])
                    choosed_hsic.append(overall_hsic[i])
            info(f'using {sigma}-sigma to remove outlier')
            info(f'the best removed is {max(outlier_x)}, the original_best is {max(overall_performance)}')
            info(f'selected length is {max_size}')
            max_length = max_size
        elif rec_length_strategy == 'length':
            # Filter out overly long sequences
            max_size = pandas.DataFrame(size).describe(percentiles=[percentage]).values[-2].item()
            for indice, i in tqdm.tqdm(enumerate(overall_processed_seq), desc=f'lengthy[{percentage}] processing'):
                if len(i) < max_size:
                    choosed_seq.append(i)
                    choosed_performance.append(overall_performance[indice])
                    choosed_hsic.append(overall_hsic[indice])
                else:
                    outlier.append(size[indice])
                    outlier_x.append(overall_performance[indice])
            info(f'selected length is {max_size}')
            info('using lengthy to remove outlier')
            info(f'the best removed is {max(outlier_x)}, the original_best is {max(overall_performance)}')
            max_length = max_size

        if len(outlier_x) != 0 and (max(overall_performance)[0] - max(outlier_x)[0]) < 0.000001:
            if rec_length_strategy == 'sigma':
                info(f'for setting strategy as {rec_length_strategy} with {sigma} '
                     f'is not suitable for task {self.task_name}')
            else:
                info(
                    f'for setting strategy as {rec_length_strategy} with {percentage} '
                    f'is not suitable for task {self.task_name}')
            okay = False
        overall_performance = choosed_performance
        overall_processed_seq = choosed_seq
        overall_hsic = choosed_hsic
        return overall_processed_seq, overall_performance, overall_hsic, max_length, okay


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    fe = TransformationFeatureEvaluator(args.name, rec_length_strategy='length', percentage=0.95,
                                        from_scratch=True)
    info(f'{fe.task_name} : {fe.best_grfg}')
    info(f'{fe.task_name} : {len(fe.records)}')

 