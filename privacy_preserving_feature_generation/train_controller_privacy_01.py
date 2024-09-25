import argparse
import os
import sys

import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from lstm.dataset_privacy_01 import DenoiseDataModule
from utils.datacollection.logger import info, error
import warnings
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
import pdb
import random
import sys
from typing import List
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from tqdm import tqdm
from controller_privacy_01 import GAFS
from feature_env_privacy_01 import base_path
from lstm.utils_meter import AvgrageMeter, pairwise_accuracy, hamming_distance, count_parameters_in_MB
from Record_privacy_01 import SelectionRecord, TransformationRecord
import wandb
import pdb

# from tools import hsic

total_step = 0

baseline_name = [
	'kbest',
	'mrmr',
	'lasso',
	'rfe',
	# 'gfs',
	'lassonet',
	'sarlfs',
	'marlfs',

]


import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.distributions import Normal

def center_kernel(K):
    n = K.size(0)
    one_n = torch.ones((n, n)) / n
    return K - torch.mm(K, one_n) - torch.mm(one_n, K) + torch.mm(one_n, torch.mm(K, one_n))
def hsic_ori(X, Y, sigma=1.0):
	X_tensor = torch.tensor(X.values, dtype=torch.float32)
	Y_tensor = torch.tensor(Y.values.reshape(-1, 1), dtype=torch.float32)

	K = torch.tensor(rbf_kernel(X, gamma=1.0 / (2 * sigma ** 2)), dtype=torch.float32)
	L = torch.tensor(rbf_kernel(Y_tensor, gamma=1.0 / (2 * sigma ** 2)), dtype=torch.float32)

	Kc = center_kernel(K)
	Lc = center_kernel(L)

	hsic_value = torch.trace(torch.mm(Kc, Lc)) / (X_tensor.size(0) - 1) ** 2
	return hsic_value.item()
def hsic(X, Y):
	X_tensor = torch.tensor(X.values, dtype=torch.float32)
	Y_tensor = torch.tensor(Y.values.reshape(-1, 1), dtype=torch.float32)

	sigma_x = np.var(X.values)
	sigma_y = np.var(Y.values)

	K = torch.tensor(rbf_kernel(X, gamma=1.0 / (2 * sigma_x)), dtype=torch.float32)
	L = torch.tensor(rbf_kernel(Y_tensor, gamma=1.0 / (2 * sigma_y)), dtype=torch.float32)

	Kc = center_kernel(K)
	Lc = center_kernel(L)

	hsic_value = torch.trace(torch.mm(Kc, Lc)) / (X_tensor.size(0) - 1) ** 2
	return hsic_value.item()

def hsic_poly(X, Y, degree=3):
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values.reshape(-1, 1), dtype=torch.float32)

    K = torch.tensor(polynomial_kernel(X, degree=degree), dtype=torch.float32)
    L = torch.tensor(polynomial_kernel(Y_tensor, degree=degree), dtype=torch.float32)

    Kc = center_kernel(K)
    Lc = center_kernel(L)

    hsic_value = torch.trace(torch.mm(Kc, Lc)) / (X_tensor.size(0) - 1) ** 2
    return hsic_value.item()

def hsic_normalized(X, Y):
	scaler = StandardScaler()

	X_scaled = scaler.fit_transform(X)
	Y_scaled = scaler.fit_transform(Y.values.reshape(-1, 1))
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
	Y_tensor = torch.tensor(Y_scaled.reshape(-1, 1), dtype=torch.float32)

	sigma_x = np.var(X_scaled)
	sigma_y = np.var(Y_scaled)

	K = torch.tensor(rbf_kernel(X, gamma=1.0 / (2 * sigma_x)), dtype=torch.float32)
	L = torch.tensor(rbf_kernel(Y_tensor, gamma=1.0 / (2 * sigma_y)), dtype=torch.float32)

	Kc = center_kernel(K)
	Lc = center_kernel(L)

	hsic_value = torch.trace(torch.mm(Kc, Lc)) / (X_tensor.size(0) - 1) ** 2
	return hsic_value.item()

def hsic_multiscale(X, Y, sigmas=[0.1, 1.0, 10.0]):
    hsic_values = []
    for sigma in sigmas:
        hsic_value = hsic_ori(X, Y, sigma=sigma)
        hsic_values.append(hsic_value)
    
    return np.mean(hsic_values)

def kl_divergence(X, Y):
    X_mean = torch.mean(torch.tensor(X.values, dtype=torch.float32), dim=0)
    X_std = torch.std(torch.tensor(X.values, dtype=torch.float32), dim=0)

    Y_mean = torch.mean(torch.tensor(Y.values, dtype=torch.float32))
    Y_std = torch.std(torch.tensor(Y.values, dtype=torch.float32))

    X_std = torch.clamp(X_std, min=1e-5)
    Y_std = torch.clamp(Y_std, min=1e-5)

    dist_X = Normal(X_mean, X_std)
    dist_Y = Normal(Y_mean, Y_std)

    kl_div = torch.distributions.kl_divergence(dist_X, dist_Y).mean()
    return kl_div.item()

def pearson_corr(X, Y):
    X_avg = torch.mean(torch.tensor(X.values, dtype=torch.float32), dim=1).numpy()
    Y_np = Y.values
    pearson_corr_value, _ = pearsonr(X_avg, Y_np)
    return pearson_corr_value
def mse(X, Y):
    X_avg = torch.mean(torch.tensor(X.values, dtype=torch.float32), dim=1).numpy()
    Y_np = Y.values
    mse_value = mean_squared_error(X_avg, Y_np)
    return mse_value

def compute_metrics(X, Y):
	if isinstance(X, np.ndarray):
		X =	pandas.DataFrame(X)  
	if isinstance(Y, np.ndarray):
		Y = pandas.Series(Y)  
	hsic_value = hsic(X.copy(), Y.copy())
	kl_value = kl_divergence(X.copy(), Y.copy())
	pearson_value = pearson_corr(X.copy(), Y.copy())
	mse_value = mse(X.copy(), Y.copy())
	hsic_poly_value = hsic_poly(X.copy(), Y.copy())
	hsic_normalized_value = hsic_normalized(X.copy(),Y.copy())
	hsic_multiscale_value = hsic_multiscale(X.copy(),Y.copy())
	return {
		"HSIC": hsic_value,
		"HSIC_poly": hsic_poly_value,
		"HISC_norm": hsic_normalized_value,
		"HSIC_muli": hsic_multiscale_value,
		"KL": kl_value,
		"Corr": pearson_value,
		"MSE": mse_value
	}

def gafs_train(train_queue, model: GAFS, optimizer):
	global total_step
	start_time= time.time()
	objs = AvgrageMeter()
	mse = AvgrageMeter()
	nll = AvgrageMeter()
	mse_sens = AvgrageMeter()
	model.train()
	for sample in train_queue:
		total_step += 1
		encoder_input = sample['encoder_input']
		encoder_target = sample['encoder_target']
		decoder_input = sample['decoder_input']
		decoder_target = sample['decoder_target']
		privacy_boundary = sample['privacy_boundary']

		encoder_input = encoder_input.cuda(model.gpu)
		encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
		decoder_input = decoder_input.cuda(model.gpu)
		decoder_target = decoder_target.cuda(model.gpu)
		privacy_boundary = privacy_boundary.cuda(model.gpu)

		optimizer.zero_grad()
		predict_value, log_prob, arch, predict_sens = model.forward(encoder_input, decoder_input)
		loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())  # mse loss
		loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))  # ce loss
		loss_3 = F.mse_loss(predict_sens.squeeze(), privacy_boundary.squeeze())  # mse loss

		loss = (1 - args.beta) * loss_1 + (args.beta) * loss_2 + 0.1 * loss_3
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
		optimizer.step()

		wandb.log({'Train/loss':loss, 'Train/est_loss':(1 - args.beta) * loss_1, 'Train/rec_loss':(args.beta) * loss_2, 'Train/snes_loss': loss_3}, step = total_step)
		
		n = encoder_input.size(0)
		objs.update(loss.data, n)
		mse.update(loss_1.data, n)
		nll.update(loss_2.data, n)
		mse_sens.update(loss_3, n)
	
	return objs.avg, mse.avg, nll.avg, mse_sens.avg


def gafs_valid(queue, model: GAFS):
	pa = AvgrageMeter()
	hs = AvgrageMeter()
	mse = AvgrageMeter()

	pa_sens = AvgrageMeter()
	mse_sens = AvgrageMeter()
	with torch.no_grad():
		model.eval()
		for step, sample in enumerate(queue):
			encoder_input = sample['encoder_input']
			encoder_target = sample['encoder_target']
			decoder_target = sample['decoder_target']
			privacy_boundary = sample['privacy_boundary']



			encoder_input = encoder_input.cuda(model.gpu)
			encoder_target = encoder_target.cuda(model.gpu)
			decoder_target = decoder_target.cuda(model.gpu)
			privacy_boundary = privacy_boundary.cuda(model.gpu)

			predict_value, logits, arch, predict_sens = model.forward(encoder_input)
			n = encoder_input.size(0)
			mse_sens.update(F.mse_loss(predict_sens.data.squeeze(), privacy_boundary.data.squeeze()), n)
			pairwise_acc_snes = pairwise_accuracy(privacy_boundary.data.squeeze().tolist(),
			                                 predict_sens.data.squeeze().tolist())
			pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
			                                 predict_value.data.squeeze().tolist())
			hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
			mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
			pa.update(pairwise_acc, n)
			hs.update(hamming_dis, n)
			pa_sens.update(pairwise_acc_snes, n)

	return mse.avg, pa.avg, hs.avg, mse_sens.avg, pa_sens.avg


# def choice_to_onehot(choice: List[int]):
#     size = len(choice)
#     onehot = torch.zeros(size + 1)
#     onehot[torch.tensor(choice)] = 1
#     return onehot[:-1]
# if choice.dim() == 1:
#     selected = torch.zeros_like(choice)
#     selected[choice] = 1
#     return selected[1:-1]
# else:
#     onehot = torch.empty_like(choice)
#     for i in range(choice.shape[0]):
#         onehot[i] = choice_to_onehot(choice[i])
#     return onehot


def gafs_infer(queue, model, step, direction='+', beams=None):
	new_gen_list = []
	original_transformation = []
	model.eval()
	for i, sample in enumerate(queue):
		encoder_input = sample['encoder_input']
		privacy_boundary = sample['privacy_boundary']

		encoder_input = encoder_input.cuda(model.gpu)
		model.zero_grad()
		new_gen = model.generate_new_feature(encoder_input, privacy_boundary, predict_lambda=step, direction=direction, beams=beams)
		new_gen_list.append(new_gen.data)
		original_transformation.append(encoder_input)
	return torch.cat(new_gen_list, 0), torch.cat(original_transformation, 0)


def select_top_k(choice: Tensor, labels: Tensor, k: int):
	values, indices = torch.topk(labels, k, dim=0)
	return choice[indices.squeeze()], labels[indices.squeeze()]


def main(args):
	global total_step
	# region init training environment
	if not torch.cuda.is_available():
		info('No GPU found!')
		sys.exit(1)
	# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True
	device = int(args.gpu)
	info(f"Args = {args}")
	# endregion
	dm = DenoiseDataModule(args)
	fe = dm.fe
	info(f'the original performance is : {fe.get_performance(fe.test)}')
	info(f'the original sens is : {fe.get_sensitive(pandas.concat([fe.test.iloc[:, 1:],fe.test.iloc[:, 0]], axis=1))}')

	# max_p = -1
	# best_trans:TransformationRecord = None
	#
	# for i in list(fe.records.r_list):
	#     i.valid = True
	#     performance = fe.get_performance(i.op(fe.test))
	#     if performance > max_p:
	#         best_trans = i
	#         max_p = performance
	#         info(f'the best trans performance is : {performance}')
	
	model = GAFS(fe, args, dm.tokenizer)
	train_queue = dm.train_dataloader()
	valid_queue = dm.val_dataloader()
	
	maybe_load_from = os.path.join(f'{base_path}', 'history', f'{dm.fe.task_name}', f'model_dmp{args.keyword}',
	                               f'{dm.fe.task_name}_{args.load_epoch}.encoder.pt')
	info(f'we load model from {maybe_load_from}:{os.path.exists(maybe_load_from)}')
	if args.load_epoch > 0 and os.path.exists(maybe_load_from):
		base_load_path = os.path.join(f'{base_path}', 'history')
		start_epoch = args.load_epoch
		model = model.from_pretrain(base_load_path, fe, args, dm.tokenizer, start_epoch, keyword=args.keyword)
		model = model.cuda(device)
		mse, pa, hs, mse_sens, pa_sens = gafs_valid(valid_queue, model)
		info("Evaluation on valid data")
		info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f} | mse_sens {:.6f} pairwise accuracy sens{:.6f}'.format(start_epoch, mse, pa,
		                                                                                       hs, mse_sens, pa_sens))
	else:
		start_epoch = 0
		model = model.cuda(device)
	
	info(f"param size = {count_parameters_in_MB(model)}MB")
	info('Training Encoder-Predictor-Decoder')
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
	for epoch in range(start_epoch + 1, args.epochs + 1):
		nao_loss, nao_mse, nao_ce, sens_mse = gafs_train(train_queue, model, optimizer)
		if epoch % 100 == 0 or epoch == 1:
			model.save_to(f'{base_path}/history', epoch, keyword=args.keyword)
			info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f} sens_mse {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce, sens_mse))
		if epoch % 100 == 0 or epoch == 1:
			mse, pa, hs, mse_sens, pa_sens = gafs_valid(valid_queue, model)
			wandb.log({'valid/mse':mse, 'valid/pair_wise':pa, 'valid/hamming_dis':hs, 'valid/mse_sens':mse_sens, 'valid/pa_sens':pa_sens}, step = total_step)

			info("Evaluation on valid data")
			info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f} | mse_sens {:.6f} pairwise accuracy sens{:.6f}'.format(start_epoch, mse, pa,
		                                                                                       hs, mse_sens, pa_sens))
	infer_queue = dm.infer_dataloader()
	new_selection = []
	new_choice = []
	predict_step_size = 0
	original_transformation = []
	total_num = 0
	valid_num = 0
	
	# print("\033[96m {}\033[00m".format(args.new_gen))
	start_time = time.time()
	info(start_time)
	while len(new_selection) < args.new_gen:
		predict_step_size += 1
		info('Generate new architectures with step size {:d}'.format(predict_step_size))
		new_record, original_record = gafs_infer(infer_queue, model, direction='+', step=predict_step_size,
		                                         beams=args.beams)
		
		# print("\033[96m {}\033[00m".format(original_record))
		new_choice.append((new_record, original_record))
		for choice, original_choice in zip(new_record, original_record):
			# print("\033[96m {}\033[00m".format(choice))

			record_ = TransformationRecord.from_tensor(choice, dm.tokenizer)
			original_record_ = TransformationRecord.from_tensor(original_choice, dm.tokenizer)
			if record_ not in new_selection:  # record_ not in fe.records.r_list and
				new_selection.append((record_, original_record_))
				valid_num += len(record_.ops)


				total_num += len(record_.input_ops)
				info(f'gen {record_.valid}: {len(record_.ops)}/{len(record_.input_ops)}')
				info(f'{len(new_selection)} new choice gener ated now', )
		if predict_step_size > args.max_step_size:
			break
	# info('----------------------------------------------------------------------')
	# info(f'------------------{model.encoder.sens_bound2}---------------------------')
	# info(f'------------------{model.encoder.sens_bound}---------------------------')
	# info('----------------------------------------------------------------------')
	info(f'build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')
	# pdb.set_trace()
	# new_choice_pt = torch.cat(new_choice, dim=0)
	#
	# choice_path = f'{base_path}/history/{fe.task_name}/generated_choice.pt'
	# torch.save(new_choice_pt, choice_path)
	# info(f'save generated choice to {choice_path}')
	
	# torch.save(model.state_dict(), f'{base_path}/history/{fe.task_name}/GAFS.model_dict')
	best_selection = None
	best_optimal = -1000
	best_sens = 1000
	infer_step = 0
	# previous_optimal = max(dm.train_dataset.original_performance)[0]
	previous_optimal = dm.fe.best_grfg
	info(f'the best performance for this task is {previous_optimal}')
	count = 0
	sens_v, hsic_v, corr_v, hsic_p_v, hsic_m_v, hsic_n_v, kl_v, mse_v = [], [], [], [], [], [], [], [] 
	for record, original_record_ in new_selection:
		infer_step += 1
		# train_data = fe.generate_data(s.operation, 'train')
		# result = fe.get_performance(train_data)
		# test_data = fe.generate_data(s.operation, 'test')
		# test_result = fe.get_performance(test_data)
		# record = TransformationRecord.from_tensor(s, tokenizer=dm.tokenizer)
		if not record.valid:
			count += 1
			info(f'invalid percentage as : {count}/{len(new_selection)}')
			continue
		# print("\033[96m {}\033[00m".format(record))

		test_data = record.op(fe.test.copy(), args.add_origin)

		sensitive_feature = test_data.iloc[:,0]
		
		test_data = test_data.iloc[:,1:]
		# print("\033[96m {}\033[00m".format(test_data))

		original_trans = original_record_.op(fe.test.copy())
		cols_gen = list(test_data.columns[:-1])
		cols_ori = list(original_trans.columns[:-1])
		final_cols = []
		data = []
		for index, i in enumerate(cols_gen):
			if final_cols.__contains__(i):
				continue
			else:
				final_cols.append(i)
				data.append(test_data.iloc[:, index])
		
		if args.add_transformed:
			for index, i in enumerate(cols_ori):
				if final_cols.__contains__(i):
					continue
				else:
					final_cols.append(i)
					data.append(original_trans.iloc[:, index])
		data.append(test_data.iloc[:, -1])
		final_cols.append('label')
		final_ds = pandas.concat(data, axis=1)
		final_ds.columns = final_cols
		# pdb.set_trace()
		sens_df = pandas.concat([final_ds.iloc[:,:-1], sensitive_feature], axis=1)
		sens_result = fe.get_sensitive(sens_df)
		# print(final_ds.iloc[:,:-1])



		# print(sens_result, hsic_value)
		try:
			result = fe.get_performance(final_ds)

			# if result > previous_optimal:
			#     optimal_selection = s.operation
			#     previous_optimal = result
			#     info(f'found optimal selection! the choice is {s.operation}, the performance on train is {result}')
			if sens_result < best_sens:
				best_sens = sens_result
				wandb.log({'best_sens/best_sensitive_per':best_sens, 'best_sens/performance_when_best_sen':result})


			if result > best_optimal:
				best_selection = final_ds
				best_optimal = result
				wandb.log({'best_p/best_performance':best_optimal, 'best_p/sen_when_best_p':sens_result})

				info(f'found best performance on {dm.fe.task_name} : {best_optimal}')
				info(f'the column is {final_ds.columns}')
		except:
			error('something wrong with this feature set, e.g., Nan or Inf')

	best_str = '{:.4f}'.format(best_optimal * 100)

	info(f'build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')
	info(f'the original performance is : {fe.get_performance(fe.test)}')
	# info(f'the original sens is : {fe.get_sensitive(pandas.concat([fe.test.iloc[:, 1:],fe.test.iloc[:, 0]]))}')

	info(f'found best performance on {dm.fe.task_name} : {best_optimal}')


# opt_path = f'{base_path}/history/{fe.task_name}/best-ours.hdf'
# ori_p = fe.report_performance(best_selection, flag='test')
# info(f'found train generation in our method! the choice is {best_selection}, the performance is {ori_p}')
# fe.generate_data(best_selection, 'train').to_hdf(opt_path, key='train')
# fe.generate_data(best_selection, 'test').to_hdf(opt_path, key='test')

# opt_path_test = f'{base_path}/history/{fe.task_name}/best-ours-test.hdf'
# test_p = fe.report_performance(best_selection_test, flag='test')
# info(f'found test generation in our method! the choice is {best_selection_test}, the performance is {test_p}')
# fe.generate_data(best_selection_test, 'train').to_hdf(opt_path_test, key='train')
# fe.generate_data(best_selection_test, 'test').to_hdf(opt_path_test, key='test')
# ps = []
# info('given overall validation')
# report_head = 'RAW\t'
# raw_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key='raw_test')
# ps.append('{:.2f}'.format(fe.get_performance(raw_test) * 100))
# for method in baseline_name:
#     report_head += f'{method}\t'
#     spe_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_test')
#     ps.append('{:.2f}'.format(fe.get_performance(spe_test) * 100))
# report_head += 'Ours\tOurs_Test'
# report = ''
# print(report_head)
# for per in ps:
#     report += f'{per}&\t'
# report += '{:.2f}&\t'.format(ori_p * 100)
# report += '{:.2f}&\t'.format(test_p * 100)
# print(report)


#  gen 25
# 0.4341 [1. 1. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4357  [1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4301 gen 100

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task_name', type=str, default='housing_boston')
	parser.add_argument('--mask_whole_op_p', type=float, default=0.0)
	parser.add_argument('--mask_op_p', type=float, default=0.0)
	parser.add_argument('--disorder_p', type=float, default=0.0)
	parser.add_argument('--num', type=int, default=12)
	
	parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')
	
	parser.add_argument('--encoder_layers', type=int, default=1)
	parser.add_argument('--encoder_hidden_size', type=int, default=64)
	parser.add_argument('--encoder_emb_size', type=int, default=32)
	parser.add_argument('--mlp_layers', type=int, default=2)
	parser.add_argument('--mlp_hidden_size', type=int, default=200)
	parser.add_argument('--decoder_layers', type=int, default=1)
	parser.add_argument('--decoder_hidden_size', type=int, default=64)
	
	parser.add_argument('--encoder_dropout', type=float, default=0)
	parser.add_argument('--mlp_dropout', type=float, default=0)
	parser.add_argument('--decoder_dropout', type=float, default=0)
	
	parser.add_argument('--new_gen', type=int, default=500)
	
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--beta', type=float, default=0.95)
	parser.add_argument('--grad_bound', type=float, default=5.0)
	parser.add_argument('--l2_reg', type=float, default=0.0)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--top_k', type=int, default=20)
	
	parser.add_argument('--load_epoch', type=int, default=2000)
	parser.add_argument('--train_top_k', type=int, default=512)
	parser.add_argument('--epochs', type=int, default=2000)
	parser.add_argument('--eval', type=bool, default=False)
	parser.add_argument('--max_step_size', type=int, default=5)
	
	parser.add_argument('--beams', type=int, default=5)
	parser.add_argument('--add_origin', type=bool, default=True)
	parser.add_argument('--add_transformed', type=bool, default=False)
	parser.add_argument('--gpu', type=int, default=0)
	
	parser.add_argument('--keyword', type=str, default='')
	
	args = parser.parse_args()
	wandb.login(key='')
	# params = vars(init_param())
	wandb.init(project="",config=args)
	main(args)
	wandb.finish()