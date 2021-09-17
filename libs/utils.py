import argparse
import random

import math
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def str2bool(v):
	if v.lower() in ['yes', 'true', 't', 'y', '1']:
		return True
	elif v.lower() in ['no', 'false', 'f', 'n', '0']:
		return False
	else:
		raise arparse.ArgumentTypeError('Boolean value expected')


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.random.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def set_device(
		use_gpu,
		gpu_idx
	):
	if use_gpu:
		device = torch.device('cuda:'+str(gpu_idx))
		print ("PyTorch version:", torch.__version__)
		print ("PyTorch GPU count:", torch.cuda.device_count())
		print ("PyTorch Current GPU:", device)
		print ("PyTorch GPU name:", torch.cuda.get_device_name(device))
		return device
	else:
		device = torch.device('cpu')
		return device


def sigmoid(x):
	return 1./1.+np.exp(-x)


def calibration(
		label, 
		pred, 
		bins=10
	):

	width = 1.0 / bins
	bin_centers = np.linspace(0, 1.0-width, bins) + width/2

	conf_bin = []
	acc_bin = []
	counts = []
	for	i, threshold in enumerate(bin_centers):
		bin_idx = np.logical_and(
			threshold - width/2 < pred, 
			pred <= threshold + width
		)
		conf_mean = pred[bin_idx].mean()
		conf_sum = pred[bin_idx].sum()
		if (conf_mean != conf_mean) == False:
			conf_bin.append(conf_mean)
			counts.append(pred[bin_idx].shape[0])

		acc_mean = label[bin_idx].mean()
		acc_sum = label[bin_idx].sum()
		if (acc_mean != acc_mean) == False:
			acc_bin.append(acc_mean)

	conf_bin = np.asarray(conf_bin)
	acc_bin = np.asarray(acc_bin)
	counts = np.asarray(counts)

	ece = np.abs(conf_bin - acc_bin)
	ece = np.multiply(ece, counts)
	ece = ece.sum()
	ece /= np.sum(counts)
	return conf_bin, acc_bin, ece


def evaluate_classification(
		y_list,
		pred_list,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

	auroc = roc_auc_score(y_list, pred_list)
	_, _, ece = calibration(y_list, pred_list)

	'''
	To calculate metric in the below,
	scores should be presented in integer type
	'''
	y_list = y_list.astype(int)
	pred_list = np.around(pred_list).astype(int)

	accuracy = accuracy_score(y_list, pred_list)
	precision = precision_score(y_list, pred_list)
	recall = recall_score(y_list, pred_list)
	f1 = 2.0 * precision * recall / (precision + recall)
	return accuracy, auroc, precision, recall, f1, ece


def evaluate_regression(
		y_list,
		pred_list,
	):
	y_list = torch.cat(y_list, dim=0).detach().cpu().numpy()
	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()

	mse = mean_squared_error(y_list, pred_list)
	rmse = math.sqrt(mse)
	r2 = r2_score(y_list, pred_list)
	return mse, rmse, r2
