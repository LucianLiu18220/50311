

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import datetime
from torch_geometric.loader import DataLoader
import pickle
import json
from torchsummary import summary
import time
from thop import profile

from dataprep.BasicSettings import resultcolumns_train_valid_test
from dataprep.BasicSettings import seed_torch, vulnamechange, get_logger, loggerpath_config, get_modelpara_account, metric

from config import get_config, get_device_config, get_dataset, get_model_optim_schedu_criter


device_cpu, device_cuda = get_device_config()


config = get_config()
seed_torch(seed=config["seed"])
vul_ann = config["vul_ann"]
logtest = config["logtest"]

Log_path_Current = loggerpath_config(vul_ann, logtest)
logger = get_logger(Log_path_Current + "/conf.log")

dataset = get_dataset(config)
if config["previous_dataset"]:
	dataset_train_valid_test = pickle.load(open(config["previous_dataset_path"], 'rb'))
	Final_index_list = dataset.Final_index_list
	trainset_item_index_list = dataset_train_valid_test["data_index"]["train"]
	validset_item_index_list = dataset_train_valid_test["data_index"]["valid"]
	testset_item_index_list = dataset_train_valid_test["data_index"]["test"]
	if set(Final_index_list) != set(trainset_item_index_list + validset_item_index_list + testset_item_index_list):
		assert False, "The dataset is not consistent with the previous dataset."
	else:
		trainset_indices_list = []
		validset_indices_list = []
		testset_indices_list = []
		for indice in tqdm(range(len(dataset))):
			index = dataset.get(indice)["index"]
			if index in trainset_item_index_list:
				trainset_indices_list.append(indice)
			elif index in validset_item_index_list:
				validset_indices_list.append(indice)
			elif index in testset_item_index_list:
				testset_indices_list.append(indice)
			else:
				pass
		train_dataset = data.Subset(dataset, trainset_indices_list)
		valid_dataset = data.Subset(dataset, validset_indices_list)
		test_dataset = data.Subset(dataset, testset_indices_list)
else:
	train_ratio, test_ratio, valid_ratio = config["dataset_split"]
	total_size = len(dataset)
	train_size = int(train_ratio * total_size)
	valid_size = int(valid_ratio * total_size)
	test_size = total_size - train_size - valid_size

	True_indices_list = []
	False_indices_list = []
	print("Finding True and False indices...")
	for indice in tqdm(range(len(dataset))):
		data_temp = dataset.get(indice)
		if data_temp["label"][dataset.target_vul] == 1:
			True_indices_list.append(indice)
		else:
			False_indices_list.append(indice)

	np.random.shuffle(True_indices_list)
	np.random.shuffle(False_indices_list)
	trainset_indices_list = True_indices_list[:train_size // 2]
	trainset_indices_list.extend(False_indices_list[:train_size // 2])
	validset_indices_list = True_indices_list[train_size // 2:(train_size // 2 + valid_size // 2)]
	validset_indices_list.extend(False_indices_list[train_size // 2:(train_size // 2 + valid_size // 2)])
	testset_indices_list = True_indices_list[(train_size // 2 + valid_size // 2):]
	testset_indices_list.extend(False_indices_list[(train_size // 2 + valid_size // 2):])

	np.random.shuffle(trainset_indices_list)
	np.random.shuffle(validset_indices_list)
	np.random.shuffle(testset_indices_list)

	train_dataset = data.Subset(dataset, trainset_indices_list)
	valid_dataset = data.Subset(dataset, validset_indices_list)
	test_dataset = data.Subset(dataset, testset_indices_list)

	trainset_item_index_list = []
	for item in train_dataset:
		trainset_item_index_list.append(item['index'])
	validset_item_index_list = []
	for item in valid_dataset:
		validset_item_index_list.append(item['index'])
	testset_item_index_list = []
	for item in test_dataset:
		testset_item_index_list.append(item['index'])
	trainset_indices_list = train_dataset.indices
	validset_indices_list = valid_dataset.indices
	testset_indices_list = test_dataset.indices
dataset_train_valid_test = {
	"data_index": {"train": trainset_item_index_list, "valid": validset_item_index_list, "test": testset_item_index_list},
	"dataset_indices": {"train": trainset_indices_list, "valid": validset_indices_list, "test": testset_indices_list}
}
trainset_validset_testset_path = Log_path_Current + "/trainset_validset_testset.pkl"
pickle.dump(dataset_train_valid_test, open(trainset_validset_testset_path, 'wb'))


logger.info("")
logger.info("Binary Classification model training.")
logger.info("{:*^40}".format("Basic settings"))
formatted_config = json.dumps(config, indent=4)
logger.info("{:<25}: {}".format("config", formatted_config))
logger.info("{:<25}: {}".format("dataset dir", dataset.root))
logger.info("")


num_epochs = config["num_epochs"] + 1
batch_size = config["batch_size"]
lr = config["lr"]
model, optimizer, scheduler, criterion = get_model_optim_schedu_criter()
model = model.to(device_cuda[0])
criterion = criterion.to(device_cuda[0])
Trainable_params, NonTrainable_params, Total_params = get_modelpara_account(model)


logger.info("{:*^40}".format("Model hyper-parameters"))
logger.info("{:<25}: {}".format("training device", device_cuda))
logger.info("{:<25}: {}".format("optimizer", optimizer.__str__()))
logger.info("{:<25}: {}".format("loss function", criterion._get_name()))
logger.info("")
logger.info("{:*^40}".format("Model structure"))
logger.info(model)
logger.info("{:<25}: {:,}".format("Trainable params", Trainable_params))
logger.info("{:<25}: {:,}".format("Non-trainable params", NonTrainable_params))
logger.info("{:<25}: {:,}".format("Total params", Total_params))
logger.info("")


logger.info("{:*^40}".format("Training"))
if config["previous_model"]:
	checkpoint = torch.load(config["previous_model_path"])
	start_epoch = checkpoint["epoch"] + 1
	result = checkpoint["result"]
	alpha_list = checkpoint["alpha_list"]
else:
	start_epoch = 0
	result = pd.DataFrame(columns=resultcolumns_train_valid_test)
	alpha_list = pd.DataFrame(columns=["alpha"])
starttime = datetime.datetime.now()


def inference(config, model, dataset, mode="train"):
	model.eval()
	pred_all = torch.tensor([]).to(device_cuda[0]).to(dtype=torch.int64)
	target_all = torch.tensor([]).to(device_cuda[0]).to(dtype=torch.int64)
	loss_sum = 0
	d_loss_sum = 0
	with torch.no_grad():
		with tqdm(total=len(dataset), leave=False) as p_bar:
			for i, data in enumerate(dataset):
				p_bar.set_description("{} inferencing".format(mode))

				# data_SCcfg_graph = data['SCcfg_Stat_graph_StatmNode']['SCcfg_Stat_graph_StatmNode']
				data_SCcfg_graph = data['SCcfg_graph_ExprsNode']['SCcfg_graph_ExprsNode']
				pred_float, d_loss = model(data_SCcfg_graph)  # shape=(bs, 2), type: torch.float32

				mask = (pred_float == pred_float.max(dim=1, keepdim=True)[0]).to(dtype=torch.int64)  # shape=(bs, 2), type: torch.int64
				pred = torch.argmax(mask, -1)  # shape=(bs,), type: torch.int64

				target = data["label"][config["vul"]].to(device_cuda[0])  # shape=(bs,), type: torch.int64
				loss = criterion(pred_float, target) + d_loss * config["wight_pen"]
				loss_sum += loss.detach().cpu().numpy()
				d_loss_sum += d_loss.detach().cpu().numpy()

				pred_all = torch.cat((pred_all, pred), dim=0)
				target_all = torch.cat((target_all, target), dim=0)

				torch.cuda.empty_cache()

				allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)
				p_bar.set_postfix({'allocated GPU memory': '%.4fMB' % allocated_memory})

				p_bar.update(1)
	loss = loss_sum / len(dataset)
	d_loss = d_loss_sum / len(dataset)
	return pred_all, target_all, loss, d_loss


for epoch in range(start_epoch, num_epochs):
	logger.info(f'【Epoch: {epoch:02d}】')

	old_time = time.time()
	pred_all, target_all, loss_train, d_loss_train = inference(config, model, train_dataset, mode="train")
	current_time = time.time()
	execute_time = current_time - old_time  # unit: second
	logger.info("\ttrain set evaluation finished, time: {0:.2f} mins".format(execute_time/60))
	metric_train_TFPN, metric_train_APRSF = metric(pred_all, target_all)

	old_time = time.time()
	pred_all, target_all, loss_valid, d_loss_valid = inference(config, model, valid_dataset, mode="valid")
	current_time = time.time()
	execute_time = current_time - old_time  # unit: second
	logger.info("\ttest set evaluation finished, time: {0:.2f} mins".format(execute_time / 60))
	metric_valid_TFPN, metric_valid_APRSF = metric(pred_all, target_all)
	scheduler.step(loss_valid)

	old_time = time.time()
	pred_all, target_all, loss_test, d_loss_test = inference(config, model, test_dataset, mode="test")
	current_time = time.time()
	execute_time = current_time - old_time  # unit: second
	logger.info("\ttest set evaluation finished, time: {0:.2f} mins".format(execute_time/60))
	metric_test_TFPN, metric_test_APRSF = metric(pred_all, target_all)

	param_groups = optimizer.param_groups
	for i, param_group in enumerate(param_groups):
		logger.info(f"\tlearning rate: group {i} : {param_group['lr']}")

	alpha = float(model.dihgnn.alpha)
	alpha_list = pd.concat([alpha_list, pd.DataFrame([alpha], columns=["alpha"])], axis=0)
	alpha_list = alpha_list.reset_index(drop=True)
	logger.info(f"\talpha: {alpha:.4f}")
	logger.info(f"\talpha_list: {[alpha for alpha in list(alpha_list['alpha'])]}")

	logger.info("\t{:^19}   |   {:^18}   |   {:^18}".format("Train", "Valid", "Test"))
	logger.info(f"\tTP: {metric_train_TFPN['TP']:=5d}, FP: {metric_train_TFPN['FP']:=5d}  |  TP: {metric_valid_TFPN['TP']:=5d}, FP: {metric_valid_TFPN['FP']:=5d}  |  TP: {metric_test_TFPN['TP']:=5d}, FP: {metric_test_TFPN['FP']:=5d}")
	logger.info(f"\tFN: {metric_train_TFPN['FN']:=5d}, TN: {metric_train_TFPN['TN']:=5d}  |  FN: {metric_valid_TFPN['FN']:=5d}, TN: {metric_valid_TFPN['TN']:=5d}  |  FN: {metric_test_TFPN['FN']:=5d}, TN: {metric_test_TFPN['TN']:=5d}")
	logger.info(f"\tTrain -- Accu: {metric_train_APRSF['Accu']:.4f}, Prec: {metric_train_APRSF['Prec']:.4f}, Reca: {metric_train_APRSF['Reca']:.4f}, Spec: {metric_train_APRSF['Spec']:.4f}, F1Sc: {metric_train_APRSF['F1Sc']:.4f}")
	logger.info(f"\tValid -- Accu: {metric_valid_APRSF['Accu']:.4f}, Prec: {metric_valid_APRSF['Prec']:.4f}, Reca: {metric_valid_APRSF['Reca']:.4f}, Spec: {metric_valid_APRSF['Spec']:.4f}, F1Sc: {metric_valid_APRSF['F1Sc']:.4f}")
	logger.info(f"\tTest  -- Accu: {metric_test_APRSF['Accu']:.4f}, Prec: {metric_test_APRSF['Prec']:.4f}, Reca: {metric_test_APRSF['Reca']:.4f}, Spec: {metric_test_APRSF['Spec']:.4f}, F1Sc: {metric_test_APRSF['F1Sc']:.4f}")
	logger.info(f"\tAverage train Loss: {loss_train:.7f}, d_loss: {d_loss_train:.7f}")
	logger.info(f"\tAverage valid Loss: {loss_valid:.7f}, d_loss: {d_loss_valid:.7f}")
	logger.info(f"\tAverage test  Loss: {loss_test:.7f}, d_loss: {d_loss_test:.7f}")

	""" training """
	model.train()
	old_time = time.time()
	with tqdm(total=len(train_dataset), leave=False) as p_bar:
		for i, data in enumerate(train_dataset):
			p_bar.set_description("training")

			optimizer.zero_grad()

			# data_SCcfg_graph = data['SCcfg_Stat_graph_StatmNode']['SCcfg_Stat_graph_StatmNode']
			data_SCcfg_graph = data['SCcfg_graph_ExprsNode']['SCcfg_graph_ExprsNode']
			pred_float, d_loss = model(data_SCcfg_graph)  # shape=(bs, 2), type: torch.float32

			target = data["label"][config["vul"]].to(device_cuda[0])  # shape=(bs,), type: torch.int64
			loss = criterion(pred_float, target) + d_loss * config["wight_pen"]
			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)
			p_bar.set_postfix({'allocated GPU memory': '%.4fMB' % allocated_memory})

			p_bar.update(1)

	current_time = time.time()
	execute_time = current_time - old_time  # unit: second
	logger.info("\ttraining finished, time: {0:.2f} mins".format(execute_time/60))

	a = [
		loss_train,
		loss_valid,
		loss_test,
		metric_train_TFPN['TP'],
		metric_train_TFPN['FN'],
		metric_train_TFPN['FP'],
		metric_train_TFPN['TN'],
		metric_train_APRSF['Accu'],
		metric_train_APRSF['Prec'],
		metric_train_APRSF['Reca'],
		metric_train_APRSF['Spec'],
		metric_train_APRSF['F1Sc'],
		metric_valid_TFPN['TP'],
		metric_valid_TFPN['FN'],
		metric_valid_TFPN['FP'],
		metric_valid_TFPN['TN'],
		metric_valid_APRSF['Accu'],
		metric_valid_APRSF['Prec'],
		metric_valid_APRSF['Reca'],
		metric_valid_APRSF['Spec'],
		metric_valid_APRSF['F1Sc'],
		metric_test_TFPN['TP'],
		metric_test_TFPN['FN'],
		metric_test_TFPN['FP'],
		metric_test_TFPN['TN'],
		metric_test_APRSF['Accu'],
		metric_test_APRSF['Prec'],
		metric_test_APRSF['Reca'],
		metric_test_APRSF['Spec'],
		metric_test_APRSF['F1Sc']
	]
	tempresult = pd.DataFrame([a], columns=resultcolumns_train_valid_test)
	result = pd.concat([result, tempresult], axis=0)
	result = result.reset_index(drop=True)

	# save training result
	endtime = datetime.datetime.now()
	tainingtime = (endtime - starttime).seconds  # unit: second
	path = Log_path_Current + "/trainingresult.pkl"
	pickle.dump([tainingtime, result, alpha_list], open(path, 'wb'))

	checkpoint = {
		"net": model.state_dict(),
		'optimizer': optimizer.state_dict(),
		"scheduler": scheduler.state_dict(),
		"epoch": epoch,
		"result": result,
		"alpha_list": alpha_list
	}
	saved_model_folder = os.path.join(Log_path_Current, "saved_model")
	if not os.path.isdir(saved_model_folder):
		os.mkdir(saved_model_folder)
	saved_model_path = os.path.join(saved_model_folder, "saved_model_epoch_%s.pth" % (str(epoch)))
	torch.save(checkpoint, saved_model_path)
	epoch_list = []
	for file in os.listdir(saved_model_folder):
		if file.endswith(".pth"):
			epoch_list.append(int(file.split("_")[-1].split(".")[0]))
	epoch_list.sort()
	if len(epoch_list) > config["max_saved_model"]:
		for epoch in epoch_list[:-config["max_saved_model"]]:
			os.remove(os.path.join(saved_model_folder, "saved_model_epoch_%s.pth" % (str(epoch))))
	pass

endtime = datetime.datetime.now()
logger.info("training time: {0:.2f} minutes".format((endtime - starttime).seconds / 60))




