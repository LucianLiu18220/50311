

def get_device_config():
	# device config
	import torch

	device_cpu = torch.device('cpu')
	device_cuda = []
	for i in range(torch.cuda.device_count()):
		device_cuda.append(torch.device(f'cuda:{i}'))

	return device_cpu, device_cuda


def get_config():
	# vul_ann, vul = 'Ree', 'reentrancy'
	vul_ann, vul = 'Tim', 'time_manipulation'

	return {
		# log config
		"logtest": False,
		"seed": 1000,

		# dataset config
		"previous_dataset": False,
		"previous_dataset_path": r"",
		"vul_ann": vul_ann,
		"vul": vul,
		"dataset_name": "SmartBugs",
		"dataset_split": [0.7, 0.15, 0.15],

		# model parameters config
		"previous_model": False,
		"previous_model_path": r"",
		"max_saved_model": 10,

		# model hyper-parameters config
		"num_labels": 2,
		"d_node_feat": 1024,
		"d_node_reshape": 128,
		"d_output": 128,

		# model training config
		"batch_size": 8,
		"num_epochs": 50,
		"wight_pen": 0.05,  # weight penalty
		"lr": 10**-4,
		"optimizer": {
			"weight_decay": 10**-4,
		},
		"scheduler": {
			"factor": 0.5,
			"patience": 1,
			"verbose": True,
			"min_lr_rate": 10**-8
		},
		"criterion": {
			"alpha": [1, 1],
			"gamma": 0,
			"reduction": "sum"
		},
	}


def get_model_optim_schedu_criter():
	import torch
	from model import Model_GCNorGAT, Model_DIHGNN

	config = get_config()
	device_cpu, device_cuda = get_device_config()

	Mymodel = Model_DIHGNN(d_node_feat=config["d_node_feat"], d_node_reshape=config["d_node_reshape"], d_output=config["d_output"])

	if config["previous_model"]:
		checkpoint = torch.load(config["previous_model_path"])
		Mymodel.load_state_dict(checkpoint["net"])
	Mymodel.to(device_cuda[0])

	special_params = [Mymodel.dihgnn.a]
	default_params = [param for name, param in Mymodel.named_parameters() if name not in ['dihgnn.a']]
	# optimizer: Adam, Adadelta, SGD
	optimizer = torch.optim.Adam([
		{'params': default_params},
		{'params': special_params, 'lr': 10 * config["lr"]}
	], lr=config["lr"], weight_decay=config["optimizer"]["weight_decay"])
	# optimizer = torch.optim.Adam(Mymodel.parameters(), lr=config["lr"], weight_decay=config["optimizer"]["weight_decay"])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"],
		verbose=config["scheduler"]["verbose"], min_lr=config["lr"] * config["scheduler"]["min_lr_rate"]
	)
	criterion = torch.nn.CrossEntropyLoss(reduction=config["criterion"]["reduction"]).to(device_cuda[0])

	if config["previous_model"]:
		checkpoint = torch.load(config["previous_model_path"])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])

	Mymodel = Mymodel.to(device_cuda[0])
	criterion = criterion.to(device_cuda[0])
	return Mymodel, optimizer, scheduler, criterion


def get_dataset(config):
	# Due to the storage limitations of the anonymous repository, we are unable to upload the processed dataset files to
	# this repository. After the paper is accepted, we will make the full dataset available on GitHub
	return dataset


if __name__ == '__main__':
	config = get_config()
	device_cpu, device_cuda = get_device_config()
	dataset = get_dataset(config)
	get_model_optim_schedu_criter()

	pass


