

def get_device_config():
	# device config
	import torch

	device_cpu = torch.device('cpu')
	device_cuda = []
	for i in range(torch.cuda.device_count()):
		device_cuda.append(torch.device(f'cuda:{i}'))

	return device_cpu, device_cuda


def get_config():
	from dataprep.BasicSettings import vulnamechange

	vul_type_Tsinghua = [
		['Int', 'Integer overflow or underflow'],
		['Unc', 'Unchecked call return value'],
		['Ree', 'Reentrancy'],
		['Ass', 'Assert violation'],
		['DoS', 'DoS with failed call'],
		['Tim', 'Timestamp dependence'],
		['Use', 'Use of call function with no data'],
		['Mul', 'Multiplication after division'],
		['ERC', 'Using approve function of the ERC-20 token standard'],
		['Ret', 'Return value is always false'],
		['Ext', 'Extra gas consumption'],
		['Loc', 'Locked money'],
		['Ovp', 'Overpowered role'],
		['Red', 'Redundant fallback function'],
		['Uns', 'Unsafe send'],
		['Wor', 'Worse readability with revert']
	]

	vul_type_SmartBugs = [
		['Ari', 'arithmetic'],
		['Ree', 'reentrancy'],
		['Tim', 'time_manipulation']
	]

	dataset_name = "Tsinghua"

	vul_ann = "Unc"
	vul = vulnamechange(vul_ann, dataset_name)

	# vul_ann = "ALLvul"
	# vul = "ALLvul"

	return {
		# log config
		"logtest": False,
		"seed": 1000,

		# dataset config
		"previous_dataset": False,
		"previous_dataset_path": r"",
		"vul_ann": vul_ann,
		"vul": vul,
		"dataset_name": dataset_name,
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
		"wight_pen": 0.05,  # 惩罚函数的权重
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
	from SCcfg_graph_V14_25_model import Model
	from Classification import MultiClassFocalLossWithAlpha

	config = get_config()
	device_cpu, device_cuda = get_device_config()

	Mymodel = Model(d_node_feat=config["d_node_feat"], d_node_reshape=config["d_node_reshape"], d_output=config["d_output"])
	# 读取模型参数
	if config["previous_model"]:
		checkpoint = torch.load(config["previous_model_path"])
		Mymodel.load_state_dict(checkpoint["net"])
	Mymodel.to(device_cuda[0])

	# 为特定层设置不同的学习率
	special_params = [Mymodel.mhgnn.alpha]
	default_params = [param for name, param in Mymodel.named_parameters() if name not in ['mhgnn.alpha']]
	# optimizer: Adam, Adadelta, SGD
	optimizer = torch.optim.Adam([
		{'params': default_params},  # 其他参数，使用默认学习率
		{'params': special_params, 'lr': 10 * config["lr"]}  # 使用特殊学习率
	], lr=config["lr"], weight_decay=config["optimizer"]["weight_decay"])
	# optimizer = torch.optim.Adam(Mymodel.parameters(), lr=config["lr"], weight_decay=config["optimizer"]["weight_decay"])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"],
		verbose=config["scheduler"]["verbose"], min_lr=config["lr"] * config["scheduler"]["min_lr_rate"]
	)
	# criterion: MSELoss, BCELoss
	criterion = torch.nn.CrossEntropyLoss(reduction=config["criterion"]["reduction"]).to(device_cuda[0])
	# criterion = torch.nn.MSELoss(reduction='sum').to(device_cuda[0])
	# criterion = MultiClassFocalLossWithAlpha(
	# 	alpha=config["criterion"]["alpha"], gamma=config["criterion"]["gamma"], reduction=config["criterion"]["reduction"]
	# )

	# 读取模型参数
	if config["previous_model"]:
		checkpoint = torch.load(config["previous_model_path"])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])

	Mymodel = Mymodel.to(device_cuda[0])
	criterion = criterion.to(device_cuda[0])
	return Mymodel, optimizer, scheduler, criterion


def get_dataset(config):
	# from dataprep.datasetgenerate.FourPhase.Phase1.SmartBugs_ast import SmartBugs_ast
	# from dataprep.datasetgenerate.FourPhase.Phase1.SmartBugs_BCcfg import SmartBugs_BCcfg
	# from dataprep.datasetgenerate.FourPhase.Phase1.SmartBugs_SCcfg import SmartBugs_SCcfg
	# from dataprep.datasetgenerate.FourPhase.Phase2.Multimodal_base import Multimodal_base
	# from dataprep.datasetgenerate.FourPhase.Phase3.BCcfg_graph_BBInsNode import BCcfg_graph_BBInsNode
	# from dataprep.datasetgenerate.FourPhase.Phase3.SCcfg_graph_ExprsNode_V1 import SCcfg_graph_ExprsNode_V1
	# from dataprep.datasetgenerate.FourPhase.Phase4.Multimodal_Fusion import Multimodal_Fusion
	#
	# # Phase1
	# dataset_Phase1_SmartBugs_ast = SmartBugs_ast()
	# dataset_Phase1_SmartBugs_BCcfg = SmartBugs_BCcfg()
	# dataset_Phase1_SmartBugs_SCcfg = SmartBugs_SCcfg()
	# # Phase2
	# IRs = [dataset_Phase1_SmartBugs_ast, dataset_Phase1_SmartBugs_BCcfg, dataset_Phase1_SmartBugs_SCcfg]
	# dataset_Multimodal_base = Multimodal_base(IRs=IRs)
	# # Phase3
	# dataset_BCcfg_graph_BBInsNode = BCcfg_graph_BBInsNode(dataset_base=dataset_Multimodal_base)
	# dataset_SCcfg_graph_ExprsNode_V1 = SCcfg_graph_ExprsNode_V1(dataset_base=dataset_Multimodal_base)
	# # Phase4
	# IR_Methods = [dataset_SCcfg_graph_ExprsNode_V1]
	# dataset_Multimodal_Fusion = Multimodal_Fusion(IR_Methods=IR_Methods, target_vul=config["vul"])

	from dataprep.datasetgenerate.FourPhase.Phase1.Tsinghua_SCcfg import Tsinghua_SCcfg
	from dataprep.datasetgenerate.FourPhase.Phase2.Multimodal_base import Multimodal_base
	from dataprep.datasetgenerate.FourPhase.Phase3.SCcfg_graph_ExprsNode_V1 import SCcfg_graph_ExprsNode_V1
	from dataprep.datasetgenerate.FourPhase.Phase4.Multimodal_Fusion import Multimodal_Fusion

	# Phase1
	dataset_Phase1_Tsinghua_SCcfg = Tsinghua_SCcfg()
	# Phase2
	IRs = [dataset_Phase1_Tsinghua_SCcfg]
	dataset_base = Multimodal_base(IRs=IRs)
	# Phase3
	dataset_SCcfg_graph_ExprsNode_V1 = SCcfg_graph_ExprsNode_V1(dataset_base=dataset_base)
	# Phase4
	IR_Methods = [dataset_SCcfg_graph_ExprsNode_V1]
	dataset_Multimodal_Fusion = Multimodal_Fusion(IR_Methods=IR_Methods, target_vul=config["vul"])

	# from dataprep.datasetgenerate.FourPhase.Phase1.SmartBugs_mini_Ree_SCcfg_Stat import SmartBugs_mini_Ree_SCcfg_Stat
	# from dataprep.datasetgenerate.FourPhase.Phase2.Multimodal_base import Multimodal_base
	# from dataprep.datasetgenerate.FourPhase.Phase3.SCcfg_graph_StatmNode_V1 import SCcfg_graph_StatmNode_V1
	# from dataprep.datasetgenerate.FourPhase.Phase4.Multimodal_Fusion import Multimodal_Fusion
	#
	# # Phase1
	# dataset_Phase1_SmartBugs_mini_Ree_SCcfg_Stat = SmartBugs_mini_Ree_SCcfg_Stat()
	# # Phase2
	# IRs = [dataset_Phase1_SmartBugs_mini_Ree_SCcfg_Stat]
	# dataset_Multimodal_base = Multimodal_base(IRs=IRs)
	# # Phase3
	# dataset_SCcfg_graph_StatmNode = SCcfg_graph_StatmNode_V1(dataset_base=dataset_Multimodal_base)
	# # Phase4
	# IR_Methods = [dataset_SCcfg_graph_StatmNode]
	# dataset_Multimodal_Fusion = Multimodal_Fusion(IR_Methods=IR_Methods, target_vul=config["vul"])

	# from dataprep.datasetgenerate.FourPhase.Phase1.SmartBugs_mini_Tim_SCcfg_Stat import SmartBugs_mini_Tim_SCcfg_Stat
	# from dataprep.datasetgenerate.FourPhase.Phase2.Multimodal_base import Multimodal_base
	# from dataprep.datasetgenerate.FourPhase.Phase3.SCcfg_graph_StatmNode_V1 import SCcfg_graph_StatmNode_V1
	# from dataprep.datasetgenerate.FourPhase.Phase4.Multimodal_Fusion import Multimodal_Fusion
	#
	# # Phase1
	# dataset_Phase1_SmartBugs_mini_Tim_SCcfg_Stat = SmartBugs_mini_Tim_SCcfg_Stat()
	# # Phase2
	# IRs = [dataset_Phase1_SmartBugs_mini_Tim_SCcfg_Stat]
	# dataset_Multimodal_base = Multimodal_base(IRs=IRs)
	# # Phase3
	# dataset_SCcfg_graph_StatmNode = SCcfg_graph_StatmNode_V1(dataset_base=dataset_Multimodal_base)
	# # Phase4
	# IR_Methods = [dataset_SCcfg_graph_StatmNode]
	# dataset_Multimodal_Fusion = Multimodal_Fusion(IR_Methods=IR_Methods, target_vul=config["vul"])

	return dataset_Multimodal_Fusion


if __name__ == '__main__':
	config = get_config()
	device_cpu, device_cuda = get_device_config()
	dataset = get_dataset(config)
	get_model_optim_schedu_criter()

	pass

	from tqdm import tqdm
	Ture_index_list = []
	False_index_list = []
	for index in tqdm(range(len(dataset))):
		data = dataset.get(index)
		if data["label"][config["vul"]] == 1:
			Ture_index_list.append(index)
		elif data["label"][config["vul"]] == 0:
			False_index_list.append(index)

