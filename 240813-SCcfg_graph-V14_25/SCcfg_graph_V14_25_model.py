

import numpy as np
import math
from torch import nn
import torch
from SCcfg_graph_V14_25_config import get_device_config
import os
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TransformerConv,FiLMConv, TAGConv
# from torch_geometric.nn.conv import gatv2_conv
# from torch_geometric.nn.conv import transformer_conv
import torch_geometric.nn as pyg_nn
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from dataprep.BasicSettings import dataset_folder
from transformers import BertModel, BertTokenizer


device_cpu, device_cuda = get_device_config()


class Model(torch.nn.Module):
	def __init__(self, d_node_feat=148, d_node_reshape=128, d_output=148):
		super(Model, self).__init__()
		self.linear_1 = nn.Linear(d_node_feat, d_node_reshape)
		self.mhgnn = MHGNN(d_node_feat=d_node_reshape, output_channels=d_output)
		self.linear_2 = nn.Linear(d_output, 2)

	def prms(self):
		params_and_grads = {
			name: {
				"param": param.detach().cpu().numpy(),
				"grad": (param.grad.detach().cpu().numpy() if param.grad is not None else None)
			}
			for name, param in self.named_parameters()
		}
		return params_and_grads

	def forward(self, data):
		node_feat = data["node_feat_BERT_fine_grained"]
		edge_dict = data["edge_dict"]

		# 初始化edge_index，它是一个tensor，用于存储边的index，每一列是一条边，第一行是起始节点的index，第二行是终止节点的index
		edge_index = torch.tensor([[], []], dtype=torch.int64).to(device_cuda[0])
		for key, value in edge_dict.items():
			out_node_index = edge_dict[key][0]
			in_node_index = edge_dict[key][1]
			edge_index = torch.cat((edge_index, torch.tensor([[out_node_index], [in_node_index]], dtype=torch.int64).to(device_cuda[0])), dim=1)

		batch = torch.tensor([0], dtype=torch.int64).to(device_cuda[0])
		x = node_feat.squeeze(0).to(device_cuda[0])

		x = self.linear_1(x)
		x, loss_similarity_scores = self.mhgnn(x, edge_index, num_node=node_feat.shape[1])
		x = torch.relu(x)
		x = pyg_nn.global_max_pool(x, batch)
		x = self.linear_2(x)

		return x, loss_similarity_scores


class CustomTanh(nn.Module):
	def __init__(self):
		super().__init__()
		# 自定义 tanh 实现
		self.a = torch.tensor([2], dtype=torch.float32).to(device_cuda[0])

	def forward(self, x):
		# 初始化一个值为0.5的tensor
		b = torch.tensor([0.5], dtype=torch.float32).to(device_cuda[0])
		exp_x = torch.exp(self.a*x).to(device_cuda[0])
		exp_minus_x = torch.exp(-self.a*x).to(device_cuda[0])

		return b + b * (exp_x - exp_minus_x) / (exp_x + exp_minus_x).to(device_cuda[0])


class MHGNN(nn.Module):

	def __init__(self, d_node_feat, output_channels):
		super().__init__()
		self.d_node_feat = d_node_feat
		initial_a = 0.9
		self.a = nn.Parameter(torch.tensor([initial_a], dtype=torch.float32))
		self.alpha = torch.tensor([initial_a], dtype=torch.float32)
		self.customtanh = CustomTanh()
		self.linear = nn.Linear(d_node_feat, output_channels)

	def forward(self, x, edge_index, num_node):
		r"""
		GIHN: Graph Infinite Hop Network

		Math:
		"""
		# 根据edge_index构造对称邻接矩阵
		adj_mat = torch.zeros((num_node, num_node), dtype=torch.int64).to(device_cuda[0])
		for i in range(edge_index.shape[1]):
			adj_mat[edge_index[0, i], edge_index[1, i]] = 1
			adj_mat[edge_index[1, i], edge_index[0, i]] = 1
		adj_mat = adj_mat * 1.0

		deg = adj_mat.sum(dim=1).to(device_cuda[0])
		deg_pow = deg.pow(-0.5)
		# 将deg_pow等于inf的值替换为0
		deg_pow = torch.where(torch.isinf(deg_pow), torch.zeros_like(deg_pow), deg_pow)
		deg_inv_sqrt = torch.diag(deg_pow)
		adj_mat_norm = torch.matmul(torch.matmul(deg_inv_sqrt, adj_mat), deg_inv_sqrt)
		# 尝试对矩阵进行小的扰动来改善条件数，例如添加微小的值到对角线元素（对角加载），保证数值稳定性
		adj_mat_norm += torch.eye(adj_mat_norm.size(0)).to(device_cuda[0]) * 1e-8

		alpha = self.customtanh(self.a)
		self.alpha = alpha

		# 使用 torch.linalg.eigh 进行A_bar_norm的特征值分解, A_bar_norm = U * diag(lamda) * U^T
		eigenvalues, eigenvectors = torch.linalg.eigh(adj_mat_norm)
		# 将eigenvalues中大于1的值置1, 小于-1的值置-1
		eigenvalues = torch.clamp(eigenvalues, max=1.0)
		eigenvalues = torch.clamp(eigenvalues, min=-1.0)
		eigenvalues = alpha * eigenvalues
		# U 是包含特征向量的酉矩阵
		U = eigenvectors

		# 计算 (aλ)^n+1
		lamda_n_plus_1 = eigenvalues.pow(num_node + 1)
		# 计算 (aλ)^n+1 - 1
		lamda_n_plus_1_min_1 = lamda_n_plus_1 - torch.ones(num_node).to(device_cuda[0])
		# 计算 aλ - 1
		lamda_min_1 = eigenvalues - torch.ones(num_node).to(device_cuda[0])
		# 确保除数不为零，替换为一个非常小的非零值
		lamda_min_1 = torch.where(lamda_min_1 == 0, torch.tensor(1e-10, dtype=lamda_min_1.dtype).to(device_cuda[0]), lamda_min_1).to(device_cuda[0])
		# 计算 ((aλ)^n+1 - 1)/(aλ - 1)
		lamda_n_plus_1_min_1_div_lamda_min_1 = lamda_n_plus_1_min_1 / lamda_min_1
		# 计算 (1-a)((aλ)^n+1 - 1)/(aλ - 1)
		lamda_n_plus_1_min_1_div_lamda_min_1 = lamda_n_plus_1_min_1_div_lamda_min_1
		# 计算 U * Q * U^T, 其中 Q = diag ( (1-a)((aλ)^n+1 - 1)/(aλ - 1) )
		U_Q = torch.matmul(U, torch.diag(lamda_n_plus_1_min_1_div_lamda_min_1))
		U_Q_Ut = torch.matmul(U_Q, U.t())

		# 计算 U * Q * U^T * X
		x = torch.matmul(U_Q_Ut, x)
		# 计算 U * Q * U^T * X * W
		x = self.linear(x)

		# 计算图节点特征相似度
		similarity_scores = (x @ x.t()) / math.sqrt(self.d_node_feat)
		# 初始化一个全一矩阵，形状和adj_mat相同
		mask = torch.ones_like(adj_mat) - torch.eye(num_node).to(device_cuda[0])
		similarity_scores.masked_fill_(mask == 0, 0)
		np_similarity_scores = similarity_scores.detach().cpu().numpy()
		similarity_scores = torch.relu(similarity_scores)
		loss_similarity_scores = torch.sum(similarity_scores) / (num_node * (num_node - 1))


		x = x * (1 - alpha)

		return x, loss_similarity_scores


if __name__ == '__main__':

	import os
	from tqdm import tqdm
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
	# dataset_SCcfg_graph_ExprsNode_V1 = SCcfg_graph_ExprsNode_V1(dataset_base=dataset_Multimodal_base)
	# # Phase4
	# IR_Methods = [dataset_SCcfg_graph_ExprsNode_V1]
	# dataset_Multimodal_Fusion = Multimodal_Fusion(IR_Methods=IR_Methods, target_vul="reentrancy")

	from dataprep.func.GeneralFunc import FindFile

	dataset_path = r"F:\Project\SmtCon_dataset\SmartContract\SmartBugs\Phase4\Multimodal_Fusion__time_manipulation_SCcfg_graph_ExprsNode\processed"
	filepaths, _, _ = FindFile(dataset_path, ".pt")
	# fileTitles是类似data_47423的文件名，但还有两个元素是pre_transform.pt和pre_filter.pt，所以要去掉
	filepath_list = [filepath for filepath in filepaths if "data" in filepath]

	# model = Model(d_node_feat=896, d_node_reshape=128, d_output=128)
	# model.to(device_cuda[0])
	# for filepath in tqdm(filepath_list):
	# 	data = torch.load(filepath)
	# 	data_SCcfg_graph = data['SCcfg_graph_ExprsNode']['SCcfg_graph_ExprsNode']
	# 	model(data_SCcfg_graph)
	# 	torch.cuda.empty_cache()

	torch.autograd.set_detect_anomaly(True)

	model = Model(d_node_feat=1024, d_node_reshape=128, d_output=128)
	model.to(device_cuda[0])
	optimizer = torch.optim.Adam(model.parameters(), lr=10**-4, weight_decay=10**-4)
	criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device_cuda[0])
	model.train()
	for filepath in tqdm(filepath_list):
		data = torch.load(filepath)
		data_SCcfg_graph = data['SCcfg_graph_ExprsNode']['SCcfg_graph_ExprsNode']
		pred_float = model(data_SCcfg_graph)  # shape=(bs, 2), type: torch.float32

		target = data["label"]["reentrancy"].to(device_cuda[0])  # shape=(bs,), type: torch.int64
		loss = criterion(pred_float, target)  # Focal loss
		loss.backward()
		optimizer.step()


	# pt_index = 0
	# pt_folder = r"E:\Project\SmtCon_dataset\SmartContract\SmartBugs\Phase3\SCcfg_graph_ExprsNode\processed"
	# pt_path = os.path.join(pt_folder, "data_" + str(pt_index) + ".pt")
	# data = torch.load(pt_path)
	#
	# IR = "SCcfg"
	# Method = "graph_ExprsNode"
	# IR_Method = IR + "_" + Method
	# data_IR_Method = data[IR_Method][IR_Method]

	pass






