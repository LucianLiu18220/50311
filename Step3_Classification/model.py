

import numpy as np
import math
from torch import nn
import torch
from config import get_device_config
import os
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TransformerConv,FiLMConv, TAGConv
import torch_geometric.nn as pyg_nn
from dataprep.BasicSettings import dataset_folder


device_cpu, device_cuda = get_device_config()


class Model_GCNorGAT(torch.nn.Module):
	def __init__(self, num_GCN=3, d_node_feat=896, d_node_reshape=128, d_output=148):
		super(Model_GCNorGAT, self).__init__()
		self.num_GCN = num_GCN
		self.linear_1 = nn.Linear(d_node_feat, d_node_reshape)
		self.multi_GCN_layer = nn.ModuleList()
		for _ in range(num_GCN):
			self.multi_GCN_layer.append(GCNConv(d_node_reshape, d_node_reshape))
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

		edge_index = torch.tensor([[], []], dtype=torch.int64).to(device_cuda[0])
		for key, value in edge_dict.items():
			out_node_index = edge_dict[key][0]
			in_node_index = edge_dict[key][1]
			edge_index = torch.cat((edge_index, torch.tensor([[out_node_index], [in_node_index]], dtype=torch.int64).to(device_cuda[0])), dim=1)

		batch = torch.tensor([0], dtype=torch.int64).to(device_cuda[0])
		x = node_feat.squeeze(0).to(device_cuda[0])

		x = self.linear_1(x)
		for i in range(self.num_GCN):
			x = self.multi_GCN_layer[i](x, edge_index)
			x = torch.relu(x)
		x = pyg_nn.global_max_pool(x, batch)
		x = self.linear_2(x)

		return x


class Model_DIHGNN(torch.nn.Module):
	def __init__(self, d_node_feat=148, d_node_reshape=128, d_output=148):
		super(Model_DIHGNN, self).__init__()
		self.linear_1 = nn.Linear(d_node_feat, d_node_reshape)
		self.dihgnn = DIHGNN(d_node_feat=d_node_reshape, output_channels=d_output)
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

		edge_index = torch.tensor([[], []], dtype=torch.int64).to(device_cuda[0])
		for key, value in edge_dict.items():
			out_node_index = edge_dict[key][0]
			in_node_index = edge_dict[key][1]
			edge_index = torch.cat((edge_index, torch.tensor([[out_node_index], [in_node_index]], dtype=torch.int64).to(device_cuda[0])), dim=1)

		batch = torch.tensor([0], dtype=torch.int64).to(device_cuda[0])
		x = node_feat.squeeze(0).to(device_cuda[0])

		x = self.linear_1(x)
		x, loss_similarity_scores = self.dihgnn(x, edge_index, num_node=node_feat.shape[1])
		x = torch.relu(x)
		x = pyg_nn.global_max_pool(x, batch)
		x = self.linear_2(x)
		return x, loss_similarity_scores


class CustomTanh(nn.Module):
	def __init__(self):
		super().__init__()
		self.a = torch.tensor([2], dtype=torch.float32).to(device_cuda[0])

	def forward(self, x):
		b = torch.tensor([0.5], dtype=torch.float32).to(device_cuda[0])
		exp_x = torch.exp(self.a*x).to(device_cuda[0])
		exp_minus_x = torch.exp(-self.a*x).to(device_cuda[0])

		return b + b * (exp_x - exp_minus_x) / (exp_x + exp_minus_x).to(device_cuda[0])


class DIHGNN(nn.Module):

	def __init__(self, d_node_feat, output_channels):
		super().__init__()
		self.d_node_feat = d_node_feat
		initial_a = 0.9
		self.a = nn.Parameter(torch.tensor([initial_a], dtype=torch.float32))
		self.alpha = torch.tensor([initial_a], dtype=torch.float32)
		self.customtanh = CustomTanh()
		self.linear = nn.Linear(d_node_feat, output_channels)

	def forward(self, x, edge_index, num_node):
		adj_mat = torch.zeros((num_node, num_node), dtype=torch.int64).to(device_cuda[0])
		for i in range(edge_index.shape[1]):
			adj_mat[edge_index[0, i], edge_index[1, i]] = 1
		adj_mat = adj_mat * 1.0
		adj_mat_bar = adj_mat
		L = adj_mat_bar.sum(dim=0)
		R = adj_mat_bar.sum(dim=1)
		L_inv_sqrt = torch.pow(L, -0.5)
		R_inv_sqrt = torch.pow(R, -0.5)
		L_inv_sqrt = torch.where(torch.isinf(L_inv_sqrt), torch.zeros_like(L_inv_sqrt), L_inv_sqrt)
		R_inv_sqrt = torch.where(torch.isinf(R_inv_sqrt), torch.zeros_like(R_inv_sqrt), R_inv_sqrt)
		L_mat_expand = L_inv_sqrt.unsqueeze(1).repeat(1, L.shape[0])
		R_mat_expand = R_inv_sqrt.unsqueeze(0).repeat(R.shape[0], 1)
		adj_mat_bar_norm = L_mat_expand * adj_mat_bar * R_mat_expand

		alpha = self.customtanh(self.a)
		self.alpha = alpha

		adj_mat_bar_norm_inv = torch.inverse(torch.eye(num_node).to(device_cuda[0]) - alpha * adj_mat_bar_norm)
		x = adj_mat_bar_norm_inv @ x
		x = self.linear(x)

		similarity_scores = (x @ x.t()) / math.sqrt(self.d_node_feat)
		mask = torch.ones_like(adj_mat) - torch.eye(num_node).to(device_cuda[0])
		similarity_scores.masked_fill_(mask == 0, 0)
		np_similarity_scores = similarity_scores.detach().cpu().numpy()
		similarity_scores = torch.relu(similarity_scores)
		loss_similarity_scores = torch.sum(similarity_scores) / (num_node * (num_node - 1))

		x = x*(1 - alpha)

		return x, loss_similarity_scores


if __name__ == '__main__':

	pass






