import torch
from tqdm import tqdm
import pandas as pd
import os, sys
import numpy as np
import re
import subprocess
from transformers import BertModel, BertTokenizer
from Step1_CFGgeneration.SCFG_ECFG import CFG_sourcecode_generator_expression, draw_ECFG
from Step1_CFGgeneration.SCFG_ECFG import CFG_sourcecode_generator_statement, draw_SCFG
from Step2_NodeEmbedding.normalize_CFG import normSCFG, normECFG


def set_compiler(version):
	command = f'solc-select use {version}'
	subprocess.run(command, stdout=subprocess.DEVNULL, shell=True, check=True)


def get_installed_versions():
	# Call the command line to get installed versions
	result = subprocess.run(["solc-select", "versions"], capture_output=True, text=True)
	installed_versions = re.findall(r'\d+\.\d+\.\d+', result.stdout)
	installed_versions.sort()
	return installed_versions


def get_bert_features(text, tokenizer, model):
	inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
	with torch.no_grad():
		outputs = model(**inputs)
	return outputs.last_hidden_state[:, 0, :]  # dimension (1, 128)

def get_SCFGnodefeat_bert(node_dict, tokenizer, model):
	# Initialize node_feat with shape (batch, len(node_dict), feat_dim)
	node_feat = torch.zeros((1, len(node_dict), 640), dtype=torch.float32)
	for node_index in node_dict.keys():
		node = node_dict[node_index]

		temp_feat = torch.zeros((1, 0), dtype=torch.float32)

		text_list = [
			node['Cont']['str'],
			node['Func']['Funct_Type'],
			node['Func']['str'],
			node['Stat']['State_Type'],
			node['Stat']['str']
		]

		for text in text_list:
			temp_feat = torch.cat((temp_feat, get_bert_features(text, tokenizer, model)), dim=1)

		node_feat[0, node_index, :] = temp_feat

	return node_feat


def SCFG_Embed(graph_data):
	node_dict, edge_dict = graph_data

	# Initialize tokenizer and model
	current_folder = os.path.dirname(os.path.abspath(__file__))
	main_folder = os.path.dirname(current_folder)
	model_directory = os.path.join(main_folder, 'bert-tiny')
	tokenizer = BertTokenizer.from_pretrained(model_directory)
	model = BertModel.from_pretrained(model_directory)

	'''The following process performs BERT-tiny embedding for node_dict of each node'''
	node_feat_fine_grained = get_SCFGnodefeat_bert(node_dict, tokenizer, model)

	graph_data_New = {
		"node_dict": node_dict,
		"node_feat_BERT_fine_grained": node_feat_fine_grained,
		"edge_dict": edge_dict  # torch.tensor([batch, len(node_dict), feat_dim], dtype=torch.int64)
	}
	return graph_data_New


def get_ECFGnodefeat_bert(node_dict, tokenizer, model):
	# Initialize node_feat with shape (batch, len(node_dict), feat_dim)
	node_feat = torch.zeros((1, len(node_dict), 1024), dtype=torch.float32)
	for node_index in node_dict.keys():
		node = node_dict[node_index]

		temp_feat = torch.zeros((1, 0), dtype=torch.float32)

		'''Find the attributes of Expr'''
		Expr_Type = node['Expr']['Exprs_Type']
		Expr_attr_all = node['Expr']['Exprs_attr']
		Expr_attr = None
		if Expr_Type == 'AssignmentOperation':
			Expr_attr = Expr_attr_all["type"].name
		elif Expr_Type == 'BinaryOperation':
			Expr_attr = Expr_attr_all["type"].name
		elif Expr_Type == 'UnaryOperation':
			Expr_attr = Expr_attr_all['type'].name
		elif Expr_Type == 'Identifier':
			Expr_attr = Expr_attr_all["value"]["type"]
		elif Expr_Type in ["CallExpression", "SuperCallExpression"]:
			Expr_attr = Expr_attr_all["type_call"]
		elif Expr_Type == 'Literal':
			Expr_attr = Expr_attr_all["type"]
		elif Expr_Type == 'MemberAccess':
			Expr_attr = Expr_attr_all['type']
		elif Expr_Type == 'NewArray':
			Expr_attr = Expr_attr_all['array_type']
		elif Expr_Type == 'NewContract':
			Expr_attr = Expr_attr_all["contract_name"]
		elif Expr_Type == 'NewElementaryType':
			Expr_attr = Expr_attr_all['type']
		elif Expr_Type == 'TypeConversion':
			Expr_attr = Expr_attr_all['type']

		if type(Expr_attr) is not str:
			Expr_attr = str(Expr_attr)

		text_list = [
			node['Cont']['str'],
			node['Func']['Funct_Type'],
			node['Func']['str'],
			node['Stat']['State_Type'],
			node['Stat']['str'],
			node['Expr']['Exprs_Type'],
			node['Expr']['str'],
			Expr_attr
		]

		for text in text_list:
			temp_feat = torch.cat((temp_feat, get_bert_features(text, tokenizer, model)), dim=1)

		node_feat[0, node_index, :] = temp_feat

	return node_feat


def ECFG_Embed(graph_data):
	node_dict, edge_dict = graph_data

	# Initialize tokenizer and model
	current_folder = os.path.dirname(os.path.abspath(__file__))
	main_folder = os.path.dirname(current_folder)
	model_directory = os.path.join(main_folder, 'bert-tiny')
	tokenizer = BertTokenizer.from_pretrained(model_directory)
	model = BertModel.from_pretrained(model_directory)

	'''The following process performs BERT embedding for node_dict of each node'''
	node_feat_fine_grained = get_ECFGnodefeat_bert(node_dict, tokenizer, model)

	graph_data_New = {
		"node_dict": node_dict,
		"node_feat_BERT_fine_grained": node_feat_fine_grained,
		"edge_dict": edge_dict  # torch.tensor([batch, len(node_dict), feat_dim], dtype=torch.int64)
	}
	return graph_data_New


if __name__ == '__main__':

	versions = get_installed_versions()
	set_compiler("0.4.24")

	current_folder = os.path.dirname(os.path.abspath(__file__))
	main_folder = os.path.dirname(current_folder)
	dataset_folder = os.path.join(main_folder, "dataset")
	contract_filebasename = "CodeExample.sol"
	contract_file = os.path.join(dataset_folder, contract_filebasename)

	"""SCFG"""
	# full_graph = CFG_sourcecode_generator_statement(contract_file)
	# nG = normSCFG(full_graph)
	# Graph = (nG.node_dict, nG.edge_dict)
	# SCFG_embed = SCFG_Embed(Graph)

	"""ECFG"""
	full_graph = CFG_sourcecode_generator_expression(contract_file)
	nG = normECFG(full_graph)
	Graph = (nG.node_dict, nG.edge_dict)
	ECFG_embed = ECFG_Embed(Graph)

	pass
