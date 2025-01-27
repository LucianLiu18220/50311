import re
import subprocess
import os
from tqdm import tqdm
from solc import compile_files  # Do not install solc, instead install py-solc!
from slither import Slither  # Do not install slither, instead install slither-analyzer!
import networkx as nx
from slither.core.cfg.node import NodeType
from slither.core import expressions
import matplotlib.pyplot as plt


def set_compiler(version):
	command = f'solc-select use {version}'
	subprocess.run(command, stdout=subprocess.DEVNULL, shell=True, check=True)


def get_installed_versions():
	result = subprocess.run(["solc-select", "versions"], capture_output=True, text=True)
	installed_versions = re.findall(r'\d+\.\d+\.\d+', result.stdout)
	installed_versions.sort()
	return installed_versions


Expression_type = {
	'AssignmentOperation': ["type"],
	'BinaryOperation': ["type"],
	'CallExpression': ["names", "type_call"],
	'ConditionalExpression': [],
	'ElementaryTypeNameExpression': [],
	'Identifier': ["value"],
	'IndexAccess': [],
	'Literal': ["value", "converted_value", "type", "subdenomination"],
	'MemberAccess': ["member_name", "type"],
	'NewArray': ["array_type"],
	'NewContract': ["contract_name", "call_value", "call_salt"],
	'NewElementaryType': ["type"],
	'SuperCallExpression': ["names", "type_call"],
	'TupleExpression': [],
	'TypeConversion': ["type"],
	'UnaryOperation': ["type", "is_prefix"],
	'NoneType': []
}


class StatementNode:
	def __init__(self, statementnode):
		self.statementnode = statementnode
		# Function to which this statement belongs
		self.function = statementnode.function
		self.function_type = self.function.function_type
		# Contract to which this statement belongs
		self.contract = statementnode.function.contract
		self.contract_kind = self.contract.contract_kind
		# Full description of the statement
		self.state_str = statementnode.__str__()
		self.state_id = statementnode.node_id

		self.contract_uniqID = "Cont: " + self.contract_kind + "--" + self.contract.__str__()
		self.function_ID = "Func: " + self.function_type.name + "--" + self.function.__str__()
		self.function_uniqID = self.contract_uniqID + "__" + self.function_ID
		self.statement_ID = "Stat: " + str(self.state_id) + "--" + self.state_str
		self.statement_uniqID = self.function_uniqID + "__" + self.statement_ID

		# Find all expressions in StatementNode and construct ExpressionNode
		self.set_expnode = []
		self.exp_id = 0  # Used to mark the order of expressions, ensuring unique identification within the same statement
		if self.statementnode.expression:
			self.contruct_expnode(self.statementnode.expression)
		# If set_expnode has no expression nodes, add a None ExpressionNode to set_expnode
		if not self.set_expnode:
			self.exp_id += 1
			self.set_expnode.append(ExpressionNode(self, None, self.exp_id))

	def contruct_expnode(self, exprs):
		self.exp_id += 1
		exp_id = self.exp_id
		exprs_type = type(exprs).__name__
		# Construct ExpressionNode based on exprs
		exprs_node = ExpressionNode(self, exprs, exp_id)
		# Check if there are expressions inside exprs based on exprs_type. If there are, recursively call contruct_expnode
		if exprs_node.inside_exprs_list:
			for inside_exprs in exprs_node.inside_exprs_list:
				self.contruct_expnode(inside_exprs)
		self.set_expnode.append(exprs_node)
		# If there are no expressions, recursion ends. On the way back, starting from the deepest expression, each layer of ExpressionNode constructs an edge pointing to the previous layer
		pass

	def __repr__(self):
		return f"StatementNode(statement: {self.state_str}, function: {self.function.__str__()}, contract: {self.contract.__str__()})"

	def __hash__(self):
		return hash(self.statement_uniqID)

	def __eq__(self, other):
		return isinstance(other, StatementNode) and self.statement_uniqID == other.statement_uniqID

	def is_same_function(self, other):
		return isinstance(other, StatementNode) and self.function_uniqID == other.function_uniqID

	def is_same_contract(self, other):
		return isinstance(other, StatementNode) and self.contract_uniqID == other.contract_uniqID

class ExpressionNode:
	def __init__(self, statementnode, exprs, exp_id):
		# Attributes inherited from StatementNode
		self.statementnode = statementnode
		# Function to which this expression belongs
		self.function = statementnode.function
		self.function_type = self.function.function_type
		# Contract to which this expression belongs
		self.contract = statementnode.function.contract
		self.contract_kind = statementnode.contract_kind
		# Full description of the statement
		self.state_str = statementnode.state_str
		self.state_id = statementnode.state_id
		# Attributes specific to ExpressionNode
		self.exprs = exprs
		self.exp_id = exp_id
		self.exprs_type = type(exprs).__name__
		self.exprs_str = exprs.__str__()
		self.exprs_attr = self.get_exprs_attr()

		self.contract_uniqID = statementnode.contract_uniqID
		self.function_ID = statementnode.function_ID
		self.function_uniqID = statementnode.function_uniqID
		self.statement_ID = statementnode.statement_ID
		self.statement_uniqID = statementnode.statement_uniqID
		self.expression_ID = "Expr: " + str(self.exp_id) + "--" + self.exprs_type + "--" + self.exprs_str
		self.expression_uniqID = self.statement_uniqID + "__" + self.expression_ID

		# Process expressions inside the current expression
		self.expressions_inside()

	def get_exprs_attr(self):
		# Extract attributes based on the type of expression
		if self.exprs_type == "AssignmentOperation":
			exprs_attr = {
				"type": getattr(self.exprs, "type")
			}
		elif self.exprs_type == "BinaryOperation":
			exprs_attr = {
				"type": getattr(self.exprs, "type")
			}
		elif self.exprs_type == "Identifier":
			exprs_attr = {
				"value": {
					"type": type(self.exprs.value).__name__,
					"name": self.exprs.value.__str__()
				}
			}
		elif self.exprs_type in ["CallExpression", "SuperCallExpression"]:
			exprs_attr = {
				"names": getattr(self.exprs, "names", None),  # List or None
				"type_call": getattr(self.exprs, "type_call"),  # str
			}
		elif self.exprs_type == "Literal":
			exprs_attr = {
				"value": getattr(self.exprs, "value"),  # int or str
				"converted_value": getattr(self.exprs, "converted_value"),  # int or str
				"type": self.exprs.type.__str__(),
				"subdenomination": getattr(self.exprs, "subdenomination", None)  # str or None
			}
		elif self.exprs_type == "MemberAccess":
			exprs_attr = {
				"member_name": getattr(self.exprs, "member_name"),  # str
				"type": self.exprs.type.__str__()
			}
		elif self.exprs_type == "NewArray":
			exprs_attr = {
				"array_type": self.exprs.array_type.__str__()
			}
		elif self.exprs_type == "NewContract":
			exprs_attr = {
				"contract_name": getattr(self.exprs, "contract_name"),  # str
			}
		elif self.exprs_type == "NewElementaryType":
			exprs_attr = {
				"type": self.exprs.type.__str__()
			}
		elif self.exprs_type == "TypeConversion":
			exprs_attr = {
				"type": self.exprs.type.__str__()
			}
		elif self.exprs_type == "UnaryOperation":
			exprs_attr = {
				"type": getattr(self.exprs, "type"),
				"is_prefix": getattr(self.exprs, "is_prefix")  # bool
			}
		elif self.exprs_type in ['ConditionalExpression', 'ElementaryTypeNameExpression', 'IndexAccess', 'TupleExpression', "NoneType"]:
			exprs_attr = {}
		else:
			exprs_attr = {}

		return exprs_attr

	def expressions_inside(self):
		# Based on the type of slither.core.expressions.Expression, process the expressions contained within the current node
		'''
		AssignmentOperation: 			expression_left: expression, expression_right: expression
		BinaryOperation: 				expression_left: expression, expression_right: expression
		CallExpression: 				called: expression, arguments: List[expression]
		ConditionalExpression: 			if_expression: expression, then_expression: expression, else_expression: expression
		ElementaryTypeNameExpression: 	None
		Identifier: 					None
		IndexAccess: 					expression_left: expression, expression_right: expression
		Literal: 						None
		MemberAccess: 					expression: expression
		NewArray: 						None
		NewContract: 					None
		NewElementaryType: 				None
		SuperCallExpression: 			called: expression, arguments: List[expression]
		SuperIdentifier: 				None
		TupleExpression: 				expressions: List[expression]
		TypeConversion: 				expression: expression
		UnaryOperation: 				expression: expression
		'''
		if isinstance(self.exprs, expressions.AssignmentOperation)\
				or isinstance(self.exprs, expressions.BinaryOperation)\
				or isinstance(self.exprs, expressions.IndexAccess):
			self.inside_exprs_list = [self.exprs.expression_right, self.exprs.expression_left]
		elif isinstance(self.exprs, expressions.CallExpression)\
				or isinstance(self.exprs, expressions.SuperCallExpression):
			self.inside_exprs_list = self.exprs.arguments + [self.exprs.called]
		elif isinstance(self.exprs, expressions.ConditionalExpression):
			self.inside_exprs_list = [self.exprs.else_expression, self.exprs.then_expression, self.exprs.if_expression]
		elif isinstance(self.exprs, expressions.MemberAccess):
			self.inside_exprs_list = [self.exprs.expression]
		elif isinstance(self.exprs, expressions.TupleExpression):
			self.inside_exprs_list = self.exprs.expressions
		elif isinstance(self.exprs, expressions.TypeConversion)\
				or isinstance(self.exprs, expressions.UnaryOperation):
			self.inside_exprs_list = [self.exprs.expression]
		else:
			self.inside_exprs_list = []

	def __repr__(self):
		return f"ExpressionNode({self.expression_uniqID})"

	def __hash__(self):
		return hash(self.expression_uniqID)

	def __eq__(self, other):
		return isinstance(other, ExpressionNode) and self.expression_uniqID == other.expression_uniqID

	def is_same_statement(self, other):
		return isinstance(other, ExpressionNode) and self.statement_uniqID == other.statement_uniqID

def CFG_sourcecode_generator_statement(contract_file_path):
	slither = Slither(contract_file_path)

	merge_contract_graph = None
	statementNode_list = []
	for contract in slither.contracts:
		merged_graph = None
		for function in contract.functions + contract.modifiers:

			nx_g = nx.MultiDiGraph()
			for node in function.nodes:
				statementNode = StatementNode(node)
				if statementNode not in statementNode_list:
					statementNode_list.append(statementNode)
					nx_g.add_node(statementNode)

			nx_graph = nx_g
			if merged_graph is None:
				merged_graph = nx_graph.copy()
			else:
				merged_graph = nx.compose(merged_graph, nx_graph)

		if merge_contract_graph is None:
			if merged_graph is not None:
				merge_contract_graph = merged_graph.copy()
			else:
				merge_contract_graph = None
		elif merged_graph is not None:
			merge_contract_graph = nx.compose(merge_contract_graph, merged_graph)

	# At this point, all nodes in the graph have been constructed. Next, construct the edges.
	# Construct edges based on the StatementNode within statementNode
	# Iterate through all StatementNodes
	for statementNode in statementNode_list:
		current_node = statementNode.statementnode
		# Check the edges in statementNode
		if current_node.type in [NodeType.IF, NodeType.IFLOOP]:
			son_true_node = current_node.son_true
			if son_true_node:
				son_true_statementNode_list = [node for node in statementNode_list if node.statementnode == son_true_node]
				assert len(son_true_statementNode_list) == 1
				son_true_statementNode = son_true_statementNode_list[0]
				merge_contract_graph.add_edge(statementNode, son_true_statementNode, edge_type='if_true')
			son_false_node = current_node.son_false
			if son_false_node:
				son_false_statementNode_list = [node for node in statementNode_list if node.statementnode == son_false_node]
				assert len(son_false_statementNode_list) == 1
				son_false_statementNode = son_false_statementNode_list[0]
				merge_contract_graph.add_edge(statementNode, son_false_statementNode, edge_type='if_false')
		else:
			for son_node in current_node.sons:
				if son_node:
					# Find the StatementNode in statementNode_list where statementNode.statementnode == son_node
					son_statementNode_list = [node for node in statementNode_list if node.statementnode == son_node]
					if len(son_statementNode_list) == 0:
						continue
					son_statementNode = son_statementNode_list[0]
					# Construct an edge
					merge_contract_graph.add_edge(statementNode, son_statementNode, edge_type='next')

	return merge_contract_graph


def CFG_sourcecode_generator_expression(contract_file_path):
	slither = Slither(contract_file_path)

	merge_contract_graph = None
	statementNode_list = []
	expressionNode_list = []
	for contract in slither.contracts:
		merged_graph = None
		for function in contract.functions + contract.modifiers:

			nx_g = nx.MultiDiGraph()
			for node in function.nodes:
				statementNode = StatementNode(node)
				if statementNode not in statementNode_list:
					statementNode_list.append(statementNode)
				for expnode in statementNode.set_expnode:
					if expnode not in expressionNode_list:
						expressionNode_list.append(expnode)
						nx_g.add_node(expnode)

			nx_graph = nx_g
			if merged_graph is None:
				merged_graph = nx_graph.copy()
			else:
				merged_graph = nx.compose(merged_graph, nx_graph)

		if merge_contract_graph is None:
			if merged_graph is not None:
				merge_contract_graph = merged_graph.copy()
			else:
				merge_contract_graph = None
		elif merged_graph is not None:
			merge_contract_graph = nx.compose(merge_contract_graph, merged_graph)

	'''
		At this point, all nodes in the graph have been constructed. Next, construct edges in 3 steps:
	'''
	# 1. Iterate through all StatementNodes to find edges between StatementNodes
	for statementNode in statementNode_list:
		current_node = statementNode.statementnode
		# Check the edges in statementNode
		if current_node.type in [NodeType.IF, NodeType.IFLOOP]:
			son_true_node = current_node.son_true
			if son_true_node:
				son_true_statementNode_list = [node for node in statementNode_list if node.statementnode == son_true_node]
				if len(son_true_statementNode_list) == 1:
					son_true_statementNode = son_true_statementNode_list[0]
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_true_statementNode.set_expnode[0], edge_type='if_true')
			son_false_node = current_node.son_false
			if son_false_node:
				son_false_statementNode_list = [node for node in statementNode_list if node.statementnode == son_false_node]
				if len(son_false_statementNode_list) == 1:
					son_false_statementNode = son_false_statementNode_list[0]
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_false_statementNode.set_expnode[0], edge_type='if_false')
		else:
			for son_node in current_node.sons:
				if son_node:
					# Find the StatementNode in statementNode_list where statementNode.statementnode == son_node
					son_statementNode_list = [node for node in statementNode_list if node.statementnode == son_node]
					if len(son_statementNode_list) == 0:
						continue
					son_statementNode = son_statementNode_list[0]
					# Construct an edge
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_statementNode.set_expnode[0], edge_type='next')

	# 2. Iterate through all StatementNodes to find edges within ExpressionNodes inside StatementNode
	for statementNode in statementNode_list:
		# statementNode.set_expnode is a list, and it needs to be connected sequencially
		for i in range(len(statementNode.set_expnode) - 1):
			merge_contract_graph.add_edge(statementNode.set_expnode[i], statementNode.set_expnode[i + 1], edge_type='next')

	# 3. Iterate through all ExpressionNodes to find CallExpression and SuperCallExpression
	#    Use their `called` attribute to find corresponding ExpressionNodes and construct edges
	for expnode_idx, expnode in enumerate(expressionNode_list):
		if expnode.exprs_type in ["CallExpression", "SuperCallExpression"]:
			# If the called function is not an Identifier, skip this call to avoid special cases
			if str(expnode.exprs.called.__class__) != "<class 'slither.core.expressions.identifier.Identifier'>":
				continue

			FunctionCall = expnode.exprs.called.value

			# Find the first ExpressionNode of the function's initial statement and save it in first_expnode_of_func_list
			# This is used for the outgoing edge of the CallNode
			first_expnode_of_func_list = []
			for son_expnode in expressionNode_list:
				if son_expnode.function == FunctionCall and son_expnode.state_id == 0 and son_expnode.contract_uniqID == expnode.contract_uniqID:
					first_node = son_expnode.statementnode.set_expnode[0]
					first_expnode_of_func_list.append(first_node)
			# If the called function has no ExpressionNode, skip this call
			if len(first_expnode_of_func_list) == 0:
				continue
			first_expnode_of_func = first_expnode_of_func_list[0]

			# Find the last ExpressionNode of the function's final statement
			# Save it in last_expnode_of_func_list for incoming edges to subsequent nodes
			last_expnode_of_func_list = []
			statment_max_id = max([node.state_id for node in statementNode_list if node.function == FunctionCall and node.contract_uniqID == expnode.contract_uniqID])
			for son_expnode in expressionNode_list:
				if son_expnode.function == FunctionCall and son_expnode.state_id == statment_max_id and son_expnode.contract_uniqID == expnode.contract_uniqID:
					last_node = son_expnode.statementnode.set_expnode[-1]
					last_expnode_of_func_list.append(last_node)
			last_expnode_of_func_list = list(set(last_expnode_of_func_list))
			last_expnode_of_func = last_expnode_of_func_list[0]

			# Find orig_son_expnode that has an edge with the current expnode, and ensure it's an outgoing edge
			# Then delete these edges
			orig_son_of_expnode_list = []
			edges = list(merge_contract_graph.edges(data=True))
			for edge in edges:
				if edge[0] == expnode:
					son_of_expnode = edge[1]
					attr = edge[2]
					orig_son_of_expnode_list.append(son_of_expnode)
					merge_contract_graph.remove_edge(expnode, son_of_expnode)

			# Reconstruct edges of two types:
			# 1. expnode -> first_expnode_of_func_list, edge_type='call'
			# 2. last_expnode_of_func -> orig_son_of_expnode, edge_type='callback'
			merge_contract_graph.add_edge(expnode, first_expnode_of_func, edge_type='call', last_expnode_of_func=last_expnode_of_func, orig_son_of_expnode_list=orig_son_of_expnode_list)
			for orig_son_of_expnode in orig_son_of_expnode_list:
				merge_contract_graph.add_edge(last_expnode_of_func, orig_son_of_expnode, edge_type='callback', first_expnode_of_func=first_expnode_of_func, expnode=expnode)

	return merge_contract_graph

def draw_ECFG(G):
	# Generate layout for visualization
	pos = nx.spring_layout(G, k=0.7, iterations=50)
	node_names = {
		node:
			node.contract_uniqID + "\n" +
			node.function_ID + "\n" +
			node.statement_ID + "\n" +
			node.expression_ID
		for node in G.nodes
	}

	# Draw nodes with labels
	nx.draw(G, pos, labels=node_names, with_labels=True, node_size=800)
	# Build edge labels
	edge_labels = {(u, v): data['edge_type'] for u, v, data in G.edges(data=True)}
	# Draw edge labels
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

	# Display the graph
	plt.show()


def draw_SCFG(G):
	# Generate layout for visualization
	pos = nx.spring_layout(G, k=0.7, iterations=50)
	node_names = {
		node:
			node.contract_uniqID + "\n" +
			node.function_ID + "\n" +
			node.statement_ID
		for node in G.nodes
	}

	# Draw nodes with labels
	nx.draw(G, pos, labels=node_names, with_labels=True, node_size=800)
	# Build edge labels
	edge_labels = {(u, v): data['edge_type'] for u, v, data in G.edges(data=True)}
	# Draw edge labels
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

	# Display the graph
	plt.show()


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
	# # Visualize SCFG
	# draw_SCFG(full_graph)

	"""ECFG"""
	full_graph = CFG_sourcecode_generator_expression(contract_file)
	# Visualize ECFG
	draw_ECFG(full_graph)

	pass
