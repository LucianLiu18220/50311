import re
import subprocess
import os
from slither.core.cfg.node import NodeType
from slither.core.declarations.function import FunctionType
from slither.core import expressions
from Step1_CFGgeneration.SCFG_ECFG import CFG_sourcecode_generator_expression, draw_ECFG
from Step1_CFGgeneration.SCFG_ECFG import CFG_sourcecode_generator_statement, draw_SCFG


def set_compiler(version):
	command = f'solc-select use {version}'
	subprocess.run(command, stdout=subprocess.DEVNULL, shell=True, check=True)


def get_installed_versions():
	# Call the command line to get installed versions
	result = subprocess.run(["solc-select", "versions"], capture_output=True, text=True)
	installed_versions = re.findall(r'\d+\.\d+\.\d+', result.stdout)
	installed_versions.sort()
	return installed_versions


Funct_Type = [
	"NORMAL",
	"CONSTRUCTOR",
	"FALLBACK",
	"RECEIVE",
	"CONSTRUCTOR_VARIABLES",  # Fake function to hold variable declaration statements
	"CONSTRUCTOR_CONSTANT_VARIABLES",
]

State_Type = [
	"ENTRY_POINT",  # no expression

	# Nodes that may have an expression
	"EXPRESSION",  # normal case
	"RETURN",  # RETURN may contain an expression
	"IF",
	"NEW VARIABLE",  # Variable declaration
	"INLINE ASM",
	"IF_LOOP",

	# Nodes where control flow merges
	# Can have phi IR operation
	"END_IF",  # ENDIF node source mapping points to the if/else "body"
	"BEGIN_LOOP",  # STARTLOOP node source mapping points to the entire loop "body"
	"END_LOOP",  # ENDLOOP node source mapping points to the entire loop "body"

	# Below the nodes do not have an expression but are used to express CFG structure.

	# Absorbing node
	"THROW",

	# Loop related nodes
	"BREAK",
	"CONTINUE",

	# Only modifier node
	"_",

	"TRY",
	"CATCH",

	# Node not related to the CFG
	# Use for state variable declaration
	"OTHER_ENTRYPOINT"
]

Exprs_Type = [
	"AssignmentOperation",
	"BinaryOperation",
	"CallExpression",
	"ConditionalExpression",
	"ElementaryTypeNameExpression",
	"Identifier",
	"IndexAccess",
	"Literal",
	"MemberAccess",
	"NewArray",
	"NewContract",
	"NewElementaryType",
	"SuperCallExpression",
	"SuperIdentifier",
	"TupleExpression",
	"TypeConversion",
	"UnaryOperation",
	'NoneType'
]

AssignmentOperationType = [
	"ASSIGN",  # = 0  # =
	"ASSIGN_OR",  # = 1  # |=
	"ASSIGN_CARET",  # = 2  # ^=
	"ASSIGN_AND",  # = 3  # &=
	"ASSIGN_LEFT_SHIFT",  # = 4  # <<=
	"ASSIGN_RIGHT_SHIFT",  # = 5  # >>=
	"ASSIGN_ADDITION",  # = 6  # +=
	"ASSIGN_SUBTRACTION",  # = 7  # -=
	"ASSIGN_MULTIPLICATION",  # = 8  # *=
	"ASSIGN_DIVISION",  # = 9  # /=
	"ASSIGN_MODULO"  # = 10  # %=
]

BinaryOperationType = [
	"POWER",  # = 0  # **
	"MULTIPLICATION",  # = 1  # *
	"DIVISION",  # = 2  # /
	"MODULO",  # = 3  # %
	"ADDITION",  # = 4  # +
	"SUBTRACTION",  # = 5  # -
	"LEFT_SHIFT",  # = 6  # <<
	"RIGHT_SHIFT",  # = 7  # >>>
	"AND",  # = 8  # &
	"CARET",  # = 9  # ^
	"OR",  # = 10  # |
	"LESS",  # = 11  # <
	"GREATER",  # = 12  # >
	"LESS_EQUAL",  # = 13  # <=
	"GREATER_EQUAL",  # = 14  # >=
	"EQUAL",  # = 15  # ==
	"NOT_EQUAL",  # = 16  # !=
	"ANDAND",  # = 17  # &&
	"OROR"  # = 18  # ||
]

UnaryOperationType = [
	"BANG",  # = 0  # !
	"TILD",  # = 1  # ~
	"DELETE",  # = 2  # delete
	"PLUSPLUS_PRE",  # = 3  # ++
	"MINUSMINUS_PRE",  # = 4  # --
	"PLUSPLUS_POST",  # = 5  # ++
	"MINUSMINUS_POST",  # = 6  # --
	"PLUS_PRE",  # = 7  # for stuff like uint(+1)
	"MINUS_PRE"  # = 8  # for stuff like uint(-1)
]


class normECFG:
	def __init__(self, origGraph):
		self.nxGraph = origGraph
		self.nxGnodes = list(origGraph.nodes)
		self.nxGedges = list(origGraph.edges(data=True))

		self.Funct_Type = Funct_Type
		self.State_Type = State_Type
		self.Exprs_Type = Exprs_Type

		self.node_dict = {}
		self.node_generate()

		self.edge_dict = {}
		self.edge_generate()

		self.Node_match_Edge()

	def node_generate(self):
		for idx, node in enumerate(self.nxGnodes):
			node_feat = {}
			node_feat["idx"] = idx
			node_feat["name"] = str(node)

			node_feat["Cont"] = {}
			node_feat["Cont"]["str"] = str(node.contract)
			node_feat["Cont"]["ID"] = node.contract_uniqID

			node_feat["Func"] = {}
			node_feat["Func"]["str"] = str(node.function)
			node_feat["Func"]["ID"] = node.function_ID
			node_feat["Func"]["Funct_Type"] = node.function_type.name
			node_feat["Func"]["Funct_Type_index"] = self.Funct_Type.index(node_feat["Func"]["Funct_Type"])

			node_feat["Stat"] = {}
			node_feat["Stat"]["str"] = node.state_str
			node_feat["Stat"]["state_sequenceNumb"] = node.state_id
			node_feat["Stat"]["State_Type"] = node.statementnode.statementnode.type.value
			node_feat["Stat"]["State_Type_index"] = self.State_Type.index(node_feat["Stat"]["State_Type"])

			node_feat["Expr"] = {}
			node_feat["Expr"]["str"] = node.exprs_str
			node_feat["Expr"]["Exprs_Type"] = node.exprs_type
			node_feat["Expr"]["Exprs_Type_index"] = self.Exprs_Type.index(node_feat["Expr"]["Exprs_Type"])
			node_feat["Expr"]["Exprs_attr"] = node.exprs_attr

			self.node_dict[idx] = node_feat

	def edge_generate(self):
		# Replace nodes in self.nxGedges with their indices from node_dict and store the result in self.edge_list
		edges = self.nxGedges
		for index_edge, edge in enumerate(edges):
			edge_feat = edge[2]
			# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[0]
			outNode_idx = -1
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[0]):
					outNode_idx = node["idx"]
					break
			inNode_idx = -1
			# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[1]
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[1]):
					inNode_idx = node["idx"]
					break

			if edge_feat["edge_type"] == "call":
				# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[2]["last_expnode_of_func"]
				expnode = edge[2]["last_expnode_of_func"]
				last_expnode_of_func = None
				for idx, node in self.node_dict.items():
					if node["name"] == str(expnode):
						last_expnode_of_func = node["idx"]
						break
				# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[2]["orig_son_of_expnode_list"]
				orig_son_of_expnode_list = []
				for expnode in edge[2]["orig_son_of_expnode_list"]:
					orig_son_of_expnode = None
					for idx, node in self.node_dict.items():
						if node["name"] == str(expnode):
							orig_son_of_expnode = node["idx"]
							break
					orig_son_of_expnode_list.append(orig_son_of_expnode)

				edge_feat["last_expnode_of_func"] = last_expnode_of_func
				edge_feat["orig_son_of_expnode_list"] = orig_son_of_expnode_list
			elif edge_feat["edge_type"] == "callback":
				# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[2]["first_expnode_of_func"]
				first_expnode_of_func = -1
				for idx, node in self.node_dict.items():
					if node["name"] == str(edge[2]["first_expnode_of_func"]):
						first_expnode_of_func = node["idx"]
						break
				# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[2]["expnode"]
				expnode = -1
				for idx, node in self.node_dict.items():
					if node["name"] == str(edge[2]["expnode"]):
						expnode = node["idx"]
						break

				edge_feat["first_expnode_of_func"] = first_expnode_of_func
				edge_feat["expnode"] = expnode

			New_edge = (outNode_idx, inNode_idx, edge_feat)
			self.edge_dict[index_edge] = New_edge

		# Bind the call edges and callback edges in self.edge_list
		for index_edge in self.edge_dict.keys():
			edge = self.edge_dict[index_edge]
			outNode_idx = edge[0]
			inNode_idx = edge[1]
			edge_feat = edge[2]
			edge_type = edge_feat["edge_type"]

			if edge_type != "call":
				continue

			callback_outNode_idx = edge_feat["last_expnode_of_func"]
			callback_inNode_idx_list = edge_feat["orig_son_of_expnode_list"]

			callback_edge_index_list = []
			for callback_inNode_idx in callback_inNode_idx_list:
				for idx in self.edge_dict.keys():
					edge_callback = self.edge_dict[idx]
					if edge_callback[0] == callback_outNode_idx and edge_callback[1] == callback_inNode_idx and edge_callback[2]["edge_type"] == "callback":
						callback_edge_index_list.append(idx)
						self.edge_dict[idx][2]["call_edge_index"] = index_edge
						# break

			self.edge_dict[index_edge][2]["callback_edge_index_list"] = callback_edge_index_list

			pass

	def Node_match_Edge(self):
		# Iterate over all nodes, find all edges starting from each node, and append these edges to the node's dictionary
		for idx in range(len(self.node_dict)):
			edge_list = []
			self.node_dict[idx]["edge_list"] = edge_list

		for edge in self.edge_dict.values():
			node_idx = edge[0]
			self.node_dict[node_idx]["edge_list"].append(edge)

		# # Print the edge status for each node
		# for node_idx in range(len(self.node_dict)):
		#     edge_list = self.node_dict[node_idx]["edge_list"]
		#     print("Node: ", node_idx)
		#     for edge in edge_list:
		#         print("\tIn: {}, Out: {}, type: {}".format(edge[0], edge[1], edge[2]["edge_type"]))


class normSCFG:
	def __init__(self, origGraph):
		self.nxGraph = origGraph
		self.nxGnodes = list(origGraph.nodes)
		self.nxGedges = list(origGraph.edges(data=True))

		self.Funct_Type = Funct_Type
		self.State_Type = State_Type
		self.Exprs_Type = Exprs_Type

		self.node_dict = {}
		self.node_generate()

		self.edge_dict = {}
		self.edge_generate()

		self.Node_match_Edge()

	def node_generate(self):
		for idx, node in enumerate(self.nxGnodes):
			node_feat = {}
			node_feat["idx"] = idx
			node_feat["name"] = str(node)

			node_feat["Cont"] = {}
			node_feat["Cont"]["str"] = str(node.contract)
			node_feat["Cont"]["ID"] = node.contract_uniqID

			node_feat["Func"] = {}
			node_feat["Func"]["str"] = str(node.function)
			node_feat["Func"]["ID"] = node.function_ID
			node_feat["Func"]["Funct_Type"] = node.function_type.name
			node_feat["Func"]["Funct_Type_index"] = self.Funct_Type.index(node_feat["Func"]["Funct_Type"])

			node_feat["Stat"] = {}
			node_feat["Stat"]["str"] = node.state_str
			node_feat["Stat"]["state_sequenceNumb"] = node.state_id
			node_feat["Stat"]["State_Type"] = node.statementnode.type.value
			node_feat["Stat"]["State_Type_index"] = self.State_Type.index(node_feat["Stat"]["State_Type"])

			self.node_dict[idx] = node_feat

	def edge_generate(self):
		# Replace nodes in self.nxGedges with their indices from node_dict and store the result in self.edge_list
		edges = self.nxGedges
		for index_edge, edge in enumerate(edges):
			edge_feat = edge[2]
			# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[0]
			outNode_idx = -1
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[0]):
					outNode_idx = node["idx"]
					break
			inNode_idx = -1
			# Find the idx in self.node_dict where self.node_dict[idx]["ID"] == edge[1]
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[1]):
					inNode_idx = node["idx"]
					break

			New_edge = (outNode_idx, inNode_idx, edge_feat)
			self.edge_dict[index_edge] = New_edge

	def Node_match_Edge(self):
		# Iterate over all nodes, find all edges starting from each node, and append these edges to the node's dictionary
		for idx in range(len(self.node_dict)):
			edge_list = []
			self.node_dict[idx]["edge_list"] = edge_list

		for edge in self.edge_dict.values():
			node_idx = edge[0]
			self.node_dict[node_idx]["edge_list"].append(edge)

		# # Print the edge status for each node
		# for node_idx in range(len(self.node_dict)):
		#     edge_list = self.node_dict[node_idx]["edge_list"]
		#     print("Node: ", node_idx)
		#     for edge in edge_list:
		#         print("\tIn: {}, Out: {}, type: {}".format(edge[0], edge[1], edge[2]["edge_type"]))


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

	"""ECFG"""
	full_graph = CFG_sourcecode_generator_expression(contract_file)
	nG = normECFG(full_graph)

	pass
