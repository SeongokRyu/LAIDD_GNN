import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from libs.layers import GraphConvolution
from libs.layers import GraphIsomorphism
from libs.layers import GraphIsomorphismEdge
from libs.layers import GraphAttention


class MyModel(nn.Module):
	def __init__(
			self, 
			model_type,
			num_layers=4, 
			hidden_dim=64,
			num_heads=4, # Only used for GAT
			dropout_prob=0.2,
			bias_mlp=True,
			readout='sum',
			act=F.relu,
			initial_node_dim=58,
			initial_edge_dim=6,
			is_classification=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.embedding_node = nn.Linear(initial_node_dim, hidden_dim, bias=False)
		self.embedding_edge = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
		self.readout = readout

		self.mp_layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			mp_layer = None
			if model_type == 'gcn':
				mp_layer = GraphConvolution(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
				)
			elif model_type == 'gin':
				mp_layer = GraphIsomorphism(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gin_e':
				mp_layer = GraphIsomorphismEdge(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gat':
				mp_layer = GraphAttention(
					hidden_dim=hidden_dim,
					num_heads=num_heads,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			else:
				raise ValueError('Invalid model type: you should choose model type in [gcn, gin, gin_e, gat, ggnn]')
			self.mp_layers.append(mp_layer)

		self.linear_out = nn.Linear(hidden_dim, 1, bias=False)

		self.is_classification = is_classification
		if self.is_classification:
			self.sigmoid = F.sigmoid
	

	def forward(
			self, 
			graph,
			training=False,
		):
		h = self.embedding_node(graph.ndata['h'].float())
		e_ij = self.embedding_edge(graph.edata['e_ij'].float())
		graph.ndata['h'] = h
		graph.edata['e_ij'] = e_ij

		# Update the node features
		for i in range(self.num_layers):
			graph = self.mp_layers[i](
				graph=graph,
				training=training
			)

		# Aggregate the node features and apply the last linear layer to compute the logit
		out = dgl.readout_nodes(graph, 'h', op=self.readout)
		out = self.linear_out(out)

		if self.is_classification:
			out = self.sigmoid(out)
		return out
