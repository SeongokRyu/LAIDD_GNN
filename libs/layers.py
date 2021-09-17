import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class MLP(nn.Module):
	def __init__(
		self, 
		input_dim, 
		hidden_dim, 
		output_dim,
		bias=True,
		act=F.relu,
	):
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.act = act

		self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
		self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)
	
	def forward(self, h):
		h = self.linear1(h)
		h = self.act(h)
		h = self.linear2(h)
		return h


class GraphConvolution(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			dropout_prob=0.2,
		):
		super().__init__()

		self.act = act
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.act(self.linear(graph.ndata['u_'])) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphism(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act
		)
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.mlp(graph.ndata['u_']) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphismEdge(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
		graph.update_all(fn.copy_edge('e_ij', 'm_e'), fn.sum('m_e', 'u_'))
		u_ = graph.ndata['neigh'] + graph.ndata['u_']
		h = self.mlp(u_) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphAttention(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_heads=4,
			bias_mlp=True,
			dropout_prob=0.2,
			act=F.relu,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=2*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.splitted_dim = hidden_dim // num_heads

		self.prob = dropout_prob

		self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

		self.act = F.elu
		self.norm = nn.LayerNorm(hidden_dim)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']
		e_ij = graph.edata['e_ij']

		graph.ndata['u'] = self.w1(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.ndata['v'] = self.w2(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w3(e_ij).view(-1, self.num_heads, self.splitted_dim)

		graph.apply_edges(fn.v_add_e('v', 'x_ij', 'm'))
		graph.apply_edges(fn.u_mul_e('u', 'm', 'attn'))
		graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.splitted_dim))
	

		graph.ndata['k'] = self.w4(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w5(e_ij).view(-1, self.num_heads, self.splitted_dim)
		graph.apply_edges(fn.v_add_e('k', 'x_ij', 'm'))

		graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
		graph.update_all(fn.copy_edge('m', 'm'), fn.sum('m', 'h'))
		
		h = self.w6(h0) + graph.ndata['h'].view(-1, self.hidden_dim)
		h = self.norm(h)

		# Add and Norm module
		h = h + self.mlp(h)
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h 
		return graph
