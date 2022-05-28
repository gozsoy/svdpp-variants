import torch
import torch.nn as nn


class RegularizedSVD(nn.Module):
	def __init__(self, num_users, num_items, global_mean, embedding_dim):
		super(RegularizedSVD, self).__init__()
		self.gm = global_mean

		self.P = nn.Embedding(num_users, embedding_dim)
		self.Q = nn.Embedding(num_items, embedding_dim)
		self.B_U = nn.Embedding(num_users, 1)
		self.B_I = nn.Embedding(num_items, 1)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def forward(self, x):

		# user and item indices start with 1 in dataset,embedding index starts with 0
		# so embedding of user 1 is stored at self.p(0)
		user_id, item_id = x[0]-1, x[1]-1
		user_id, item_id = user_id.to(self.device), item_id.to(self.device)

		p_u = self.P(user_id)
		q_i = self.Q(item_id)
		b_u = self.B_U(user_id)
		b_i = self.B_I(item_id)

		pred_r_ui = torch.sum(p_u * q_i, axis=1) + \
			torch.squeeze(b_u) + torch.squeeze(b_i) + self.gm

		return pred_r_ui


class SVDPP(nn.Module):
	def __init__(self, num_users, num_items, global_mean, embedding_dim):
		super(SVDPP, self).__init__()
		self.gm = global_mean

		# +1 because of padding
		self.P = nn.Embedding(num_users+1, embedding_dim)
		self.Q = nn.Embedding(num_items+1, embedding_dim)
		self.B_U = nn.Embedding(num_users+1, 1)
		self.B_I = nn.Embedding(num_items+1, 1)
		self.Y = nn.Embedding(num_items+1, embedding_dim, padding_idx=0)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def forward(self, x):

		# user and item indices start with 1 in dataset,embedding index starts with 1
		user_id, item_id = x[0], x[1]
		rated_items, rated_counts = x[2], x[3]
		user_id, item_id = user_id.to(self.device), item_id.to(self.device)
		rated_items = rated_items.to(self.device), 
		rated_counts = rated_counts.to(self.device)

		p_u = self.P(user_id)
		q_i = self.Q(item_id)
		b_u = self.B_U(user_id)
		b_i = self.B_I(item_id)

		y_j_sum = torch.sum(self.Y(rated_items), dim=1)
		y_j_sum = torch.mul(y_j_sum, torch.unsqueeze(
			torch.div(1, torch.sqrt(rated_counts)), dim=1))

		pred_r_ui = torch.sum((p_u+y_j_sum) * q_i, axis=1) + \
			torch.squeeze(b_u) + torch.squeeze(b_i) + self.gm

		return pred_r_ui

