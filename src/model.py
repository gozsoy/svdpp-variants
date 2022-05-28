import torch
import torch.nn as nn
import torch.nn.functional as F


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
		rated_items = rated_items.to(self.device)
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


class Bayesian_SVDPP(nn.Module):

	def __init__(self, num_users, num_items, global_mean, embedding_dim):
		super(Bayesian_SVDPP, self).__init__()
		self.gm = global_mean
		self.embedding_dim = embedding_dim

		self.P_mu = nn.Embedding(num_users+1, embedding_dim)
		nn.init.uniform_(self.P_mu.weight, -0.6, 0.6)
		self.P_rho = nn.Embedding(num_users+1, embedding_dim)
		nn.init.constant_(self.P_rho.weight, -3.)

		self.Q_mu = nn.Embedding(num_items+1, embedding_dim)
		nn.init.uniform_(self.Q_mu.weight, -0.6, 0.6)
		self.Q_rho = nn.Embedding(num_items+1, embedding_dim)
		nn.init.constant_(self.Q_rho.weight, -3.)

		self.B_U_mu = nn.Embedding(num_users+1, 1)
		nn.init.uniform_(self.B_U_mu.weight, -0.6, 0.6)
		self.B_U_rho = nn.Embedding(num_users+1, 1)
		nn.init.constant_(self.B_U_rho.weight, -3.)

		self.B_I_mu = nn.Embedding(num_items+1, 1)
		nn.init.uniform_(self.B_I_mu.weight, -0.6, 0.6)
		self.B_I_rho = nn.Embedding(num_items+1, 1)
		nn.init.constant_(self.B_I_rho.weight, -3.)

		self.Y_mu = nn.Embedding(num_items+1, embedding_dim, padding_idx=0)
		nn.init.uniform_(self.Y_mu.weight, -0.6, 0.6)
		self.Y_rho = nn.Embedding(num_items+1, embedding_dim, padding_idx=0)
		nn.init.constant_(self.Y_rho.weight, -3.)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def forward(self, x):

		# user and item indices start with 1 in dataset,embedding index starts with 1
		user_id, item_id = x[0], x[1]
		rated_items, rated_counts = x[2], x[3]
		user_id, item_id = user_id.to(self.device), item_id.to(self.device)
		rated_items = rated_items.to(self.device)
		rated_counts = rated_counts.to(self.device)

		p_u = self.P_mu(user_id) + F.softplus(self.P_rho(user_id)) * \
			torch.normal(mean=0.0, std=1.0, 
						 size=(len(user_id), self.embedding_dim)).to(self.device)
		q_i = self.Q_mu(item_id) + F.softplus(self.Q_rho(item_id)) * \
			torch.normal(mean=0.0, std=1.0, 
						 size=(len(user_id), self.embedding_dim)).to(self.device)
		b_u = self.B_U_mu(user_id) + F.softplus(self.B_U_rho(user_id)) * \
			torch.normal(mean=0.0, std=1.0, size=(len(user_id), 1)).to(self.device)
		b_i = self.B_I_mu(item_id) + F.softplus(self.B_I_rho(item_id)) * \
			torch.normal(mean=0.0, std=1.0, size=(len(user_id), 1)).to(self.device)
		y_j = self.Y_mu(rated_items) + F.softplus(self.Y_rho(rated_items)) * \
			torch.normal(
				mean=0.0, std=1.0, size=(len(user_id), 
										 rated_items.shape[1],
										 self.embedding_dim)).to(self.device)
			
		y_j_sum = torch.sum(y_j, dim=1)
		y_j_sum = torch.mul(y_j_sum, torch.unsqueeze(
			torch.div(1, torch.sqrt(rated_counts)), dim=1))

		pred_r_ui = torch.sum((p_u+y_j_sum) * q_i, axis=1) + \
			torch.squeeze(b_u) + torch.squeeze(b_i) + self.gm

		return pred_r_ui

	def compute_total_kl_loss(self):
		param_tuples= [(self.P_mu, self.P_rho), (self.Q_mu, self.Q_rho),
					(self.B_U_mu, self.B_U_rho), (self.B_I_mu, self.B_I_rho), 
					(self.Y_mu, self.Y_rho)]

		kl_loss_list = list(map(lambda mu_rho: self.compute_layer_kl_loss(mu_rho[0],mu_rho[1]), param_tuples))

		return torch.sum(torch.stack(kl_loss_list))

	# total kl loss for the weights in this layer
	def compute_layer_kl_loss(self, X_mu,X_rho):
		layer_kl_loss = torch.sum(self._kl_loss(X_mu.weight,X_rho.weight))

		return layer_kl_loss

  # kl loss between one weight's posterior and unit Gaussian prior (closed form complexity cost)
	def _kl_loss(self,temp_mu, temp_rho):
		sigma_squared = F.softplus(temp_rho) ** 2

		return -0.5 * (1 + torch.log(sigma_squared) - temp_mu ** 2 - sigma_squared)