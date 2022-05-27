import torch
import torch.nn as nn

class RegularizedSVD(nn.Module):
  def __init__(self, num_users, num_items, global_mean, embedding_dim):
    super().__init__()
    self.gm = global_mean

    # cfg should handle these num_users, num_movies, embedding_dim
    self.P = nn.Embedding(num_users, embedding_dim)
    self.Q = nn.Embedding(num_items, embedding_dim)
    self.B_U = nn.Embedding(num_users, 1)
    self.B_I = nn.Embedding(num_items, 1)

    self.device = next(self.P.parameters()).device

  def forward(self, x):
  
    # user and item indices start with 1 in dataset, embedding index starts with 0 
    # so embedding of user 1 is stored at self.p(0)
    user_id, item_id = x[0]-1, x[1]-1
    user_id, item_id = user_id.to(self.device), item_id.to(self.device)

    p_u = self.P(user_id)
    q_i = self.Q(item_id)
    b_u = self.B_U(user_id)
    b_i = self.B_I(item_id)

    pred_r_ui = torch.sum(p_u * q_i, axis=1) + torch.squeeze(b_u) + torch.squeeze(b_i) + self.gm

    return pred_r_ui
