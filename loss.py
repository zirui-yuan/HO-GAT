from networkx.algorithms.traversal.edgebfs import FORWARD
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class HOGATloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Ls = nn.BCELoss()
        self.La = nn.MSELoss()
        
        
        


    def forward(self, rec_adj, rec_features, features, adj):
        rec_adj_flatten = torch.flatten(rec_adj)
        adj_flatten = torch.flatten(adj)
        Ls = self.Ls(rec_adj_flatten, adj_flatten)
        La = self.La(rec_features, features)
        loss = Ls + La
        return loss
        

