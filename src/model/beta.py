import torch, sys, os
from torch.nn import Embedding, Linear
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from model_utils import gaussian_rbf

class Model(MessagePassing):
    """fka EGNN2. 32-dimensional embedding with one round of message passing.
    """
    def __init__(self):
        super().__init__()
        
        # initialize layers
        self.embedding = Embedding(118,32)
        self.message_lin = Linear(32 + 8, 32)
        self.update_lin = Linear(32 + 32, 32)
        self.compress_lin = Linear(32, 1)
        
    def forward(self, data):
        # get relevent parts from data arg
        edge_index = data.edge_index
        z = data.z
        pos = data.pos
        pos.requires_grad_(True)
        
        # make edge vector 
        idx1, idx2 = edge_index
        edge_distance = torch.norm(pos[idx1] - pos[idx2], p=2, dim=-1).view(-1, 1)
        gaussian_edge_attr = gaussian_rbf(x=edge_distance)
        
        # embedding, message passing, aggregation
        E_hat = self.embedding(z)
        E_hat = self.propagate(edge_index, x=E_hat, edge_attr=gaussian_edge_attr)
        E_hat = self.compress_lin(E_hat)
        E_hat = global_add_pool(E_hat, data.batch)
        
        # calculate force on each atom, which is negative gradient of atom's position
        F_hat = -torch.autograd.grad(E_hat.sum(), pos, retain_graph=True)[0]
        
        # return predictions
        return E_hat, F_hat
    
    def message(self, x_j, edge_attr):
        # input tensor consists of neighbor node embedding and edge tensor
        lin_in = torch.cat((x_j, edge_attr), dim=1).float()
        
        # put through learnable linear layer
        out = self.message_lin(lin_in)
        
        # return message
        return out
    
    def update(self, aggr_out, x):
        # input tensor consists of aggregated message tensor and current embedding
        lin_in = torch.cat((aggr_out, x), dim=1).float()
        
        # put through learnable linear layer
        out = self.update_lin(lin_in)
        
        # return new embedding
        return out