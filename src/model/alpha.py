import torch
from torch_geometric.nn import GCNConv
from torch.nn import Module, Embedding, Linear, LeakyReLU
from torch_geometric.nn import global_add_pool

class Alpha(Module):
    """fka ToyGCN. 16-dimensional embedding with two GCN conv layers.
    """
    def __init__(self):
        super().__init__()
        
        # initialize layers
        self.embedding = Embedding(118, 16)
        self.conv1 = GCNConv(16, 16)
        self.lin1 = Linear(16, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin2 = Linear(16, 4)
        self.lin3 = Linear(4, 1)
        self.non_linearity = LeakyReLU()

    # forward pass
    def forward(self, data):
        # get relevant parts from data arg
        edge_index = data.edge_index        
        pos = data.pos
        pos.requires_grad_(True)
        
        # distances between nodes
        edge_attr = torch.sqrt(torch.sum(torch.square(pos[edge_index[0,:]] - pos[edge_index[1,:]]),dim=1))
        
        # initialize E_hat
        E_hat = data.z

        # embed E_hat
        E_hat = self.embedding(E_hat)
        
        # conv layer 1
        E_hat = self.conv1(E_hat, edge_index, edge_attr)
        E_hat = self.non_linearity(E_hat)
        
        # linear layer 1
        E_hat = self.lin1(E_hat)
        E_hat = self.non_linearity(E_hat)
        
        # conv layer 2
        E_hat = self.conv2(E_hat, edge_index, edge_attr)
        E_hat = self.non_linearity(E_hat)
        
        # linear layer 2
        E_hat = self.lin2(E_hat)
        E_hat = self.non_linearity(E_hat)
        
        # linear layer 3: compression
        E_hat = self.lin3(E_hat)
        E_hat = self.non_linearity(E_hat)
        
        # combine representations of all nodes
        # into single graph-level prediction
        E_hat = global_add_pool(E_hat, data.batch)
        
        # calculate force on each atom, which is negative gradient of atom's position
        F_hat = -torch.autograd.grad(E_hat.sum(), pos, retain_graph=True)[0]
        
        # return predictions
        return E_hat, F_hat