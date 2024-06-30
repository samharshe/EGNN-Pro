import torch
from torch.nn import Embedding, Linear, SiLU
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from .utils.model_utils import gaussian_rbf

class Gamma(MessagePassing):
    """fka EGNN3. 32-dimensional embedding with SiLU activation and one round of message passing.
    """
    def __init__(self):
        super().__init__()
        
        # activation function
        self.act = SiLU()
        
        # initialize layers
        # 118 atomic numbers into 32-dimensional space
        self.embedding = Embedding(118,32)
        
        # 32 dimensions for the embedding of the neighbor
        # 8 for the embedding of the distance
        self.message_lin = Linear(32 + 8, 32)
        
        # 32 dimensions for the current node embedding
        # 32 for the message
        self.update_lin = Linear(32 + 32, 32)
        
        # compress the 32-dimensional node embedding to 1 dimension
        self.compress_lin = Linear(32, 1)
        
    def forward(self, data):
        # get attributes out of data object
        edge_index = data.edge_index
        z = data.z
        pos = data.pos
        
        # force is negative gradient of energy with respect to position, so pos must be on computational graph
        pos.requires_grad_(True)
        
        # calculate edge distances and turn them into a vector through Gaussian RBF
        idx1, idx2 = edge_index
        edge_attr = torch.norm(pos[idx1] - pos[idx2], p=2, dim=-1).view(-1, 1)
        gaussian_edge_attr = gaussian_rbf(edge_attr)
        
        # forward pass proper
        E_hat = self.embedding(z)
        E_hat = self.act(E_hat)
        E_hat = self.propagate(edge_index, x=E_hat, edge_attr=gaussian_edge_attr)
        E_hat = self.act(E_hat)
        E_hat = self.compress_lin(E_hat)
        E_hat = self.act(E_hat)
        E_hat = global_add_pool(E_hat, data.batch)
        
        # calculate energy prediction as negative gradient of energy with respect to position, retaining computational graph for backprop
        F_hat = -torch.autograd.grad(E_hat.sum(), pos, retain_graph=True)[0]
        
        # return a tuple of the predictions
        return E_hat, F_hat
    
    def message(self, x_j, edge_attr):
        # concatenate vectors
        lin_in = torch.cat((x_j, edge_attr), dim=1).float()
        
        # pass into linear layer
        out = self.message_lin(lin_in)
        
        # return output
        return out
    
    def update(self, aggr_out, x):
        # concatenate vectors
        lin_in = torch.cat((aggr_out, x), dim=1).float()
        
        # pass into linear layer
        out = self.update_lin(lin_in)
        
        # return the output
        return out