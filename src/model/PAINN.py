import torch
from torch.nn import Module, Linear, SiLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool

class PAINNBlock(MessagePassing):
   def __init__():
        super().__init__()
        
        self.act = SiLU()
        
        self.message_linear_1 = Linear(128, 128)
        self.message_linear_2 = Linear(20, 384)
        self.message_linear_3 = Linear(128, 384)
        
        self.update_linear_1 = Linear(256, 128)
        self.update_linear_2 = Linear(128, 384)
        
        self.update_U = Linear()
        
    def forward(self, data):
        z = data.z
        pos = data.pos
        pos.requires_grad_(True)
        
        idx1, idx2 = data.edge_index
        edge_attr = torch.norm(pos[idx1] - pos[idx2], p=2, dim=-1).view(-1,1)
        
    def message(self, x_i, x_j, edge_attr):
        
    def update(self, aggr_out, x):
        
class PAINNPrediction(Module):
    def __init__():
        super().__init__()
        
        self.act = SiLU()
        
        self.linear_1 = Linear(128, 128)
        self.linear_2 = Linear(128, 128)
    
    def forward(self, data):
        x = self.linear_1(data)
        x = self.act(x)
        x = self.linear_2(data)
        x = global_add_pool(x, data.batch)
        
        return x
        
    
class PAINN(MessagePassing):
    def __init__():
        super().__init__()
        
        self.block_1 = PAINNBlock()
        self.block_2 = PAINNBlock()
        self.block_3 = PAINNBlock()
    
        self.prediction = PAINNPrediction()
        
     def forward(self, data)
     