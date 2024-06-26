import torch
from torch.nn import Module, Linear, SiLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool

class PAINNMessage(MessagePassing):
   def __init__():
        super(aggr='sum').__init__()
        
        self.act = SiLU()
        
        self.linear_1 = Linear(128, 128)
        self.linear_2 = Linear(20, 384)
        self.linear_3 = Linear(128, 384)
        
    def forward(self, data):
        edge_index = data.edge_index
        x = self.propagate(edge_index, x=data)
        return
        
    def message(self, x_i, x_j, edge_attr):
        s_j = x_j.s_j
        v_j = x_j.v_j
        r_hat_ij = r_ij / torch.norm(r_ij)
        
        s_j = self.linear_1(s_j)
        s_j = self.act(s_j)
        phi = self.linear_3(s_j)
        
        edge_attr = edge_attr
        rbf = rbf(edge_attr) # IMPLEMENT
        W = self.linear_2(edge_attr)
        
        split = phi * W
        split_1 = split[:128]
        split_2 = split[128:256]
        split_3 = split[256:]
        
        split_3 = split_3.view(128, 1) * r_hat_ij
        
        v_j = split_1 * v_j
        v_j = v_j + split_3
        s_j = split_2
        
        return v_j, s_j

    def update(self, aggr_out, x):
        x.v_i = x.v_i + aggr_out[0]
        x.s_i = x.s_i + aggr_out[1]
        
        return x

class PAINNUpdate(MessagePassing):
    def __init__():
        super().__init__()
        
        self.V = Linear(128,128, bias=False)
        self.U = Linear(128,128, bias=False)
        
        self.linear_1 = Linear(256, 128)
        self.linear_2 = Linear(128, 384)

    def forward(self, data):
        edge_index = data.edge_index
        v_i = data.v_i
        s_i = data.s_i
        
        v_i, s_i = self.propogate(edge_index, x=x_i)
        
    def message(self, x_i, x_j, edge_attr):
        
    def update()
    
class PAINNBlock():
    def __init__():
        super().__init__()
        
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
     