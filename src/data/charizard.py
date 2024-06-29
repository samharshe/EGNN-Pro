from typing import Any
import torch
from torch_geometric.datasets import MD17
import torch_geometric.transforms as T
from torch_geometric.transforms import RadiusGraph, NormalizeScale

# minumum energy in benzene, uracil, and aspirin datasets is -406757.5938
max_abs_energy = -406757.5938

class NormalizeForce(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.force = torch.div(data.force, max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class NormalizeEnergy(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.energy = torch.div(data.energy, max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class DoubleDistance(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.pos = data.pos*2
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
    
class MakeEdgeAttributes(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        idx1 = data.edge_index[0]
        idx2 = data.edge_index[1]
        data.edge_vec = data.pos[idx1] - data.pos[idx2]
        data.edge_vec_length = torch.norm(data.edge_vec, dim=1)
        data.unit_edge_vec = torch.div(data.edge_vec, data.edge_vec_length.view(-1,1))
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

transform = T.Compose([RadiusGraph(1.8100), NormalizeScale(), DoubleDistance(), NormalizeEnergy(), NormalizeForce(), MakeEdgeAttributes()])

benzene_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/charizard/', name='benzene', pre_transform=transform)
uracil_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/charizard/', name='uracil', pre_transform=transform)
aspirin_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/charizard/', name='aspirin', pre_transform=transform)