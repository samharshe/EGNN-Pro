from typing import Any
import torch
from torch_geometric.datasets import MD17
import torch_geometric.transforms as T
from torch_geometric.transforms import RadiusGraph, NormalizeScale
import os

# minumum energy in benzene, uracil, and aspirin datasets is -406757.5938
# we divide by 10 because I think mapping to (0,1) is causing weird flop errors
mod_max_abs_energy = -406757.5938 / 10

class NormalizeForce(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.force = torch.div(data.force, mod_max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class NormalizeEnergy(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.energy = torch.div(data.energy, mod_max_abs_energy)
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class DoubleDistance(T.BaseTransform):
    def __call__(self, data: Any) -> Any:
        data.pos = data.pos*2
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

transform = T.Compose([RadiusGraph(1.8100), NormalizeScale(), DoubleDistance(), NormalizeEnergy(), NormalizeForce()])

benzene_dataset = MD17(root='data/drip/', name='benzene', pre_transform=transform)
uracil_dataset = MD17(root='data/drip/', name='uracil', pre_transform=transform)
aspirin_dataset = MD17(root='data/drip/', name='aspirin', pre_transform=transform)