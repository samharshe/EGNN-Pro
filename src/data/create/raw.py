from torch_geometric.datasets import MD17

benzene_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/raw/', name='benzene')
uracil_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/raw/', name='uracil')
aspirin_dataset = MD17(root='/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/raw/', name='aspirin')