from torch_geometric.datasets import MD17

benzene_dataset = MD17(root='data/apricot/', name='benzene')
uracil_dataset = MD17(root='data/apricot/', name='uracil')
aspirin_dataset = MD17(root='data/apricot/', name='aspirin')