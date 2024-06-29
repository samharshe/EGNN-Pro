from torch import manual_seed
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.utils.data import random_split
from typing import Tuple

def get_dataloaders(version: str, molecule: str, train_split: float, val_split: float, test_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """returns a 3-tuple of train, val, and test DataLoader objects as specified in function call.
    
    parameters
    ----------
    version : str
         which of the pre-processed datasets (raw, apricot, brisket, etc.) to fetch.
    molecule : str
        which of molecule datasets (benzene, uracil, aspirin) to fetch.
    train_split : float
        proportion of dataset to be allocated to train dataloader.
    val_split : float
        proportion of dataset to be allocated to val dataloader.
    test_split : float
        proportion of dataset to be allocated to val dataloader.
    batch_size : int
        self-explanatory.
        
    returns
    -------
    3-tuple of train, val, and test loader objects as specified in function call.
    """
    # reproducibility
    manual_seed(2002)
    
    # make sure the splits make sense
    assert train_split + val_split + test_split == 1, f"train_split, val_split, and test_split must sum to 1. got: {train_split + val_split + test_split}."
    
    # load in dataset
    dataset = MD17(root=f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/{version}/', name=f'{molecule}')

    # split defined by the argument of function
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # build train, val, test datasets out of main dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # turn into DataLoaders for batching efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # return DataLoaders
    return train_loader, val_loader, test_loader

def get_datasets(version: str, molecule: str, train_split: float, val_split: float, test_split: float) -> Tuple[Dataset, ...]:
    """returns a 3-tuple of train, val, and test Dataset objects as specified in function call.
    
    parameters
    ----------
    version : str
         which of pre-processed datasets (raw, apricot, brisket, etc.) to fetch.
    molecule : str
        which of molecule datasets (benzene, uracil, aspirin) to fetch.
    train_split : float
        proportion of dataset to be allocated to train dataloader.
    val_split : float
        proportion of dataset to be allocated to val dataloader.
    test_split : float
        proportion of dataset to be allocated to val dataloader.
        
    returns
    -------
    3-tuple of train, val, and test Dataset objects as specified in function call.
    """
    # reproducibility
    manual_seed(2002)
    
    # make sure splits make sense
    assert train_split + val_split + test_split == 1, f"train_split, val_split, and test_split must sum to 1. got: {train_split + val_split + test_split}"
    
    # load in dataset
    dataset = MD17(root=f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/{version}/', name=f'{molecule}')

    # split defined by argument of function
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # build train, val, test datasets out of main dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # return Datasets
    return train_dataset, val_dataset, test_dataset

def get_dataset(version: str, molecule: str) -> Dataset:
    """returns a Dataset object as specified in the function call.
    
    parameters
    ----------
    version : str
         which of pre-processed datasets (raw, apricot, brisket, etc.) to fetch.
    molecule : str
        which of molecule datasets (benzene, uracil, aspirin) to fetch.
        
    returns
    -------
    Dataset object as specified in the function call.
    """
    # reproducibility
    manual_seed(2002)
    
    # load in dataset
    dataset = MD17(root=f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/{version}/', name=f'{molecule}')

    # return Dataset
    return dataset

def get_mini_dataloader(version: str, molecule: str, num_items: int, batch_size: int) -> DataLoader:
    """returns a DataLoader object as specified in function call; especially useful for getting small DataLoader objects to use in experimentation.
    
    parameters
    ----------
    version : str
         which of pre-processed datasets (raw, apricot, brisket, etc.) to fetch.
    molecule : str
        which of molecule datasets (benzene, uracil, aspirin) to fetch.
    num_items : int
        self-explanatory. 
    batch_size : int
        self-explanatory.
        
    returns
    -------
    DataLoader object as specified in function call.
    """
    # reproducibility
    manual_seed(2002)
    
    # load in the dataset
    dataset = MD17(root=f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/data/{version}/', name=f'{molecule}')

    # make mini_dataset out of dataset
    mini_dataset, _ = random_split(dataset, [num_items, len(dataset)-num_items])
    
    # make min_dataloader out of mini_dataset
    mini_dataloader = DataLoader(mini_dataset, batch_size=batch_size)

    # return DataLoader
    return mini_dataloader