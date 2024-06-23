import torch
from torch.nn import L1Loss
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
import wandb
from typing import Callable, Dict
import sys
import os
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/model'))
from model_utils import F_loss_fn

def evaluate(model: MessagePassing, loss_fn: Callable, test_dataloader: DataLoader, rho: float) -> None:
    """evaluates model on dataloader and logs results via wandb.
    
    evaluates loss according to loss_fn and also according to L1Loss, since MAE is often desired for comparison with other models.
    
    parameters
    ----------
    model : MessagePassing
        self-explanatory.
    loss_fn : Callable
        loss function used in training, for apples-to-apples comparison with training loss.
    test_dataloader : DataLoader
        self-explanatory.
    rho : float
        loss = (1-rho)*E_loss + rho*F_loss.
    """
    # do not track gradients
    model.eval()
    
    # test statistics using the same loss function as training
    losses = []
    E_losses = []
    F_losses = []
    
    # test statistics using MAE for comparison with other benchmarks
    total_absolute_loss = 0
    # mean absolute error 
    absolute_loss_fn = L1Loss()    
    
    # iterate through test_dataloader        
    for data in test_dataloader:
        # target values
        E = data.energy
        F = data.force
        
        # predictions from the model
        E_hat, F_hat = model(data)
        E_hat.squeeze_(dim=1)
        
        # loss_fn error for energy loss
        E_loss = loss_fn(E_hat, E)
        
        # a version of loss_fn error for force loss
        F_loss = F_loss_fn(F_hat, F, loss_fn)
        
        # canonical loss
        loss = (1-rho)*E_loss + rho*F_loss
        
        # absolute error for energy loss
        E_absolute_loss = absolute_loss_fn(E_hat, E)
        
        # a version of absolute error for force loss
        F_absolute_loss = F_loss_fn(F_hat, F, absolute_loss_fn)
        
        # absolute loss
        total_absolute_loss += len(data)*((1-rho)*E_absolute_loss + rho*F_absolute_loss).item()
        
        # save loss_fn losses
        losses.append(loss.item())
        E_losses.append(E_loss.item())
        F_losses.append(F_loss.item())
        
    # calculate and log test mean losses
    # very slightly incorrect in the case that batch_size does not divide len(test_dataloader.dataset)
    # this is properly accounted for in MAE but is not worth properly accounting for in these statistics
    mean_loss = torch.mean(torch.tensor(losses)).item()
    wandb.log({'test_mean_loss': mean_loss})
    mean_E_loss = torch.mean(torch.tensor(E_losses)).item()
    wandb.log({'test_mean_E_loss': mean_E_loss})
    mean_F_loss = torch.mean(torch.tensor(F_losses)).item()
    wandb.log({'test_mean_F_loss': mean_F_loss})

    # calculate and log test mean absolute error
    mean_absolute_loss = float(total_absolute_loss) / len(test_dataloader.dataset)
    wandb.log({'absolute_loss': mean_absolute_loss})
    
    # print final results
    print(f'TEST MEAN LOSS: {mean_loss}')
    print(f'TEST MEAN ABSOLUTE LOSS: {mean_absolute_loss}')
    
    # end wandb run
    wandb.finish()