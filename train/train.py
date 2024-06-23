import torch, sys, os, wandb
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Dict
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/model'))
from model_utils import F_loss_fn

def train(model: MessagePassing, optimizer: Optimizer, scheduler: LRScheduler, loss_fn: Callable, train_dataloader: DataLoader, val_dataloader: DataLoader, rho: float, num_epochs: int, name: str) -> None:
    """trains model on dataloader, saves weights of the best-performing model, and logs ongoing results through wandb.
    
    parameters
    ----------
    model : MessagePassing
        self-explanatory.
    optimizer : Optimizer
        self-explanatory.
    scheduler : LRScheduler
        self-explanatory.
    loss_fn : Callable
        self-explanatory.
    train_dataloader : DataLoader
        self-explanatory.
    val_dataloader : DataLoader
        self-explanatory.
    rho : float
        loss = (1-rho)*E_loss + rho*F_loss.
    num_epochs : int
        self-explanatory.
    name : str
        best-performing weights are saved to f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/weights/{name}.pth'.
    """
    # keep track of the best val performance to know when to save weights
    val_min_loss = sys.float_info.max

    # training loop occurs num_epochs times
    for epoch in range(num_epochs):
        # track gradients
        model.train()
        
        # dummy variable to track loss every 100 batches
        batch_counter = 0
        
        # iterate through test_dataloader        
        for data in train_dataloader:
            # clear gradients
            optimizer.zero_grad()

            # target values
            E = data.energy
            F = data.force
            
            # predictions from the model
            E_hat, F_hat = model(data)
            E_hat.squeeze_(dim=1)

            # squared error for energy loss
            E_loss = loss_fn(E_hat, E)

            # a version of squared error for force loss
            F_loss = F_loss_fn(F_hat, F, loss_fn)
            
            # canonical loss
            loss = (1-rho)*E_loss + rho*F_loss
        
            # calculate gradients
            loss.backward()
            
            # update
            optimizer.step()
            
            # log loss every 100 batches
            if batch_counter == 0:
                # log losses and learning rate
                wandb.log({"train_losses": loss.item(), "E_train_losses": E_loss.item(), "F_train_losses": F_loss.item(), "learning_rates": optimizer.param_groups[0]['lr']})
            batch_counter+=1
            batch_counter%=100
        
        # VAL
        epoch_losses = []
        epoch_E_losses = []
        epoch_F_losses = []
        
        # do not track gradients
        model.eval()
        
        # iterate through val_dataloader
        for data in val_dataloader:
            # targets
            E = data.energy
            F = data.force
            
            # predictions
            E_hat, F_hat = model(data)
            E_hat.squeeze_(dim=1)
            
            # loss_fn error for energy loss
            E_loss = loss_fn(E_hat, E)
            
            # a version of loss_fn error for force loss
            F_loss = F_loss_fn(F_hat, F, loss_fn)
            
            # canonical loss
            loss = (1-rho)*E_loss + rho*F_loss
            
            # track F_loss, E_loss, canonical loss
            epoch_losses.append(loss.item())
            epoch_E_losses.append(E_loss.item())
            epoch_F_losses.append(F_loss.item())
        
        # calculate and log mean losses from this epoch
        epoch_mean_loss = torch.mean(torch.tensor(epoch_losses)).item()
        epoch_mean_E_loss = torch.mean(torch.tensor(epoch_E_losses)).item()
        epoch_mean_F_loss = torch.mean(torch.tensor(epoch_F_losses)).item()
        wandb.log({"epoch_mean_loss": epoch_mean_loss, "epoch_mean_E_loss": epoch_mean_E_loss, "epoch_mean_F_loss": epoch_mean_F_loss})
        
        # print out results of epoch
        print(f'EPOCH {epoch+1} OF {num_epochs} | VAL MEAN LOSS: {epoch_mean_loss}')
        
        # if this is best val performance yet, save weights
        if val_min_loss > epoch_mean_loss:
            val_min_loss = epoch_mean_loss
            torch.save(model, f'/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/weights/{name}.pth')
            
        # update lr based on mean loss of previous epoch
        scheduler.step(epoch_mean_loss)