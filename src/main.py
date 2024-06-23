import torch
from torch.nn import MSELoss
import sys
import os
from EGNN5 import EGNN5
import wandb
from test_model import test_model
from train_model import train_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/data/get/'))
from utils import get_dataloaders
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/train/'))
from train import train
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/evaluate'))
from evaluate import evaluate

# reproducibility
torch.manual_seed(2002)

# hyperparameters saved to config dict
config = {
    'name': 'EGNN5',
    'base_learning_rate': 0.001,
    'num_epochs': 5,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_mode': 'min',
    'scheduler_factor': 0.32, 
    'scheduler_patience': 1,
    'scheduler_threshold': 0,
    'training_loss_fn': 'MSELoss',
    'rho': 1-1e-1,
    'batch_size': 32
}

# initialize the star of the show
model = EGNN5()

# I couldn't think of a concise way to initialize optimizer, scheduler, and loss_fn based on the contents of config
# this is all for show anyway, but it would be nice to have a natural way of doing this that generalizes when I am selecting hyperparameters more carefully
optimizer = Adam(model.parameters(), lr=config['base_learning_rate'])

# initialize scheduler
scheduler = ReduceLROnPlateau(
    optimizer=optimizer, 
    mode=config['scheduler_mode'], 
    factor=config['scheduler_factor'], 
    patience=config['scheduler_patience'], 
    threshold=config['scheduler_threshold']
    )

# initialize loss function
loss_fn = MSELoss()

# wandb
os.environ['WANDB_NOTEBOOK_NAME'] = 'main.py'
wandb.init(
    project = "EGNN Pro",
    config = config,
)

# create dataloaders
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(version='apricot', molecule='benzene', train_split=0.8, val_split=0.1, test_split=0.1, batch_size=32)

# train and evaluate model
train(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader, rho=config['rho'], num_epochs=config['num_epochs'], name=config['name'])
evaluate(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader, rho=config['rho'])