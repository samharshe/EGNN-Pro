import torch, sys, os, wandb
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/src/data/get'))
from data_get_utils import get_dataloaders
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/train'))
from train import train
sys.path.append(os.path.abspath('/Users/samharshe/Documents/Gerstein Lab/EGNN Pro/evaluate'))
from evaluate import evaluate
from model.delta import Delta

# reproducibility
torch.manual_seed(2002)

# hyperparameters saved to config dict
config = {
    'name': 'Delta',
    'base_learning_rate': 0.001,
    'max_epochs': 20,
    'early_stop_patience': 2,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_mode': 'min',
    'scheduler_factor': 0.32, 
    'scheduler_patience': 1,
    'scheduler_threshold': 0,
    'training_loss_fn': 'MSELoss',
    'rho': 1-1e-1,
    'batch_size': 32}

# initialize the star of the show
model = Delta()

# initialize optimizer, scheduler, loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config['base_learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, 
    mode=config['scheduler_mode'], 
    factor=config['scheduler_factor'], 
    patience=config['scheduler_patience'], 
    threshold=config['scheduler_threshold'])
loss_fn = torch.nn.MSELoss()

# wandb
os.environ['WANDB_NOTEBOOK_NAME'] = 'main.py'
wandb.init(project="EGNN", config=config)

# create dataloaders
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(version='apricot', molecule='benzene', train_split=0.8, val_split=0.1, test_split=0.1, batch_size=32)

# train and evaluate model
train(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader, rho=config['rho'], max_epochs=config['max_epochs'], early_stop_patience=config['early_stop_patience'], name=config['name'])
evaluate(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader, rho=config['rho'])

# end wandb run
wandb.finish()