import torch, os, sys
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.optim import Optimizer, SGD
import matplotlib.pyplot as plt
from typing import Callable
from .data_get_utils import get_mini_dataloader

def sanity_check(model: MessagePassing, rho:float=1-1e-2, num_items:int=1024, batch_size:int=32, num_epochs:int=10) -> None:
    """puts the model through a very brief training run to check for elementary bugs, making a matplotlib plot of loss.
    
    parameters
    ----------
    model : MesagePassing
        the model to be checked.
    rho : float, optional
        default : 1-1e-2.
        loss = (1-rho)*E_loss + rho*F_loss.
    num_items : int, optional
        default : 1024
        the number of data items in each epoch.
    batch_size : int, optional
        default : 32
        self-explanatory.
    num_epochs : int, optional
        default : 10
        self-explanatory.
    
    returns
    -------
    None; prints matplotlib plot of loss.
    """
    
    # make dataloader
    dataloader = get_mini_dataloader(version='apricot', molecule='benzene', num_items=num_items, batch_size=batch_size)
    
    # SGD for maximal simplicity
    optimizer = SGD(model.parameters(), lr=0.001)
    
    # MSE for maximal simplicity
    loss_fn = torch.nn.MSELoss()
        
    # track losses
    losses = []
    
    # training loop
    for _ in range(num_epochs):
        for data in dataloader:
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
            
            # track losses
            losses.append(loss.item())
            
    # make plot of losses to check for convergence
    plt.plot(range(len(losses)), losses)
    plt.show()
    
def gaussian_rbf(x: Tensor, sigma: float = 0.05, c_min: float = 0, c_max: float = 1.6, num_c: int = 8) -> Tensor:
    """takes in a tensor representing distance and expands it into a vector (Tensor) in a Gaussian radial basis.
    
    formula for a Gaussial radial basis function: 
    
    ..math:: 
    
        \\exp(-\\frac{{(x-c)}^2}{2\\sigma^2}).
        
    this method creates `num_c` Gaussian radial basis functions with centers evenly spaced from `c_min` to `c_max` and returns a vector (tensor) whose i-th element is the value of `x` written in the i-th basis element.
    default values work well for distances in the range :math:`(0, 1.6)`.
    
    parameters
    ----------
    x : Tensor
        1-element tensor, representing distance, to be expanded in the Gaussian radial basis.
    sigma : float, optional
        default : 0.05.
        see formula.
    c_min : float, optional
        default : 0
        center of the leftmost basis function.
    c_max : float, optional
        default : 1.6
        center of the rightmost basis function.
    num_c : int, optional
        default : 8
        size of the basis.
    
    returns
    -------
    vector (Tensor) representing input distance in a Gaussian radial basis, as specified in function call.
    """
    # size of the increments
    c_inc = c_max / num_c
    
    # tensor of the centers 
    cs = torch.arange(c_min, c_max, c_inc)
    
    # returning a vectorized implementation of basis projection, using broadcasting
    return torch.exp(-torch.square((x - cs)) / (2*sigma**2)).float()

def F_loss_fn(F: Tensor, F_hat: Tensor, loss_fn: Callable) -> Tensor:
    """calculates the atomwise Euclidean distances between the predicted and actual force vectors and returns a loss via loss_fn.
    
    parameters
    ----------
    F : Tensor
        target atomwise force vector.
        dimensions are [3, num_atoms].
    F_hat : Tensor
        predicted atomwise force vector.
        dimensions are [3, num_atoms].
    loss_fn : Callable
        takes in [num_atoms] tensor and returns 1-item loss tensor.
        
    returns:
    --------
    1-item Tensor containing loss.
    """
    # avoid bugs when the parameters do not make sense
    assert F.size() == F_hat.size(), f'expected F and F_hat to be the same size. got F.size()={F.size()} and F_hat.size()={F_hat.size()}'
    
    # Euclidean distance between the target and predicted force vectors
    F_error = torch.sqrt(torch.sum(torch.square(F - F_hat), dim=1))
    
    # takes in [num_atoms] tensor and returns 1-item loss tensor
    F_loss = loss_fn(F_error, torch.zeros_like(F_error))
    
    # return F_loss
    return F_loss

def bessel_rbf(x: Tensor, n: int, r_cut: float) -> Tensor:
    """takes in a tensor representing distance and expands it into a vector (tensor) in a Bessel radial basis.
    
    formula for a Bessel radial basis function: 
    
    ..math:: 
    
        \\sin(\\frac{(n\\pi)}{r_{\\mathrm{cut}}} \\Vert \\vec{r}_{ij} \\Vert) / \\Vert \\vec{r}_{ij} \\Vert.
        
    notation consistent with page 4 of https://arxiv.org/pdf/2102.03150, which follows the lead of page 5 of https://arxiv.org/pdf/2003.03123.
        
    this method creates `n` Bessel radial basis functions and returns a vector (tensor) whose i-th element is the value of `x` written in the i-th basis element.
    
    parameters
    ----------
    x : Tensor
        1-element tensor, representing distance, to be expanded in the Bessel radial basis.
    n : int
        cardinality of Bessel basis.
    r_cut : float
        cutoff distance, representing the maximum distance between two connected nodes.
        
    returns
    -------
    vector (Tensor) representing input distance in a Bessel radial basis, as specified in function call.
    """
    # frequency tensor of length n
    ns = torch.arange(1, n+1).view(1,-1).float()
    
    # output as defined in Bessel radial basis function equation
    out = torch.div(torch.sin(torch.div(torch.matmul(x,ns) * torch.pi, r_cut)), x)
    
    # return
    return out

def cosine_cutoff(x: Tensor, r_cut: float) -> Tensor:
    """takes in a tensor representing distance and returns its coefficient under a cosine cutoff.
    
    formula for a cosine cutoff function:
    
    ..math::
    
        0.5 \\cos(\\frac{\\pi x}{r_{\mathrm{cut}}} + 1)
        
    for :math:`x \leq r_{\mathrm{cut}}`, and :math:`0` for :math:`x > r_{\mathrm{cut}}`.
    
    it is desirable to set the value of the basis for all values greater than `r_cut` to 0 without introducing a discontinuity at `r_cut`.
    cosine cutoff maps 0 to 1, leaving distances near 0 minimally affected, and maps `r_cut` to 0, giving distances slightly smaller than `r_cut` values near 0. it maps values greater than `r_cut` uniformly to 0.
    it is :math:`C^\\infty`, which allows it to interact nicely with all basis functions.
    
    parameters
    ----------
    x : Tensor
        1-element tensor, representing basis, whose coefficient under a cosine cutoff is to be calculated.
    r_cut : float
        maximum distance between connected nodes.
        
    returns
    -------
    Tensor representing coefficient of input distance under cosine cutoff with `r_cut` as specified in function call.
    """
    # f(0) = 1 and f(r_cut) = 0 smoothly
    cutoff_distances = 0.5 * (torch.cos(torch.pi * x / r_cut) + 1)
    
    # truncate everything beyond r_cut
    cutoff_distances[x > r_cut] = 0.0
    
    # return
    return cutoff_distances