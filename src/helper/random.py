import random 
import numpy as np 
import torch 
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """ 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)