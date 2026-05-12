import torch
from torch import nn as nn, softmax
from torch.nn import functional as F
from torch.distributions import Categorical

def categorical_sample(logits_BlV: torch.Tensor, num_samples=1, rng=None) -> torch.Tensor:
    """
     logits  softmax  top-k/top-p 
    
    :
        logits_BlV:  (B, L, V)  logits 
        num_samples: 1
        rng: None
        
    :
        idx:  (B, L, num_samples) 
    """
    B, L, V = logits_BlV.shape
    
    # Step 1: 
    probs = torch.softmax(logits_BlV, dim=-1)  #  (B, L, V)
    
    # Step 2:  Categorical 
    dist = Categorical(probs=probs)
    
    # Step 3: 
    # sample_shape=(num_samples,)  num_samples 
    idx = dist.sample(sample_shape=(num_samples,))  #  (num_samples, B, L)
    
    # Step 4:  (B, L, num_samples)
    idx = idx.permute(1, 2, 0)  # (B, L, num_samples)
    
    return idx

def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    return categorical_sample(logits_BlV, num_samples=num_samples)

def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'