"""Exploration-only stateless cell using the production hard spike/surrogate."""
import torch
from torch import nn,Tensor
from models.neurons.lif import LIFCell,surrogate_spike

class StatelessSurrogateCell(nn.Module):
 def __init__(self,threshold=1.0,surrogate_beta=10.0):
  super().__init__();self.threshold=float(threshold);self.surrogate_beta=float(surrogate_beta)
 def forward(self,current:Tensor,detach_temporal:bool=False)->Tensor:
  if current.ndim<3:raise ValueError(f'expected [T,B,...], got {tuple(current.shape)}')
  return surrogate_spike(current-self.threshold,self.surrogate_beta)

def make_stateless(model):
 replaced=[]
 for name,module in model.named_modules():
  if hasattr(module,'neuron') and isinstance(module.neuron,LIFCell):
   old=module.neuron;module.neuron=StatelessSurrogateCell(old.threshold,old.surrogate_beta);replaced.append(name+'.neuron')
 if not replaced:raise RuntimeError('no LIF cells replaced')
 return replaced
