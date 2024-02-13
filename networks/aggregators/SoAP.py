import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _l2norm(x):
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x
    
    
class SoAP(nn.Module):
    def __init__(self, 
                 p=0.75,
                 epsilon=1e-8, 
                 do_fc=True, 
                 do_log=True, 
                 do_power_norm = True,
                 do_pe = True,
                 input_dim=16, 
                 output_dim=256,
                 **kwargs):
        super(SoAP, self).__init__()
        
        self.do_fc = do_fc
        self.do_log = do_log
        self.do_pe = do_pe
        self.do_power_norm = do_power_norm
        # power norm over PE
        self.do_pe = False if do_power_norm or do_log else self.do_pe
        
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.fc = nn.LazyLinear( output_dim)
        self.p = nn.Parameter(torch.ones(1) * p)
    
    def __str__(self):
        
        stack = ["SoAP" ,
                 "log" if self.do_log else "no_log",
                 "pownorm" if self.do_power_norm else "no_pownorm",
                 "pe" if self.do_pe else "no_pe",
                  "fc" if self.do_fc else "no_fc",
                 ]
        return '-'.join(stack)
    
    def _pe(self,x):
        u_, s_, v_ = torch.svd(x)
        s_alpha = torch.pow(s_, 0.5)
        x =torch.matmul(torch.matmul(u_, torch.diag_embed(s_alpha)), v_.transpose(-2, -1))
        return x
  
            
    def _log(self,x):
        
        # Inspired by -> Semantic Segmentation with Second-Order Pooling
        # Implementation -> https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
        u, s, v = torch.linalg.svd(x)
        s = s.clamp(min=self.epsilon)  # clamp to avoid log(0)
        x=torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
        x = x.clamp(min=self.epsilon)
        # Power Normalization.
        #h=0.75 
        self.p.clamp(min=self.epsilon, max=1.0)
        x = torch.sign(x)*torch.pow(torch.abs(x),self.p)
        x = x.clamp(min=self.epsilon)
            
        return x.float()

    def _pow_norm(self,x):
        # Power Normalization.
        # Semantic Segmentation with Second-Order Pooling
        #h=0.75 
        self.p.clamp(min=self.epsilon, max=1.0)
        x = torch.sign(x)*torch.pow(torch.abs(x),self.p)
        x = x.clamp(min=self.epsilon)
            
        return x.float()
    
    
    def forward(self, x):
        x = x.double()
        x = x.clamp(min=self.epsilon)
        
        # Outer product
        batchSize, nPoints, dimFeat = x.data.shape
        x = x.unsqueeze(-1)
        x = x.matmul(x.transpose(3, 2))

        
        # Average pooling
        #x = torch.reshape(x, (batchSize, nPoints, dimFeat, dimFeat))
        x = torch.mean(x, 1)
        #print(x.data.shape)
        #x = torch.reshape(x, (-1, dimFeat, dimFeat))
        
        if self.do_log:
            x = self._log(x)
            
        if self.do_power_norm:
            x = self._pow_norm(x)
        
        if self.do_pe:
            x = self._pe(x)
        
        # Flatten
        x = x.reshape(batchSize, -1)   
         
        if self.do_fc:
            x =  self.fc(x)
            #x = x*F.softmax(self.fc(x),1)
        
        x = _l2norm(x)
        return torch.squeeze(x).float()
    
   
 