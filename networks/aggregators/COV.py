import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class COVtorch(nn.Module):
    def __init__(self,   do_fc=True, output_dim=256,**kwargs):
        super(COV, self).__init__()
        self.do_fc = do_fc
        self.fc = nn.LazyLinear( output_dim)

    def _l2norm(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        cov = []
        for y in x:
            c = torch.cov(y.T)
            cov.append(c.flatten())
        x = torch.stack(cov).squeeze()
        
        if self.do_fc:
            x = self.fc(x)
            
        x = self._l2norm(x)
        return torch.squeeze(x)
    
    def __str__(self):
        
        stack = ["COVtroch",
                "fc" if self.do_fc else "no_fc",
                ]
        return '-'.join(stack)
    

class COV(nn.Module):
    def __init__(self, thresh=1e-8, do_pe=True,  do_fc=True, input_dim=16, is_tuple=False,output_dim=256):
        super(COV, self).__init__()
        self.thresh = thresh
        self.sop_dim = input_dim * input_dim
        self.do_fc = do_fc
        self.do_pe = do_pe
        self.is_tuple = is_tuple
        self.fc = nn.LazyLinear( output_dim)

    def _so_meanpool(self, x):
        batchSize, nFeat, dimFeat = x.data.shape
        x = torch.reshape(x, (-1, dimFeat))
        # de-mean
        xmean = torch.mean(x, 0)
        x = x - xmean.unsqueeze(0)
        
        x = x.unsqueeze(-1)
        x = x.matmul(x.transpose(2, 1))

        x = torch.reshape(x, (batchSize, nFeat, dimFeat, dimFeat))
        x = torch.mean(x, 1)
        x = torch.reshape(x, (-1, dimFeat, dimFeat))

        # Normalize covariance
        if self.do_pe:
            x = x.double()
            # For pytorch versions < 1.9
            u_, s_, v_ = torch.svd(x)
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

        x = torch.reshape(x, (batchSize, -1))
        return x.float()
    
    def _l2norm(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self._so_meanpool(x)
        if self.do_fc:
            x = self.fc(x)
        x = self._l2norm(x)
        return torch.squeeze(x)
    
    def __str__(self):
        
        stack = ["COV" ,
                 "fc" if self.do_fc else "no_fc",
                 "pe" if self.do_pe else "no_pe",
                 ]
        return '-'.join(stack)
 
