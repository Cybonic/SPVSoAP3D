import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _so_layer_cov(x,do_pe=True,do_dm=True,do_log=False, epsilon = 1e-8):
        batchSize, nFeat, dimFeat = x.data.shape
        #x = torch.reshape(x, (-1, dimFeat))
        #x = torch.reshape(x, (-1, dimFeat))
        # de-mean
        if do_dm:
            xmean = torch.mean(x, 1)
            x = x - xmean.unsqueeze(1)
        
        x = x.unsqueeze(-1)
        x = x.matmul(x.transpose(3, 2))

        x = torch.reshape(x, (batchSize, nFeat, dimFeat, dimFeat))
        x = torch.mean(x, 1)
        x = torch.reshape(x, (-1, dimFeat, dimFeat))
        
        if do_log:
            # https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
            u, s, v = torch.linalg.svd(x)
            x=torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
            x = x.clamp(min=epsilon)
            # Power Normalization.
            # Semantic Segmentation with Second-Order Pooling
            h=0.75 
            
            x = torch.sign(x)*torch.pow(torch.abs(x),h)
            x = x.clamp(min=epsilon)
            
          
        
        # Normalize covariance
        if do_pe:
            x = x.double()
            # For pytorch versions < 1.9
            u_, s_, v_ = torch.svd(x)
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

        x = torch.reshape(x, (batchSize, -1))
        return x.float()

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
    def __init__(self, 
                 thresh=1e-8, 
                 do_pe=True,  
                 do_fc=True, 
                 do_dm=True,
                 do_log=False, 
                 input_dim=16, 
                 is_tuple=False,
                 output_dim=256,
                 pooling='bach_cov',
                 **kwargs):
        super(COV, self).__init__()
        self.thresh = thresh
        self.sop_dim = input_dim * input_dim
        self.do_fc = do_fc
        self.do_pe = do_pe
        self.do_dm = do_dm
        self.do_log = do_log
        self.is_tuple = is_tuple
        self.fc = nn.LazyLinear( output_dim)
        self.pooling = pooling

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
        if self.pooling == 'layer_cov':
            x = _so_layer_cov(x,do_pe=self.do_pe,do_dm = self.do_dm,do_log=self.do_log)
        else:
            x = self._so_meanpool(x)
            
        if self.do_fc:
            x =  self.fc(x)
            #x = x*F.softmax(self.fc(x),1)
        x = self._l2norm(x)
        return torch.squeeze(x)
    
    def __str__(self):
        
        stack = ["COV" ,
                 self.pooling,
                 "fc" if self.do_fc else "no_fc",
                 "pe" if self.do_pe else "no_pe",
                 "dm" if self.do_dm else "no_dm",
                 "log" if self.do_log else "no_log",
                 ]
        return '-'.join(stack)
 
