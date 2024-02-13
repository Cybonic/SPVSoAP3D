import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from networks.aggregators.SoAP import *

from networks.backbones.spvnas.model_zoo import spvcnn,spvcnnx
from networks.pipelines.pipeline_utils import *

__all__ = ['LOGG3D']

    

class SPCov3Dx(nn.Module):
    def __init__(self, output_dim=256,n_seg_class=2,local_feat_dim=16,do_fc=True,do_pe = True, do_dm = True,pres=1,vres=1,cr=0.64,**kwargs):
        super(SPCov3Dx, self).__init__()

        #self.backbone = spvcnn(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        self.backbone = spvcnnx(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        self.head = SoAP(do_fc=do_fc, do_pe = do_pe,do_dm=do_dm, input_dim=local_feat_dim, is_tuple=False,output_dim=output_dim,**kwargs)
        
        
    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x,features = self.backbone(x)
        # map feature from the middle of the network to batch format
        y = torch.split(features, list(counts))
        mfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        # map feature from the end of the network to batch format
        y = torch.split(x, list(counts))
        lfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.head(lfeat)
        
        out = {'out':x,'feat':mfeat}
        return out #, y[:2]

    def __str__(self):
        
        stack = ["SPCov3Dx" ,
                 str(self.head),
                 ]
        return '-'.join(stack)
    
    
class SPVSoAP3D(nn.Module):
    def __init__(self, output_dim=256,
                 local_feat_dim=16,
                 do_fc =True,
                 do_pe =True,
                 do_pwnorm = True,
                 do_log = False,
                 pres=1,
                 vres=1,
                 cr=0.64,
                 **kwargs):
        super(SPVSoAP3D, self).__init__()

        self.backbone = spvcnn(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        
        self.head = SoAP(do_fc = do_fc, 
                        do_pe  = do_pe, 
                        do_log = do_log,
                        do_power_norm = do_pwnorm,
                        input_dim=local_feat_dim,
                        output_dim=output_dim,
                        **kwargs)
        
        
    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.backbone(x)
        y = torch.split(x, list(counts))
        lfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.head(lfeat)
        
        #print(x.shape)
        out = {'out':x,'feat':lfeat}
        return out #, y[:2]

    def __str__(self):
        
        stack = ["SPVSoAP3D" ,
                 str(self.head),
                 ]
        return '-'.join(stack)
    
    



