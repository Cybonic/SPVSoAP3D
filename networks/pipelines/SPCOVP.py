import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from networks.aggregators.COV import *
from networks.aggregators.pooling import SPoC
from networks.backbones.spvnas.model_zoo import spvcnn,spvcnnx
from networks.pipelines.pipeline_utils import *
from networks.backbones.pointnet import PointNet_features

__all__ = ['LOGG3D']


class SPGAP(nn.Module):
    def __init__(self, output_dim=256,local_feat_dim=16,pres=0.05,vres=0.05,cr=0.64):
        super(SPGAP, self).__init__()

        self.backbone = spvcnnx(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        
        self.head = SPoC(output_dim=output_dim)
        
        #self.head = COV(do_fc=do_fc, do_pe = do_pe, input_dim=local_feat_dim, is_tuple=False,output_dim=output_dim)
        
        
    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)

        x = self.backbone(x)
        y = torch.split(x, list(counts))
        lfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 2,0)
        x = self.head(lfeat)
        
        out = {'out':x,'feat':lfeat}
        return out #, y[:2]

    def __str__(self):
        
        stack = ["SPCov3D" ,
                 str(self.head),
                 ]
        return '-'.join(stack)
    

class SPCov3D(nn.Module):
    def __init__(self, output_dim=256,n_seg_class=2,local_feat_dim=16,do_fc=False,do_pe = False,pres=1,vres=1,cr=0.64):
        super(SPCov3D, self).__init__()

        self.backbone = spvcnnx(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        self.head = COV(do_fc=do_fc, do_pe = do_pe, input_dim=local_feat_dim, is_tuple=False,output_dim=output_dim)
        
        
    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)

        x = self.backbone(x)
        y = torch.split(x, list(counts))
        lfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.head(lfeat)
        
        out = {'out':x,'feat':lfeat}
        return out #, y[:2]

    def __str__(self):
        
        stack = ["SPCov3D" ,
                 str(self.head),
                 ]
        return '-'.join(stack)
    
    
class fc_net(nn.Module):
    def __init__(self, dim, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PointNetCov3D(nn.Module):
    def __init__(self, in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetCov3D, self).__init__()
        self.backbone = self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        self.head = COV(do_fc=False, input_dim=feat_dim, is_tuple=False,output_dim=output_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def __str__(self):
        return "PointNetCov3D"
    
    
    
class PointNetCov3DC(nn.Module):
    def __init__(self, in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetCov3DC, self).__init__()

        self.backbone = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        self.head = COV(do_fc=False, input_dim=feat_dim, is_tuple=False,output_dim=output_dim)
        self.head_fc = fc_net(feat_dim,output_dim)
        
    def forward(self, x):
        
        x = self.backbone(x)
        d = self.head(x)
        x = self.head_fc(x)
        return d,x

    def __str__(self):
        return "PointNetCov3DFC"
    

