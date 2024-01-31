import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.pooling import *
from ..backbones.pointnet import *
from ..backbones import resnet
from networks.utils import *

class PointNetSPoC(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetSPoC, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.head = SPoC(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x)
        x = self.head(x)
        return x
  
    def __str__(self):
        return "PointNetSPoC"

class PointNetAP(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetAP, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.head = SPoC(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x) # B X Features X Points
        #x = x.permute(0,2,1)
        x = self.head(x)
        return x
  
    def __str__(self):
        return "PointNetSPoC"
    

class ResNet50SPoC(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, output_dim=1024):
        super(ResNet50SPoC, self).__init__()

        return_layers = {'layer4': 'out'}
        param = {'pretrained_backbone': False,
                    'out_dim': output_dim,
                    'feat_dim': feat_dim,
                    'in_channels': in_dim,
                    'max_points': num_points,
                    'modality': 'bev'} 
         
        #pretrained = resnet50['pretrained_backbone']

        #max_points = model_param['max_points']
        backbone = resnet.__dict__['resnet50'](param)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.head = SPoC(outdim=output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x['out']
        x = self.head(x)
        return x
    
    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.net_vlad.parameters()
  
    def __str__(self):
        return "ResNet50SPoC"
    
    
    
    
class PointNetCGAP(torch.nn.Module):
    def __init__(self, num_c, feat_dim,use_tnet,output_dim=1024,pooling='max'):
        super().__init__()
        self.features = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        list_layers = mlp_layers(feat_dim, [512, 256], b_shared=False, bn_momentum=0.01, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)
        self.fc = nn.LazyLinear(output_dim)
        
        self.classloss = torch.nn.NLLLoss()
        self.head = SPoC(outdim=output_dim)
        self.pooling = pooling

    def forward(self, points,):
        feat = self.features(points)
        if self.pooling == 'max':
            feat_int = torch.max(feat, dim=-1, keepdim=False)[0]
        elif self.pooling == 'mean':
            feat_int = torch.mean(feat, dim=-1, keepdim=False)
        else:
            raise NotImplementedError
        #feat = self.head(feat)
        out = self.classifier(feat_int)
        d = self.head(feat)
        
        return {'c':out,'d':d}

    def __str__(self):
        return f"PointNetCGAP-{self.pooling}-nlog"
    
    def loss(self, out, target, w=0.1):
        
        target = target.long()
        correct_targets = target > -1
        target = target[correct_targets]
        out = out[correct_targets]
        
        #F.nll_loss(F.log_softmax(input, dim=1), target)
        loss_c = self.classloss(
            torch.nn.functional.log_softmax(out, dim=1), target)
        return loss_c


