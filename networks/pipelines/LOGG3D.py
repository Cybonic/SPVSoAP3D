import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from networks.aggregators.SOP import *
from networks.backbones.spvnas.model_zoo import spvcnn
from networks.pipelines.pipeline_utils import *

__all__ = ['LOGG3D']


class LOGG3D(nn.Module):
    def __init__(self, output_dim=256):
        super(LOGG3D, self).__init__()

        self.backbone = spvcnn(output_dim=16)
        self.head = SOP(
            signed_sqrt=False, do_fc=False, input_dim=16, is_tuple=False)

    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)

        x = self.backbone(x)
        y = torch.split(x, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.head(x)
        
        out = {'out':x,'feat':[]}
        return out

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.head.parameters()
  
    def __str__(self):
        return "LOGG3D"
    
