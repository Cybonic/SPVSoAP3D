import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _l2norm(x):
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x
def is_batch_symmetric(matrices):
  """Checks if all matrices in a batch are symmetric.

  Args:
      matrices: A PyTorch tensor of shape (batch_size, matrix_dim, matrix_dim).

  Returns:
      A boolean tensor of shape (batch_size,) indicating symmetry for each matrix.
  """
  return torch.allclose(matrices, matrices.transpose(1, 2))

def is_symmetric(matrix):
  """Checks if a matrix is symmetric by comparing it with its transpose.

  Args:
      matrix: A PyTorch tensor representing the matrix.

  Returns:
      True if the matrix is symmetric, False otherwise.
  """
  return torch.allclose(matrix, matrix.T)

def is_positive_definite_direct(matrix):
  """Checks if a matrix is positive definite using direct verification (expensive).

  Args:
      matrix: A PyTorch tensor representing the matrix.

  Returns:
      True if the matrix is positive definite, False otherwise.
  """
  for v in torch.randn(matrix.data.size(1), **matrix.data.size()):  # Generate random vectors
    if torch.dot(v.T @ matrix @ v, v) <= 0:
      return False
  return True


    
class SoAP(nn.Module):
    def __init__(self, 
                 p=0.75,
                 epsilon=1e-8, 
                 do_fc=True, 
                 do_log=True, 
                 do_pn = True,
                 do_epn = True,
                 input_dim=16, 
                 output_dim=256,
                 **kwargs):
        super(SoAP, self).__init__()
        
        self.do_fc = do_fc
        self.do_log = do_log
        self.do_epn = do_epn
        self.do_pn = do_pn
        # power norm over  eigen-value power normalization
        self.do_epn = False if do_pn or do_log else self.do_epn
        
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.fc = nn.LazyLinear( output_dim)
        self.p = nn.Parameter(torch.ones(1) * p)
    
    def __str__(self):
        
        pn_str = "pn" 
        if self.do_epn and not self.do_pn:
            pn_str = "epn"
        elif not self.do_epn and not self.do_pn:
            pn_str = "no_pn"
        
        stack = ["SoAP" ,
                 "log" if self.do_log else "no_log",
                 pn_str,
                  "fc" if self.do_fc else "no_fc",
                 ]
        return '-'.join(stack)
    
    def _epn(self,x):
        """
        Eigen-value Power Normalization over the positive semi-definite matrix.
        """
        u_, s_, v_ = torch.svd(x)
        s_alpha = torch.pow(s_, 0.5)
        x =torch.matmul(torch.matmul(u_, torch.diag_embed(s_alpha)), v_.transpose(-2, -1))
        return x
  
            
    def _log(self,x):
        # Log-Euclidean Tangent Space Mapping
        # Inspired by -> Semantic Segmentation with Second-Order Pooling
        # Implementation -> https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
        # x must be a symmetric positive definite (SPD) matrix
        # assert is_batch_symmetric(x) # and is_positive_definite_direct(x)
        
        u, s, v = torch.linalg.svd(x)
        s = s.clamp(min=self.epsilon)  # clamp to avoid log(0)
        x=torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
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

        # Averaging over the points
        x = torch.mean(x, 1) 

        if self.do_log:
            x = self._log(x)
            
        if self.do_pn:
            x = self._pow_norm(x)
        
        if self.do_epn:
            x = self._epn(x)
        
        # Flatten
        x = x.reshape(batchSize, -1)   
         
        if self.do_fc:
            x =  self.fc(x)
            #x = x*F.softmax(self.fc(x),1)
        
        x = _l2norm(x)
        return torch.squeeze(x).float()
    
   
 