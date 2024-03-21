

# INSTALLATION

### Set up environment
- python 3.9.4
- Cuda 11.7
- pytorch 2.0.1
- open3d
- torchpack

### Create conda environment with python:
```
conda create -n pr_env python=3.9.4
```
```
conda activate pr_env
```
### Install Conda 11.7.0
```
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```
### Install Pytorch 2.0.1

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
### Install Sparse
```
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

### Install remaining libs
```
pip install -r requirements.txt
```



# Download 

### Dataset



### Checkpoints 
The checkpoints can be downloaded from  [SPVSoAP3D_iros24.zip](https://nas-greenbotics.isr.uc.pt/drive/d/s/xkN8AYuu7uiP9n4kp2Am1fUNFxE2dLaa/bRgEMDjkuiBPCYZb9qKxFg7_3cZ50SXd-DLkgwc17OQs).

SPVSoAP3D_iros24.zip contains additionally the descriptors and results obtained with SPVSoAP3D on all sequences. SPVSoAP3D_iros24.zip has the following strucutre:
```
24SPVSoAP3D_iros24.zip
├── GT23 
├── OJ22
├── OJ23
├── ON22
├── ON23
└── SJ23
    ├── checkpoints.pth # trained weights
    ├── descriptors.torch # descriptors
    └── place
        ├── results_precision.csv # precision 
        ├── results_recall.csv # recall 
        ├── loops.csv # predicted loops
        ├── scores.csv # similarity scores  
        └── target.csv # ground-truth loops
```



The compressed file iros24_spvsoap_checkpoints.zip contains the following checkpoint files: 

| file     | Sequence |
|:--------:|:--------:|
| gtj23-spvsoap3d.pth   | GTJ23  |
| oj22-spvsoap3d.pth    | OJ22   |
| oj23-spvsoap3d.pth    | OJ23   |
| on22-spvsoap3d.pth    | ON22   |
| on23-spvsoap3d.pth    | ON23   |
| sj23-spvsoap3d.pth    | SJ23   |


##  Inference 

python 
