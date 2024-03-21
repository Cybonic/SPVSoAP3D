## Install NVIDIA Drivers 
ubuntu-drivers devices

### Clean/remove installed version

https://askubuntu.com/questions/206283/how-can-i-uninstall-a-nvidia-driver-completely

sudo apt-get remove --purge '^nvidia-.*'
sudo reboot

sudo dpkg -P $(dpkg -l | grep nvidia-driver | awk '{print $2}')
sudo apt autoremove

### INSTALL NVIDIA DRIVER
#### on Ubuntu 20.04 

sudo apt install nvidia-driver-530 -> Worked 


######################################################33
After installing the above driver, conda environments;


## Set up environment
- python 3.9.4
- Cuda 11.7
- pytorch 2.0.1
- open3d
- torchpack


## Important Commands
nvidia-smi
nvcc --version

## Create conda environment with python:
```
conda create -n pr_env python=3.9.4
conda activate pr_env
```
## Install Conda 11.7.0

```
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit -> Ubuntu 22.04
conda install -c "nvidia/label/cuda-11.4.0" cuda-toolkit -> ubuntu 20.02 drivers 450.80.02
```
## Install Pytorch 2.0.1

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
import torch
python -c "import torch; print(torch.cuda.is_available())
```
## Install sparse
Both Cuda11.7 and cuda.11
```
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Install remaining libs
```
pip install -r requirements.txt
```



# DATASET

# Data Visualization:

## Meaning of the files:
 - gps.txt: [lat lon alt] This file contains lat lon alt  coordinates mapped to utm, the first data point is the origin of the reference. The values are seperated by a space;
 - raw_gps.txt: [lat lon alt] file contains lat lon alt coordinates in original GNSS coordinate frame, can be used to generate kml file;

## Modality synchronization 
Synchronization was obtained by retrieving the nearest neigbhor timestamp between a query modality and second modality. All modalities were synchronized to the point cloud timestamps. 
Note: In same cases where there are more data points in the query than in the reference modality, duplications of data may occur.
E.g.: when there are more point clouds than GPS measurements, the same position can be atributed to different point clouds. 

orchards/sum22/extracted/ 
 path: gps.txt
 sync data points: 4361

orchards/aut22/extracted/
 path file: gps.txt
 sync data points: 7974

orchards/june23/extracted/
 path file: gps.txt
 sync data points 7229

Strawberry/june23/extracted/
 path file: gps.txt
 sync data points 6389

# Triplet Ground Truth: 

Pre-generated triplet files: 
A loop exists (ie anchor-positive pair) whenever  two samples from different revisits are whithin a range of 2 meters;   
For each anchor, exists: 
 - 1 positive, nearest neigbor within a range of 2 meters,
 - 20 negatives, selected from outside a range of 10 meters,

For the purpose of this work, 4 pre-defined triplet data where generated  and stored in the pickled files.
The files names incode information regarding the selection process of the data. 
E.g., the file "ground_truth_ar0.1m_nr10m_pr2m.pkle" comprises anchors (ar) are sperated by at least 0.1m.
the negatives where generated from outside a range of 10m and the positive was selected whithin a range of 2m.

The four predefined triplet data files are the following:
 - ground_truth_ar0.1m_nr10m_pr2m.pkle
 - ground_truth_ar0.5m_nr10m_pr2m.pkle
 - ground_truth_ar1m_nr10m_pr2m.pkle
 - ground_truth_ar5m_nr10m_pr2m.pkle

## Number of anchor positive pairs (ie loops):
The following information represents the number of loops that exist for each ground truth file. The number of loops also correspond to the number of training samples in each file. 

Orchards/aut22:\
AR5.0m: 45
AR1.0m: 252
AR0.5m: 495
AR0.1m: 1643

Orchards/sum22:
AR5.0m: 10
AR1.0m: 53
AR0.5m: 100
AR0.1m: 391

Orchards/june23:
AR5.0m: 54
AR1.0m: 268
AR0.5m: 512
AR0.1m: 1982

Strawberry/june23:
AR5.0m: 66
AR1.0m: 311
AR0.5m: 598
AR0.1m: 1942

greenhouse/e3:
AR5.0m: 14
AR1.0m: 73
AR0.5m: 117
AR0.1m: 242