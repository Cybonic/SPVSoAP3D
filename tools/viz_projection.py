#!/usr/bin/env python3

from tqdm import tqdm
import argparse
import numpy as np


import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio
from PIL import Image,ImageOps
import os, sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))



from tools.utils_dataset import lidar_loader

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/deep/Dropbox/SHARE/DATASET')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'greenhouse',
                                    type=str,
                                    help=' dataset root directory .'
                                    )
    parser.add_argument('--seq',    
                                default  = 'GEORGIA-FR/husky/orchards/10nov23/00/submaps',
                                type = str)
    parser.add_argument('--plot',default  = True ,type = bool)
    parser.add_argument('--loop_thresh',default  = 1 ,type = float)
    parser.add_argument('--record_gif',default  = False ,type = bool)
    parser.add_argument('--option',default  = 'compt' ,type = str,choices=['viz','compt'])
    parser.add_argument('--pose_file',default  = 'gps' ,type = str)
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot
    record_gif_flag = args.record_gif
    option = args.option
    loop_thresh = args.loop_thresh

    modality = "bev"

    save_root = os.path.join('log',dataset,seq,'viz')
    os.makedirs(save_root,exist_ok=True)
    fig = Figure(figsize=(5, 4), dpi=25,)
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(1, 1)
    
    filename = 'projection.gif'
    canvas = FigureCanvasAgg(fig)
    writer = imageio.get_writer(filename, mode='I')
        
    loader = lidar_loader(root,seq,modality_name = modality,num_points=10000,square_roi=[{'xmin':-30,'xmax':30,'ymin':-30,'ymax':30,'zmax':30}],position_file=args.pose_file)
    
    n_samples = len(loader)
    
    for i in tqdm(range(0,n_samples,1)):
        
        bev = loader.load_bev(i)
        height = bev['height']
        img = np.concatenate([bev['height'],bev['height'],bev['height']],axis = 2)
        
        sp = loader.load_spherical(i,Width=1024,Height=128,fov_up=30.6,fov_down=-30.6,max_depth=30,max_rem=1)
        img = np.concatenate([sp['range'],sp['range'],sp['range']],axis = 2)
        
        pil_range = Image.fromarray(img.astype(np.uint8))
        
        pil_range.save(os.path.join(save_root,f'{i}.png'))
        #pil_range = ImageOps.colorize(pil_range, black="white", white="black")
        #plt.savefig(buffer, format='png')
        X = np.asarray(pil_range)
        writer.append_data(X)

  #loader = OrchardDataset(root,'',sequence,sync = True,modality = 'bev',square_roi = [{'xmin':-15,'xmax':15,'ymin':-15,'ymax':15,'zmax':1}]) #cylinder_roi=[{'rmax':10}])

  






  
  