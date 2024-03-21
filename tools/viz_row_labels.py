import os, sys
import numpy as np
import argparse

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


from utils.viz import myplot
from dataloader.utils import extract_points_in_rectangle_roi
import yaml

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def load_row_bboxs(seq_name):
    from dataloader import row_segments
    seq_structure = seq_name.split('/')
    # remove string 'extracted'
    if 'extracted' in seq_structure:
        seq_structure.remove('extracted')
    else:
        print("[ERR] Sequence name does not contain 'extracted'")
        seq_structure = seq_structure[:-1]
    seq_name = '_'.join(seq_structure)
    #seq = seq.replace('/','_')
    print("[INF] Loading row segments from: %s"% seq_name)
    segment =  row_segments.__dict__[seq_name]
    return(np.array(segment['rows']))

def generate_labels(seq,poses):
    # Load row ids
    rectangle_rois = load_row_bboxs(seq)
    # Load aline rotation
    labels  = extract_points_in_rectangle_roi(poses,rectangle_rois)
    
    return(labels,poses)


def viz_overlap(xy, loops, record_gif= False, file_name = 'anchor_positive_pair.gif',frame_jumps=50):


    mplot = myplot(delay=0.2)
    mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'whitesmoke')
    mplot.xlabel('m'), mplot.ylabel('m')
        
    if record_gif == True:
        mplot.record_gif(file_name)
    
    indices = np.array(range(xy.shape[0]-1))
    ROI = indices[2:]
    positives  = []
    anchors    = []
    pos_distro = []
    n_points   = ROI.shape[0]

    color = np.array(['y' for ii in range(0,n_points)])
    scale = np.array([10 for ii in range(0,n_points)])
    
    for i in tqdm(ROI,desc='Vizualizing Loop Closure'):
        idx = loops[i]        
        
        positives_idx = np.where(idx>0)[0]
  
        if len(positives_idx)>0:
            positives.extend(positives_idx)
            anchors.append (i)
            pos_distro.append(positives_idx)
        
        if i % frame_jumps != 0:
            continue

        color[:i]= 'k'
        scale[:i]= 10
        # Colorize the N smallest samples
        color[anchors]   = 'b'
        color[positives] = 'g'

        scale[positives] = 100

        x = xy[:i,1]
        y = xy[:i,0]

        mplot.update_plot(y,x,offset=2,color=color,zoom=0,scale=scale) 

    return(pos_distro)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/DATASETS_TO_NAS')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'HORTO',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "GTJ23",type = str)
    parser.add_argument('--show',default  = True ,type = bool)
    parser.add_argument('--pose_data_source',default  = "positions" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = True ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    parser.add_argument('--save_data',default  = False ,type = bool,
                        help='save evaluation data to a pickle file')
    parser.add_argument('--load_labels',default  = True ,type = bool,
                        help='load labels from a pickle file')
    

    args = parser.parse_args()


    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    show    = args.show

    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )
    print("[INF] Plotting Flag:   " + str(show))
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    

    print("[INF] Root directory: %s\n"% root)

    dir_path = os.path.join(root,dataset,seq,'extracted')
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path
    
    print("[INF] Loading data from directory: %s\n" % dir_path)

    # Create Save Directory
    save_root_dir  = os.path.join(root,dataset,seq,"tempv2")
    if args.debug_mode:
        save_root_dir = os.path.join('temp',dataset,seq)
    
    os.makedirs(save_root_dir,exist_ok=True)
    print("[INF] Saving data to directory: %s\n" % save_root_dir)

    # Loading DATA
    from dataloader.horto3dlm.dataset import load_positions
   
    
    assert args.pose_data_source in ['gps','poses','positions'], "Invalid pose data source"
    
    
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    
    
    poses = load_positions(pose_file)

    print("[INF] Reading poses from: %s"% pose_file)
    print("[INF] Number of poses: %d"% poses.shape[0])
    
    # ========================================
    # Load Row Labels
    if args.load_labels:
        labels = []
        row_label_file = os.path.join(dir_path,'point_row_labels.pkl')
        assert os.path.isfile(row_label_file), "Row label file does not exist " + row_label_file
        with open(row_label_file, 'rb') as f:
            labels = pickle.load(f)
    else:
        labels,poses = generate_labels(seq,poses)
        
    unique_labels = np.unique(labels)
    n_points      = poses.shape[0]

    # Generate Ground-truth Table
    if args.save_data:
        file = os.path.join(save_root_dir,'point_row_labels.pkl')
         # save the numpy arrays to a file using pickle
        with open(file, 'wb') as f:
            pickle.dump(labels, f)
    
        print("[INF] saved ground truth at:" + file)

    # ========================================
    # ========================================

    if show:
        print("[INF] Plotting data...")
        import matplotlib.colors as mcolors
        #color pallet based on the number of unique labels
        #color_pallet = np.array(['y','b','g','r','c','m','k','w'])
        color_pallet = np.array(['yellow','blue','green','red','cyan','magenta','pink','peru'])
        colors = np.linspace(0, 1, 8)[::-1]
        #color_pallet = np.array([ mcolors.CSS4_COLORS[v] for v in color_pallet])
        # colors with  color=plt.cm.viridis(i / 5.0)
        color_pallet = np.array([plt.cm.tab20c(colors[i]) for i in range(0,unique_labels.shape[0])])
        color = color_pallet[labels]
        
        # print color hex and label
        for i in range(0,colors.shape[0]):
            hex_code = mcolors.to_hex(np.array([plt.cm.viridis(colors[i])]))
            print(hex_code,i)
        
        point_color = np.array([color_pallet[labels[ii]] for ii in range(0,n_points)])
        # Plot the data
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(poses[:,0],poses[:,1],s=10,c=point_color)
        ax.set_aspect('equal')
        plt.savefig(os.path.join(save_root_dir,'point_row_labels.png'))
        plt.savefig('point_row_labels.png')
        print("[INF] Saved figure to: %s"% os.path.join(save_root_dir,'point_row_labels.png'))
        print("[INF] Saved figure to: %s"% 'point_row_labels.png')
        plt.show()


    

