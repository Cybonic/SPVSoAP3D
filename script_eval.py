
import os


# Define the path to the checkpoints
chkpt_root = '/home/tiago/workspace/SPCoV/predictions/iros24/' # Path to the checkpoints or descriptors
save_path  = 'predictions/iros24' # Path to save the predictions
experiment = 'cross_validation'
resume     = "checkpoints.pth" # choise [checkpoints.pth, descriptors.torch]
model      = "SPVSoAP3D"
  
test_sequences = ['GTJ23','OJ22','OJ23','ON22','ON23','SJ23']

for seq in test_sequences:
        func_arg = [ 
                f'--val_set {seq}',
                f'--network {model}', # Network
                f'--resume {chkpt_root}/{model}/{seq}/{resume}'
                '--memory DISK', # [DISK, RAM] 
                '--device cuda', # Device
                '--eval_roi_window 100', # Evaluation ROI window
                f'--save_predictions {os.path.join(save_path,experiment)}', # Save predictions
                f'-e {experiment}', # Experiment
                ' --roi 0 --augmentation 0 --pcl_norm 0' # Preprocessing
                ]
        
        func_arg_str = ' '.join(func_arg)
        os.system('python3 eval_knn.py ' + func_arg_str)