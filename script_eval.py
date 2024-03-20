
import os


# Define the path to the checkpoints
chkpt_root = '/home/tiago/workspace/SPCoV/checkpoints/iros24_spvsoap_checkpoints'

save_path = '~/workspace/SPCoV/predictions'

test_sequences = [
        f'--val_set GTJ23  --resume {chkpt_root}/gtj23-spvsoap3d.pth',
        f'--val_set OJ22   --resume {chkpt_root}/oj22-spvsoap3d.pth',
        f'--val_set OJ23   --resume {chkpt_root}/oj23-spvsoap3d.pth',
        f'--val_set ON22   --resume {chkpt_root}/on22-spvsoap3d.pth',
        f'--val_set ON23   --resume {chkpt_root}/on23-spvsoap3d.pth',
        f'--val_set SJ23   --resume {chkpt_root}/sj23-spvsoap3d.pth', 
]

experiment = 'iros24/cross_validation'

for seq in test_sequences:
        func_arg = [ 
                seq,
                f'--network SPVSoAP3D', # Network
                '--memory DISK', # [DISK, RAM] 
                '--device cpu', # Device
                '--eval_roi_window 100', # Evaluation ROI window
                #f'--chkpt_root {chkpt_root}', # Checkpoint root
                f'--save_predictions {os.path.join(save_path,experiment)}', # Save predictions
                f'-e {experiment}', # Experiment
                ' --roi 0 --augmentation 0 --pcl_norm 0' # Preprocessing
                ]
        
        func_arg_str = ' '.join(func_arg)
        os.system('python3 eval_knn.py ' + func_arg_str)