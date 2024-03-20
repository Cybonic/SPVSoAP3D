
import os
import subprocess

full_cap = '--epoch 20'
args = [
        '--network SPVSoAP3D',
]       

losses = ['LazyTripletLoss']

density = ['10000']

evaluation_type = "cross_validation"
experiment      = f'-e iros24/{evaluation_type}'
input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1 --pcl_norm 0'


chk_dir = '~/workspace/SPCoV/checkpoints/iros24_published/sj23-spvsoap3d.pth'

test_sequrnces = [
        #'--val_set GEORGIA-FR/husky/orchards/10nov23/00/submaps',
        f'--val_set uk/strawberry/june23/extracted --resume {chk_dir}/sj23-spvsoap3d.pth',
        #'--val_set greenhouse/e3/extracted', 
        #'--val_set uk/orchards/aut22/extracted',
        #'--val_set uk/orchards/sum22/extracted',
        #'--val_set uk/orchards/june23/extracted'
]

for seq in test_sequrnces:
        for arg in args:
                func_arg = [arg,
                            f'--model_evaluation {evaluation_type}',
                            '--memory DISK',
                            '--device cpu',
                            seq,
                            experiment,
                            full_cap,
                            #resume,
                            input_preprocessing
                ]
                
                func_arg_str = ' '.join(func_arg)        
                
                os.system('python3 train_knn.py ' + func_arg_str)