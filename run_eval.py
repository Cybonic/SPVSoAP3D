
import os

chkpt_root = '~/workspace/SPCoV/checkpoints'

save_path = '~/workspace/SPCoV/predictions'
evaluation_type = "cross_validation"

experiments = [f'iros24/{evaluation_type}-nonorm-10m-aug-noroi',
              #f'iros24/{evaluation_type}-nonorm-10m-aug'     
]

input_preprocessings = [' --roi 0 --augmentation 1 --pcl_norm 0',
                       #' --roi 30 --augmentation 1 --pcl_norm 0'
]

args = [
        f'--network scancontext',
        #f'--network SPCov3D',
        #f'--network PointNetMAC',
        #f'--network PointNetSPoC',
        #f'--network overlap_transformer --modality bev',
        #f'--network LOGG3D',
]

losses = ['LazyTripletLoss']

density = ['10000']

test_sequences = [
        '--val_set uk/orchards/sum22/extracted',
        '--val_set uk/orchards/june23/extracted' ,
        '--val_set uk/orchards/aut22/extracted',
        '--val_set uk/strawberry/june23/extracted',
        '--val_set greenhouse/e3/extracted', 
        '--val_set GEORGIA-FR/husky/orchards/10nov23/00/submaps',  
]


for input_preprocessing,experiment in zip(input_preprocessings,experiments):
        for seq in test_sequences:
                for arg in args:
                        func_arg = [arg,
                                seq,
                                f'--model_evaluation {evaluation_type}',
                                '--memory RAM',
                                '--device cuda',
                                '--eval_roi_window 100',
                                '--resume best_model',
                                f'--chkpt_root {chkpt_root}',
                                f'--save_predictions {os.path.join(save_path,experiment)}',
                                f'-e {experiment}',
                                input_preprocessing
                                ]
                
                        func_arg_str = ' '.join(func_arg)
                        #print(func_arg_str + '\n')
                        os.system('python3 eval_knnv2.py ' + func_arg_str)