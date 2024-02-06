
import os

full_cap = '--epoch 50'
args = [
        '--network PointNetCov3DC',
        #'--network PointNetVLAD',
        #'--network LOGG3D',
        #'--network SPCov3D',
        #'--network PointNetMAC',
        #'--network PointNetGeM',
        #'--network overlap_transformer --modality bev',

        #'--network PointNetORCHNet',
        #'--network ResNet50ORCHNet --modality bev'
        #'--network ResNet50ORCHNetMaxPooling --modality bev',
        #'--network ResNet50GeM --modality bev',
        #' --network overlap_transformer',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model MuHA_resnet50',
]

#losses = ['PositiveLoss','LazyTripletLoss','LazyQuadrupletLoss']
#losses = ['LazyTripletLoss','LazyQuadrupletLoss']
losses = ['LazyTripletLoss']

density = ['10000']

evaluation_type = "cross_validation"
experiment      = f'-e iros24/{evaluation_type}-nonorm-10m-aug-noroi'
input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1 --pcl_norm 0'

resume  = '--resume best_model'

test_sequrnces = [
        '--val_set GEORGIA-FR/husky/orchards/10nov23/00/submaps',
        '--val_set uk/orchards/aut22/extracted',
        '--val_set uk/strawberry/june23/extracted',
        '--val_set greenhouse/e3/extracted', 
        '--val_set uk/orchards/sum22/extracted',
        '--val_set uk/orchards/june23/extracted'
]

for seq in test_sequrnces:
        for arg in args:
                func_arg = [arg,
                            f'--model_evaluation {evaluation_type}',
                            '--memory RAM',
                            '--device cuda',
                            seq,
                            experiment,
                            full_cap,
                            resume,
                            input_preprocessing
                ]
                
                func_arg_str = ' '.join(func_arg)        
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg_str)