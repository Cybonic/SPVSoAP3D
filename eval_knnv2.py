#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/


'''

Scancontext requires Batch size of 1
 
'''
import argparse
import yaml
import os
import torch 

#from networks.orchnet import *
from trainer import Trainer
from networks import contrastive
from utils import loss as losses
from pipeline_factory import model_handler,dataloader_handler
import numpy as np

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    
    parser.add_argument(
        '--network', '-m',
        type=str,
        required=False,
        default='scancontext',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=False,
        default=f'iros24/cross_validation-nonorm-10m-aug-noroi',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        required=False,
        default='best_model',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--memory',
        type=str,
        required=False,
        default='DISK',
        choices=['DISK','RAM'],
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='uk', # uk
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        required=False,
        default=22000, #  Max size (based on the negatives)
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--loss',
        type=str,
        required=False,
        default = 'LazyTripletLoss',
        choices=['LazyTripletLoss','LazyQuadrupletLoss','PositiveLoss'],
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--max_points',
        type=int,
        required=False,
        default = 10000,
        help='sampling points.'
    )
    parser.add_argument(
        '--feat_dim',
        type=int,
        required=False,
        default = 1024,
        help='sampling points.'
    )
    parser.add_argument(
        '--modality',
        type=str,
        required=False,
        default = "pcl",
        help='sampling points.'
    )

    parser.add_argument(
        '--triplet_file',
        type=str,
        required=False,
        default = "triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl",
        help='sampling points.'
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        required=False,
        default = "eval/ground_truth_loop_range_10m.pkl",
        help='sampling points.'
    )

    parser.add_argument(
        '--monitor_loop_range',
        type=float,
        required=False,
        default = 10,
        help='sampling points.'
    )

    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default = 'uk/orchards/sum22/extracted',
    )

    parser.add_argument(
        '--roi',
        type=float,
        required=False,
        default = 0,
    )
    parser.add_argument(
        '--model_evaluation',
        type=str,
        required=False,
        default = "cross_validation",
        choices = ["cross_validation"]
    )
    parser.add_argument(
        '--chkpt_root',
        type=str,
        required=False,
        default = "None"
    )
    
    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default = "ukfrpt",
    )
    
    parser.add_argument(
        '--eval_roi_window',
        type=float,
        required=False,
        default = 100,
    )
    
    parser.add_argument(
        '--eval_warmup_window',
        type=float,
        required=False,
        default = 100,
    )
    
    parser.add_argument(
        '--eval_protocol',
        type=str,
        required=False,
        choices=['place','relocalization'],
        default = 'place',
    )
    
    parser.add_argument(
        '--save_predictions',
        type=str,
        required=False,
        default = 'saved_model_data',
    )


    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    SESSION['save_predictions'] = FLAGS.save_predictions
    
    if FLAGS.chkpt_root != "None":
        SESSION['trainer']['save_dir'] =  FLAGS.chkpt_root
    # Update config file with new settings
    SESSION['experiment'] = FLAGS.experiment
    SESSION['modelwrapper']['minibatch_size']  = FLAGS.mini_batch_size
    SESSION['modelwrapper']['feat_dim']  = FLAGS.feat_dim
    
    # Define evaluation mode: cross_validation or split
    SESSION['model_evaluation'] = FLAGS.model_evaluation
    SESSION['train_loader']['sequence'] = SESSION['cross_validation'][FLAGS.val_set]
    SESSION['val_loader']['sequence']   = [FLAGS.val_set]
   
    SESSION['val_loader']['batch_size'] = FLAGS.batch_size
    SESSION['train_loader']['triplet_file'] = FLAGS.triplet_file
    SESSION['val_loader']['ground_truth_file'] = FLAGS.eval_file
    SESSION['val_loader']['augmentation'] = False
    SESSION['val_loader']['roi'] = FLAGS.roi
    SESSION['loss']['type'] = FLAGS.loss
    SESSION['max_points'] = FLAGS.max_points
    SESSION['memory']     = FLAGS.memory
    SESSION['loop_range'] = FLAGS.monitor_loop_range
    SESSION['eval_roi_window'] = FLAGS.eval_roi_window
    


    print("----------")
    print("Saving Predictions: %s"%FLAGS.save_predictions)
    print("\n======= VAL LOADER =======")
    print("Sequence : ", SESSION['val_loader']['sequence'])
    print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
    print("Max Points: " + str(SESSION['max_points']))
    print("\n========== MODEL =========")
    print("Backbone : ", FLAGS.network)
    print("Resume: ",  FLAGS.resume )
    print("Loss: ",FLAGS.loss)
    print("MiniBatch Size: ", str(SESSION['modelwrapper']['minibatch_size']))
    print("\n==========================")
    print(f'Eval Protocal: {FLAGS.eval_protocol}')
    print(f'Memory: {FLAGS.memory}')
    print(f'Device: {FLAGS.device}')
    print("Loss: %s" %(SESSION['loss']['type']))
    print("Experiment: %s" %(FLAGS.experiment))
    print("----------\n")

    # For repeatability
    torch.manual_seed(0)
    np.random.seed(0)


    ###################################################################### 

    # The development has been made on different PCs, each has some custom settings
    # e.g the root path to the dataset;
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))

    root_dir = pc_config[device_name]
    
    # Build the model and the loader
    model = model_handler(FLAGS.network,
                            num_points = SESSION['max_points'],
                            output_dim = 256,
                            feat_dim   = FLAGS.feat_dim,
                            device = FLAGS.device,
                            loss = SESSION['loss'], # Loss is required to build the model name correctly (change this in the future)
                            modelwrapper = SESSION['modelwrapper']
                            )

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)


    loader = dataloader_handler(root_dir,FLAGS.network,FLAGS.dataset,SESSION, roi = FLAGS.roi, pcl_norm = False)

    run_name = {'dataset': '-'.join(str(SESSION['val_loader']['sequence'][0]).split('/')),
                'experiment':os.path.join(FLAGS.experiment,FLAGS.triplet_file,str(FLAGS.max_points)), 
                'model': str(model)
            }

    trainer = Trainer(
            model        = model,
            train_loader = None,#loader.get_train_loader(),
            val_loader   = loader.get_val_loader(),
            resume = FLAGS.resume,
            monitor_range = FLAGS.monitor_loop_range,
            config = SESSION,
            device = FLAGS.device,
            run_name = run_name,
            train_epoch_zero = False,
            eval_protocol='place',
            window_roi=  SESSION['eval_roi_window']
            )
    
    loop_range = list(range(0,120,1))
    
    best_model_filename = trainer.save_best_model_filename 
    # Generate descriptors, predictions and performance for the best weights
    print(f'\nLoading best model: {best_model_filename}\n')
    trainer.eval_approach.load_pretrained_model(best_model_filename)
    #loop_range = [1,5,10,15,20,500]
    #os.makedirs(save_to,exist_ok=True)
    #trainer.eval_approach.load_descriptors(load_from)
    trainer.eval_approach.run(loop_range=loop_range)
    
    save_to = FLAGS.save_predictions
    trainer.eval_approach.save_params(save_to)
    trainer.eval_approach.save_descriptors(save_to)
    trainer.eval_approach.save_predictions_cv(save_to)
    trainer.eval_approach.save_results_cv(save_to)
