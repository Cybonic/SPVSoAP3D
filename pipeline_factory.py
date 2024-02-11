
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan
from dataloader.kitti.kitti import cross_validation,split


from networks.pipelines.PointNetVLAD import PointNetVLAD
from networks.pipelines.LOGG3D import LOGG3D
from networks.pipelines import ORCHNet
from networks.pipelines.GeMNet import PointNetGeM,ResNet50GeM
from networks.pipelines.overlap_transformer import featureExtracter
from networks.pipelines.SPoCNet import PointNetSPoC,ResNet50SPoC,PointNetAP,PointNetCGAP
from networks.pipelines.MACNet import PointNetMAC,ResNet50MAC 
from networks.pipelines.PointNetSOP import PointNetSOP
from networks.pipelines.SPCOVP import SPCov3D,PointNetCov3D, PointNetCovTroch3DC,PointNetCov3DC,SPGAP,PointNetPCACov3DC,SPCov3Dx
from networks.scancontext.scancontext import SCANCONTEXT
#from networks.pipelines.Steerable import SO3MLP
#from networks.pipelines.Steerable import SO3MLP
import yaml

from utils import loss as losses
from networks import contrastive

# ==================================================================================================
MODELS = ['LOGG3D',
          'PointNetVLAD',
          'ORCHNet_PointNet',
          'ORCHNet_ResNet50',
          'overlap_transformer',
          'ORCHNet']

# ==================================================================================================
# ======================================== PIPELINE FACTORY ========================================
# ==================================================================================================

def model_handler(pipeline_name, num_points=4096,output_dim=256,feat_dim=1024,device='cuda',**argv):
    """
    This function returns the model 
    
    Parmeters:
    ----------
    pipeline_name: str
        Name of the pipeline to be used
    num_points: int
        Number of points to be used as input
    output_dim: int
        Dimension of the output feature vector
    feat_dim: int
        Dimension of the hidden feature vector

    Returns:
    --------
    pipeline: object
        Pipeline object
    """
    
    print("\n**************************************************")
    print(f"Model: {pipeline_name}")
    print(f"N.points: {num_points}")
    print(f"Dpts: {output_dim}")
    print(f"Feat Dim: {feat_dim}")
    print("**************************************************\n")

    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name == 'SPGAP':
        pipeline = SPGAP(output_dim=output_dim)
    elif pipeline_name == 'PointNetCGAP':
        pipeline = PointNetCGAP(num_c = 7,feat_dim = 1024,use_tnet=False,output_dim=output_dim,pooling='max')
    elif pipeline_name == 'PointNetCov3D':
        pipeline = PointNetCov3D(output_dim=output_dim, feat_dim = 1024)
    elif pipeline_name == 'PointNetCovTorch3D':
        pipeline = PointNetCovTroch3DC(output_dim=output_dim, feat_dim = 1024)
    elif pipeline_name == 'PointNetCov3DC':
        pipeline = PointNetCov3DC(output_dim=output_dim, feat_dim = 512)
    elif pipeline_name == 'PointNetPCACov3DC':
        pipeline = PointNetPCACov3DC(output_dim=output_dim, feat_dim = 512)
    elif pipeline_name == 'SPCov3D':
        pipeline = SPCov3D(output_dim=output_dim,
                           local_feat_dim=16,
                           do_fc = True,
                           do_pe = False,
                           do_dm = True,
                           pres=0.1,
                           vres=0.1,
                           pooling = 'layer_cov')
    elif pipeline_name == 'SPCov3Dx':
        pipeline = SPCov3Dx(output_dim=output_dim,
                           local_feat_dim=16,
                           do_fc = True,
                           do_pe = True,
                           do_dm = True,
                           pres=0.1,
                           vres=0.1,
                           pooling = 'layer_cov')
    elif pipeline_name == 'PointNetAP':
        pipeline = PointNetAP(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == 'PointNetSOP':
        pipeline = PointNetSOP(output_dim=output_dim, num_points = num_points, feat_dim = 16)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD(use_tnet=True, output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "PointNetGeM":
        pipeline = PointNetGeM(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50GeM": 
        pipeline = ResNet50GeM(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == "PointNetSPoC":
        pipeline = PointNetSPoC(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50SPoC":    
        pipeline = ResNet50SPoC(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == "PointNetMAC":
        pipeline = PointNetMAC(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50MAC":
        pipeline = ResNet50MAC(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == 'overlap_transformer':
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)
    elif pipeline_name == "scancontext":
        pipeline = SCANCONTEXT(max_length=80, ring_res=20, sector_res=60,lidar_height=2.0)
    else:
        raise NotImplementedError("Network not implemented!")

    loss = None
    if 'loss' in argv:
        loss_type  = argv['loss']['type']
        loss_param = argv['loss']['args']

        loss = losses.__dict__[loss_type](**loss_param,device = device)

    print("*"*30)
    print(f'Loss: {loss}')
    print("*"*30)

    descriptor = {
            'in_dim':2*256,
            'kernels':[512,256],
            'representation':'descriptor'
        }
    
    if pipeline_name in ['PointNetCGAP','PointNetCov3DC','PointNetPCACov3DC']:
        
        features = {
            'in_dim':2*16,
            'kernels':[32,16],
            'representation':'features'
        }
        
        model = contrastive.ModelWrapperLoss(pipeline,
                                             loss =loss,
                                             device = device, 
                                             margin = 0.5,
                                             class_loss_on = True,
                                             representation = 'descriptor',
                                             pooling = 'max',
                                             **argv['modelwrapper'])

    elif pipeline_name in ['SPCov3D','SPGAP'] or pipeline_name.startswith("SPCov3D"):
        #model_name,lossname = pipeline_name.split("_")
        
        features_l1 = { # features are pooled from the L1 layer
            'in_dim':2*163,
            'kernels':[163,64],
            'representation':'features'
        }
        
        model = contrastive.SparseModelWrapperLoss(pipeline, 
                                               loss = loss,
                                               device = device,
                                               aux_loss_on = None, # None 'pairloss' or 'segmentloss'
                                               class_loss_margin = 0.1, 
                                               pooling = 'max',
                                               **features_l1,
                                               **argv['modelwrapper'])
    
    elif pipeline_name in ['LOGG3D']:
        model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['modelwrapper'])
    elif pipeline_name != "scancontext":
        model = contrastive.ModelWrapper(pipeline,loss =loss,device = device, **argv['modelwrapper'])
    else: 
        model = pipeline

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)

    return model

# ==================================================================================================
# ======================================== DATALOADER FACTORY ======================================
# ==================================================================================================

def dataloader_handler(root_dir,network,dataset,session,pcl_norm=False,**args):

    assert dataset in ['kitti','orchard-uk','uk','GreenHouse','greenhouse'],'Dataset Name does not exist!'

    sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)

    roi = None
    if 'roi' in args and args['roi'] > 0:
        roi = {}
        #sensor_pram = sensor_pram[dataset]
        #roi = sensor_pram['square_roi']
        print(f"\nROI: {args['roi']}\n")
        roi['xmin'] = -args['roi']
        roi['xmax'] = args['roi']
        roi['ymin'] = -args['roi']
        roi['ymax'] = args['roi']

    if network in ['ResNet50_ORCHNet','overlap_transformer',"ResNet50GeM"] or network.startswith("ResNet50"):
        # These networks use proxy representation to encode the point clouds
        
        if session['modality'] == "bev" or network == "overlap_transformer":
            #sensor_pram = sensor_pram[dataset]
            #bev_pram = sensor_pram['BEV']
            modality = BEVProjection(width=256,height=256,square_roi=roi)
        elif session['modality'] == "spherical" or network != "overlap_transformer":
            modality = SphericalProjection(256,256,square_roi=roi)
            
    elif network in ['LOGG3D','SPGAP'] or network.startswith("SPCov3D"):
        # Get sparse (voxelized) point cloud based modality
        num_points=session['max_points']
        modality = SparseLaserScan(voxel_size=0.1,max_points=num_points, pcl_norm = pcl_norm)
    
    elif network in ['PointNetVLAD','PointNet_ORCHNet',"PointNetGeM","scancontext"] or network.startswith("PointNet"):
        
        # Get point cloud based modality
        num_points = session['max_points']
        modality = Scan(max_points=num_points,square_roi=roi, pcl_norm = pcl_norm)
    else:
        raise NotImplementedError("Modality not implemented!")

    dataset = dataset.lower()

    # Select experiment type by default is cross_validation
    model_evaluation = "cross_validation" # Default

    if "model_evaluation" in session:
        model_evaluation = session['model_evaluation']

    print(f"\n[INFO]Model Evaluation: {model_evaluation}")

    if model_evaluation == "cross_validation":
        loader = cross_validation( root = root_dir,
                                    dataset = dataset,
                                    modality = modality,
                                    memory   = session['memory'],
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
        
    elif model_evaluation == "split":
        loader = split( root = root_dir,
                                    dataset = dataset,
                                    modality = modality,
                                    memory   = session['memory'],
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
    else:
        raise NotImplementedError("Model Evaluation not implemented!")

    return loader



if __name__=="__main__":
    import yaml,os
    
    dataset = 'kitti'
    session_cfg_file = os.path.join('sessions', dataset.lower() + '.yaml')
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    Model,dataloader = pipeline('LOGG3D','kitti',SESSION)
    print(Model)
    print(dataloader)
    
    #assert str(Model)=="LO"