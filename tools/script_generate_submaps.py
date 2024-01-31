#!/bin/python3


"""

ToDo:   

"""


import open3d as o3d
import numpy as np
import os,sys
import tqdm
import copy


# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from dataloader.laserscan import Scan
from dataloader.kitti import kitti_dataset
import logging
from datetime import date



def timestamp_match(source,reference,verbose=True):
    """
    Compute the timestamp match between query and reference

    Args:
        query (np.array): query timestamps
        reference (np.array): reference timestamps
        verbose (bool, optional): Print error statistics. Defaults to True.

    Returns:
        np.array: nearest indices of the reference array
    """
    from scipy.spatial.distance import cdist
    # point to point distance, where rows are the query points and columns are the reference points
    source = np.array(source)
    source = source.reshape(-1,1)
    reference = np.array(reference)
    reference = reference.reshape(-1,1)

    distances = cdist(source,reference, 'euclidean')
    
    nearest_indices = np.argmin(distances, axis=1)
    # Compute error
    if verbose:
        error = abs(source-reference[nearest_indices])
        print("Mean %f"%np.mean(error))
        print("STD %f"%np.std(error))
        print("MAX %f"%np.max(error))
        print("MIN %f"%np.min(error))
    
    src_idx = np.arange(len(source))
    return src_idx,nearest_indices


def save_pcd_kitti_format(file,data):
    # cast to float23
    data = data.astype(np.float32)
    pcl = open(file,'wb')
    data.tofile(pcl)
    pcl.close()


def np_to_pcd(pts):
    pcd = pts
    if isinstance(pts,np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def calculate_angle(x, y):
    angles = np.arctan2(y, x) * (180 / np.pi)
    if angles < 0:
        angles += 360
    return angles

def filter_points_by_angle(point_cloud, angle_range=(150, 180)):
    angles = np.apply_along_axis(lambda point: calculate_angle(point[0], point[1]), axis=1, arr=point_cloud)
    mask = np.logical_or(angles < angle_range[0], angles > angle_range[1])
    #filtered_point_cloud = point_cloud[mask]
    return mask


class lidar_loader():
    def __init__(self,root,sequence, num_points, kitti_format=True,filter_zeros=True,position_file="positions"):
        self.root = root
        self.kitti_format = kitti_format
        self.filter_zeros = filter_zeros
        # root,dataset,sequence,position_file="positions.txt",verbose=False
        dataloader = kitti_dataset.kittidataset(root,'',sequence,position_file=position_file,verbose=False)
        self.modality = Scan(max_points=num_points,
                        aug_flag=False,square_roi=None,pcl_norm = False)
        
        self.pcl_files,name = dataloader._get_point_cloud_file_()
        self.t_gps = dataloader._get_gps_timestamps_()
        self.t_pcl = dataloader._get_pcl_timestamps_()
        self.poses = dataloader._get_pose_()
        
        self.pcl_idx,self.gps_idx = timestamp_match(self.t_pcl,self.t_gps)
 
        print("Filtering zeros ",str(filter_zeros))

    def __getitem__(self, index):
        # Get the point cloud
        pcd,intensity = self.modality.load(self.pcl_files[index])
        if self.filter_zeros:
            r = np.linalg.norm(pcd[:,:3],axis=1)
            nonezero = r!=0.0
            pcd = pcd[nonezero,:]
            intensity = intensity[nonezero]
        
        mask = filter_points_by_angle(pcd, angle_range=(160, 200))
        pcd = pcd[mask]
        intensity = intensity[mask]
        return pcd,intensity
    
    def __len__(self):
        return len(self.pcl_files)

    def get_matching_indices(self):
        return self.gps_idx,self.pcl_idx
    
    
    
class registration():
    def __init__(self,max_iterations,tolerance,max_correspondence_distance,voxel_size,weighted_flag):
        self.max_iterations = max_iterations
        self.tolerance      = tolerance
        self.voxel_size     = voxel_size
        self.weighted_flag  = weighted_flag
        self.fitness_scores = []
        self.rmse_scores    = []
        self.max_correspondence_distance = max_correspondence_distance
    
    
    def preprocess_point_cloud(self,pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    
    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result.transformation

    def execute_fast_global_registration(self,source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    
    def point_to_point(self, source, target, init_transform):
        """
        This function is used to register point clouds using ICP based on Point to Point
        param: source: point cloud
        param: target: point cloud
        param: max_correspondence_distance: maximum correspondence distance
        param: max_iterations: maximum number of iterations
        param: init_transform: initial transformation
        return: transformation matrix
        """
        
        source_down, pcd_fpfh =self.preprocess_point_cloud(source, self.voxel_size)
        target_down, pcd_fpfh =self.preprocess_point_cloud(target, self.voxel_size)
            
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source_down, target_down, self.max_correspondence_distance, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))

        return icp_coarse.transformation


def gen_local_maps(loader,registration,dist,log_dir,submap_size=100000):
    from utils.viz import map_viz
    mapvis = map_viz(plot=False)
    
    num_pcds = len(loader)
    assert num_pcds>1, "At least two point clouds are required"

    pcd_dir = os.path.join(log_dir,'point_cloud')
    os.makedirs(pcd_dir,exist_ok=True)
    
    positions = loader.poses
    gps_idx_array,pcd_idx_array = loader.get_matching_indices()
    
    temp_positions = loader.t_gps 
    temp_pcd = loader.t_pcl
    
    local_map  = o3d.geometry.PointCloud()
    temp_file  = os.path.join("",'temp.pcd')
    
    new_pose_file = os.path.join(log_dir,'positions.txt')
    fd = open(new_pose_file,'w')
    fd_time_positions =  open(os.path.join(log_dir,'positions_timestamp.txt'),'w')
    fd_time_pcd =  open(os.path.join(log_dir,'point_cloudtimestamp.txt'),'w')
    
    global_tf = []
    
    itr = 0
    base_pose_idx =0
    base_pcd_idx = 0
    
    local_map_intensities = []
    num_points = []
    
    
    for i in tqdm.tqdm(range(0,num_pcds-1),"Computing Odometry"):
         # Get the point cloud
        pcd_idx = pcd_idx_array[i]
        pcd_np, intensity = loader[pcd_idx]
        
        # Get the point cloud and transform to open3d format
        source = np_to_pcd(pcd_np)

        # First iteration
        if len(local_map.points) == 0:
            # ADD FIRST POINT CLOUD TO THE MAP
            local_map += source
            global_tf = np.eye(4)
            mapvis.init(local_map)
            local_map_intensities = intensity.tolist()
            continue
        
        # accumulate intensities
        local_map_intensities.extend(intensity)
        
        # Get map 
        target = copy.deepcopy(local_map)
        # Transform source to global frame
        source = source.transform(global_tf)
    
        # Get the transformation
        delta_TF = registration.point_to_point(copy.deepcopy(source),target,global_tf)
        # Transform source to local frame
        source = source.transform(delta_TF)
        # Add to local map
        local_map += source
        # Update global transform
        global_tf=np.dot(delta_TF,global_tf)
        # Update visualization
        mapvis.update(local_map,source)
        
        
        if len(local_map.points) > submap_size:
            print("Using gps idx: ",base_pose_idx)
            # Save pose
            selected_pose = positions[base_pose_idx,:]
            fd.write("%f %f %f\n"%(selected_pose[0],selected_pose[1],selected_pose[2]))
            
            tp = temp_pcd[pcd_idx_array[base_pcd_idx]]
            print("temp pcd: ",tp)
            fd_time_pcd.write("%f\n"%temp_pcd[pcd_idx_array[base_pcd_idx]])
            
            tpp = temp_positions[base_pose_idx]
            print("temp pose: ",tpp)
            fd_time_positions.write("%f\n"%temp_positions[base_pose_idx])
            # SAVE PCD
            
            # concatenate intensities to the point cloud
            local_map_intensities = np.array(local_map_intensities)
            local_map_np = np.concatenate((np.array(local_map.points),local_map_intensities.reshape(-1,1)),axis=1)
            
            num_points.append(local_map_np.shape[0])
            print("Saving local map with %d points"%local_map_np.shape[0])
            file_name = os.path.join(pcd_dir,'{0:09d}.bin'.format(itr))
            save_pcd_kitti_format(file_name,local_map_np)
            # Reset local map
            mapvis.save(temp_file)
            local_map  = o3d.geometry.PointCloud()
            base_pose_idx = gps_idx_array[i+1]
            base_pcd_idx  = i+1
            global_tf = np.eye(4)
            itr+=1
    
    fd.close()
    fd_time_positions.close()
    fd_time_pcd.close()
            
    logging.info("************")
    logging.info("Finished")
    # save computed parameters
    logging.info("Number of submaps: %d"%itr)
    avg_points = np.mean(num_points)
    std_points = np.std(num_points)
    
    num_pts_str = "Average number of points per submap: %f +/- %f"%(avg_points,std_points)
    logging.info(num_pts_str)
    logging.info("************")



if __name__=="__main__":
    # Usage example
  
    parameters = {}

    parameters['root']          = "/home/deep/Dropbox/SHARE/DATASET"  # "/media/tiago/vbig/dataset/" 
    parameters['sequence']      = "GEORGIA-FR/husky/orchards/10nov23/00"
    parameters['max_iterations']= 100000
    parameters['tolerance']     = 0.0001
    parameters['corr_distance_threshold'] = 1000000000
    parameters['submap_size']   = 100000
    parameters['voxel_size']    = 0.1
    sequence = os.path.join(parameters['sequence'],'extracted')
    pcd_loader = lidar_loader(parameters['root'],sequence,-1,kitti_format=True,filter_zeros=True,position_file="gps")

    new_sequence = os.path.join(parameters['sequence'],'submaps_{0:03d}'.format(parameters['submap_size']))
    #root = os.path.join(parameters['experiment'],'open3d',new_sequence)

    log_dir  =  os.path.join(parameters['root'],new_sequence)
    os.makedirs(log_dir,exist_ok=True)
    # Create a logger
    # Name of the log file is the current time
    log_file  = os.path.join(log_dir,date.today().strftime("%Y-%m-%d_%H-%M-%S") + ".log")

    logger = logging.basicConfig(level = logging.INFO, filename = log_file)

    logging.info("....")
    logging.info("Parameters: ")
    logging.info(parameters)
    logging.info("....")
    

    
    parameter_file = os.path.join(log_dir,"parameters.json")
    # save parameters
    import json
    with open(parameter_file, 'w') as fp:
        json.dump(parameters, fp)
        

    registration_algo = registration(
                                    max_iterations=parameters['max_iterations'],
                                    tolerance= parameters['tolerance'],
                                    max_correspondence_distance=parameters['corr_distance_threshold'],
                                    voxel_size = parameters['voxel_size'],
                                    weighted_flag=False)

    gen_local_maps(pcd_loader,registration_algo,0.5,log_dir,parameters['submap_size'])
    

    
    
    
    

    