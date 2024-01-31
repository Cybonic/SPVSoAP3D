
import os,sys
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
    
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


from dataloader.laserscan import Scan
from dataloader.projections import  get_bev_proj,get_spherical_proj
from dataloader.kitti import kitti_dataset
from datetime import date



def timestamp_match(source, reference, verbose=True):
    """
    Compute the timestamp match between a source and reference array.

    Parameters:
        source (array-like): The source array containing timestamps.
        reference (array-like): The reference array containing timestamps.
        verbose (bool, optional): Whether to print the error statistics. Defaults to True.

    Returns:
        tuple: A tuple containing the source indices and the nearest indices in the reference array.
    """

    # point to point distance, where rows are the query points and columns are the reference points
    source = np.array(source)
    source = source.reshape(-1, 1)
    reference = np.array(reference)
    reference = reference.reshape(-1, 1)
    distances = cdist(source, reference, 'euclidean')

    nearest_indices = np.argmin(distances, axis=1)
    
    # Compute error
    if verbose:
        error = abs(source - reference[nearest_indices])
        print("Mean %f" % np.mean(error))
        print("STD %f" % np.std(error))
        print("MAX %f" % np.max(error))
        print("MIN %f" % np.min(error))
    
    src_idx = np.arange(len(source))
    return src_idx, nearest_indices



def save_pcd_kitti_format(file, data):
    """
    Save point cloud data in KITTI format to a file.

    Args:
        file (str): The path to the output file.
        data (numpy.ndarray): The point cloud data to be saved.

    Returns:
        None
    """
    # cast to float32
    data = data.astype(np.float32)
    pcl = open(file, 'wb')
    data.tofile(pcl)
    pcl.close()
    




def np_to_pcd(pts):
    """
    Convert a NumPy array of points to an Open3D PointCloud object.

    Args:
        pts (numpy.ndarray): Array of points.

    Returns:
        o3d.geometry.PointCloud: Open3D PointCloud object.
    """
    pcd = pts
    if isinstance(pts,np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd




def calculate_angle(x, y):
    """
    Calculates the angle in degrees between the positive x-axis and the vector (x, y).

    Parameters:
    x (float): The x-coordinate of the vector.
    y (float): The y-coordinate of the vector.

    Returns:
    float: The angle in degrees between the positive x-axis and the vector (x, y).
    """
    angles = np.arctan2(y, x) * (180 / np.pi)
    if angles < 0:
        angles += 360
    return angles





def filter_points_by_angle(point_cloud, angle_range=(150, 180)):
    """
    Filters the points in a point cloud based on their angle with respect to the origin.

    Args:
        point_cloud (numpy.ndarray): The input point cloud.
        angle_range (tuple): The range of angles (in degrees) within which the points should be filtered.
            Defaults to (150, 180).

    Returns:
        numpy.ndarray: A boolean mask indicating which points pass the angle filter.
    """
    angles = np.apply_along_axis(lambda point: calculate_angle(point[0], point[1]), axis=1, arr=point_cloud)
    mask = np.logical_or(angles < angle_range[0], angles > angle_range[1])
    return mask




class lidar_loader():
    def __init__(self,root,sequence, num_points, modality_name= "scan", kitti_format=True,filter_zeros=True,position_file="positions",square_roi=None):
        self.root = root
        self.kitti_format = kitti_format
        self.filter_zeros = filter_zeros
        # root,dataset,sequence,position_file="positions.txt",verbose=False
        dataloader = kitti_dataset.kittidataset(root,'',sequence,position_file=position_file,verbose=False)
        
        self.pcl_files,name = dataloader._get_point_cloud_file_()
        self.t_gps = dataloader._get_gps_timestamps_()
        self.t_pcl = dataloader._get_pcl_timestamps_()
        self.poses = dataloader._get_pose_()
        
        
        
        self.modality_name = modality_name
        #if modality == "scan":
        self.modality = Scan(max_points=num_points,
                        aug_flag=False,square_roi=square_roi,pcl_norm = False)
        
        
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
    
    
    def load_bev(self, index,Width=512,Height=512,fov_up=3.0,fov_down=-25.0):
        # Get the point cloud
        points,intensities = self.__getitem__(index)
        bev = get_bev_proj(points,intensities,Width=Width,Height=Height)
        return bev

    def load_spherical(self, index,Width=1024,Height=64,fov_up=-22,fov_down=-22.0,max_depth=30,max_rem=1):
        # Get the point cloud
        points,intensities = self.__getitem__(index)
        sp = get_spherical_proj(points,intensities,Width=Width,Height=Height,fov_up=fov_up,fov_down=fov_down,max_depth=max_depth,max_rem=max_rem)
        return sp