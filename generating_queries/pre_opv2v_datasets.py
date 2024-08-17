import re
import yaml
import os
import math
import random
from collections import OrderedDict
import open3d as o3d
import cv2
import lidar_projection
from ground_removal import Processor
from scipy.spatial.transform import Rotation as R
import numpy as np
import csv

def load_yaml(file):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


class BevPreprocessor:
    def __init__(self, preprocess_params):
        self.lidar_range = preprocess_params['cav_lidar_range']
        self.geometry_param = preprocess_params["geometry_param"]

    def preprocess(self, pcd_cloud):
        """
        Preprocess the lidar points to BEV representations.

        Parameters
        ----------
        pcd_cloud : open3d.geometry.PointCloud
            The raw lidar point cloud.

        Returns
        -------
        data_dict : the structured output dictionary.
        """
        bev = np.zeros(self.geometry_param['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((bev.shape[0], bev.shape[1]), dtype=int)
        bev_origin = np.array(
            [self.geometry_param["L1"], self.geometry_param["W1"], self.geometry_param["H1"]]).reshape(1, -1)

        points = np.asarray(pcd_cloud.points)
        intensities = np.ones(points.shape[0]) if not pcd_cloud.has_colors() else np.asarray(pcd_cloud.colors)[:,
                                                                                  0]  # Use colors as intensities if available

        indices = ((points[:, :3] - bev_origin) / self.geometry_param["res"]).astype(int)

        valid_indices_mask = (indices[:, 0] >= 0) & (indices[:, 0] < bev.shape[0]) & \
                             (indices[:, 1] >= 0) & (indices[:, 1] < bev.shape[1])
        valid_indices = indices[valid_indices_mask]
        valid_intensities = intensities[valid_indices_mask]
        valid_heights = points[valid_indices_mask][:, 2]

        for i in range(valid_indices.shape[0]):
            bev[valid_indices[i, 0], valid_indices[i, 1], -1] += valid_intensities[i]
            intensity_map_count[valid_indices[i, 0], valid_indices[i, 1]] += 1

            bev[valid_indices[i, 0], valid_indices[i, 1], -2] = max(bev[valid_indices[i, 0], valid_indices[i, 1], -2],
                                                                    valid_heights[i])


        divide_mask = intensity_map_count != 0
        bev[divide_mask, -1] = np.divide(bev[divide_mask, -1], intensity_map_count[divide_mask])

        intensity_image = bev[:, :, -1]
        height_image = bev[:, :, -2]

        intensity_image = cv2.normalize(intensity_image, None, 0, 255, cv2.NORM_MINMAX)

        height_image = cv2.normalize(height_image, None, 0, 255, cv2.NORM_MINMAX)

        intensity_image = intensity_image.astype(np.uint8)
        height_image = height_image.astype(np.uint8)

        # Create a final 3-channel image with height, intensity, and zero channels
        bev_image = np.zeros((intensity_image.shape[0], intensity_image.shape[1], 3), dtype=np.uint8)
        bev_image[:, :, 0] = height_image
        bev_image[:, :, 1] = intensity_image
        bev_image[:, :, 2] = 0

        data_dict = {"bev_input": np.transpose(bev, (2, 0, 1)), "bev_image": bev_image}
        return data_dict

 
def main():
    root_dir = '../mydatasets'
    dist_dir = '../bindatasets'
    scenario_folders = sorted([os.path.join(root_dir, x)
                               for x in os.listdir(root_dir) if
                               os.path.isdir(os.path.join(root_dir, x))])
    scenario_database = OrderedDict()

    
        
    # loop over all scenarios
    for (i, scenario_folder) in enumerate(scenario_folders):
        print(scenario_folder)
        bin_folder = os.path.join(dist_dir, scenario_folder.split('\\')[-1])
        print(f"Scenario {i}: {scenario_folder}")
        print(f"Bin folder: {bin_folder}")
        
        if not os.path.exists(bin_folder):
            os.makedirs(bin_folder)
            
        scenario_database.update({i: OrderedDict()})

        cav_list = sorted([x for x in os.listdir(scenario_folder)
                        if os.path.isdir(os.path.join(scenario_folder, x))])
        if int(cav_list[0]) < 0:
            cav_list = cav_list[1:] + [cav_list[0]]
        print(cav_list)
        max_cav = len(cav_list)
        print(max_cav)
        with open(f'{bin_folder}/lidar_data.csv', mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'x', 'y']) # write the header
                
            for (j, cav_id) in enumerate(cav_list):
                if j > max_cav - 1:
                    print('too many cavs')
                    break
                scenario_database[i][cav_id] = OrderedDict()

                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                lidar_files = sorted([os.path.join(cav_path, x)
                                    for x in os.listdir(cav_path) if
                                    x.endswith('.pcd') and 'additional' not in x])

                parent_folder = os.path.dirname(cav_path)
                if j == 0:
                    output_folder_ego_non_ground = os.path.join(parent_folder, f"scenario_{i}_cav_ego_non_ground")
                    if not os.path.exists(output_folder_ego_non_ground):
                        os.makedirs(output_folder_ego_non_ground)
                elif j > 0:
                    output_folder_nei_non_ground = os.path.join(parent_folder, f"scenario_{i}_cav_nei_non_ground")
                    if not os.path.exists(output_folder_nei_non_ground):
                        os.makedirs(output_folder_nei_non_ground)
                output_folder_all_non_ground = os.path.join(parent_folder, f"scenario_{i}_cav_all_non_ground")
                if not os.path.exists(output_folder_all_non_ground):
                    os.makedirs(output_folder_all_non_ground)

                    # 滤除地面点后直接存储xyz生成BEV图像
                for pointcloud_path in lidar_files:
                    pcd = o3d.io.read_point_cloud(pointcloud_path)
                    vel_msg = np.asarray(pcd.points)

                    vel_msg = vel_msg * np.array([1, 1, -1])  # revert the z axis

                    process = Processor(n_segments=180, n_bins=80, r_max=150, r_min=0.5,
                                        line_search_angle=0.3, max_dist_to_line=0.8,
                                        max_slope=2.0, max_error=0.1, long_threshold=8,
                                        max_start_height=0.5, sensor_height=1.93)
                    vel_non_ground = process(vel_msg)

                    res = pointcloud_path.split('\\')[-1]
                    timestamp = res.replace('.pcd', '')
                    image_non_ground_filename = f"cav_{j}_{timestamp}_non_ground.png"
                    preprocess_params = {
                        "cav_lidar_range": [-140.8, -38.4, -3, 140.8, 38.4, 1],
                        "geometry_param": {
                            "input_shape": [704, 192, 24],
                            "L1": -140.8,
                            "W1": -38.4,
                            "H1": -3,
                            "res": 0.4
                        }
                    }
                    preprocessor = BevPreprocessor(preprocess_params)
                    img_non_ground = preprocessor.preprocess(pcd)

                    # 将处理后的雷达数据保存为.bin文件
                    bin_filepath = f"{bin_folder}/{timestamp}.bin"
                    print(f"Bin file path: {bin_filepath}")
                    print(f"Bin folder: {bin_folder}")  
                    img_non_ground['bev_input'].astype(np.float32).tofile(bin_filepath)

                    # 记录timestamp和真实坐标到csv
                    for file in yaml_files:
                        ego_pose = load_yaml(file)['lidar_pose']
                        res = file.split('\\')[-1]
                        timestamp = res.replace('.yaml', '')
                        true_ego_pos = np.zeros(7)
                        true_ego_pos[:3] = ego_pose[:3]
                        r4 = R.from_euler('zxy', ego_pose[3:6], degrees=True)
                        true_ego_pos[3:7] = np.array(r4.as_quat())

                        csv_writer.writerow([timestamp, true_ego_pos[0], true_ego_pos[1]])
            
            csv_writer.close()

if __name__ == "__main__":
    main()
