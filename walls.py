import os
import cv2
import numpy as np
import json
import torch

import sys
sys.path.append('/home/g/gajdosech2/Depth-Anything-V2/metric_depth')
sys.path.append('/home/g/gajdosech2/image-stiching-superglue')
from depth_anything_v2.dpt import DepthAnythingV2

from utils import depth_to_pointcloud, load_data_log, load_angles_log, find_closest_angle

IMG_WIDTH = 640
IMG_HEIGHT = 480

FX = 400
FY = 400
CX = 320.0
CY = 240.0 
TOP_PERCENTAGE = 25 

DE_PATH = '/home/g/gajdosech2/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'


def find_valls(images_folder, data_log_path, angles_log_path, wall_log_file, M=1):
    image_data = load_data_log(data_log_path)
    angles_data = load_angles_log(angles_log_path)

    image_indices = [0, 0, 0]
    angles = [float('inf'), float('inf'), float('inf')]
    wall_coords = []

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 20})
    depth_anything.load_state_dict(torch.load(DE_PATH, map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()

    for i in range(0, len(image_data), M):
        image_info = image_data[i]
        filename = image_info['filename'].replace('.ppm', '.jpg')

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
            continue

        timestamp = image_info['timestamp']
        closest_yaw = find_closest_angle(timestamp, angles_data)['yaw']

        if abs(closest_yaw + 60) < abs(angles[0] + 60):
            angles[0] = closest_yaw
            image_indices[0] = i

        if abs(closest_yaw - 60) < abs(angles[1] - 60):
            angles[1] = closest_yaw
            image_indices[1] = i

        if abs(closest_yaw) < abs(angles[2]):
            angles[2] = closest_yaw
            image_indices[2] = i


    for w, i in enumerate(image_indices):
        image_info = image_data[i]
        filename = image_info['filename'].replace('.ppm', '.jpg')

        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
    
        depth = depth_anything.infer_image(image, 518)

        boundary_row = int(IMG_HEIGHT * (TOP_PERCENTAGE / 100))
        valid_pixels = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        if w == 0:
            valid_pixels[:boundary_row, :80] = 1 
        elif w == 1:
            valid_pixels[:boundary_row, IMG_WIDTH-80:] = 1 
        elif w == 2:
            valid_pixels[40:boundary_row+40, IMG_WIDTH//2-40:IMG_WIDTH//2+40] = 1  

        depth = cv2.bitwise_and(depth, depth, mask=valid_pixels)

        depth_debug = (depth - 0.0) / (20.0 - 0.0) * 255.0
        depth_debug = depth_debug.astype(np.uint8)
        cv2.imwrite("/home/g/gajdosech2/image-stitching-supeglue/masked_depths/depth_" + str(w) + filename, depth_debug)
        cv2.imwrite("/home/g/gajdosech2/image-stitching-supeglue/masked_depths/rgb_" + str(w) + filename, image)

        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        o3d_cloud = depth_to_pointcloud(depth, CX, CY, FX, FY)
        plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        wall_coords.append(d)

        inlier_cloud = o3d_cloud.select_by_index(inliers)

        coords = np.asarray(inlier_cloud.points)
        wall_coords.append(coords[:3].tolist())

    print(wall_coords)
    with open(wall_log_file, 'w') as log_file:
        json.dump(wall_coords, log_file)


if __name__ == '__main__':
    images_folder = '/home/g/gajdosech2/datasets/icup/UKBA/leftCamera/'
    data_log_path = '/home/g/gajdosech2/datasets/icup/UKBA/leftCamera/data.log'
    angles_log_path = '/home/g/gajdosech2/datasets/icup/UKBA/neckAngles/data.log'
    wall_log_file = '/home/g/gajdosech2/image-stitching-supeglue/wall_pixels.json'
    find_valls(images_folder, data_log_path, angles_log_path, wall_log_file)
