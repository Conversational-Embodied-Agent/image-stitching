import numpy as np
import open3d as o3d


def find_closest_angle(timestamp, angles_data):
    return min(angles_data, key=lambda x: abs(x['timestamp'] - timestamp))

def load_data_log(data_log_path):
    image_data = []
    with open(data_log_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_data.append({
                'timestamp': float(parts[1]),
                'filename': parts[2]
            })
    return image_data

def load_angles_log(angles_log_path):
    angles_data = []
    with open(angles_log_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            angles_data.append({
                'timestamp': float(parts[1]),
                'yaw': float(parts[2]),
                'pitch': float(parts[3])
            })
    return angles_data

def depth_to_pointcloud(depth_image, CX, CY, FX, FY):
    point_cloud = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            depth = depth_image[y, x][0]
            if depth > 0: 
                z = depth
                x_point = (x - CX) * z / FX
                y_point = (y - CY) * z / FY
                point_cloud.append([x_point, y_point, z])

    point_cloud = np.array(point_cloud)
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    return o3d_cloud