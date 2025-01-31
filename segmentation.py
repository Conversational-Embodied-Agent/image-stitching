import numpy as np
import cv2
import json
import random
import open3d as o3d

IMG_WIDTH = 640
IMG_HEIGHT = 480
PANORAMA_WIDTH = 1700
PANORAMA_HEIGHT = 800

FX = 400
FY = 400
CX = 320.0
CY = 240.0 

OUT_WORK_DIR = 'output/'


def apply_mask_and_sample(panorama, mask, log_data):
    masked_panorama = cv2.bitwise_and(panorama, panorama, mask=mask)
    
    non_black_pixels = np.argwhere(masked_panorama > 0)
    non_black_pixels = np.unique(non_black_pixels[:, :2], axis=0)
    
    sampled_pixel = random.choice(non_black_pixels)
    px_y, px_x = sampled_pixel
    
    log_entry = log_data[px_y * PANORAMA_WIDTH + px_x]
    return log_entry, masked_panorama

def process_depth_and_create_pointcloud(depth_path, image_pixels, scaling_matrix=None, rgb_path=None):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_array = np.load(depth_path.replace('.jpg', '.npy'))
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=image_pixels)

    cv2.imwrite(OUT_WORK_DIR + 'masked_depth.jpg', masked_depth)

    if rgb_path:
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    point_cloud = []
    colors = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            depth = masked_depth[y, x][0] / 255 * 20
            if depth > 0:
                z = depth
                if depth_array is not None:
                    z = depth_array[y, x]
                x_point = (x - CX) * z / FX
                y_point = (y - CY) * z / FY
                point_cloud.append([x_point, y_point, z])

                if rgb_path:
                    colors.append(rgb_image[y, x] / 255.0)

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np.array(point_cloud))

    if rgb_path:
        o3d_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    if scaling_matrix is not None:
        o3d_cloud = o3d_cloud.transform(scaling_matrix)

    o3d.visualization.draw_geometries([o3d_cloud])
    return o3d_cloud

def fit_cuboid(cube_cloud):
    aabb = cube_cloud.get_axis_aligned_bounding_box()

    extent = aabb.get_extent()
    print("AABB extents:", extent)

    target_size = 5
    scaling_factors = [target_size / size if size != 0 else 1 for size in extent]
    print("Scaling factors for each axis:", scaling_factors)
    scaling_matrix = np.diag(scaling_factors + [1])

    aabb.color = (1, 0, 0)
    cube_cloud.paint_uniform_color([0, 1, 0]) 
    o3d.visualization.draw_geometries([cube_cloud, aabb])
    return scaling_matrix


if __name__ == '__main__':
    panorama_path = '/home/g/gajdosech2/image-stitching-supeglue/output/rgb_panorama.jpg'
    mask_path = '/home/g/gajdosech2/image-stitching-supeglue/output/masks/mask_1.png'
    log_file_path = '/home/g/gajdosech2/image-stitching-supeglue/output/panorama_pixel_log.json'
    depths_folder = '/home/g/gajdosech2/image-stitching-supeglue/output/depths/'
    rgbs_folder = '/home/g/gajdosech2/image-stitching-supeglue/output/rgbs/'

    with open(log_file_path, 'r') as log_file:
        log_data = json.load(log_file)

    panorama = cv2.imread(panorama_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    log_entry, masked_panorama = apply_mask_and_sample(panorama, mask, log_data)
    cv2.imwrite(OUT_WORK_DIR + 'masked_panorama.jpg', masked_panorama)

    image_pixels = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    neck_yaw, neck_pitch = 0, 0
    for x in range(0, masked_panorama.shape[1]):
        for y in range(0, masked_panorama.shape[0]):
            rgb_value = np.sum(masked_panorama[y, x])
            if rgb_value:
                log_index = y * PANORAMA_WIDTH + x
                if (log_data[log_index]['filename'] == log_entry['filename']):
                    image_x = int(log_data[log_index]['image_x'])
                    image_y = int(log_data[log_index]['image_y'])
                    image_pixels[image_y, image_x] = 1
                    neck_yaw, neck_pitch = log_data[log_index]['neck_yaw'], log_data[log_index]['neck_pitch']

    cv2.imwrite(OUT_WORK_DIR + 'image_pixels.jpg', image_pixels * 255)

    depth_path = f"{depths_folder}/{log_entry['filename']}"

    point_cloud = process_depth_and_create_pointcloud(depth_path, image_pixels)
    scaling_matrix = fit_cuboid(point_cloud)

    point_cloud = process_depth_and_create_pointcloud(depth_path, np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8), scaling_matrix=None, rgb_path=f"{rgbs_folder}/{log_entry['filename']}")
    # rotate with respect to neck yaw angle
    print(neck_yaw, neck_pitch)
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(neck_pitch) , np.radians(neck_yaw) , 0))
    point_cloud.rotate(rotation, center=(0, 0, 0))
    o3d.io.write_point_cloud(OUT_WORK_DIR + "point_cloud.ply", point_cloud)