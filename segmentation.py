import numpy as np
import cv2
import json
import random
import open3d as o3d

IMG_WIDTH = 640
IMG_HEIGHT = 480
PANORAMA_WIDTH = 1300
PANORAMA_HEIGHT = 480

FX = 400
FY = 400
CX = 320.0
CY = 240.0 

OUT_WORK_DIR = ''


def apply_mask_and_sample(panorama, mask, log_data):
    masked_panorama = cv2.bitwise_and(panorama, panorama, mask=mask)
    
    non_black_pixels = np.argwhere(masked_panorama > 0)
    non_black_pixels = np.unique(non_black_pixels[:, :2], axis=0)
    
    sampled_pixel = random.choice(non_black_pixels)
    px_y, px_x = sampled_pixel
    
    log_entry = log_data[px_y * PANORAMA_WIDTH + px_x]
    return log_entry, masked_panorama

def process_depth_and_create_pointcloud(depth_path, image_pixels):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=image_pixels)

    cv2.imwrite(OUT_WORK_DIR + 'masked_depth.jpg', masked_depth)

    point_cloud = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            depth = masked_depth[y, x][0]
            if depth > 0: 
                z = depth
                x_point = (x - CX) * z / FX
                y_point = (y - CY) * z / FY
                point_cloud.append([x_point, y_point, z])

    point_cloud = np.array(point_cloud)
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)

    o3d.visualization.draw_geometries([o3d_cloud])


if __name__ == '__main__':
    panorama_path = '/home/g/gajdosech2/image-stitching-supeglue/rgb_panorama.jpg'
    mask_path = '/home/g/gajdosech2/image-stitching-supeglue/mask_0.png'
    log_file_path = '/home/g/gajdosech2/image-stitching-supeglue/panorama_pixel_log.json'
    depths_folder = '/home/g/gajdosech2/image-stitching-supeglue/depths/'

    with open(log_file_path, 'r') as log_file:
        log_data = json.load(log_file)

    panorama = cv2.imread(panorama_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    log_entry, masked_panorama = apply_mask_and_sample(panorama, mask, log_data)
    cv2.imwrite(OUT_WORK_DIR + 'masked_panorama.jpg', masked_panorama)

    image_pixels = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for x in range(0, masked_panorama.shape[1]):
        for y in range(0, masked_panorama.shape[0]):
            rgb_value = np.sum(masked_panorama[y, x])
            if rgb_value:
                log_index = y * PANORAMA_WIDTH + x
                image_x = int(log_data[log_index]['image_x'])
                image_y = int(log_data[log_index]['image_y'])
                image_pixels[image_y, image_x] = 1

    cv2.imwrite(OUT_WORK_DIR + 'image_pixels.jpg', image_pixels * 255)

    depth_path = f'{depths_folder}/{log_entry['filename']}'

    process_depth_and_create_pointcloud(depth_path, image_pixels)