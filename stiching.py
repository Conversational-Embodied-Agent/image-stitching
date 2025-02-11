import os
import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO
from ultralytics import SAM
import sys
sys.path.append('/home/g/gajdosech2/Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

os.chdir("/home/g/gajdosech2/image-stitching-supeglue/")


IMG_WIDTH = 640
IMG_HEIGHT = 480
PANORAMA_WIDTH = 1700
PANORAMA_HEIGHT = 800

LEFT_YAW = -60
RIGHT_YAW = 60
BOTTOM_PITCH = -20
TOP_PITCH = 20

PIXEL_LOG_FILE = 'output/panorama_pixel_log.json'
YOLO_PATH = '/home/g/gajdosech2/image-stitching-supeglue/models/yolov8l-worldv2.pt'
DE_PATH = '/home/g/gajdosech2/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
SAM_PATH = '/home/g/gajdosech2/image-stitching-supeglue/models/sam2.1_l.pt'
DEPTHS_OUT_WORK_DIR = 'output/depths/'
RGBS_OUT_WORK_DIR = 'output/rgbs/'
MASKS_OUT_WORK_DIR = 'output/masks/'
PANORAMAS_OUT_WORK_DIR = 'output/'


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

def find_closest_angle(timestamp, angles_data):
    return min(angles_data, key=lambda x: abs(x['timestamp'] - timestamp))

def map_yaw_to_x(yaw):
    return int(((yaw - LEFT_YAW) / (RIGHT_YAW - LEFT_YAW)) * (PANORAMA_WIDTH - IMG_WIDTH))

def map_pitch_to_y(pitch):
    return int((1 - ((pitch - BOTTOM_PITCH) / (TOP_PITCH - BOTTOM_PITCH))) * (PANORAMA_HEIGHT - IMG_HEIGHT))

def place_image_on_panorama(panorama, image, x_start, y_start, log_array, image_filename, timestamp, yaw, pitch):
    img_height, img_width = image.shape[:2]
    x_end = min(x_start + img_width, PANORAMA_WIDTH)
    y_end = min(y_start + img_height, PANORAMA_HEIGHT)

    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            panorama[y, x] = image[y - y_start, x - x_start]
            log_index = y * PANORAMA_WIDTH + x
            log_array[log_index] = {'filename': image_filename, 'timestamp': timestamp, 'image_x': x - x_start, 'image_y': y - y_start, 'neck_yaw': yaw, 'neck_pitch': pitch}


def yolo_sam(rgb_panorama):
    yolo_world = YOLO(YOLO_PATH) 
    yolo_world.set_classes(['person'])
    print("\nYOLOWorld Prediction")
    classes = yolo_world.predict(rgb_panorama, conf=0.6)[0]
    classes.save(PANORAMAS_OUT_WORK_DIR + 'person_detection.jpg')

    if (len(classes.boxes)):
        sam = SAM(SAM_PATH)
        print("\nSAM Prediction")
        masks = sam(rgb_panorama, bboxes=classes.boxes.xyxy.detach().cpu().numpy())[0]

        for i, mask in enumerate(masks.masks.data.detach().cpu().numpy()):
            cv2.imwrite(MASKS_OUT_WORK_DIR + f'mask_{i}.png', mask.astype(np.uint8) * 255)

    yolo_world = YOLO(YOLO_PATH) 
    yolo_world.set_classes(['blue cube'])
    print("\nYOLOWorld Prediction")
    classes = yolo_world.predict(rgb_panorama, conf=0.0008)[0]
    classes.save(PANORAMAS_OUT_WORK_DIR + 'cube_detection.jpg')

    if (len(classes.boxes)):
        sam = SAM(SAM_PATH)
        print("\nSAM Prediction")
        masks = sam(rgb_panorama, bboxes=classes.boxes.xyxy.detach().cpu().numpy())[0]

        for i, mask in enumerate(masks.masks.data.detach().cpu().numpy()):
            cv2.imwrite(MASKS_OUT_WORK_DIR + f'cube_{i}.png', mask.astype(np.uint8) * 255)



def stitch_panorama(images_folder, data_log_path, angles_log_path, M):
    image_data = load_data_log(data_log_path)
    angles_data = load_angles_log(angles_log_path)

    rgb_panorama = np.zeros((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3), dtype=np.uint8)
    depth_panorama = np.zeros((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3), dtype=np.uint8) 
    log_array = [{'filename': None, 'timestamp': None, 'image_x': None, 'image_y': None, 'neck_yaw': None, 'neck_pitch': None} for _ in range(PANORAMA_WIDTH * PANORAMA_HEIGHT)]

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 20})
    depth_anything.load_state_dict(torch.load(DE_PATH, map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()

    sequence_length = len(image_data)
    for i in range(0, sequence_length, M):
        image_info = image_data[i]
        timestamp = image_info['timestamp']
        filename = image_info['filename'].replace('.ppm', '.jpg')

        closest_angle = find_closest_angle(timestamp, angles_data)
        yaw = closest_angle['yaw']
        pitch = closest_angle['pitch']
        x_start = map_yaw_to_x(yaw)
        y_start = map_pitch_to_y(pitch)

        image_path = os.path.join(images_folder, filename)

        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
    
        depth = depth_anything.infer_image(image, 480)
        np.save(DEPTHS_OUT_WORK_DIR + filename.replace('.jpg', '.npy'), depth)
        depth = (depth - 0.0) / (20.0 - 0.0) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        cv2.imwrite(DEPTHS_OUT_WORK_DIR + filename, depth)
        cv2.imwrite(RGBS_OUT_WORK_DIR + filename, image)

        place_image_on_panorama(depth_panorama, depth, x_start, y_start, log_array, filename, timestamp, yaw, pitch)
        place_image_on_panorama(rgb_panorama, image, x_start, y_start, log_array, filename, timestamp, yaw, pitch) 

    cv2.imwrite(PANORAMAS_OUT_WORK_DIR + 'rgb_panorama.jpg', rgb_panorama)
    cv2.imwrite(PANORAMAS_OUT_WORK_DIR + 'depth_panorama.jpg', depth_panorama)

    yolo_sam(rgb_panorama)
   
    with open(PIXEL_LOG_FILE, 'w') as log_file:
        json.dump(log_array, log_file)


if __name__ == '__main__':
    images_folder = '/home/g/gajdosech2/datasets/icup/CUBES/leftCam/'
    data_log_path = '/home/g/gajdosech2/datasets/icup/CUBES/leftCamppm/data.log'
    angles_log_path = '/home/g/gajdosech2/datasets/icup/CUBES/neckAngles/data.log'
    M = 90
    stitch_panorama(images_folder, data_log_path, angles_log_path, M)