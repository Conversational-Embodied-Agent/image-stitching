To begin using these scripts, please create your environment from the provided *.yml* file `conda env create --name panoicup --file=environments.yml` and download model checkpoints from [https://drive.google.com/drive/folders/1_pjllybtXvf5r1xU0gLlJAOL05ixVeBJ?usp=share_link
](https://drive.google.com/drive/folders/1_pjllybtXvf5r1xU0gLlJAOL05ixVeBJ?usp=share_link)

> Script files found in this repository are designed as a part of a larger pipeline for **multiparty conversation architecture**. In their current state, they work with files in a local storage both on their input and output, hence various paths must be configured within the scripts. 
In a real-time deployment, they will be reworked to input/output images and other numpy arrays directly during runtime, instead of files on disk. 

Description of individual script files:

- `convert.py`: conversion from *.ppm* image files to *.jpg* with compression (PROBABLY USELESS IN ACTUAL DEPLOYMENT)
- `segmentation.py`: example for Laxmi on how to retrieve person coordinates from a 2D binary segmentation masks and depth maps (PROBABLY USELESS IN ACTUAL DEPLOYMENT)
- `ulils.py`: common function used by both `stiching.py` and `walls.py`
- `walls.py`: script for automatic detection of distances of 3 surrounding walls (left, front, right), designed to be run once after placing a robot into a new environment
- `stichingy.py`: main script, which takes a sequence of RGB frames from the robot and creates a panorama, depth frames, and segmentation masks for people.

Please make sure to set CONSTANTS before running the `stichingy.py` script (these are supposed to stay the same for each usage on a single machine):

```
PIXEL_LOG_FILE = 'panorama_pixel_log.json'
YOLO_PATH = '/home/g/gajdosech2/image-stitching-supeglue/yolov8l-worldv2.pt'
DE_PATH = '/home/g/gajdosech2/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
SAM_PATH = '/home/g/gajdosech2/image-stitching-supeglue/sam2.1_l.pt'
DEPTHS_OUT_WORK_DIR = 'depths/'
MASKS_OUT_WORK_DIR = 'masks/'
PANORAMAS_OUT_WORK_DIR = ''
```

and variables for the actual `stitching` method:

```
images_folder = '/home/g/gajdosech2/datasets/icup/UKBA/leftCamera/'
data_log_path = '/home/g/gajdosech2/datasets/icup/UKBA/leftCamera/data.log'
angles_log_path = '/home/g/gajdosech2/datasets/icup/UKBA/neckAngles/data.log'
```

which are supposed to be dynamically changed for each sequence and in actual deployment, they will probably be substituted with numpy arrays. Lastly, the stitching method needs an `M` parameter, denoting the sampling of RGB frames for panorama creation (i.e. `M=1` means use every single frame). 
See its usage under `if __name__ == '__main__':`
