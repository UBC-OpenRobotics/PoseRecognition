# TensorFlow-2.x-YOLOv3 and YOLOv4 tutorials

YOLOv3 and YOLOv4 implementation in TensorFlow 2.x, with support for training, transfer training, object tracking mAP and so on...
Code was tested with following specs:
- i7-7700k CPU and Nvidia 1080TI GPU
- OS Ubuntu 18.04
- CUDA 10.1
- cuDNN v7.6.5
- TensorRT-6.0.1.5
- Tensorflow-GPU 2.3.1
- Code was tested on Ubuntu and Windows 10 (TensorRT not supported officially)

## Installation
Clone (or download) this repo. The model we are using and all weights are already in the repo.

```
pip install -r ./requirements.txt

```

## Quick start
Cd into where you've cloned the repository. The pose estimation script is a modified version of Nick's object tracker.
By default, the script will take webcam footage as an input. Once it detects a person, it will crop out the bounding box
as a separate image and perform pose prediction. Including an optional argument "--image" will instead use any images in the
'IMAGES' folder. The optional argument "--video" will look in the 'IMAGES' folder for a video named "input_vid" and it will take
individual frames and do the pose estimation on them. Change the "frame_skip" variable on line 41 to change how many frames it will
skip before each estimate. Before running the script, ensure you have a directory called "output_images" in the repo.
```
python3 object_tracker.py
python3 object_tracker.py --images 
python3 object_tracker.py --video
```
The script runs and crops images correctly and labels with decent accuracy. The only issue is with blurry images.

To do:
- Change how we output the labels (Instead of directly drawing on the cropped photos)
