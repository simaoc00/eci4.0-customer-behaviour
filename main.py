from utilities import *
from framework import run

import os
import re
import sys
import cv2
import argparse
import numpy as np
from dataclasses import dataclass

import torch
from mmpose.apis import init_pose_model
from mmaction.apis import init_recognizer

sys.path.append(os.path.abspath("libs/byte_track/yolox/tracker"))
from byte_tracker import BYTETracker


def parse_args():
    parser = argparse.ArgumentParser(description='ECI4.0 Customer Behaviour Framework')
    parser.add_argument('video', help='video name and extension')
    parser.add_argument('--homography', help='homography matrix path', required=False)
    parser.add_argument('--action-recognizer', help='action recognizer name (stgcn|2sagcn|posec3d)', default='2sagcn')
    args = parser.parse_args()
    return args


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    args = parse_args()
    video_dir = "demos/videos"
    video_category = "/"
    video_name = args.video
    for root, _, files in os.walk(video_dir):
        if video_name in files:
            video_category = f"/{os.path.split(root)[-1]}/"
            break
    # extract video frames and related information
    source_video_path = f"{video_dir}{video_category}{video_name}"
    video_capture = video.get_video_capture(source_video_path)
    frame_list = list(video.generate_frames(video_capture))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_capture.release()
    # extract homography matrix
    homography_path = args.homography or (f"demos/homography/VIRAT_{video_name.split('_')[2][0:4]}_homography_img2world.txt" if re.match(r'^VIRAT_S_\d{6}_\d{2}_\d{6}_\d{6}(.*)$', video_name) else None)
    homography = None
    if homography_path:
        with open(homography_path, 'r') as f:
            homography = np.array([[float(num) for num in line.split(',')] for line in f])
    if homography is None:
        print(f"\n(notice) homography is either not available or has not been provided for the selected video.\n")
    # create object detector instance (YOLOv5)
    object_detector = torch.hub.load("ultralytics/yolov5", "yolov5x6")
    # create object tracker instance (ByteTrack)
    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.5
        track_buffer: int = fps
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 1.6
        min_box_area: float = 10
        mot20: bool = False
    tracker = BYTETracker(BYTETrackerArgs(), fps)
    # create pose estimation model instance (HRNet)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pose_config = "libs/mmpose/configs/hrnet_w32_coco_384x288.py"
    pose_checkpoint = "libs/mmpose/weights/hrnet_w32_coco_384x288.pth"
    pose_model = init_pose_model(pose_config, pose_checkpoint, device)
    # create action recognizer instance (ST-GCN|2s-AGCN|PoseC3D)
    recognizer_config = f"libs/mmaction/configs/{args.action_recognizer}_pip12_keypoint.py"
    recognizer_checkpoint = f"libs/mmaction/weights/{args.action_recognizer}_pip12_keypoint.pth"
    recognizer = init_recognizer(recognizer_config, recognizer_checkpoint, device)
    # run framework
    run(video_name, homography, frame_list, fps, width, height, object_detector, tracker, pose_model, recognizer)


if __name__ == '__main__':
    main()
