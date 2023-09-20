import os
import cv2
import numpy as np
from typing import Generator


def get_video_capture(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


def generate_frames(video_capture: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_video_writer(fps: int, width: int, height: int, output_path: str) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc=codec, fps=fps, frameSize=(width, height), isColor=True)
