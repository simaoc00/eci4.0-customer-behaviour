from utilities import *

import cv2
import mmcv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

smoothing_window = 8
group_distance_factor = 1.35
group_scale_threshold = 1.5
pose_threshold = 0.3

cmap = plt.get_cmap("tab20")
colors = [cmap(c)[:3] for c in np.linspace(0, 1, 20)]


def run(video_name, homography, frame_list, fps, width, height, object_detector, tracker, pose_model, recognizer):
    target_video_path = f"results/{video_name}"
    target_data_path = f"results/{video_name.split('.')[0]}.xlsx"

    print("\nextracting data from the video...")
    prog_bar = mmcv.ProgressBar(len(frame_list))
    groups = []
    paths = defaultdict(list)
    skeleton_sequences = defaultdict(list)
    action_labels = defaultdict(str)
    data = pd.DataFrame(columns=["frame_id", "track_id", "bbox", "frame_tracklet", "world_tracklet", "pose", "action"])
    for frame_id, frame in enumerate(frame_list):
        objects = object_detector(frame, size=1280)
        objects = objects.pred[0].cpu().numpy()
        detections = []
        for x_min, y_min, x_max, y_max, confidence, class_id in objects:
            if object_detector.names[int(class_id)] == "person":
                detections.append([x_min, y_min, x_max, y_max, confidence])
        tracked_bboxes = defaultdict(object)
        if detections:
            tracks = tracker.update(output_results=np.array(detections), img_info=frame.shape, img_size=frame.shape)
            for track in tracks:
                track_id = track.track_id
                bbox, pose = skeleton.get_pose_estimation_prediction(pose_model, frame, [dict(bbox=track.tlwh)], pose_threshold)
                x_min, y_min, x_max, y_max = tracked_bboxes[track_id] = bbox
                raw_tracklet = [((x_max + x_min) / 2), y_max] if y_max < frame.shape[0] - 10 else None
                skeleton_sequences[track_id].append([{"bbox": np.array(bbox), "keypoints": np.array(pose)}])
                if len(skeleton_sequences[track_id]) == fps * 2:
                    height, width = frame.shape[0], frame.shape[1]
                    action_labels[track_id] = action_recognition.get_action_prediction(recognizer, skeleton_sequences[track_id], fps * 2, height, width)
                    skeleton_sequences[track_id].clear()
                action = action_labels[track_id]
                data.loc[len(data)] = [frame_id, track_id, bbox, raw_tracklet, None, pose, action]
        frame_groups = tracking.generate_groups(tracked_bboxes, group_distance_factor, group_scale_threshold)
        for i, group in enumerate(frame_groups):
            x_min, y_min, x_max, y_max = tracking.get_group_bbox(group, tracked_bboxes)
            frame_groups[i] = [x_min, y_min, x_max, y_max]
        groups.append(frame_groups)
        prog_bar.update()

    print("\n\napplying trajectory smoothing...")
    prog_bar = mmcv.ProgressBar(2)
    original_tracklets = tracking.apply_trajectory_smoothing(data, len(frame_list), smoothing_window)
    prog_bar.update()
    reversed_tracklets = tracking.apply_trajectory_smoothing(data, len(frame_list), smoothing_window, reverse=True)
    prog_bar.update()

    print("\n\ngenerating output files...")
    prog_bar = mmcv.ProgressBar(len(frame_list))
    video_writer = video.get_video_writer(fps, width, height, target_video_path)
    for frame_id, frame in enumerate(frame_list):
        for track_id in set(data.loc[data["frame_id"] == frame_id]["track_id"]):
            bbox = x_min, y_min, x_max, y_max = dataframe.get_item(data, frame_id, track_id, "bbox")
            raw_tracklet = dataframe.get_item(data, frame_id, track_id, "frame_tracklet")
            pose = dataframe.get_item(data, frame_id, track_id, "pose")
            action = dataframe.get_item(data, frame_id, track_id, "action")
            if raw_tracklet is not None:
                original_tracklet = dataframe.get_item(original_tracklets, frame_id, track_id, "tracklet")
                reversed_tracklet = dataframe.get_item(reversed_tracklets, frame_id, track_id, "tracklet")
                weight = tracking.get_coordinate_weight(data, frame_id, track_id, smoothing_window)
                x_coordinate = (original_tracklet[0] * weight) + (reversed_tracklet[0] * (1 - weight))
                y_coordinate = (original_tracklet[1] * weight) + (reversed_tracklet[1] * (1 - weight))
                frame_tracklet = [x_coordinate, y_coordinate]
                world_tracklet = tracking.convert_img2world(frame_tracklet, homography)
                i = dataframe.get_index(data, frame_id, track_id)
                data.at[i, "frame_tracklet"] = frame_tracklet
                data.at[i, "world_tracklet"] = world_tracklet
                paths[track_id].append(world_tracklet)
            frame_tracklets = dataframe.get_current_list(data, frame_id, track_id, "frame_tracklet")
            world_tracklets = dataframe.get_current_list(data, frame_id, track_id, "world_tracklet")
            vel = float("{:.3f}".format(tracking.get_world_velocity(world_tracklets, fps)))
            color = colors[int(track_id) % len(colors)]
            color = [c * 255 for c in color]
            for coordinate in frame_tracklets:
                cv2.circle(frame, (round(coordinate[0]), round(coordinate[1])), radius=1, color=color, thickness=2)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            skeleton.draw_pose([{"bbox": np.array(bbox), "keypoints": np.array(pose)}], frame, pose_threshold)
            cv2.putText(frame, f"{str(track_id)} ({action}): v = {vel}m/s", (int(x_min), int(y_min) - 10), 5, 0.7, color, 2, 2, False)
        frame_groups = groups[frame_id]
        for group in frame_groups:
            cv2.rectangle(frame, (int(group[0]), int(group[1])), (int(group[2]), int(group[3])), (255, 255, 255), 2)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        prog_bar.update()
    video_writer.release()
    data.to_excel(target_data_path)
    if homography is not None:
        tracking.plot_homography(video_name, paths, colors)
    print("\n\nfiles stored in the results folder!\n")
