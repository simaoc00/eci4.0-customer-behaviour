from utilities import video

import cv2
import numpy as np
import pandas as pd


def calculate_bbox_iou(groundtruth_bbox, predicted_bbox):
    x_left = max(groundtruth_bbox[0], predicted_bbox[0])
    y_top = max(groundtruth_bbox[1], predicted_bbox[1])
    x_right = min(groundtruth_bbox[2], predicted_bbox[2])
    y_bottom = min(groundtruth_bbox[3], predicted_bbox[3])
    interception = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    groundtruth_area = (groundtruth_bbox[2] - groundtruth_bbox[0] + 1) * (groundtruth_bbox[3] - groundtruth_bbox[1] + 1)
    predicted_area = (predicted_bbox[2] - predicted_bbox[0] + 1) * (predicted_bbox[3] - predicted_bbox[1] + 1)
    return interception / float(groundtruth_area + predicted_area - interception)


def get_iou_values(frame_groundtruth, frame_predictions):
    iou_values = []
    for groundtruth_bbox in frame_groundtruth:
        candidates = []
        for predicted_bbox in frame_predictions:
            candidates.append(calculate_bbox_iou(groundtruth_bbox, predicted_bbox))
        if len(candidates) != 0:
            iou_values.append(max(candidates))
    return iou_values


def calculate_average_iou(iou_values):
    return sum(iou_values) / len(iou_values)


def get_cumulative_measures(iou_values, iou_threshold):
    total_tp = 0
    total_fp = 0
    cumulative_tp = []
    cumulative_fp = []
    for iou in iou_values:
        if iou >= iou_threshold:
            total_tp += 1
        if 0 < iou < iou_threshold:
            total_fp += 1
        cumulative_tp.append(total_tp)
        cumulative_fp.append(total_fp)
    return cumulative_tp, cumulative_fp


def calculate_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def calculate_recall(tp, n_gt):
    return tp / n_gt if n_gt != 0 else 0


def tlwh_to_tlbr(bboxes):
    return [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]


def calculate_average_precision(recalls, precision):
    eleven_recalls = np.flip(np.linspace(0, 1, 11), axis=0)
    rho_interp = []
    for r in eleven_recalls:
        greater_recalls = np.argwhere(recalls >= r)
        max_precision = 0
        if greater_recalls.size != 0:
            max_precision = max(precision[greater_recalls.min():])
        rho_interp.append(max_precision)
    # ap = sum(max(precision whose recall is above r))/11
    return sum(rho_interp) / 11


def get_object_detector_mAP(video_name, predicted_detections, iou_threshold):
    with open(f"demos/annotations/{video_name}.viratdata.objects.txt", 'r') as f:
        groundtruth = np.array([[int(num) for num in line.split(' ')] for line in f])
    df = pd.DataFrame(groundtruth)
    APs = []
    for i in range(len(predicted_detections)):
        frame_groundtruth = df.loc[df[2] == i].iloc[:, 3:7].values.tolist()
        frame_predictions = predicted_detections[i]
        iou_values = get_iou_values(tlwh_to_tlbr(frame_groundtruth), tlwh_to_tlbr(frame_predictions))
        cumulative_tp, cumulative_fp = get_cumulative_measures(iou_values, iou_threshold)
        precisions = []
        recalls = []
        for tp, fp in zip(cumulative_tp, cumulative_fp):
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, len(frame_groundtruth))
            precisions.append(precision)
            recalls.append(recall)
        APs.append(calculate_average_precision(recalls, precisions))
    mAP = np.average(np.array(APs))
    return mAP


def get_iou_values_by_id(frame_groundtruth, frame_tracks, iou_threshold):
    matching_info = []
    for id, predicted_bbox in frame_tracks.items():
        candidates = []
        for idx in range(len(frame_groundtruth)):
            candidates.append([id, calculate_bbox_iou(frame_groundtruth[idx], predicted_bbox), idx])
        candidates = np.array(candidates)
        if len(candidates) != 0:
            i = np.argmax(candidates[:, 1], axis=0)
            matching_info.append(candidates[i] if candidates[i][1] >= iou_threshold else [-1, candidates[i][1], -1])
            continue
        matching_info.append([-1, -1, -1])
    return matching_info


def get_total_measures(iou_values, total_groundtruth, iou_threshold):
    total_fp = 0
    total_tp = 0
    for iou in iou_values:
        if iou != -1:
            if iou >= iou_threshold:
                total_tp += 1
            elif iou == 0:
                total_fp += 1
    total_fn = total_groundtruth - total_tp
    return total_fn, total_fp


def get_tracker_metrics(video_name, predicted_tracks, iou_threshold):
    with open(f"demos/annotations/{video_name}.viratdata.objects.txt", 'r') as f:
        groundtruth = np.array([[int(num) for num in line.split(' ')] for line in f])
    df = pd.DataFrame(groundtruth)
    id_history = {}
    total_fn, total_fp, total_ids, total_groundtruth = 0, 0, 0, 0
    for i in range(len(predicted_tracks)):
        frame_groundtruth = np.array(df.loc[(df[2] == i) & (df[7] == 1)].values.tolist())
        frame_groundtruth_bboxes = tlwh_to_tlbr(frame_groundtruth[:, 3:7]) if len(frame_groundtruth) != 0 else []
        frame_predicted_tracks = predicted_tracks[i]
        matching_info = np.array(get_iou_values_by_id(frame_groundtruth_bboxes, frame_predicted_tracks, iou_threshold))
        iou_values = list(matching_info[:, 1]) if len(matching_info) != 0 else []
        frame_fn, frame_fp = get_total_measures(iou_values, len(frame_groundtruth), iou_threshold)
        frame_ids = 0
        for j in range(len(frame_predicted_tracks)):
            if len(matching_info) != 0 and matching_info[j][1] != -1:
                predicted_id, iou = int(matching_info[j][0]), matching_info[j][1]
                groundtruth_id = int(frame_groundtruth[int(matching_info[j][2]), 0])
                if predicted_id != -1 and iou != 0 and groundtruth_id in id_history and id_history[groundtruth_id] != predicted_id:
                    frame_ids += 1
                if predicted_id != -1 and iou != 0:
                    id_history[groundtruth_id] = predicted_id
        total_fn += frame_fn
        total_fp += frame_fp
        total_ids += frame_ids
        total_groundtruth += len(frame_groundtruth)
    mota = 1 - ((total_fn + total_fp + total_ids) / total_groundtruth)
    return [mota, total_groundtruth, total_fn, total_fp, total_ids]


def save_results(video_name, detector_name, tracker_name, detector_metric, tracker_metrics):
    f = open(f"results/{tracker_name}_results.txt", "a+")
    f.write(f"**{video_name}_{detector_name}_{tracker_name}**")
    f.write("\n**Object Detector**\n")
    f.write(f"mAP: {detector_metric}")
    f.write("\n**Tracker**\n")
    f.write(f"MOTA: {tracker_metrics[0]}\n"
            f"Total Groundtruth: {tracker_metrics[1]}\n"
            f"Total FN: {tracker_metrics[2]}\n"
            f"Total FP: {tracker_metrics[3]}\n"
            f"Total IDS: {tracker_metrics[4]}\n\n")
    f.close()


def get_occlusion_awareness_iou(video_name, predicted_tracks, occluded_id, frame_ids):
    video_capture = video.get_video_capture(f"demos/videos/{video_name}.mp4")
    frame_list = list(video.generate_frames(video_capture))
    video_capture.release()
    with open(f"demos/annotations/{video_name}.viratdata.objects.txt", 'r') as f:
        groundtruth = np.array([[int(num) for num in line.split(' ')] for line in f])
    df = pd.DataFrame(groundtruth)
    iou_values = []
    for frame_id in range(len(predicted_tracks)):
        if frame_id in frame_ids:
            frame = frame_list[frame_id]
            frame_groundtruth = df.loc[(df[0] == occluded_id) & (df[2] == frame_id)].iloc[:, 3:7].values.tolist()
            frame_predictions = [list(p) for p in predicted_tracks[frame_id].values()]
            iou_value = get_iou_values(tlwh_to_tlbr(frame_groundtruth), frame_predictions)
            if len(iou_value) > 0:
                iou_values.append(max(iou_value))
            color = (0, 0, 0)
            for bbox in tlwh_to_tlbr(frame_groundtruth):
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                color = (255, 0, 0) if iou_values[-1] < 0.5 else (0, 255, 0)
                cv2.putText(frame, f"{iou_values[-1]}", (int(x_min), int(y_min) - 10), 5, 0.7, color, 2, 2, False)
            color = (255, 255, 255)
            for bbox in frame_predictions:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(frame, f"{frame_id}", (15, 25), 5, 1, color, 2, 2, False)
            cv2.imshow('preview', cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (1280, 720)))
            cv2.waitKey(0)
    print(iou_values)
    print(sum(iou_values)/len(iou_values))
    return sum(iou_values)/len(iou_values)


def get_trajectory_curvature_metrics(points):
    num_points = len(points)
    if num_points < 3:
        return 0
    curvatures = []
    for i in range(1, num_points - 1):
        x0, y0 = points[i - 1][0], points[i - 1][1]
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[i + 1][0], points[i + 1][1]
        vec1 = np.array([x1 - x0, y1 - y0])
        vec2 = np.array([x2 - x1, y2 - y1])
        # calculate the angle between the vectors using the dot product
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        angle = np.arccos(dot_product / norm_product)
        # calculate the curvature as the reciprocal of the radius of the circle defined by the angle
        vec3 = np.array([x2 - x0, y2 - y0])
        distance = np.linalg.norm(vec3)
        curvature = 2 * np.sin(angle) / distance
        curvatures.append(curvature)
    total_curvature = np.sum(np.abs(curvatures))
    average_curvature = np.mean(np.abs(curvatures))
    return total_curvature, average_curvature


def get_velocity_stdev(velocities):
    print(np.std(velocities))
    return np.std(velocities)
