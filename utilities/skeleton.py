import cv2
import math
from mmpose.apis import inference_top_down_pose_model

coco_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

num_kpts = 17

skeleton = [[5, 7], [6, 8], [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [1, 3], [1, 0], [2, 4], [2, 0], [0, 5],
            [0, 6], [7, 9], [8, 10], [13, 15], [14, 16]]

coco_colors = [[66, 133, 244], [52, 168, 83], [251, 188, 5], [234, 67, 53]]


def get_pose_estimation_prediction(pose_model, image, bbox, threshold, occlusion_awareness=True):
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model, image, bbox)
    t, l, w, h = bbox[0]['bbox']
    if occlusion_awareness:
        occluded_body_part = False
        if (not check_left_arm(pose_results, threshold)) and (not check_right_arm(pose_results, threshold)):
            occluded_body_part = True
            t -= w / 8
            w += w / 4
        if not check_lower_part(pose_results, threshold):
            occluded_body_part = True
            aspect_ratio = w / h
            x = 7.5 * (aspect_ratio - 0.75)
            sigmoid = (1.5 / (1 + math.exp(-x)))
            h *= 1 + sigmoid
        if not check_upper_part(pose_results, threshold):
            occluded_body_part = True
            aspect_ratio = w / h
            x = 7.5 * (aspect_ratio - 0.75)
            sigmoid = (1.5 / (1 + math.exp(-x)))
            l -= abs(h - (h * (1 + sigmoid)))
            h *= 1 + sigmoid
        if occluded_body_part:
            pose_results, returned_outputs = inference_top_down_pose_model(pose_model, image, [dict(bbox=[t, l, w, h])])
    return pose_results[-1]["bbox"].tolist(), pose_results[-1]["keypoints"].tolist()


def draw_pose(pose, image, threshold):
    kpts = pose[-1]["keypoints"][:, :2]
    scores = pose[-1]["keypoints"][:, 2]
    assert kpts.shape == (num_kpts, 2)
    for i in range(len(skeleton)):
        kpt_a, kpt_b = skeleton[i][0], skeleton[i][1]
        x_a, y_a = kpts[kpt_a][0], kpts[kpt_a][1]
        x_b, y_b = kpts[kpt_b][0], kpts[kpt_b][1]
        color = (0, 0, 0)
        if i in range(7, 13):
            color = coco_colors[0]
        elif i == 13:
            color = coco_colors[1]
        elif i == 14:
            color = coco_colors[2]
        elif i in range(15, 17):
            color = coco_colors[3]
        if scores[kpt_a] >= threshold and scores[kpt_b] >= threshold:
            cv2.line(image, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)
        if scores[kpt_a] >= threshold:
            cv2.circle(image, (int(x_a), int(y_a)), 3, color, -1)
        if scores[kpt_b] >= threshold:
            cv2.circle(image, (int(x_b), int(y_b)), 3, color, -1)


def check_upper_part(pose, threshold):
    scores = pose[-1]["keypoints"][:7, 2]
    return sum(score >= threshold for score in scores)/len(scores) > 0.5


def check_lower_part(pose, threshold):
    scores = pose[-1]["keypoints"][13:, 2]
    return sum(score >= threshold for score in scores)/len(scores) > 0.5


def check_left_arm(pose, threshold):
    scores = pose[-1]["keypoints"][(7, 9), 2]
    return all(score >= threshold for score in scores)


def check_right_arm(pose, threshold):
    scores = pose[-1]["keypoints"][(8, 10), 2]
    return all(score >= threshold for score in scores)
