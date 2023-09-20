from utilities import dataframe

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict


def get_bbox_center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def generate_groups(tracked_bboxes: dict, distance_factor: float, scale_threshold: float):
    pairs = []
    ids = list(tracked_bboxes.keys())
    for i in range(len(ids)):
        i_bbox = tracked_bboxes[ids[i]]
        i_bbox_area = (i_bbox[2] - i_bbox[0]) * (i_bbox[3] - i_bbox[1])
        for j in range(i + 1, len(ids)):
            j_bbox = tracked_bboxes[ids[j]]
            j_bbox_area = (j_bbox[2] - j_bbox[0]) * (j_bbox[3] - j_bbox[1])
            distance = math.dist(get_bbox_center(i_bbox), get_bbox_center(j_bbox))
            max_width = max(i_bbox[2] - i_bbox[0], j_bbox[2] - j_bbox[0])
            area_diff = max(i_bbox_area, j_bbox_area) / min(i_bbox_area, j_bbox_area)
            if distance < max_width * distance_factor and area_diff < scale_threshold:
                pairs.append((ids[i], ids[j]))
    return list(merge_common(pairs))


def merge_common(pairs):
    neigh = defaultdict(set)
    visited = set()
    for each in pairs:
        for item in each:
            neigh[item].update(each)
    def comp(node, neigh=neigh, visited=visited, vis=visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node
    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))


def get_group_bbox(ids: list, tracked_bboxes: dict):
    coordinates = []
    for id in ids:
        x_min, y_min, x_max, y_max = tracked_bboxes.get(id)
        coordinates.append([x_min, y_min, x_max, y_max])
    df = pd.DataFrame(data=coordinates, columns=["x_min", "y_min", "x_max", "y_max"])
    return min(df["x_min"]), min(df["y_min"]), max(df["x_max"]), max(df["y_max"])


def get_smooth_tracklet(tracklets: list, window: int):
    if len(tracklets) < window:
        window = len(tracklets)
    return [sum(np.array(list(tracklets))[-window:, 0]) / window, sum(np.array(list(tracklets))[-window:, 1]) / window]


def apply_trajectory_smoothing(data, n_frames, smoothing_window, reverse=False):
    smooth_tracklets = pd.DataFrame(columns=["frame_id", "track_id", "tracklet"])
    for frame_id in range(n_frames):
        if reverse:
            frame_id = n_frames - frame_id - 1
        for track_id in set(data.loc[data["frame_id"] == frame_id]["track_id"]):
            raw_tracklet = dataframe.get_item(data, frame_id, track_id, "frame_tracklet")
            if raw_tracklet is None:
                smooth_tracklets.loc[len(smooth_tracklets)] = [frame_id, track_id, None]
                continue
            previous = dataframe.get_full_list(smooth_tracklets, track_id, "tracklet")
            previous.append(raw_tracklet)
            smooth_tracklet = get_smooth_tracklet(previous, smoothing_window)
            smooth_tracklets.loc[len(smooth_tracklets)] = [frame_id, track_id, smooth_tracklet]
    return smooth_tracklets


def get_coordinate_weight(data, frame_id, track_id, smoothing_window):
    tracklet_id = len(dataframe.get_current_list(data, frame_id, track_id, "frame_tracklet")) - 1
    n_tracklets = len(dataframe.get_full_list(data, track_id, "frame_tracklet")) - 1
    interval = (smoothing_window * (smoothing_window + 1)) // 2
    weight = 0.5
    if tracklet_id < interval:
        weight = 1 - ((1 / (2 * interval)) * tracklet_id)
    elif tracklet_id > n_tracklets - interval:
        weight = 0.5 - ((1 / (2 * interval)) * (tracklet_id - (n_tracklets - interval)))
    return weight


def convert_img2world(img_point, homography_matrix):
    if homography_matrix is None:
        return None
    result = np.matmul(homography_matrix, np.array([img_point[0], img_point[1], 1]))
    scale = [result[2]]
    return list(result / scale)


def plot_homography(video_name, tracklets, colors):
    plt.axis([-35, 20, -2.5, 10])
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(2)
    for track_id, tracklet in tracklets.items():
        plt.scatter(np.array(tracklet)[:, 0], np.array(tracklet)[:, 1], s=1, color=colors[int(track_id) % len(colors)])
    plt.savefig(f"results/{video_name}_homography.png")


def get_frame_velocity(vx, vy):
    return math.sqrt(vx ** 2 + vy ** 2)


def get_world_velocity(world_tracklets: list, fps: int):
    if len(world_tracklets) < 2:
        return 0
    window = len(world_tracklets) if len(world_tracklets) < fps else fps
    total_distance = 0
    for i in reversed(range(len(world_tracklets) - window, len(world_tracklets), 2)):
        total_distance += math.dist(world_tracklets[i], world_tracklets[i - 1])
    return (total_distance / window) * fps
