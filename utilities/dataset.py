import os
import random
import numpy as np
import mmcv
import vipy
import pickle
from pathlib import Path
from collections import defaultdict
from mmpose.apis import init_pose_model, inference_top_down_pose_model

label_map = "libs/mmaction/label_maps/label_map_pip12.txt"

actions = {
    0: "person_picks_object",
    1: "person_places_object",
    2: "person_carries_object",
    3: "person_transfers_object_to_person",
    4: "person_interacts_with_phone",
    5: "person_interacts_with_laptop",
    6: "person_talks_to_person",
    7: "person_talks_on_phone",
    8: "person_purchases_from_cashier",
    9: "person_sits_down",
    10: "person_stands_up",
    11: "person_walks"
}

pip_370k_filter = {
    "person_picks_up_object": 0,
    "person_picks_up_object_from_floor": 0,
    "person_picks_up_object_from_shelf": 0,
    "person_picks_up_object_from_table": 0,
    "person_puts_down_object": 1,
    "person_puts_down_object_on_floor": 1,
    "person_puts_down_object_on_shelf": 1,
    "person_puts_down_object_on_table": 1,
    "person_carries_heavy_object": 2,
    "person_transfers_object_to_person": 3,
    "person_texts_on_phone": 4,
    "person_interacts_with_laptop": 5,
    "person_talks_to_person": 6,
    "person_talks_on_phone": 7,
    "person_purchases_from_cashier": 8,
    "person_sits_down": 9,
    "person_stands_up": 10,
    "person_walks": 11
}


def load_vipy_dataset(json):
    dataset = vipy.util.load(json)
    categories = set([v.category() for v in dataset])
    return dataset, categories


def save_vipy_dataset(dataset, json):
    vipy.util.save(dataset, json)


def filter_vipy_dataset(original_dataset, original_categories, filter):
    filtered_dataset = [v for v in original_dataset if v.category() in filter]
    filtered_categories = set([v.category() for v in filtered_dataset])
    print("\n**categories**")
    print(f"original ({len(original_categories)}): {original_categories}")
    print(f"filtered ({len(filtered_categories)}): {filtered_categories}")
    print("\n**videos**")
    print(f"original ({len(original_dataset)})")
    print(f"filtered ({len(filtered_dataset)})")
    return filtered_dataset, filtered_categories


def detection_inference(track_results):
    detection_results = defaultdict(list)
    for i, track in track_results.items():
        if track.category().lower() == 'person':
            frames = track.keyframes()
            detections = track.keyboxes()
            for frame, detection in zip(frames, detections):
                detection_results[frame].append(detection.ulbr())
    return detection_results


def pose_inference(model, frames, detection_results):
    ids, detections = list(detection_results.keys()), list(detection_results.values())
    frames = [frames[i] for i in ids if i < len(frames)]
    num_frame = len(frames)
    print(num_frame)
    num_person = max([len(detection) for detection in detections], default=0)
    pose_results = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)
    for i, (frame, detection) in enumerate(zip(frames, detections)):
        detection = [dict(bbox=bbox) for bbox in list(detection)]
        pose = inference_top_down_pose_model(model, frame, detection, format='xyxy')[0]
        for j, item in enumerate(pose):
            pose_results[j, i] = item['keypoints']
    return pose_results


def extract_annotations(video, pose_model):
    path = str(video.abspath().filename())
    detection_results = detection_inference(video.tracks())
    video_capture = video.get_video_capture(path)
    pose_results = pose_inference(pose_model, list(video.generate_frames(video_capture)), detection_results)
    video_capture.release()
    anno = dict()
    anno['frame_dir'] = os.path.splitext(os.path.basename(path))[0]
    anno['label'] = pip_370k_filter[video.abspath().category()]
    anno['img_shape'] = (video.height(), video.width())
    anno['original_shape'] = (video.height(), video.width())
    anno['total_frames'] = pose_results.shape[1]
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    return anno


def generate_mmaction_skeleton_dataset(vipy_path, dataset_name, pose_config, pose_checkpoint):
    os.makedirs(dataset_name, exist_ok=True)
    previous = []
    if os.path.isfile(f"{dataset_name}/progress.txt"):
        with open(f"{dataset_name}/progress.txt", 'r') as prog_file:
            previous = prog_file.read().splitlines()
    with open(f"{dataset_name}/progress.txt", 'a') as prog_file:
        pose_model = init_pose_model(pose_config, pose_checkpoint)
        vipy_dataset = vipy.util.load(vipy_path, True)
        prog_bar = mmcv.ProgressBar(len(vipy_dataset))
        for video in vipy_dataset:
            video_dir = video.abspath().category()
            video_name = os.path.splitext(os.path.basename(str(video.abspath().filename())))[0]
            if f"{video_dir}/{video_name}" not in previous:
                os.makedirs(f"{dataset_name}/{video_dir}", exist_ok=True)
                anno = extract_annotations(video, pose_model)
                mmcv.dump(anno, f"{dataset_name}/{video_dir}/{video_name}.pkl")
                prog_file.write(f"{video_dir}/{video_name}\n")
            prog_bar.update()


def remove_flawed_annotations(dataset_path):
    flawed = []
    path_list = Path(dataset_path).rglob('*.pkl')
    for path in path_list:
        with open(path, 'rb') as f:
            annotation = pickle.load(f)
        if len(annotation['keypoint']) == 0 or annotation['total_frames'] == 0:
            os.remove(path)
            flawed.append(str(path).replace('\\', '/').replace(f"{dataset_path}/", '').split('.')[0])
    with open('pip/progress.txt', 'r') as f:
        progress = f.readlines()
    with open('pip/progress.txt', 'w') as f:
        for line in progress:
            if line.strip() not in flawed:
                f.write(line)


def check_labels(dataset_path):
    labels = defaultdict(set)
    path_list = Path(dataset_path).rglob('*.pkl')
    for path in path_list:
        with open(path, 'rb') as f:
            annotation = pickle.load(f)
        labels[str(path).replace('\\', '/').replace(f"{dataset_path}/", '').split('/')[0]].add(annotation['label'])
    return labels


def assemble_mmaction_skeleton_dataset(dataset_path, split_percent):
    path_list = list(Path(dataset_path).rglob('*.pkl'))
    total_results = defaultdict(list)
    train_results = []
    val_results = []
    prog_bar = mmcv.ProgressBar(len(path_list))
    for path in path_list:
        with open(path, 'rb') as f:
            annotation = pickle.load(f)
        total_results[annotation['label']].append(annotation)
        prog_bar.update()
    for result in total_results.values():
        random.shuffle(result)
        split_index = int(len(result) * split_percent)
        train_results.extend(result[:split_index])
        val_results.extend(result[split_index:])
    with open(f"{dataset_path}/custom_dataset_train.pkl", 'wb') as train_out:
        pickle.dump(train_results, train_out, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{dataset_path}/custom_dataset_val.pkl", 'wb') as val_out:
        pickle.dump(val_results, val_out, protocol=pickle.HIGHEST_PROTOCOL)
