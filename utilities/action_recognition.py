from utilities import skeleton
from utilities import dataset

import os
import re
import ast
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from mmaction.apis import inference_recognizer


def get_action_prediction(recognizer, pose_results, num_frames, height, width):
    annotation = dict(
        frame_dir='',
        label=-1,
        img_shape=(height, width),
        original_shape=(height, width),
        start_index=0,
        modality='Pose',
        total_frames=num_frames)
    keypoint = np.zeros((1, num_frames, skeleton.num_kpts, 2), dtype=np.float16)
    keypoint_score = np.zeros((1, num_frames, skeleton.num_kpts), dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            keypoint[j, i] = pose["keypoints"][:, :2]
            keypoint_score[j, i] = pose["keypoints"][:, 2]
    annotation['keypoint'] = keypoint
    annotation['keypoint_score'] = keypoint_score
    results = inference_recognizer(recognizer, annotation)
    labels = [x.strip() for x in open(dataset.label_map).readlines()]
    return labels[results[0][0]]


def illustrate_confusion_matrix(cm_path):
    with open(cm_path, "r") as f:
        data = f.read()
    cm = np.array(ast.literal_eval(re.sub(' +', ' ', data).replace(' [ ', '[').replace(' ', ',').replace('\n', ',').split(",,")[0]))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(11.5, 10))
    classes = dataset.actions.values()
    cm = seaborn.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap='Blues', fmt='.2f')
    cm.set(xlabel='predicted', ylabel='actual')
    cm.figure.tight_layout()
    plt.savefig(os.path.dirname(cm_path) + "/confusion_matrix.png")


def show_performance_scores(cm_path):
    with open(cm_path, "r") as f:
        data = f.read()
    cm = np.array(ast.literal_eval(re.sub(' +', ' ', data).replace(' [ ', '[').replace(' ', ',').replace('\n', ',').split(",,")[0]))
    accuracy = np.trace(cm) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
    print(f"\n** performance metrics **\n"
          f"\naccuracy: {accuracy}\n"
          f"macro-averaged precision: {np.mean(precision)}\n"
          f"macro-averaged recall: {np.mean(recall)}\n"
          f"macro-averaged f1-score: {np.mean(f1_score)}")
