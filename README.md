# ECI4.0 Customer Behaviour Framework
This project was developed as part of the <a href="https://ciencia.iscte-iul.pt/projects/smart-commercial-spaces/1736" target="_blank">ECI 4.0 - Espaços Comerciais Inteligentes</a>
project, which was conducted as a partnership between AXIANSEU Digital Solutions SA (Axians), SONAE MC Serviços Partilhados SA (Sonae) and ISCTE-IUL.
It is primarily focused on the task "Development of pose classification models for the detection of client specific behaviours".

Taking this into account, the main goal of this work was to create a framework capable of effectively extracting information regarding customer behaviour, 
based on videos acquired from high-resolution surveillance cameras. This includes trajectory-related data (location and speed) and pose-related data (actions), 
both of which require stable and accurate location data for the continuous identification of each customer.

## Methodology

The process of extracting information from video regarding the behaviour of a person is composed of a series of steps. Initially, a video is loaded and decoded 
into a sequence of frames, which are stored in a list. Then, for each frame, an object detection algorithm is applied to identify and locate people. Next, a 
multi-object tracking algorithm is applied to assign a numerical identifier (ID) to each detected person over time. Based on this result, trajectory points are 
extracted, their respective projections on the 2D floor plan (top view) are obtained and an estimate of the speed at which each person moves is calculated. In 
addition to the trajectory data, skeleton sequences regarding the pose of each tracked customer are also extracted. These sequences are then utilised to infer the 
actions being performed. Furthermore, groups are identified by analysing factors such as the distance between people and the scale of their bounding boxes.

<p align="center">
    <img src="https://github.com/simaoc00/eci4.0-customer-behaviour/assets/58070852/a7ba6e54-7def-4106-861f-5c7898ca8b5b" alt="default" width="85%"/>
</p>

When certain obstacles are present in the scene, for instance cars, lampposts, trees, and bushes (in a street scene) or items, shelves, and banners (in a retail 
shop scene), they are likely to occlude people standing close to them. This causes the detections made by the object detection algorithm to simply surround 
the visible area of the occluded individuals. To address this problem, we developed a mechanism to detect whether or not an individual is occluded and, if so, to 
automatically adjust the dimensions of its bounding box to include the occluded area.

Furthermore, there are a few circumstances that can lead to oscillations in both the location and the dimensions of the bounding boxes throughout a sequence of 
consecutive video frames. These oscillations cause the extracted points to present irregularities in the trajectory. In order to mitigate these irregularities and, 
consecutively, generate more reliable path information, a smoothing method was applied.

## Usage

So that users can experiment with the project, we provided some videos in the [demos](demos) folder. These are divided into two sections: 
[occlusion_aware_detection_and_tracking](demos/videos/occlusion_aware_detection_and_tracking) and [action_recognition](demos/videos/action_recognition) - where the videos in the
former were chosen to demonstrate the trajectory-related data (including the occlusion-aware mechanism and trajectory smoothing method), and the videos in the
latter were chosen to demonstrate the pose-related data (since the action recognition models were trained to recognise the actions depicted in them).

With this in mind, you must first install all the necessary requirements using the following command:

```
cd <project root directory>
pip install -r requirements.txt
```

(IMPORTANT) Note that in the [requirements.txt](requirements.txt) file, there are two commented lines corresponding to the CPU version of PyTorch. If you don't have a CUDA-enabled GPU, 
you need to uncomment these lines and comment the CUDA versions instead:

```
-f https://download.pytorch.org/whl/torch_stable.html
# torch==1.11.0+cu113
torch==1.11.0+cpu
-f https://download.pytorch.org/whl/torch_stable.html
# torchvision==0.12.0+cu113
torchvision==0.12.0+cpu
```

Once the requirements have been installed, the framework can then be executed by running the [main.py](main.py) script.

```
cd <project root directory>
python main.py video_name --homography (optional) --action-recognizer (options=stgcn|2sagcn) (default=2sagcn)
```

An example command is as follows:

```
python main.py VIRAT_S_010208_03_000201_000232.mp4
```

## Citation

```bibtex
@inproceedings{correia2023eci40,
    title={Occlusion-Aware Pedestrian Detection and Tracking},
    author={Correia, Simão and Mendes, Diogo and Jorge, Pedro and Brandão, Tomás and Arriaga, Patrícia and Nunes, Luís},
    booktitle={2023 30th International Conference on Systems, Signals and Image Processing (IWSSIP)},
    year={2023}
}
```

## Acknowledgement

This project uses the works of [YOLOv5](https://github.com/ultralytics/yolov5), [ByteTrack](https://github.com/ifzhang/ByteTrack), [MMPose](https://github.com/open-mmlab/mmpose) 
(including [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)), and [MMAction2](https://github.com/open-mmlab/mmaction2) (including [ST-GCN](https://github.com/yysijie/st-gcn) & 
[2s-AGCN](https://github.com/lshiwjx/2s-AGCN)).
