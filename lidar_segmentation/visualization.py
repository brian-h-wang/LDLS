import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from lidar_segmentation.segmentation import LidarSegmentationResult
import seaborn as sns
sns.set()

import numpy as np
from moviepy.editor import ImageSequenceClip

from colorsys import hsv_to_rgb

"""
Reference for visualization:
https://github.com/navoshta/KITTI-Dataset/blob/master/kitti_demo-dataset.ipynb
"""

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

COCO_CLASS_COLORS = {i: 255*hsv_to_rgb(*np.random.rand(3)) for i in range(len(CLASS_NAMES))}
COCO_CLASS_COLORS[0] = np.array([30, 30, 30]) # dark gray color for BG points

COLORS = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'b',
    'Truck': 'b',
    'Pedestrian': 'y',
    'Sitter': 'k',
    'car': 'b',
    'bicycle': 'g',
    'person': 'r',
    'truck': 'b',
    'motorcycle': 'g',
    'bus': 'b',
    'table': 'c',
    'chair': 'c',
    'BG': 'k',
    'other': 'm'
}
AXES_LIMITS_KITTI = [
    [0, 40],  # X axis range
    [-20, 20],  # Y axis range
    [-3, 10]  # Z axis range
]
AXES_LIMITS_JACKAL = [
    [0, 20],  # X axis range
    [-10, 10],  # Y axis range
    [-1, 10]  # Z axis range
]

AXES_LIMITS = AXES_LIMITS_JACKAL
AXES_STR = ['X', 'Y', 'Z']


class PointCloudVisualizer(object):

    def __init__(self):
        fig = plt.figure()
        ax = fig.add_axes(projection='3d')

def color_array(class_ids):
    # Get mapping of class ids to color strings
    id_to_color = lambda x: COLORS[CLASS_NAMES[x]] if CLASS_NAMES[x] in COLORS.keys() else COLORS['other']
    return [id_to_color(x) for x in class_ids]


def visualize(segmentation, name=None, image_name=None, class_names=[]):
    """
    Visualize a segmentation result
    
    Parameters
    ----------
    segmentation: LidarSegmentationResult

    Returns
    -------

    """
    fig = plt.figure(figsize=(20,10))
    if image_name is None:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim3d(*AXES_LIMITS[0])
    ax.set_ylim3d(*AXES_LIMITS[1])
    ax.set_zlim3d(*AXES_LIMITS[2])

    # get class ID for each point
    instance_class_ids = np.concatenate([np.array([0]), segmentation.class_ids])
    point_class_ids = instance_class_ids[segmentation.instance_labels()]
    colors = color_array(point_class_ids)
    # colors = [COCO_CLASS_COLORS[class_id] for class_id in point_class_ids]
    points = segmentation.points

    ax.scatter(points[:,0], points[:,1], points[:,2], '.', s=0.05, c=colors)
    # point_handles = []
    # for class_id in np.unique(point_class_ids):
    #     p = point_class_ids == class_id  # indices of points with this class ID
    #     color = np.array([COCO_CLASS_COLORS[class_id] for i in range(np.sum(p))])
    #     color = color / 255.
    #     handle = ax.scatter(points[p,0], points[p,1], points[p,2], '.', s=0.05,
    #                c=color)
    #     point_handles.append(handle)
    ax.view_init(elev=15, azim=-160)
    # plt.legend(point_handles, [CLASS_NAMES[i] for i in np.unique(point_class_ids)])
    if image_name is not None:
        img = plt.imread(image_name)
        img_ax = fig.add_subplot(122, xticks=[], yticks=[])
        plt.imshow(img)
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
        plt.close(fig)


if __name__ == '__main__':
    from pathlib import Path
    # base_path = Path("/Users/BrianWang/Repos/lidar-stereo-autolabel/results/kitti/2011_09_26_drive0106/")
    # image_path = Path("/Users/BrianWang/Repos/lidar-stereo-autolabel/data/kitti/2011_09_26_drive0106/2011_09_26_drive_0106_sync/image_02/data")
    base_path = Path("/Users/BrianWang/Repos/lidar-stereo-autolabel/results/jackal_data/")
    image_path = Path("/Users/BrianWang/Repos/lidar-stereo-autolabel/data/jackal_data/image")
    segmentation_path = base_path / "segmentation"

    frames_path = base_path / "frames_with_image"
    # frames_path = base_path / "frames"
    if not frames_path.is_dir():
        frames_path.mkdir(parents=True)

    print("Preparing frames...")
    frames = []
    frame_nums = range(70,2782) # 2782 frames in Jackal data in total

    results = []
    seg_file_paths = [str(segmentation_path / ("%06d.npz" % i)) for i in frame_nums]
    results = [LidarSegmentationResult.load_file(seg_file_path) for
               seg_file_path in seg_file_paths]
    all_class_ids = set()
    for result in results:
        u = np.unique(result.class_labels())
        all_class_ids = all_class_ids.union(set(u))

    all_class_names = [CLASS_NAMES[cid] for cid in all_class_ids]

    for i,result in zip(frame_nums, results):
        print("Frame %d" % i)
        frame_file_path = str(frames_path / ("%06d.png" % i))
        image_file_path = str(image_path / ("%06d.png" % i))
        # if not Path(frame_file_path).is_file():
        visualize(result, name=frame_file_path, image_name=image_file_path,
                  class_names=all_class_names)
        frames.append(frame_file_path)
    frames = [str(frames_path / ("%06d.png" % i)) for i in frame_nums]

    print(frames)
    clip = ImageSequenceClip(frames, fps=10)
    # clip.write_gif(str(base_path / "results.gif"), fps=5)
    clip.write_videofile(str(base_path / "results_with_image.mp4"), fps=10)

    # visualize(result)




