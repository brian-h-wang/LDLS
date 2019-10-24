
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
from mrcnn import visualize
from colorsys import hsv_to_rgb

import matplotlib.pyplot as plt

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

class Detections(object):
    """
    Stores object detections for an image.
    """
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def create_label_image(self, mask_shrink=0.5, mask_dilate=1.5):
        """
        Outputs a label image, i.e. a (n_rows by n_cols) matrix whose entries
        are 0 (for background pixels) or integer numbers.
        
        So if label_image[i,j] == 4, then pixel (i,j) is part of instance 4
        
        Returns
        -------
        numpy.ndarray

        """
        raise NotImplementedError


class MaskRCNNDetections(Detections):

    def __init__(self, shape, rois, masks, class_ids, scores):
        self.shape = shape
        self.rois = rois
        self.masks = masks
        self.class_ids = class_ids # stored as ints
        self.scores = scores

    def __len__(self):
        return self.masks.shape[2]

    @property
    def class_names(self):
        return [CLASS_NAMES[i] for i in self.class_ids]

    @classmethod
    def load_file(cls, filename):
        """
        Load MaskRCNN detections from a zipped Numpy file
        Parameters
        ----------
        file

        Returns
        -------

        """
        filename = str(filename)
        if not filename.endswith(".npz"):
            filename += ".npz"
        with open(filename, "rb") as loadfile:
            npzfile = np.load(loadfile)
            detections = cls(shape=tuple(npzfile["shape"]), rois=npzfile["rois"],
                             masks=npzfile["masks"],
                             class_ids=npzfile["class_ids"],
                             scores=npzfile["scores"])
        return detections

    def to_file(self, filename):
        """
        Save to a zipped Numpy file
        
        Parameters
        ----------
        filename

        Returns
        -------
        None

        """
        with open(filename, "wb") as savefile:
            np.savez_compressed(savefile, shape=np.array(self.shape), rois=self.rois,
                     masks=self.masks, class_ids=self.class_ids,
                     scores=self.scores)

    def visualize(self, image):
        # Make dictionary of class colors
        hues = np.random.rand(len(CLASS_NAMES))
        s = 0.6
        v = 0.8

        # Separate out hues that actually appear in the image
        classes_in_image = list(set(self.class_ids))
        class_hues = np.linspace(0, 1.0, num=len(classes_in_image) + 1)[:-1]
        # Randomize hues but keep them separated and between 0 and 1.0
        class_hues = np.mod(class_hues + np.random.rand(), 1.0)
        np.random.shuffle(class_hues)
        for i, c in enumerate(classes_in_image):
            hues[c] = class_hues[i]

        class_colors = np.array([hsv_to_rgb(h, s, v) for h in hues]) * 255
        class_colors[0, :] = [175, 175, 175]  # set background color to grey

        # Visualize results


        visualize.display_instances(image, self.rois, self.masks, self.class_ids,
                                    CLASS_NAMES, self.scores,
                                    colors=[class_colors[i, :] / 255. for i in
                                            self.class_ids])

    def get_background(self):
        bg_mask = np.logical_not(np.logical_or.reduce(self.masks, axis=2))
        return bg_mask

