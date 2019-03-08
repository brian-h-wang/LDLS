
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

    def create_label_image(self, mask_shrink=0.5, mask_dilate=2.0):
        """
        
        Returns
        -------
        numpy.ndarray
            -1 means no label
            0 means background
            Other positive integers mean object instances

        """
        mask_erode = MaskErode(self.masks, mask_shrink=mask_shrink,
                               mask_dilate=mask_dilate)
        label_image = mask_erode.create_label_image()
        return label_image

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


class MaskErode(object):

    def __init__(self, masks, mask_shrink=0.5, mask_dilate=2.0, ksize=5):
        assert 0 < mask_shrink <= 1.0, "Mask shrink must be between 0 and 1"
        assert 1.0 <= mask_dilate, "Mask dilate must be >= 1"
        self.ksize = ksize
        self.mask_shrink = mask_shrink
        self.mask_dilate = mask_dilate
        self.masks = masks
        self.label_image = None
        self.n_masks = masks.shape[2]

    def erode_object_mask(self, i, n_iters=None):
        kernel_size = self.ksize
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = self.masks[:,:,i]
        mask_image = mask.astype(np.uint8)
        mask_size = np.sum(mask_image)
        original_size = mask_size
        if n_iters is None:
            while mask_size > (self.mask_shrink * original_size):
                mask_image = binary_erosion(mask_image, kernel)
                # mask_image = erode(mask_image, kernel, iterations=1)
                mask_size = np.sum(mask_image)
        return mask_image

    def dilate_object_mask(self, i, n_iters=None):
        kernel_size = self.ksize
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = self.masks[:,:,i]
        mask_image = mask.astype(np.uint8)
        mask_size = np.sum(mask_image)
        original_size = mask_size
        max_size = self.masks.shape[0] * self.masks.shape[1]
        while mask_size < (self.mask_dilate * original_size):
            if mask_size >= max_size:
                break # stop if mask dilated to fill whole image
            mask_image = binary_dilation(mask_image, kernel)
            # mask_image = erode(mask_image, kernel, iterations=1)
            mask_size = np.sum(mask_image)
        return mask_image

    def erode_background_mask(self):
        kernel_size = self.ksize
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        dilated_masks = np.stack([self.dilate_object_mask(i) for i in range(self.n_masks)], axis=2)
        # import matplotlib.pyplot as plt
        # for i in range(self.n_masks):
        #     plt.subplot(self.n_masks+1, 1, i+1)
        #     plt.imshow(dilated_masks[:,:,i])
        bg_image = np.logical_not(np.logical_or.reduce(dilated_masks, axis=2)).astype(np.uint8)
        # plt.subplot(self.n_masks+1, 1, self.n_masks+1)
        # plt.imshow(bg_image)
        # plt.show()
        # If no masks, everything is background:
        # if self.masks.shape[2] == 0:
        #     return bg_image
        # bg_size = np.sum(bg_image)
        # original_size = bg_size
        # while bg_size > (self.bg_shrink * original_size):
        #     bg_image = binary_erosion(bg_image, kernel)
        #     bg_image = erode(bg_image, kernel, iterations=1)
            # bg_size = np.sum(bg_image)
        return bg_image.astype(int)


    def create_label_image(self):
        # assert 0 < self.mask_shrink <= 1.0, "Mask shrink factor must be a float between 0 and 1.0"
        # assert 0 < self. <= 1.0, "Background shrink factor must be a float between 0 and 1.0"
        rows, cols, n_masks = self.masks.shape
        if n_masks == 0:
            return np.zeros((rows, cols), dtype=int)
        # All pixels have label -1 (unknown) initially
        label_image = -np.ones((rows, cols), dtype=int)

        bg_mask = self.erode_background_mask().astype(bool)
        eroded_masks = [self.erode_object_mask(i).astype(bool)
                        for i in range(n_masks)]
        original_shape = label_image.shape
        label_image = label_image.flatten()
        for i in range(n_masks):
            label_image[eroded_masks[i].flatten()] = i + 1
        label_image[bg_mask.flatten() == 1] = 0


        label_image = label_image.reshape(original_shape)
        # label_image += 1

        # Set all pixels on border of label image to be -1
        # label_image[0, :] = -1
        # label_image[rows - 1, :] = -1
        # label_image[:, 0] = -1
        # label_image[:, cols - 1] = -1

        self.label_image = label_image
        return label_image

