"""

Module for performing object segmentation and initial labeling of images.

Reference:
Uses the Mask-RCNN detector from https://github.com/matterport/Mask_RCNN

"""

import os
import sys

import keras.backend
import tensorflow

from lidar_segmentation.detections import MaskRCNNDetections

# Leave part of the GPU memory unallocated, so can be used for label diffusion
gpu_opt = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tensorflow.ConfigProto(gpu_options=gpu_opt)

# config = tensorflow.ConfigProto()
config.inter_op_parallelism_threads = 1
keras.backend.set_session(tensorflow.Session(config=config))

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version

from mask_rcnn import coco


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNDetector(object):
    def __init__(self, use_gpu=False):
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = InferenceConfig()
        # config.display()

        # Create model object in inference mode.
        if use_gpu:
            # with tensorflow.device("/gpu:0"):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)
        else:
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.model = model

    def detect(self, images, verbose=0):
        """
        Run Mask-RCNN to detect objects.
        Input can be one image, or a list of images
        
        Parameters
        ----------
        images: numpy.ndarray or list
        verbose: int
            0 or 1.

        Returns
        -------
        MaskRCNNDetections, or list of MaskRCNNDetections objects
            

        """
        detect_multiple = type(images) == list
        if not detect_multiple:
            images = [images]
        all_results = self.model.detect(images, verbose=verbose)
        all_detections = [MaskRCNNDetections(shape=image.shape,
                                             rois=result['rois'],
                                             masks=result['masks'],
                                             class_ids=result['class_ids'],
                                             scores=result['scores'])
                          for image,result in zip(images, all_results)]
        if not detect_multiple:
            return all_detections[0]
        else:
            return all_detections


