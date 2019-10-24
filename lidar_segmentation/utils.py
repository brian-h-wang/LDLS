"""
util.py
Brian Wang

Utilities for loading in lidar and image data.

"""
import numpy as np
from skimage.io import imread

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


class Projection(object):
    """
    Class for projecting lidar points into a 2D image frame.
    
    Project points using the Projection.project() method.
    
    Attributes
    ----------
    T: numpy.ndarray
        Transformation matrix. 4 by 4
        Transforms 3D homogeneous coordinate lidar points to 3D homogeneous
        cooordinate points in the camera fram.e
    P: numpy.ndarray
        Projection matrix. 3 by 4.
        Project a 3D point (x,y,z) to 2D image coordinates by appending a 1,
        for homogeneous coordinates, and then multiplying by P.
        
        R = P * [x y z 1]'
        
        Then, the image row coordinate is R[0]/R[2],
        and the column coordinate is R[1]/R[2]
        (i.e. divide the first and second dimensions by the third dimension)
        
    """


    def __init__(self, Tr, P):
        self.transformation_matrix = Tr
        self.projection_matrix = P

    def project(self, points, remove_behind=True):
        """
        Project points from the Velodyne coordinate frame to image frame
        pixel coordinates.
        
        Parameters
        ----------
        points: numpy.ndarray
            n by 3 numpy array.
            Each row represents a 3D lidar point, as [x, y, z]
        remove_behind: bool
            If True, projects all lidar points that are behind the camera
            (checked as x <= 0) to NaN

        Returns
        -------
        numpy.ndarray
            n by 2 array.
            Each row represents a point projected to 2D camera coordinates
            as [row, col]

        """
        n = points.shape[0]
        d = points.shape[1]
        Tr = self.transformation_matrix
        P = self.projection_matrix
        if d == 3:
            # Append 1 for homogenous coordinates
            points = np.concatenate([points, np.ones((n, 1))], axis=1)
        projected = (P.dot(Tr).dot(points.T)).T

        # normalize by dividing first and second dimensions by third dimension
        projected = np.column_stack(
            [projected[:, 0] / projected[:, 2],
             projected[:, 1] / projected[:, 2]])

        if remove_behind:
            behind = points[:,0] <= 0
            projected[behind,:] = np.nan

        return projected

def load_image(image_path):
    rgb_image = imread(str(image_path))
    return rgb_image

def load_csv_lidar_data(lidar_path):
    """
    
    Parameters
    ----------
    lidar_path: str or Path
        Lidar file path

    Returns
    -------
    ndarray
        n by 3 numpy array
    """
    # skip 1 header line
    points = np.genfromtxt(lidar_path, dtype=float, delimiter=',',
                           skip_header=1)

    # csv files from Jackal also include intensity, ring
    if points.shape[1] > 3:
        points = points[:,0:3]
    return points



