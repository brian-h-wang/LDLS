"""
kitti_util.py
Brian Wang

Utilities for working with KITTI data files.

"""

from lidar_segmentation.utils import Projection, load_image
import numpy as np

from lidar_segmentation.detections import Detections, CLASS_NAMES
from scipy.spatial import Delaunay

class KittiProjection(Projection):

    def __init__(self, Tr, P):
        super().__init__(Tr, P)

    @classmethod
    def load_object(cls, calib_path):
        """
        Load a calibration file from KITTI object detection data.
        
        Parameters
        ----------
        calib_path: str
            Path to calibration file

        Returns
        -------

        """
        calib_dict = cls._file_to_calib_dict(calib_path)
        # Get transformation matrices from the calibration data
        velo_to_cam = calib_dict['Tr_velo_to_cam'].reshape((3, 4))
        Tr = np.concatenate(
            [velo_to_cam, np.array([0, 0, 0, 1]).reshape((1, 4))])
        P = calib_dict['P2'].reshape((3, 4))
        return cls(Tr, P)

    @classmethod
    def load_raw(cls, v2c_path, c2c_path):
        """
        Load a calibration file from KITTI raw data
        
        Parameters
        ----------
        v2c_path: str
            Path to the file "calib_velo_to_cam.txt"
        c2c_path: str
            Path to the file "calib_cam_to_cam.txt"

        Returns
        -------

        """
        v2c_dict = cls._file_to_calib_dict(v2c_path)
        c2c_dict = cls._file_to_calib_dict(c2c_path)

        # Get transformation matrix from velo_to_cam file
        T = v2c_dict['T']
        R = v2c_dict['R']
        Tr = np.eye(4)
        Tr[0:3,0:3] = R.reshape((3,3))
        Tr[0:3, 3] = T

        # Get projection matrix from cam_to_cam file
        P = c2c_dict['P_rect_02'].reshape((3,4))
        return cls(Tr, P)

    @staticmethod
    def _file_to_calib_dict(calib_path):
        # Read values from the KITTI-specification calibration file
        calib_dict = {}
        with open(calib_path, "r") as calib_file:
            for line in calib_file.readlines():
                s = line.split(":")
                if len(s) < 2: continue  # ignore blank lines
                name = s[0]
                try:
                    data = np.array(
                        [np.float32(num_str) for num_str in s[1].split()])
                    calib_dict[name] = data
                except ValueError:  # not a valid float value
                    continue
        return calib_dict

    def inverse_transform(self, points):
        """
        Perform inverse 3D transformation
        
        Returns
        -------
        np.ndarray
            4 by 4 matrix

        """
        Tr = self.transformation_matrix
        Rinv = Tr[0:3, 0:3].T
        d = Tr[0:3, 3]
        Tr_inv = np.zeros((4, 4))
        Tr_inv[0:3, 0:3] = Rinv
        Tr_inv[0:3, 3] = -Rinv.dot(d)
        Tr_inv[3, 3] = 1
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        return ((Tr_inv.dot(points.T)).T)[:,0:3]


def load_kitti_lidar_data(filename, verbose=False, load_reflectance=False):
    """
    Loads lidar data stored in KITTI format.
    
    Parameters
    ----------
    filename
    verbose

    Returns
    -------
    numpy.ndarray
        n_points by 4 array.
        Columns are x, y, z, reflectance

    """
    with open(filename, "rb") as lidar_file:
        # Velodyne data stored as binary float matrix
        lidar_data = np.fromfile(lidar_file, dtype=np.float32)
        # Velodyne data contains x,y,z, and reflectance
        lidar_data = lidar_data.reshape((-1,4))
    if verbose:
        print("Loaded lidar point cloud with %d points." % lidar_data.shape[0])
    if load_reflectance:
        return lidar_data
    else:
        return lidar_data[:,0:3]

def load_kitti_object_calib(calib_path):
    return KittiProjection.load_object(calib_path)

def check_points_in_box(points, box_corner_points):
    """
    Check if points are within a 3D bounding box.
    
    Parameters
    ----------
    points : ndarray(dtype=float, ndims=2)
        n_points by 3
    box_corner_points : ndarray
        8 by 3, box vertices

    Returns
    -------
    ndarray(dtype=bool)
        n_points-length array. Element i is 1 if point i is in the box and
        0 if not.
    """
    # Create Delaunay tessalation for the 3D bounding box
    # Then, can use Delaunay.find_simplex() to determine whether a point is
    # inside the box
    hull = Delaunay(box_corner_points)

    in_box = hull.find_simplex(points) >= 0
    return in_box

class KittiLabel(object):
    """
    Attribute descriptions, from KITTI object detection readme.
    Note that all 3D coordinates are given in the 3D camera frame.
    
    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
    alpha        Observation angle of object, ranging [-pi..pi]
    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
    dimensions   3D object dimensions: height, width, length (in meters)
    location     3D object location x,y,z in camera coordinates (in meters)
    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
    """

    def __init__(self, object_type, truncated, occluded, alpha, bbox,
                 dimensions, location, rotation_y):

        self.object_type = object_type  # type: str
        self.truncated = truncated  # type: bool
        self.occluded = occluded  # type: bool
        self.alpha = alpha  # type: float

        # bbox is [left, top, right, bottom] (in pixel coordinates)
        self.bbox = bbox  # type: np.ndarray

        # dimensions is [height, width, length] in 3D coordinates
        self.dimensions = dimensions  # type: np.ndarray

        # location is [x, y, z] in 3D camera coordinates
        self.location = location  # type: np.ndarray
        self.rotation_y = rotation_y  # type: float

    def box_corners(self):
        corner_points = np.empty((8, 3))
        height, width, length = self.dimensions

        k = 0
        # for dx in [-length / 2, length / 2]:
        #     for dy in [-width / 2, width / 2]:
        #         for dz in [0, height]:
        #             corner_points[k, :] = [dx, dy, dz]
        #             k += 1
        for dx in [-width / 2, width / 2]:
            for dz in [-length / 2, length / 2]:
                for dy in [0, -height]:
                    corner_points[k, :] = [dx, dy, dz]
                    k += 1

        # Rotate around z (vertical) axis
        # rotation = -obj.rotation_y + (np.pi / 2)
        rotation = -self.rotation_y + np.pi/2
        c = np.cos(rotation)
        s = np.sin(rotation)
        # R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        corner_points = (R.dot(corner_points.T)).T
        # add center coordinates to each point
        corner_points = corner_points + self.location
        return corner_points


def load_kitti_labels(filename):
    """
    
    Parameters
    ----------
    filename: str
    projection: KittiProjection
        Camera-Lidar projection. Needed to convert bounding box points from
        camera frame into lidar frames.

    Returns
    -------

    """
    with open(filename, "r") as label_file:
        lines = label_file.readlines()
    # Tr_c2v = projection.inverse_transform()  # camera to velodyne transform
    objects = []
    for line in lines:
        split_line = line.split(" ")
        if len(split_line) < 2:  # not a valid object line
            continue
        object_type = split_line[0]
        truncated = float(split_line[1])
        occluded = int(split_line[2])
        alpha = float(split_line[3])
        bbox = np.array([float(x) for x in split_line[4:8]])
        dimensions = np.array([float(x) for x in split_line[8:11]])
        location_cam = np.array([float(x) for x in split_line[11:14]])

        # transform camera frame center point to lidar frame coordinates
        # location_cam_h = np.append(location_cam, 1).reshape((4,1))
        # location_vel_h = Tr_c2v.dot(location_cam_h)
        # location = location_vel_h[0:3,0] # lidar frame 3D coordinates
        location = location_cam

        rotation_y = float(split_line[14])
        objects.append(KittiLabel(object_type, truncated, occluded,
                                  alpha, bbox, dimensions, location,
                                  rotation_y))
    return objects

class KittiLabelDetections(Detections):
    """
    Class for creating detections from KITTI annotation 2D bounding boxes.
    
    Considers Person and Car classes only
    
    """

    def __init__(self, label_file_path, image_file_path):
        image = load_image(image_file_path)
        self.shape = image.shape
        self.object_labels = load_kitti_labels(label_file_path)
        self.object_labels = [label for label in self.object_labels
                              if label.object_type in ['Pedestrian', 'Car']]
        # KITTI bounding boxes are left, top, right, bottom
        masks = np.zeros((self.shape[0], self.shape[1],
                          len(self.object_labels)),
                         dtype=int)
        self.class_ids = []
        for i, label in enumerate(self.object_labels):
            if label.object_type == 'Pedestrian':
                self.class_ids.append(CLASS_NAMES.index('person'))
            elif label.object_type == 'Car':
                self.class_ids.append(CLASS_NAMES.index('car'))
            left, top, right, bottom = [int(x) for x in label.bbox]
            masks[top:bottom, left:right, i] = 1
        self.masks = masks

    def __len__(self):
        return len(self.object_labels)


class KittiBoxSegmentationResult(object):

    def __init__(self, lidar, kitti_labels, proj):
        self.labels = kitti_labels

        car = CLASS_NAMES.index('car')
        pedestrian = CLASS_NAMES.index('person')

        inst = -np.ones(lidar.shape[0], dtype=int)
        cls = np.zeros(lidar.shape[0], dtype=int)

        for i, label in enumerate(kitti_labels):
            in_box = check_points_in_box(lidar, proj.inverse_transform(label.box_corners()))
            if label.object_type == 'Car':
                inst[in_box] = i
                cls[in_box] = car
            elif label.object_type == 'Pedestrian' or label.object_type == 'Person_sitting':
                inst[in_box] = i
                cls[in_box] = pedestrian
        self.inst = inst
        self.cls = cls

    def instance_labels(self):
        return self.inst

    def class_labels(self):
        return self.cls
