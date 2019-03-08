"""
Evaluate results.
"""

from lidar_segmentation.segmentation import LidarSegmentationResult
from lidar_segmentation.utils import CLASS_NAMES

import numpy as np
import time

from multiprocessing import Pool, cpu_count
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

GROUND_LEVEL = -1.6


class LidarSegmentationGroundTruth(object):
    """
    Segmentation ground truth.
    
    """

    def __init__(self, instance_labels, class_labels):
        self.instance_labels = np.array(instance_labels)
        self.class_labels = np.array(class_labels)

    def filter(self, filter_array):
        self.instance_labels = self.instance_labels[filter_array]
        self.class_labels = self.class_labels[filter_array]

    @classmethod
    def load_file(cls, filename):
        """
        Load ground truth from a .txt file with rows formatted as:
            instance_label class_label
        
        Instance and class labels are separated by a space.
        instance_label should be castable to an int,
        and class_label will be used as a string.
        
        Parameters
        ----------
        filename: str
            Name of file to load

        Returns
        -------
        LidarSegmentationGroundTruth

        """
        with open(filename, "r") as loadfile:
            lines = loadfile.readlines()
        splitlines = [line.split(" ") for line in lines]
        instance_labels = [int(s[0]) for s in splitlines]
        class_labels = [s[1] for s in splitlines]
        class_labels = [l[:-1] if l.endswith('\n') else l for l in class_labels]
        return cls(instance_labels, class_labels)


        # with Pool(n_workers) as p:
        #     results_list = p.map(load_avod_labeling_results, frame_range)

    @property
    def n_instances(self):
        return len(np.unique(self.instance_labels))

def get_dont_care_indices(gt):
    """
    Find indices of all points in DontCare, Van, or Cyclist regions from
    the ground truth labeling.
    """
    all_dont_care = np.array([False for i in range(len(gt.lidar))])
    for obj in gt.objects:
        if obj.object_type == "DontCare":
            left, top, right, bottom = obj.bbox
            proj_x = gt.projection['x'].values
            proj_y = gt.projection['y'].values
            in_x = np.logical_and(left < proj_x, proj_x < right)
            in_y = np.logical_and(top < proj_y, proj_y < bottom)
            dont_care = np.logical_and(in_x, in_y)
            all_dont_care = np.logical_or(all_dont_care, dont_care)
    invalid_class = np.logical_or(gt.lidar['class_label'].values == "Van",
                                  gt.lidar['class_label'].values == "Cyclist")
    invalid_class = np.logical_or(invalid_class,
                                  gt.lidar['class_label'].values == "Misc")
    all_dont_care = np.logical_or(all_dont_care, invalid_class)
    return all_dont_care


def evaluate_semantic_segmentation(results_list, gt_list, range_limit=None,
                                   cp_only=False, filter_ground=False,
                                   return_pr=False, remove_dont_care=False,
                                   return_pr_iu=False):
    """
    Evaluate labeling result as semantic segmentation (i.e. without considering object instances)

    Reports IoU over classes
    """
    # KITTI objects we don't care about
    # Any points inside the bounding box for one of these object classes will be ignored
    # This helps because e.g. KITTI Van objects can be identified by Mask-RCNN as
    # either trucks or cars, and Cyclists can be identified as bicycles, motorcycles, or persons.
    objects_to_ignore = ["DontCare", "Van", "Cyclist"]

    # Define list of classes to evaluate
    if cp_only:
        kitti_names = ['Car', 'Pedestrian']
        coco_names = ['car', 'person']
    else:
        kitti_names = ['Car', 'Pedestrian', 'Truck',
                       'Tram']  # These are KITTI object class names
        coco_names = ['car', 'person', 'truck', 'train']

    # Keep running total of intersection and union values, for each class
    i_totals = [0 for c in kitti_names]
    u_totals = [0 for c in kitti_names]
    fp_totals = [0 for c in kitti_names]
    fn_totals = [0 for c in kitti_names]
    for (results, gt) in zip(results_list, gt_list):
        # Get object class labels (not instance labels)
        results_class_ids = results.class_labels()
        results_class_labels = np.array([CLASS_NAMES[i] for i in results_class_ids])
        gt_class_labels = gt.class_labels
        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]


        # Find indices of lidar points that are in "DontCare" regions
        if remove_dont_care:
            all_dont_care = get_dont_care_indices(gt)
            # print("Removing %d DontCare, Van, Cyclist points" % np.sum(all_dont_care))
            results_class_labels[all_dont_care] = "BG"
            gt_class_labels[all_dont_care] = "BG"

        # Set Person_sitting to Pedestrian
        gt_class_labels[gt_class_labels == "Person_sitting"] = "Pedestrian"

        if range_limit is not None:
            lidar_points = results.points
            ranges = np.linalg.norm(lidar_points, axis=1)
            in_range = ranges < range_limit
            results_class_labels = results_class_labels[in_range]
            gt_class_labels = gt_class_labels[in_range]

        if filter_ground:
            lidar_points = results.points
            if range_limit is not None:
                lidar_points = lidar_points[in_range,:]
            not_ground = lidar_points[:,2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]

        # For each class C:
        for i in range(len(kitti_names)):
            kitti_class = kitti_names[i]
            coco_class = coco_names[i]
            # Find which lidar points are labelled as this class in the results,
            # and in the KITTI ground truth

            # account for results with KITTI class labels or COCO class lables
            r = np.logical_or(results_class_labels == coco_class, results_class_labels == kitti_class)
            g = gt_class_labels == kitti_class

            intersection = np.logical_and(r, g)
            union = np.logical_or(r, g)

            i_totals[i] += np.sum(intersection)
            u_totals[i] += np.sum(union)

            fp_totals[i] += np.sum(np.logical_and(r, np.logical_not(g)))
            fn_totals[i] += np.sum(np.logical_and(g, np.logical_not(r)))

            # iou = np.sum(intersection) / np.sum(union)
            # print("IoU for class %s is %.3f" % (kitti_class, iou))

    # true positives = intersection
    tp_totals = i_totals

    if return_pr:
        return tp_totals, fp_totals, fn_totals

    elif return_pr_iu:
        return tp_totals, fp_totals, fn_totals, i_totals, u_totals

    else:
        iou_list = [i / u for (i, u) in zip(i_totals, u_totals)]
        return iou_list


def semantic_tp_fp_fn(results_list, gt_list, range_limit=None,
                                   cp_only=False, filter_ground=False,
                                   return_pr=False, remove_dont_care=False,
                                   return_pr_iu=False):
    """
    Evaluate labeling result as semantic segmentation (i.e. without considering object instances)

    Reports IoU over classes
    """
    # KITTI objects we don't care about
    # Any points inside the bounding box for one of these object classes will be ignored
    # This helps because e.g. KITTI Van objects can be identified by Mask-RCNN as
    # either trucks or cars, and Cyclists can be identified as bicycles, motorcycles, or persons.
    objects_to_ignore = ["DontCare", "Van", "Cyclist"]

    # Define list of classes to evaluate
    if cp_only:
        kitti_names = ['Car', 'Pedestrian']
        coco_names = ['car', 'person']
    else:
        kitti_names = ['Car', 'Pedestrian', 'Truck',
                       'Tram']  # These are KITTI object class names
        coco_names = ['car', 'person', 'truck', 'train']

    # Keep running total of intersection and union values, for each class
    i_totals = [0 for c in kitti_names]
    u_totals = [0 for c in kitti_names]
    fp_totals = [0 for c in kitti_names]
    fn_totals = [0 for c in kitti_names]
    for (results, gt) in zip(results_list, gt_list):
        # Get object class labels (not instance labels)
        results_class_ids = results.class_labels()
        results_class_labels = np.array([CLASS_NAMES[i] for i in results_class_ids])
        gt_class_labels = gt.class_labels
        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]

        # Find indices of lidar points that are in "DontCare" regions
        if remove_dont_care:
            all_dont_care = get_dont_care_indices(gt)
            # print("Removing %d DontCare, Van, Cyclist points" % np.sum(all_dont_care))
            results_class_labels[all_dont_care] = "BG"
            gt_class_labels[all_dont_care] = "BG"

        # Set Person_sitting to Pedestrian
        gt_class_labels[gt_class_labels == "Person_sitting"] = "Pedestrian"

        if range_limit is not None:
            lidar_points = results.points
            ranges = np.linalg.norm(lidar_points, axis=1)
            in_range = ranges < range_limit
            results_class_labels = results_class_labels[in_range]
            gt_class_labels = gt_class_labels[in_range]

        if filter_ground:
            lidar_points = results.points
            if range_limit is not None:
                lidar_points = lidar_points[in_range,:]
            not_ground = lidar_points[:,2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]

        # For each class C:
        for i in range(len(kitti_names)):
            kitti_class = kitti_names[i]
            coco_class = coco_names[i]
            # Find which lidar points are labelled as this class in the results,
            # and in the KITTI ground truth

            # account for results with KITTI class labels or COCO class lables
            r = np.logical_or(results_class_labels == coco_class, results_class_labels == kitti_class)
            g = gt_class_labels == kitti_class

            intersection = np.logical_and(r, g)
            union = np.logical_or(r, g)

            i_totals[i] += np.sum(intersection)
            u_totals[i] += np.sum(union)

            fp_totals[i] += np.sum(np.logical_and(r, np.logical_not(g)))
            fn_totals[i] += np.sum(np.logical_and(g, np.logical_not(r)))

            # iou = np.sum(intersection) / np.sum(union)
            # print("IoU for class %s is %.3f" % (kitti_class, iou))

    # true positives = intersection
    tp_totals = i_totals

    if return_pr:
        return tp_totals, fp_totals, fn_totals

    elif return_pr_iu:
        return tp_totals, fp_totals, fn_totals, i_totals, u_totals

    else:
        iou_list = [i / u for (i, u) in zip(i_totals, u_totals)]
        return iou_list
def print_iou_results(iou_list, classes=('Car', 'Pedestrian', 'Truck', 'Tram')):
    for (iou, name) in zip(iou_list, classes):
        print("IoU for class %s is %.3f" % (name, iou))


class InstanceSegmentationResults(object):

    def __init__(self, iou_threshold, n_classes):
        self.iou_threshold = iou_threshold
        self.tp_totals = [0 for i in range(n_classes)]
        self.fp_totals = [0 for i in range(n_classes)]
        self.fn_totals = [0 for i in range(n_classes)]


def evaluate_instance_segmentation(results_list, gt_list,
                                   iou_threshold=0.7, range_limit=None,
                                   cp_only=False, filter_ground=False,
                                   remove_dont_care=False):
    """
    Evaluate labeling result as instance segmentation

    Reports IoU over classes

    Attributes
    ----------
    results_list: list
        List of LidarSegmentationResult
    gt_list: list
        List of LidarSegmentationGroundTruth
    iou_threshold: float
    range_limits: tuple, or None
        Specify range_limits to only look at objects at certain distances.
        Should contain two float values, e.g. (0,10) to look at objects
        from 0 to 10 meters away.
    """
    # KITTI objects we don't care about
    # Any points inside the bounding box for one of these object classes will be ignored
    # This helps because e.g. KITTI Van objects can be identified by Mask-RCNN as
    # either trucks or cars, and Cyclists can be identified as bicycles, motorcycles, or persons.
    objects_to_ignore = ["DontCare", "Van", "Cyclist"]

    # Define list of classes to evaluate
    if cp_only:
        kitti_names = ['Car', 'Pedestrian']
        coco_names = ['car', 'person']
    else:
        kitti_names = ['Car', 'Pedestrian', 'Truck',
                       'Tram']  # These are KITTI object class names
        coco_names = ['car', 'person', 'truck', 'train']

    # Keep running total of intersection and union values, for each class
    tp_totals = [0 for c in kitti_names]
    fp_totals = [0 for c in kitti_names]
    fn_totals = [0 for c in kitti_names]

    # iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # eval_results = [InstanceSegmentationResults(threshold) for threshold in iou_thresholds]

    for (results, gt) in zip(results_list, gt_list):
        # if range_limits is not None:
        #     ranges = np.linalg.norm(results[['x', 'y', 'z']].values,
        #                             axis=1)
        # Get object class labels (not instance labels)
        results_class_ids = results.class_labels()
        results_class_labels = np.array([CLASS_NAMES[i] for i in results_class_ids])
        gt_class_labels = gt.class_labels

        results_instance_labels = results.instance_labels()
        gt_instance_labels = gt.instance_labels

        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]
        if len(results_instance_labels) != len(gt_instance_labels):
            gt_instance_labels = gt_instance_labels[results.in_camera_view]
        # Find indices of lidar points that are in "DontCare" regions
        # all_dont_care = get_dont_care_indices(gt)
        # print("Removing %d DontCare, Van, Cyclist points" % np.sum(all_dont_care))
        # results_class_labels[all_dont_care] = "BG"
        # gt_class_labels[all_dont_care] = "BG"
        # results_instance_labels[all_dont_care] = -1
        # gt_instance_labels[all_dont_care] = -1

        # Set Person_sitting to Pedestrian
        gt_class_labels[gt_class_labels == "Person_sitting"] = "Pedestrian"

        if range_limit is not None:
            lidar_points = results.points
            ranges = np.linalg.norm(lidar_points, axis=1)
            in_range = ranges < range_limit
            results_class_labels = results_class_labels[in_range]
            results_instance_labels = results_instance_labels[in_range]
            gt_class_labels = gt_class_labels[in_range]
            gt_instance_labels = gt_instance_labels[in_range]

        if filter_ground:
            lidar_points = results.points
            if range_limit is not None:
                lidar_points = lidar_points[in_range,:]
            not_ground = lidar_points[:,2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            results_instance_labels = results_instance_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]
            gt_instance_labels = gt_instance_labels[not_ground]


        # For each class C:
        for i in range(len(kitti_names)):
            kitti_class = kitti_names[i]
            coco_class = coco_names[i]
            # Find which lidar points are labelled as this class in the results,
            # and in the KITTI ground truth

            # account for KITTI or COCO class labels
            r_is_class = np.logical_or(results_class_labels == coco_class, results_class_labels == kitti_class)
            g_is_class = gt_class_labels == kitti_class

            # Find instances of this class, in results and in ground truth
            r_instances = np.unique(results_instance_labels[r_is_class])
            g_instances = np.unique(gt_instance_labels[g_is_class])

            n_r = len(r_instances)
            n_g = len(g_instances)

            # Create IoU matrix
            # Is n by m, where n is the number of object instances in the segmentation results,
            # and m is the number of instances in the ground truth
            iou_matrix = np.zeros((n_r, n_g))

            for row in range(n_r):
                r_instance = results_instance_labels == r_instances[
                    row]  # Results instance number
                for col in range(n_g):
                    g_instance = gt_instance_labels == g_instances[
                        col]  # GT instance number
                    intersection = np.logical_and(r_instance, g_instance)
                    union = np.logical_or(r_instance, g_instance)
                    iou_matrix[row, col] = np.sum(intersection) / np.sum(
                        union)

            r_matching, g_matching = linear_sum_assignment(
                cost_matrix=1 - iou_matrix)
            matching_matrix = np.zeros(iou_matrix.shape, dtype=int)

            tp_count = 0

            for (r, g) in zip(r_matching, g_matching):
                iou = iou_matrix[r, g]
                # print("Maximal matching: Matched results %d to GT %d, with iou %.3f" % (r,g,iou))
                if iou > iou_threshold:
                    matching_matrix[r, g] = 1
                    tp_count += 1

            # The number of all-zero rows in the matching matrix is the
            # number of false positives
            zero_rows = ~np.any(matching_matrix, axis=1)
            fp_count = np.sum(zero_rows)

            # The number of all-zero columns in the matching matrix is
            # the number of false negatives (undetected GT objects)
            zero_cols = ~np.any(matching_matrix, axis=0)
            fn_count = np.sum(zero_cols)

            tp_totals[i] += tp_count
            fp_totals[i] += fp_count
            fn_totals[i] += fn_count

    return tp_totals, fp_totals, fn_totals
    # precision = [tp_count / (tp_count + fn_count)
    # recall = [tp_count / (tp_count + fp_count)

    # print("For class %s, precision is %.3f and recall is %.3f" % (kitti_class, precision, recall))
    # print("TP=%d, FP=%d, FN=%d" % (tp_count, fp_count, fn_count))

    # iou_list = [i/u for (i,u) in zip(i_totals, u_totals)]
    # return iou_list

# evaluate_instance_segmentation(results_list[100:200], gt_list[100:200])
# tp_totals, fp_totals, fn_totals = evaluate_instance_segmentation(
#     results_list, gt_list)


def print_pr_results(tp_totals, fp_totals, fn_totals,
                     classes=('Car', 'Pedestrian', 'Truck', 'Tram')):
    for (tp, fp, fn, name) in zip(tp_totals, fp_totals, fn_totals,
                                  classes):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("For class %s, precision is %.3f and recall is %.3f" % (
        name, precision, recall))
        print("TP=%d, FP=%d, FN=%d" % (tp, fp, fn))

# iou_list = evaluate_semantic_segmentation(results_list, gt_list)


def calculate_precision_recall(tp_totals, fp_totals, fn_totals):
    """
    Calculate list of precision and recall values from lists of true pos., 
    false pos., false neg. values.
    """
    precision = [tp / (tp+fp) if (tp+fp)>0 else 0 for (tp, fp) in zip(tp_totals, fp_totals)]
    recall = [tp / (tp+fn) if (tp+fn)>0 else 0 for (tp, fn) in zip(tp_totals, fn_totals)]
    return precision, recall



def plot_range_vs_accuracy(results_list, gt_list, filter_ground=False, cp_only=True,
                           savefile=None):
    """
    Evaluate labeling result as instance segmentation

    Reports IoU over classes

    Attributes
    ----------
    results_list: list
        List of LidarSegmentationResult
    gt_list: list
        List of LidarSegmentationGroundTruth
    iou_threshold: float
    range_limits: tuple, or None
        Specify range_limits to only look at objects at certain distances.
        Should contain two float values, e.g. (0,10) to look at objects
        from 0 to 10 meters away.
    """
    # KITTI objects we don't care about
    # Any points inside the bounding box for one of these object classes will be ignored
    # This helps because e.g. KITTI Van objects can be identified by Mask-RCNN as
    # either trucks or cars, and Cyclists can be identified as bicycles, motorcycles, or persons.
    objects_to_ignore = ["DontCare", "Van", "Cyclist"]

    # Define list of classes to evaluate
    if cp_only:
        kitti_names = ['Car', 'Pedestrian']
        coco_names = ['car', 'person']
    else:
        kitti_names = ['Car', 'Pedestrian', 'Truck',
                       'Tram']  # These are KITTI object class names
        coco_names = ['car', 'person', 'truck', 'train']

    # Keep running total of intersection and union values, for each class
    tp_totals = [0 for c in kitti_names]
    fp_totals = [0 for c in kitti_names]
    fn_totals = [0 for c in kitti_names]


    class_points = [np.empty((0,2)) for c in kitti_names]
    class_styles = ['.b', '^r']

    # iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # eval_results = [InstanceSegmentationResults(threshold) for threshold in iou_thresholds]

    for (results, gt) in zip(results_list, gt_list):
        # if range_limits is not None:
        #     ranges = np.linalg.norm(results[['x', 'y', 'z']].values,
        #                             axis=1)
        # Get object class labels (not instance labels)
        results_class_ids = results.class_labels()
        results_class_labels = np.array([CLASS_NAMES[i] for i in results_class_ids])
        gt_class_labels = gt.class_labels

        results_instance_labels = results.instance_labels()
        gt_instance_labels = gt.instance_labels

        # Find indices of lidar points that are in "DontCare" regions
        # all_dont_care = get_dont_care_indices(gt)
        # print("Removing %d DontCare, Van, Cyclist points" % np.sum(all_dont_care))
        # results_class_labels[all_dont_care] = "BG"
        # gt_class_labels[all_dont_care] = "BG"
        # results_instance_labels[all_dont_care] = -1
        # gt_instance_labels[all_dont_care] = -1

        # Set Person_sitting to Pedestrian
        gt_class_labels[gt_class_labels == "Person_sitting"] = "Pedestrian"

        # if range_limit is not None:
        #     lidar_points = results.points
        #     ranges = np.linalg.norm(lidar_points, axis=1)
        #     in_range = ranges < range_limit
        #     results_class_labels = results_class_labels[in_range]
        #     results_instance_labels = results_instance_labels[in_range]
        #     gt_class_labels = gt_class_labels[in_range]
        #     gt_instance_labels = gt_instance_labels[in_range]

        lidar_points = results.points
        ranges = np.linalg.norm(lidar_points, axis=1)

        if filter_ground:
            not_ground = lidar_points[:,2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            results_instance_labels = results_instance_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]
            gt_instance_labels = gt_instance_labels[not_ground]
            ranges = ranges[not_ground]

        # Calculate mean range to each ground truth instance
        instance_ranges = [np.mean(ranges[gt_instance_labels == i]) for i in
                           range(1, gt.n_instances)]

        # For each class C:
        for i in range(len(kitti_names)):
            kitti_class = kitti_names[i]
            coco_class = coco_names[i]
            # Find which lidar points are labelled as this class in the results,
            # and in the KITTI ground truth

            # account for KITTI or COCO class labels
            r_is_class = np.logical_or(results_class_labels == coco_class, results_class_labels == kitti_class)
            g_is_class = gt_class_labels == kitti_class

            # Find instances of this class, in results and in ground truth
            r_instances = np.unique(results_instance_labels[r_is_class])
            g_instances = np.unique(gt_instance_labels[g_is_class])

            n_r = len(r_instances)
            n_g = len(g_instances)

            # Create IoU matrix
            # Is n by m, where n is the number of object instances in the segmentation results,
            # and m is the number of instances in the ground truth
            iou_matrix = np.zeros((n_r, n_g))

            for row in range(n_r):
                r_instance = results_instance_labels == r_instances[
                    row]  # Results instance number
                for col in range(n_g):
                    g_instance = gt_instance_labels == g_instances[
                        col]  # GT instance number
                    intersection = np.logical_and(r_instance, g_instance)
                    union = np.logical_or(r_instance, g_instance)
                    iou_matrix[row, col] = np.sum(intersection) / np.sum(
                        union)

            r_matching, g_matching = linear_sum_assignment(
                cost_matrix=1 - iou_matrix)
            matching_matrix = np.zeros(iou_matrix.shape, dtype=int)

            tp_count = 0

            for (r, g) in zip(r_matching, g_matching):
                iou = iou_matrix[r, g]
                # print("Maximal matching: Matched results %d to GT %d, with iou %.3f" % (r,g,iou))
                pt = np.array([instance_ranges[g], iou]).reshape((1,2))
                class_points[i] = np.append(class_points[i], pt, axis=0)

            # The number of all-zero rows in the matching matrix is the
            # number of false positives
            # zero_rows = ~np.any(matching_matrix, axis=1)
            # fp_count = np.sum(zero_rows)

            # The number of all-zero columns in the matching matrix is
            # the number of false negatives (undetected GT objects)
            # zero_cols = ~np.any(matching_matrix, axis=0)
            # fn_count = np.sum(zero_cols)

            # tp_totals[i] += tp_count
            # fp_totals[i] += fp_count
            # fn_totals[i] += fn_count

    if savefile is not None:
        np.savez(savefile, car_pts=class_points[0], pedestrian_pts=class_points[1])

    for pts, style in zip(class_points, class_styles):
        plt.plot(pts[:,0], pts[:,1], style)
    plt.legend(kitti_names)
    plt.xlabel("Range to object centroid [m]")
    plt.ylabel("IoU")
    plt.savefig("range_scatter.eps", bbox_inches='tight')
    plt.show()


