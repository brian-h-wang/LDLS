"""
segmentation.py
Brian Wang

Lidar segmentation module.

"""

import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize

import time


NO_LABEL=-1

class LidarSegmentationResult(object):
    """
    Output from lidar segmentation
    
    """

    def __init__(self, points, projected, label_likelihoods,
                 class_ids, in_camera_view, initial_labels):
        self.points = points # all lidar points: xyz
        self.projected = projected
        self.label_likelihoods = label_likelihoods # n_iterations by n_points by n_instances
        self.class_ids = class_ids # length n_instances (NOT counting background)
        # self.confidence = confidence  # confidence at point level
        self.in_camera_view = in_camera_view # indices of lidar points that are in view of camera
        self.initial_labels = initial_labels

    @classmethod
    def load_file(cls, filename):
        """
        
        Parameters
        ----------
        filename: str
            Name of file to load

        Returns
        -------
        LidarSegmentationResult

        """
        if not filename.endswith(".npz"):
            filename += ".npz"
        with open(filename, "rb") as loadfile:
            npzfile = np.load(loadfile)
            results = cls(points=npzfile["points"],
                       projected=npzfile["projected"],
                       label_likelihoods=npzfile["label_likelihoods"],
                       class_ids=npzfile["class_ids"],
                       in_camera_view=npzfile["in_camera_view"],
                       initial_labels=npzfile["initial_labels"])
        return results

    def to_file(self, filename):
        """
        
        Parameters
        ----------
        filename: str

        Returns
        -------
        None

        """
        with open(filename, "wb") as savefile:
            np.savez_compressed(savefile, points=self.points,
                                projected=self.projected,
                                label_likelihoods=self.label_likelihoods,
                                class_ids=self.class_ids,
                                in_camera_view=self.in_camera_view,
                                initial_labels=self.initial_labels)

    def point_confidence(self, iter=-1):
        """
        Returns confidence of the instance label prediction for each point.
        Confidence is calculated using softmax function.
        
        Parameters
        ----------
        iter: int
            The iteration at which to compute results.

        Returns
        -------
        numpy.ndarray
            n_points-element vector

        """
        L = self.label_likelihoods[iter,:,:]
        labels = self.instance_labels(iter)
        L = normalize(L, axis=1)
        exp_L = np.exp(L)


    def instance_confidence(self, iter=-1):
        pass

    def instance_labels(self, iter=-1):
        L = self.label_likelihoods[iter,:,:]
        return np.argmax(L, axis=1)

    def class_labels(self, iter=-1):
        # Instance 0 is background (class 0)
        instance_labels = self.instance_labels(iter=iter) - 1
        if len(self.class_ids) == 0:  # No objects, return all zeros
            return np.zeros(instance_labels.shape, dtype=int)
        labels = self.class_ids[instance_labels]
        labels[instance_labels == -1] = 0
        return labels

    @property
    def n_iters(self):
        return self.label_likelihoods.shape[0]

    # @property
    # def instance_labels(self):
    #     return self.instance_labels(iter=-1)
    #
    # @property
    # def class_labels(self):
    #     return self.class_labels(iter=-1)



class LidarSegmentation(object):
    """
    Class for performing segmentation of lidar point clouds.
    """

    def __init__(self, projection, num_iters=-1, num_neighbors=10,
                 distance_scale=1.0,
                 mask_shrink=0.5, mask_dilate=2.0, outlier_removal=True):
        """
        
        Parameters
        ----------
        projection
        num_iters: int
            If set to <= 0, will iterate until convergence (slower)
        num_neighbors
        mask_shrink
        """
        self.projection = projection
        self.num_iters = num_iters
        self.num_neighbors = num_neighbors
        # self.x_scale = x_scale
        # self.y_scale = y_scale
        # self.z_scale = z_scale
        self.distance_scale= distance_scale
        self.mask_shrink = mask_shrink
        self.mask_dilate = mask_dilate
        self.outlier_removal = outlier_removal

    def create_initial_labeling(self, lidar, detections,
                                distance_threshold=1.0):
        """
        Create initial labeling for the lidar points.
        
        Parameters
        ----------
        lidar: np.ndarray
            xyz coordinates of the lidar points. n_points by 3 array.
        detections: Detections
            Image detections.
        distance_threshold: float
            Maximum distance for a lidar point to be matched with an image
            pixel.

        Returns
        -------
        np.ndarray
            Vector with n_points elements. Element i is the initial label of
            lidar point i, which can be NaN (no initial label), -1 (background)
            or >= 0 (instance label)

        """
        n_points = lidar.shape[0]

        # Get label image from the image detections
        label_image = detections.create_label_image(mask_shrink=self.mask_shrink,
                                                    mask_dilate=self.mask_dilate)

        img_rows, img_cols = label_image.shape

        row_mesh, col_mesh = np.meshgrid(np.arange(img_rows),
                                         np.arange(img_cols),
                                         indexing='ij')

        # Create KDTree from image pixels for NN lookup
        # label_image_kdtree = KDTree(np.column_stack([col_mesh.flatten(),
        #                                              row_mesh.flatten()]))

        projected = self.projection.project(lidar)
        # Check which lidar points project to within the image bounds
        in_frame_x = np.logical_and(projected[:,0] > 0,
                                    projected[:,0] < img_cols-1)
        in_frame_y = np.logical_and(projected[:,1] > 0,
                                    projected[:,1] < img_rows-1)

        projected_in_frame = np.logical_and(in_frame_x, in_frame_y)
        # Lidar point is in view if in front of camera (x>0) *and* projects
        # to inside the image
        in_view = np.logical_and(lidar[:,0] > 0, projected_in_frame)

        # Proceed with labeling only lidar points that are in the camera view
        lidar_in_view = lidar[in_view,:]
        proj_in_view = projected[in_view,:]

        # Query label image KDTree at projected lidar points
        # distances, nearest_neighbors = label_image_kdtree.query(proj_in_view)

        # Find which image pixel the projected lidar points are nearest to
        # Round projected coordinates to ints, then flip (to get row-column)
        nearest_pixels = np.round(proj_in_view).astype(np.uint)
        #nearest_pixels = np.fliplr(nearest_pixels)
        nearest_pixels = np.fliplr(nearest_pixels)

        # note: distances, nearest_neighbors each have shape (n_points, 1)

        # Instance labels: 0 for background, integers >0 for instances
        # image_labels = label_image.astype(int).flatten()

        # labels = image_labels[nearest_neighbors[:,0]]
        labels = label_image[nearest_pixels[:,0], nearest_pixels[:,1]].flatten()
        # labels[distances[:,0] > distance_threshold] = NO_LABEL
        # print(np.sum(distances[:,0] > distance_threshold))
        labels[labels == -1] = NO_LABEL

        return labels, projected, in_view

    def remove_outliers(self, lidar, labels, threshold=1.0):
        # Remove outliers based on median depth of each cluster
        lidar_x = lidar[:,0]
        for instance_label in np.unique(labels):
            # Ignore the background, and unlabeled points
            if instance_label == NO_LABEL or instance_label == 0:
                continue
            in_mask = labels == instance_label
            instance_points_x = lidar_x[in_mask]
            # Ignore masks that have no associated labelled lidar points
            if len(instance_points_x) == 0:
                continue
            median_depth = np.median(instance_points_x)
            is_outlier = (np.abs(lidar_x - median_depth) > threshold).flatten()
            labels[in_mask & is_outlier] = NO_LABEL
        return labels

    def weights_matrix(self, lidar, k):
        """
        Attributes
        ----------
        lidar: numpy.ndarray
            n by 3 array. Each row of data is an xyz point
        k: int
            Number of nearest neighbors to consider
            
        Returns
        -------
        scipy.sparse.csr_matrix
            n by n sparse matrix.
            Exp-weighted nearest neighbors graph
        """

        n = lidar.shape[0]

        # Define distance metric according to x,y,z scaling factors
        # metric = DistanceMetric.get_metric('wminkowski', p=2,
        #                                    w=[self.x_scale, self.y_scale, self.z_scale])
        # Rescale lidar points according to x,y,z scaling factors
        # Implements a custom distance metric

        # Create sparse matrix of nearest neighbors
        kdt = KDTree(lidar)

        # do KDT query with k+1 to account for point being own nearest neighbor
        distances, neighbors = kdt.query(lidar, k=k + 1)

        # COO matrix initialization
        d = np.exp(-(distances**2) / (self.distance_scale**2)).flatten()
        row_indices = np.indices(distances.shape)[0].flatten()
        col_indices = neighbors.flatten()
        S = coo_matrix((d, (row_indices, col_indices)))

        # Normalize rows to sum to 1
        S = S.tocsc()
        S = normalize(S, norm='l1', axis=1)

        # Convert to CSR format, should be more efficient?
        return S.tocsr()

    def run(self, lidar, detections, max_iters=100,
                      rtol=1e-02, atol=1e-06):
        # ------------------------------------
        # INITIAL LABELING AND PRE-PROCESSING
        # ------------------------------------
        num_iters = self.num_iters
        n_points = lidar.shape[0]
        n_instances = len(detections)

        # Create initial lidar points labeling
        initial_labels, projected, in_view = self.create_initial_labeling(
            lidar, detections)

        lidar = lidar[in_view,:]
        if self.outlier_removal:
            initial_labels = self.remove_outliers(lidar, initial_labels)

        initial_labeled = (initial_labels != NO_LABEL).astype(bool)
        initial_unlabeled = (initial_labels == NO_LABEL).astype(bool)

        # Create label likelihood matrix F
        # F[i,j] is 1 if point i has initial label j, and 0 otherwise
        # Note that there is one extra column for the background label
        label_likelihoods = np.array([initial_labels == l
                                     for l in range(n_instances+1)]).T.astype(float)

        initial_likelihoods = np.copy(label_likelihoods)
        all_label_likelihoods = [initial_likelihoods]
        # Create weighted neighbors graph
        G = self.weights_matrix(lidar, k=self.num_neighbors)

        # ----------------------------
        # MAIN LABEL PROPAGATION LOOP
        # ----------------------------
        # Iterate until convergence, or until specified number of iterations
        done = False
        if num_iters == 0:
            done = True
        i = 0

        while not done:
            # print("Clustering iteration %d" % (i+1))
            prev_likelihoods = np.copy(label_likelihoods)
            label_likelihoods = G.dot(label_likelihoods)

            # Clamp initially labeled points
            label_likelihoods[initial_labeled,:] = initial_likelihoods[initial_labeled,:]

            i += 1

            # Check for termination condition
            # If number of iterations specified, done if i > num_iters
            if num_iters >= 0:
                if i >= num_iters:
                    done = True
            # No number of iterations specified, so run until converging
            else:
                # Reached maximum iterations - ideally this should not happen
                if i >= max_iters:
                    # print("Warning: Reached maximum number of iterations.")
                    done = True
                # Check for convergence of label likelihoods
                else:
                    done = np.allclose(label_likelihoods, prev_likelihoods,
                                   atol=atol, rtol=rtol)
            all_label_likelihoods.append(np.copy(label_likelihoods))

        all_label_likelihoods = np.array(all_label_likelihoods)

        return LidarSegmentationResult(points=lidar, projected=projected,
                                       in_camera_view=in_view,
                                       label_likelihoods=all_label_likelihoods,
                                       class_ids=detections.class_ids,
                                       initial_labels=initial_labels)

