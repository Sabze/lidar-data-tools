#!/usr/bin/env python3
import numpy as np
import warnings
from . import utilities


class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, name=""):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.labels = None
        self.laser_info = {}
        self.num_of_lasers = None
        self.name = name
        self.reset()

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__) and \
            np.array_equal(other.points, self.points) and \
            np.array_equal(other.remissions, self.remissions) and \
            other.size() == self.size():
            if (self.labels is None) or (other.labels is None):
                warnings.warn("Labels are not set, the comparision might not be sufficient")
                #TODO: change to True
                return False
            else:
                return np.array_equal(other.labels, self.labels)
        return False

    def copy(self):
        lasercan_copy = LaserScan(project=self.project, H=self.proj_H, W=self.proj_W, fov_up=self.proj_fov_up,
                                  fov_down=self.proj_fov_down, name=self.name)
        return lasercan_copy


    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission
        self.labels = None
        self.laser_info = {}
        self.num_of_lasers = None
        self.set_projection_var(self.proj_H, self.proj_W)

    def set_projection_var(self, H, W):
        self.proj_H = H
        self.proj_W = W
        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected labels - [H,W] labels (-1 is no data)
        self.proj_label = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)
        # COLOR [H, W, 3]
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename, flip=False):
        """ Open raw scan and fill in attributes """

        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        if flip:
            scan = np.flip(scan, axis=0)
        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file) """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def count_elems_at_proj_pos(self):
        elems = np.zeros((self.proj_H, self.proj_W))
        for col, row in zip(self.proj_x, self.proj_y):
            elems[row, col] += 1
        return elems

    def open_labels(self, label_filename: str, flip=False):
        """ Open raw scan and fill in attributes """

        # check filename is string
        if not isinstance(label_filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(label_filename))))

        # check extension is a laserscan
        if not any(label_filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(label_filename, dtype=np.uint32)
        label = label.reshape((-1))
        if flip:
            label = np.flip(label)
        # set it
        self.set_labels(label)

    def set_labels(self, label):
        """ Set points for label not from file but from np """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")
        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.labels = label & 0xFFFF  # semantic label in lower half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")
        # TODO: Add support for projection
        # if self.project:
        #  self.do_label_projection()

    def _ordered_get_turn_info(self, phi_values: np.array, start_phi=0.0, acc_noise=5.0):
        """Return a dictionary with information about the revolution of each laser (channel). Expects that the points
        are sorted by the azimuth angle.

        Args:
            phi_values (np.array): The azimuthal angle of the points.
            start_phi (float): The first angle of the laser.
            acc_noise (float): The number of accepted inaccurate points.
        Returns:
            turn_info (dict): Dictionary with the laser nr. (turn nr.) as key. The items are dictionaries
             with the keys: "num_of_points", "start_index".
        """
        turn_info = {}
        counter = 0  # Counts the number of points in the half-turn
        subsequent_count = 0  # Counts the number of consecutive points with the same sign
        current_laser = 0
        start_index = 0
        prev_sign_neg = False
        for i, phi in enumerate(phi_values):
            if (phi - start_phi) > 0:
                if prev_sign_neg:   # Check if we changed sign
                    if subsequent_count > acc_noise:  # If so, check that it's not just noise
                        # We reached a full turn, add info in dictionary
                        turn_info[current_laser] = {"num_of_points": counter, "start_index": start_index}
                        # Prep for next turn
                        start_index = i
                        counter = 0
                        current_laser += 1
                        prev_sign_neg = False
                    subsequent_count = 0
            else:
                if not prev_sign_neg:
                    subsequent_count = 0
                prev_sign_neg = True
            counter += 1
            subsequent_count += 1
        if counter > 0:
            turn_info[current_laser] = {"num_of_points": counter, "start_index": start_index}
        return turn_info

    def _ordered_post_process_turns(self, turn_info: dict, phi_coords: np.array, start_phi: float, accepted_noise: float,
                                    expected_num_of_turns: int, tries=3):
        """ Tries to find exactly the expected_num_of_turns by changing the accepted noice. Expects that the points
        are sorted by the azimuth angle.
        
        Args:
            turn_info (dict):               The previous turn information.
            phi_coords (np.array):          The azimuthal angle of the points.
            start_phi (float):              The first angle of the laser.
            accepted_noise (float):         The number of accepted inaccurate points.
            expected_num_of_turns (int):    The number of laser channels in the LiDAR.
            tries (int):                    The maximum times the accepted noise will change to try to find the
                                            right number of lasers
        
         Returns:
            turn_info (dict): Dictionary with the laser nr. (turn nr.) as key. The items are dictionaries
             with the keys: "num_of_points", "start_index".
        """

        if len(turn_info) == expected_num_of_turns:  # Success, we found the right number of lasers.
            num_of_points = sum([v["num_of_points"] for (k, v) in turn_info.items()])
            assert (num_of_points == self.size()), \
                f"Expected {self.size()} number of points, got {num_of_points}"
            return turn_info
        elif tries > 0:     # We got the wrong amount of lasers, change the accepted noise.
            if len(turn_info) > expected_num_of_turns:
                accepted_noise *= 2
                turn_info = self._ordered_get_turn_info(phi_coords, start_phi, acc_noise=accepted_noise)
            else:
                accepted_noise *= 0.7
                turn_info = self._ordered_get_turn_info(phi_coords, start_phi, acc_noise=accepted_noise)
            return self._ordered_post_process_turns(turn_info, phi_coords, start_phi, accepted_noise,
                                                    expected_num_of_turns, tries - 1)
        else:    # We got the wrong amount of lasers and we have no tries left to fix it.
            if len(turn_info) > expected_num_of_turns:
                warnings.warn(f"Perfect laser-match is not possible:\nMatching the {expected_num_of_turns} "
                              f"first lasers out of {len(turn_info)}")
                rel_lasers = list(range(0, expected_num_of_turns))
                turn_info = {k: v for (k, v) in turn_info.items() if k in rel_lasers}
            else:
                warnings.warn(f"Perfect laser-match is not possible:\nMatching only {len(turn_info)} lasers")
            return turn_info

    def sort_points(self, sort_values: np.array):
        """ Sorts the points, labels and remission
        Args:
            sort_values (np.array): The points, labels and remission are sorted based on the values in sort_values.
        Returns:
            sorted sort_values
        """
        warnings.warn("Don't use this function if you don't know what you are doing!")
        order = np.argsort(sort_values)
        self.points = np.take(self.points, order, axis=0)
        self.remissions = np.take(self.remissions, order)
        if self.labels is not None:
            self.labels = np.take(self.labels, order)
        return np.take(sort_values, order)

    def _unordered_get_laserinfo(self, vert_values: np.array, angular_res: float, precision: float):
        """ Return a dictionary with information about the revolution of each laser (channel)."""
        # TODO: make sure this is correct
        vert_angles = self.sort_points(vert_values)
        diffs = np.diff(vert_angles)
        start_inds = [ind+1 for ind, diff in enumerate(diffs) if diff > angular_res*precision]
        start_inds.append(self.size())
        start_inds = [0] + start_inds
        num_points = np.diff(start_inds)
        laserinfo = {}
        for laser_num, info in enumerate(zip(start_inds[0:-1], num_points)):
            start_ind, num_point = info
            laserinfo[laser_num] = {"start_index": start_ind, "num_of_points": num_point}
        return laserinfo

    def set_laser_info(self, expected_num_of_lasers: int, start_phi=0, accepted_noise=15.0,
                       ordered=True, angular_res=0.035, precision=0.5):
        """Set laser_info, which contains information the laser channels and their points.

         Args:
            expected_num_of_lasers (int):   The number of vertical laser channels in the LiDAR.
            start_phi (float):              The start azimuthal angle of the recorded points.
            accepted_noise (float):         How many points that are allowed to be 'wrong'.
            ordered (bool):                 True if the points are ordered by laser channel (e.g. first comes the points
                                            from laser channel 1, then from laser channel 2, etc.)
            angular_res (float):            The expected vertical angle difference of the consecutive laser channels.
                                            Used if the points aren't ordered.
            precision (float):              How accurate the angular resolution actually is.
                                            Used if the points aren't ordered.
            """
        self.laser_info = {}  # Contains information the laser channels and their points.
        spherical_coords = utilities.get_spherical_coords(self.points)
        if ordered:
            turn_info = self._ordered_get_turn_info(spherical_coords[:, 1], start_phi, acc_noise=accepted_noise)
            turn_info = self._ordered_post_process_turns(turn_info, spherical_coords[:, 1], start_phi, accepted_noise,
                                                         expected_num_of_lasers, tries=3)
        else:
            # TODO: add post processing
            vert_values = spherical_coords[:, 2]
            turn_info = self._unordered_get_laserinfo(vert_values, angular_res, precision)
        self.laser_info = turn_info
        self.num_of_lasers = len(self.laser_info)
        return turn_info

    def do_range_projection(self, h_flip=False, v_flip=False, half_turn=True):
        """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]
        # get angles of all points
        yaw = np.arctan2(scan_y, scan_x)

        pitch = np.arcsin(scan_z / depth)
        # get projections in image coords
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        if half_turn:
            proj_x = -yaw / np.pi + 0.5  # pi horizontal angle
        else:
            proj_x = 0.5 * (-yaw / np.pi + 1.0)  # in [0.0, 1.0] # Original (2pi horizontal angle)

        # Works for all start angles
        # shifted_yaw = ((-yaw) - start_angle) % (2 * np.pi)
        # proj_x = shifted_yaw / horizontal_fov

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]
        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        if v_flip:
            proj_x = (self.proj_W - 1) - proj_x
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        if h_flip:
            proj_y = (self.proj_H-1) - proj_y
        self.proj_y = np.copy(proj_y)  # stope a copy in original order
        # copy of depth in original order
        self.unproj_range = np.copy(depth)
        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)
        if self.labels is not None:
            labels = self.labels[order]
            self.proj_label[proj_y, proj_x] = labels

    def do_label_projection(self):
        # TODO: FIX
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

