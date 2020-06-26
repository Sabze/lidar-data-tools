from .laserscan import LaserScan
import numpy as np
from . import utilities
import warnings
import math

EXTENSIONS_LABEL = ['.label']
REMISSION_NAMES = ["remission", "intensity", "rem", "int", "i"]
RANGE_NAMES = ["r", "radius", "range"]
THETA_NAMES = ["theta", "vertical", "ver"]
PHI_NAMES = ["phi", "hor", "horizontal", "azimuth"]
LASERCAN_FOLDER = "velodyne"


def crop_laserscan(laserscan: LaserScan, wanted_inds: list, name=""):
    """Create a new laserscan with the points, remissions and labels at the inds given by wanted_inds. """
    new_points = np.take(laserscan.points, wanted_inds, axis=0)
    new_rem = np.take(laserscan.remissions, wanted_inds)
    new_labels = np.take(laserscan.labels, wanted_inds)
    new_laserscan = laserscan.copy()
    new_laserscan.set_points(new_points, new_rem)
    new_laserscan.set_labels(new_labels)
    new_laserscan.name = f"{name}:{laserscan.name}"
    return new_laserscan


def vertical_downsample(laserscan: LaserScan, num_desired_lasers=16):
    """Downsample the laserscan to num_desired_lasers laser channels. """
    if laserscan.laser_info is None:
        raise RuntimeError("Set laser information before downsample")
    rate = math.ceil(laserscan.num_of_lasers / num_desired_lasers)
    laser_inds = list(range(0, laserscan.num_of_lasers, rate))
    desired_inds = []
    for laser_ind in laser_inds:
        start = laserscan.laser_info[laser_ind]["start_index"]
        end = start + laserscan.laser_info[laser_ind]["num_of_points"]
        desired_inds += list(range(start, end))
    if len(laser_inds) != num_desired_lasers:
        warnings.warn(f"Downsampled laserscan contains {len(laser_inds)} lasers not {num_desired_lasers}.")
    new_laserscan = crop_laserscan(laserscan, desired_inds, name=f"vert_dwn ({num_desired_lasers})")
    return new_laserscan


def crop_horizontally(laserscan: LaserScan, start_angle: float, fov=np.pi):
    """Start_angle should be value in [-pi, pi]"""
    start = start_angle + np.pi  # In interval [0, 2*pi]
    end = start + fov
    spherical_coords = utilities.get_spherical_coords(laserscan.points)
    phi_values = spherical_coords[:, 1] + np.pi  # interval between [0, 2*pi]
    desired_inds = []
    for ind, phi in enumerate(phi_values):
        if utilities.is_inbetween_angles(start, end, phi):
            desired_inds.append(ind)
    new_laserscan = crop_laserscan(laserscan, desired_inds, name=f"hori-crop ({fov:.2f})")
    return new_laserscan


def print_ranges(laserscan: LaserScan):
    x = laserscan.points[:, 0]
    y = laserscan.points[:, 1]
    z = laserscan.points[:, 2]
    intensity = laserscan.remissions
    spherical_coords = utilities.get_spherical_coords(laserscan.points)
    r = spherical_coords[:, 0]
    phi = spherical_coords[:, 1]
    theta_sin = spherical_coords[:, 2]
    print("--- RANGES ---")
    utilities.print_range(x, "x")
    utilities.print_range(y, "y")
    utilities.print_range(z, "z")
    utilities.print_range(r, "Radius")
    utilities.print_range(phi, "Phi")
    utilities.print_range(theta_sin, "Theta(sin)")
    utilities.print_range(intensity, "Intensity")
    return


def get_scan_values(scan:LaserScan, pos:str):
    if pos == "x":
        return scan.points[:, 0]
    elif pos == "y":
        return scan.points[:, 1]
    elif pos == "z":
        return scan.points[:, 2]
    elif pos in REMISSION_NAMES:
        return scan.remissions
    elif pos in RANGE_NAMES:
        return np.linalg.norm(scan.points, axis=1)
    elif pos in PHI_NAMES:
        return - np.arctan2(scan.points[:, 1], scan.points[:, 0])
    elif pos in THETA_NAMES:
        return np.arcsin(scan.points[:, 2] / np.linalg.norm(scan.points, axis=1))
    else:
        raise ValueError("Invalid pos-value")


def concatenate_scan_values(laserscans: list, pos:str):
    data_array = get_scan_values(laserscans[0], pos)
    for laserscan in laserscans[1:]:
        vals = get_scan_values(laserscan, pos)
        data_array = np.concatenate([data_array, vals], axis=0)
    return data_array


def get_remission_vals_for_label(laserscans: list, label:str, label_mapping:dict):
    """ Get remission values for a specific label/class.

    Args:
        laserscans (list):      List of laserscans
        label (str):            Name of the label/class to investigate.
        label_mapping (dict):   Dictionary with a mapping between label and label ID.

    Returns:
        np.array with the remission values for the specified label/class.
    """
    label = label.lower()
    try:
        label_id = label_mapping[label]
    except:
        raise ValueError(f"{label} is not a valid label, please use one of {label_mapping.keys()}")
    data_array = np.array([])
    for laserscan in laserscans:
        relevant_indices = np.argwhere(laserscan.labels == label_id).flatten()
        if any(relevant_indices):
            remission = np.take(laserscan.remissions, relevant_indices)
            data_array = np.concatenate([data_array, remission], axis=0)
    if (len(data_array) == 0):
        warnings.warn(f"There were no points with label {label} in the scans ", UserWarning)
    return data_array


def sum_laserscan_properties(laserscan):
    """ Helper function for calculating the mean.
    Sum the laserscan's x, y, z, range, remission values.
    Return a np.array (1x6). """
    xyz_sums = np.sum(laserscan.points, axis=0)
    rem_sum = np.array([np.sum(laserscan.remissions)])
    range_sum = np.array([np.sum(np.linalg.norm(laserscan.points, axis=1))])
    num_points = np.array([laserscan.size()])
    sum_vector = np.concatenate([xyz_sums, rem_sum, range_sum, num_points]).reshape(1, -1)
    return sum_vector


def sum_laserscan_std(laserscan, xmean, ymean, zmean, remmean, rangemean):
    """ Helper function for calculating the standard deviations. """
    xyzmean = np.array([[xmean, ymean, zmean]])
    range = np.linalg.norm(laserscan.points, axis=1)
    point_std_sum = np.sum((np.square(laserscan.points - xyzmean)), axis=0)
    rem_std_sum = np.array([np.sum(np.square(laserscan.remissions - remmean))])
    range_std_sum = np.array([np.sum(np.square(range - rangemean))])
    num_points = np.array([laserscan.size()])
    std_sum_vector = np.concatenate([point_std_sum, rem_std_sum, range_std_sum, num_points]).reshape(1, -1)
    return std_sum_vector


def get_rotated_laserscan(laserscan, x_rot=0.0, y_rot=0.0, z_rot=0.0):
    x_cos, x_sin = utilities.get_trig_angles(x_rot)
    y_cos, y_sin = utilities.get_trig_angles(y_rot)
    z_cos, z_sin = utilities.get_trig_angles(z_rot)
    rotate_x = np.array([[1, 0, 0],
                        [0, x_cos, -x_sin],
                        [0, x_sin, x_cos]])
    rotate_y = np.array([[y_cos, 0, y_sin],
                        [0, 1, 0],
                        [-y_sin, 0, y_cos]])
    rotate_z = np.array([[z_cos, -z_sin, 0],
                        [z_sin, z_cos, 0],
                        [0, 0, 1]])
    rotation_matrix = np.matmul(rotate_z, np.matmul(rotate_y, rotate_x))
    rotated_points = np.matmul(rotation_matrix, laserscan.points.T, dtype="float32").T
    rotated_laserscan = laserscan.copy()
    rotated_laserscan.set_points(rotated_points, laserscan.remissions)
    if np.any(laserscan.labels):
        rotated_laserscan.set_labels(laserscan.labels)
    return rotated_laserscan


def get_translated_laserscan(laserscan, x_diff=0, y_diff=0, z_diff=0):
    translated_points = laserscan.points + [x_diff, y_diff, z_diff]
    translated_laserscan = laserscan.copy()
    translated_laserscan.set_points(translated_points, laserscan.remissions)
    if np.any(laserscan.labels):
        translated_laserscan.set_labels(laserscan.labels)
    return translated_laserscan

