import os
import glob
import logging

import yaml

from .laserscan import LaserScan
import warnings
import time
import numpy as np
import multiprocessing as mp
import shutil
FREE_PROCESSORS = 3
SUB_FOLDER = "sequences"
LASER_FOLDER = "velodyne"
PRED_PROB_FOLDER = "probabilities"
OUTPUT_FOLDER = "x_y_z_rem_label"
LABEL_FOLDER = "labels"
LABEL_SUFFIX = ".label"
LASER_SUFFIX = ".bin"
PREDICTION_SUFFIX = ".label"
PREDICTION_FOLDER = "predictions"
EXAMPLE_FORMAT = f".../dataset/{SUB_FOLDER}/XX/"
PRED_PROB_SUFFIX = ".npy"

# --- DICTIONARY KEYS ------
PRED_PROB_NAME = "pred_probs"
LABEL_NAME = "labels"
LASER_NAME = "laserscans"
PREDICTION_NAME = "predictions"
UNKNOWN_SEQ = "Unknown"

#------ LABELING TOOL FILES -----
LABELING_TOOL_FILES = ["calib.txt", "instances.txt", "poses.txt", "times.txt"]

def get_example_format(folder_name:str, file_suffix):
    return f".../dataset/{SUB_FOLDER}/XX/{folder_name}/XXXXXX{file_suffix}"

# ------------------------------- Functions for collecting and reading laserscan data -------------------------------

def add_velodyne_files(file_dict:dict, dataset_filepath: str):
    """ Add the velodyne files in dataset_filepath to the dictionary file_dict.
    The format of file_dict is {sequence_num: {filetype1: [file_paths], filetype2: [file_paths]}}.
    Args:
        dataset_filepath (str): The filepath to data in kitti-format or to the sequence directory.
        file_dict (dict):       The velodyne files are added to this dictionary.
    """
    example_format = get_example_format(LASER_FOLDER, LASER_SUFFIX)
    add_laserscan_files(file_dict, dataset_filepath, LASER_NAME, LASER_FOLDER, LASER_SUFFIX,
                        example_format)
    return


def add_label_files(file_dict:dict, dataset_filepath: str):
    """Add the label files in dataset_filepath to the dictionary file_dict.
    The format of file_dict is {sequence_num: {filetype1: [file_paths], filetype2: [file_paths]}}.
    Args:
        dataset_filepath (str): The filepath to data in kitti-format or to the sequence directory.
        file_dict (dict):       The label files are added to this dictionary.
    """
    example_format = get_example_format(LABEL_FOLDER, LABEL_SUFFIX)
    add_laserscan_files(file_dict, dataset_filepath, LABEL_NAME, LABEL_FOLDER, LABEL_SUFFIX,
                        example_format)
    return


def add_pred_prob_files(file_dict:dict, dataset_filepath: str):
    """ Add the pred prob files in dataset_filepath to the dictionary file_dict
    Args:
        dataset_filepath (str): The filepath to data in kitti-format.
        file_dict (dict):       The pred prob files are added to this dictionary.
    """
    example_format = get_example_format(PRED_PROB_FOLDER, PRED_PROB_SUFFIX)
    add_laserscan_files(file_dict, dataset_filepath, PRED_PROB_NAME, PRED_PROB_FOLDER, PRED_PROB_SUFFIX,
                        example_format)
    return


def add_prediction_files(file_dict:dict, dataset_filepath: str):
    """Add the prediction files in dataset_filepath to the dictionary file_dict.
    The format of file_dict is {sequence_num: {filetype1: [file_paths], filetype2: [file_paths]}}.
    Args:
        dataset_filepath (str): The filepath to data in kitti-format or to the sequence directory.
        file_dict (dict):       The prediction files are added to this dictionary.
    """
    example_format = get_example_format(PREDICTION_FOLDER, PREDICTION_SUFFIX)
    add_laserscan_files(file_dict, dataset_filepath, PREDICTION_NAME, PREDICTION_FOLDER, PREDICTION_SUFFIX,
                        example_format)
    return


def add_laserscan_files(file_dict:dict, dataset_filepath: str, name:str, folder_name:str, file_suffix:str,
                        example_format:str):
    """ Add the files in dataset_filepath to the dictionary file_dict

    Returns:
        A dictionary in the format {sequence: {name: [file_path]}}
    """
    if not os.path.isdir(dataset_filepath):
        raise ValueError("The given dataset directory is not a directory.")
    sub_folders = os.listdir(dataset_filepath)
    if SUB_FOLDER in sub_folders:
        # Get files for more than one sequence
        sequences = sorted(os.listdir(os.path.join(dataset_filepath, SUB_FOLDER)))
        for sequence in sequences:
            if not sequence in file_dict:
                file_dict[sequence] = {}
            file_dict[sequence][name] = sorted(glob.glob(os.path.join(*[dataset_filepath, SUB_FOLDER, sequence, folder_name,
                                                      f"*{file_suffix}"])))
    elif folder_name in sub_folders:
        # Get files for one sequence
        sequence = os.path.basename(dataset_filepath)
        if sequence == "":
            sequence = UNKNOWN_SEQ
        if not sequence in file_dict:
            file_dict[sequence] = {}
        file_dict[sequence][name] = sorted(glob.glob(os.path.join(*[dataset_filepath, folder_name,
                                                              f"*{file_suffix}"])))
    else:
        logging.error(f"ValueError, Please have the following folder structure: {example_format}")
        raise ValueError(f"Please have the following folder structure: {example_format}")
    return


def read_label_file(label_filename:str):
    if not isinstance(label_filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(label_filename))))

    # check extension is a laserscan
    if not label_filename.endswith(LABEL_SUFFIX):
        raise RuntimeError(f"Filename extension is not valid label file. ({label_filename})")

    # if all goes well, open label
    label = np.fromfile(label_filename, dtype=np.uint32)
    label = label.reshape((-1))
    labels = label & 0xFFFF  # semantic label in lower half
    return labels


def read_laserscan(laserscan_file, label_file=None):
    laser_filename = os.path.splitext(os.path.basename(laserscan_file))[0]
    laserscan = LaserScan()
    laserscan.open_scan(laserscan_file)
    if label_file is not None:
        label_filename = os.path.splitext(os.path.basename(label_file))[0]
        if laser_filename != label_filename:
            raise ValueError(f"The laser-file {laser_filename} and label-file {label_filename} are not "
                                       f"the same.")
        laserscan.open_labels(label_file)
    return laserscan, laser_filename


def get_laserscan(dir: str, sequence_num: str, scan_num: int, name="", labels=True, fov_up=3.0, fov_down=-25.0):
    scan_file = os.path.join(*[dir, sequence_num, "velodyne", f"{scan_num:06}.bin"])
    print(f"Loading points from: {scan_file}")
    laserscan = LaserScan(name=f"{name}:{sequence_num}:{scan_num:06}", fov_up=fov_up, fov_down=fov_down)
    laserscan.open_scan(scan_file)
    if labels:
        label_file = os.path.join(*[dir, sequence_num, "labels", f"{scan_num:06}.label"])
        print(f"Loading labels from: {label_file}")
        laserscan.open_labels(label_file)
    return laserscan


def get_laserscans(dir: str, scan_nums: list, name="", labels=True, fov_up=3.0, fov_down=-25.0):
    """ Return the a list of laserscans.
    Args:
        dir (str):          The directory for the laserscans.
        scan_nums (list):   A list with the IDs/names of the wanted scans.
        name (str):         Common prefix of the name for all the loaded laserscans.
        labels (bool):      True if the labels should be loaded, false otherwise.
        fov_up (float):     Max vertical angle up.
        fov_down (float):   Max vertical angle down.

    Returns:
        list of laserscans."""
    sequence_num = os.path.basename(dir)
    seq_dir = os.path.dirname(dir)
    if sequence_num == "":
        sequence_num = os.path.basename(dir[:-1])
        seq_dir = os.path.dirname(dir[:-1])
    laserscans = []
    for scan_num in scan_nums:
        laserscan = get_laserscan(seq_dir, sequence_num, scan_num, name, labels=labels, fov_up=fov_up, fov_down=fov_down)
        laserscans.append(laserscan)
    return laserscans


def get_num_scans(dir: str, sequence_num: str):
    """ Return the number of scans in the specified sequence."""
    scan_dir = os.path.join(*[dir, sequence_num, LASER_FOLDER])
    scan_files = os.listdir(scan_dir)
    return len(scan_files)


def count_labels(labelfile:str):
    labels = read_label_file(labelfile)
    num_labels = labels.shape[0]
    labels, counts = np.unique(labels, return_counts=True)
    return labels, counts, num_labels


def calc_entropy(pred_prob_file:str):
    probs = np.load(pred_prob_file)
    log_probs = np.log(probs)
    entropy = -np.sum(np.multiply(probs, log_probs), axis=1)
    if np.any(entropy > 1.5):
        print(f"Bigger than 1.5: {np.max(entropy)}")
    return entropy


def get_weight(label_file:str, label_learning_map, label_ignore_map, label_weights):
    label_types, nums, tot_labels = count_labels(label_file)
    weight = 0
    for label_type, num in zip(label_types, nums):
        mapped_class = label_learning_map[label_type]
        if not label_ignore_map[mapped_class]:
            weight += num * 1 / label_weights[mapped_class]
    weight /= tot_labels
    return np.round(weight) ** 2


def copy_labeling_tool_files(source_dir: str, destination_dir: str):
    if not os.path.isdir(source_dir) or not os.path.isdir(destination_dir):
        raise ValueError("The given dataset directory is not a directory.")
    dst_sequences = sorted(os.listdir(os.path.join(destination_dir, SUB_FOLDER)))
    src_sequences = os.listdir(source_dir)
    has_several_sequences = SUB_FOLDER in src_sequences
    for dst_sequence in dst_sequences:
        if not has_several_sequences:
            src_seq = source_dir
        else:
            src_seq = os.path.join(*[source_dir, SUB_FOLDER, dst_sequence])
        for labeling_tool_file in LABELING_TOOL_FILES:
            src_file = os.path.join(src_seq, labeling_tool_file)
            if os.path.isfile(src_file):
                dst_file = os.path.join(*[destination_dir, SUB_FOLDER, dst_sequence, labeling_tool_file])
                shutil.copyfile(src_file, dst_file)
            else:
                warnings.warn(f"Labeling tool file {src_file} could not found "
                              f"for sequence {dst_sequence}")


# ----------------------------------- Functions for saving laserscan data-------------------------------------

def save_scan(laserscan: LaserScan, filename: str, output: str, overwrite=False):
    """ Save laserscan at output/velodyne/filename.bin and output/label/filename.label
     Args:
         laserscan (Laserscan): The laserscan to save
         filename (str): The name or number for the laserscan.
         output (str): The output directory where the laserscan will be saved.
         overwrite (boolean): If True it will overwrite files at the output-path if they exist.
     """
    # check filename is string
    save_point_data(laserscan.points, laserscan.remissions, filename, output, overwrite)
    save_label_data(laserscan.labels, filename, output, overwrite)


def save_label_data(label_data: np.array, filename, output, overwrite=False):
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))
    label_dir = os.path.join(output, LABEL_FOLDER)
    label_file = os.path.join(label_dir, filename + LABEL_SUFFIX)
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    if not overwrite and os.path.exists(label_file):
        warnings.warn(f"A file with the filename {filename} already exist in the output directory {output}, "
                      f"Aborting save...", UserWarning)
    else:
        label_data.tofile(label_file)


def save_point_data(points: np.array, remissions: np.array, filename, output, overwrite=False):
    laser_dir = os.path.join(output, LASER_FOLDER)
    laser_file = os.path.join(laser_dir, filename + LASER_SUFFIX)
    if not os.path.isdir(laser_dir):
        os.makedirs(laser_dir)
    if not overwrite and os.path.exists(laser_file):
        warnings.warn(f"A file with the filename {filename} already exist in the output directory {output}, "
                                   f"Aborting save...", UserWarning)
    else:
        # Save point cloud
        remissions = remissions.reshape(-1, 1)
        data_array = np.concatenate((points, remissions), axis=1)
        data_array = data_array.reshape(-1, 1)
        data_array.tofile(laser_file)


# ------------------------- Functions for processing several laserscan files ------------------------------


def sequential_process_scans(laserscan_files, label_files, output_path, proc_func, overwrite):
    """Process laserscan with proc_func and save the new laserscan in output_path
    Args:
        laserscan_files (list): List with the laserscans files to process.
        label_files (list): List with the label files to process.
        output_path (str): Path of the output directory.
        proc_func (func(laserfile, labelfile)): A processing function that takes a laserscan file and
                                                a label file and returns name and a new laserscan
        overwrite (bool): If True, it will overwrite files at the output-path if they exist.

    Returns:
         proc_time (float): Time it took to process the laserscans.
         saved_files (int): Number of successfully processed laserscans.
    """
    counter = 20
    saved_files = 0
    start = time.time()
    for laserscan_file, label_file in zip(laserscan_files, label_files):
        if counter == 20:
            print(".", end='', flush=True)
            counter = 0
        counter += 1
        name, proc_laserscan = proc_func(laserscan_file, label_file)
        if name is not None:
            save_scan(proc_laserscan, name, output_path, overwrite)
            saved_files +=1
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def parallel_process_scans(laserscan_files, label_files, output_path, proc_func, overwrite, batch_size,
                           num_of_processes):
    """ Process laserscan with proc_func and save the new laserscan in output_path
    Args:
        laserscan_files (list): List with the laserscans files to process.
        label_files (list): List with the label files to process.
        output_path (str): Path of the output directory.
        proc_func (func(laserfile, labelfile)): A processing function that takes a laserscan file and
                                                a label file and returns name and a new laserscan
        overwrite (bool): If True, it will overwrite files at the output-path if they exist.
        num_of_processes (int): Number of processes to use.

    Returns:
         proc_time (float): Time it took to process the laserscans.
         saved_files (int): Number of successfully processed laserscans.
    """
    saved_files = 0
    num_of_files = len(laserscan_files)
    start = time.time()
    for batch_num in range(0, num_of_files, batch_size):
        pool = mp.Pool(num_of_processes)
        end = min(num_of_files, batch_num + batch_size)
        proc_laserscans = pool.starmap(proc_func,
                                       [(laserscan_file, label_file) for (laserscan_file, label_file) in
                                        zip(laserscan_files[batch_num:end], label_files[batch_num:end])])
        for name, scan in proc_laserscans:
            if name is not None:
                save_scan(scan, name, output_path, overwrite)
                saved_files += 1
        print(".", end='', flush=True)
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


# ------------------ Data config functions ---------------------------

def read_label_config(label_config_file: str):
    """Read the label configuration file.

    Args:
        label_config_file (str): Path to the configuration file.
    Returns:
        config_dict (dict): Dictionary with the configuration values.
    """
    with open(label_config_file) as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return configs


def get_color_mapping(label_config: str):
    """ Return a dictionary with a mapping between labels and colors."""
    config_dict = read_label_config(label_config)
    return config_dict["color_map"]


def get_label_id_mapping(label_config: str):
    """ Return a dictionary with a mapping between label names and label ID."""
    config_dict = read_label_config(label_config)
    rev_label_mapping = config_dict["labels"]
    items = [item for key, item in rev_label_mapping.items()]
    if len(items) != len(set(items)):
        raise ValueError(("The label names are are not unique! Please check the config-file"))
    label_mapping = {item: key for key, item in rev_label_mapping.items()}
    return label_mapping


def get_learning_mapping(label_config:str):
    config_dict = read_label_config(label_config)
    return config_dict["learning_map"]


def get_label_name_mapping(label_config:str):
    config_dict = read_label_config(label_config)
    return config_dict["labels"]


def get_learning_content(label_config:str):
    config_dict = read_label_config(label_config)
    learning_content = {}
    for label_id, learning_map in config_dict["learning_map"].items():
        learning_content[learning_map] = learning_content.get(learning_map, 0) + config_dict["content"][label_id]
    return learning_content


def get_ignore_mapping(label_config:str):
    config_dict = read_label_config(label_config)
    return config_dict["learning_ignore"]


def get_sequences(label_config:str, split:str):
    config_dict = read_label_config(label_config)
    sequences = config_dict["split"].get(split, None)
    return sequences
