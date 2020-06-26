import argparse
import os
import warnings
import numpy as np
import time
import multiprocessing as mp

from common import laserscan_handler as lh
from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter
from common.laserscan import LaserScan

SUB_FOLDER = "sequences"
BATCH_SIZE = 200
FREE_PROCESSORS = 3
SEQUENTIAL_FEEDBACK = 20


def save_split_scans(scan1: LaserScan, scan2: LaserScan, name: str, output_path: str, overwrite=False):
    num_saved_files = 0
    if name is not None:
        fpt.save_scan(scan1, name, output_path, overwrite)
        num_saved_files += 1
        if name[0] == "0":
            name2 = f"1{name[1:]}"
            fpt.save_scan(scan2, name2, output_path, overwrite)
            num_saved_files += 1
        else:
            warnings.warn(f"To many laserscans in sequence, ignoring second split for {name}")
    return num_saved_files


def split_laserscan(laserscan_file: str, label_file: str):
    laserscan, name = fpt.read_laserscan(laserscan_file, label_file)
    front_split_laserscan = lh.crop_horizontally(laserscan, -0.5 * np.pi)
    rotated_laserscan = lh.get_rotated_laserscan(laserscan, z_rot=np.pi)
    back_split_laserscan = lh.crop_horizontally(rotated_laserscan, -0.5 * np.pi)
    return (front_split_laserscan, back_split_laserscan, name)


def parallel_split_scans(laserscan_files: list, label_files: list, output_path: str, overwrite: bool, batch_size: int,
                         num_of_processes: int):
    """ Process laserscan with proc_func and save the new laserscan in output_path
    Args:
        laserscan_files (list): List with the laserscans files to process.
        label_files (list): List with the label files to process.
        output_path (str): Path of the output directory.
        overwrite (bool): If True, it will overwrite files at the output-path if they exist.
        num_of_processes (int): Number of processes to use.
        batch_size (int): How many scans to process in parallel.

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
        proc_laserscans = pool.starmap(split_laserscan,
                                       [(laserscan_file, label_file) for (laserscan_file, label_file) in
                                        zip(laserscan_files[batch_num:end], label_files[batch_num:end])])
        for scan1, scan2, name in proc_laserscans:
            num_saved_scans = save_split_scans(scan1, scan2, name, output_path, overwrite)
            saved_files += num_saved_scans
        print(".", end='', flush=True)
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def sequential_split_scans(laserscan_files: list, label_files: list, output_path: str, overwrite: bool):
    """Process laserscan with proc_func and save the new laserscan in output_path
    Args:
        laserscan_files (list): List with the laserscans files to process.
        label_files (list): List with the label files to process.
        output_path (str): Path of the output directory.
        overwrite (bool): If True, it will overwrite files at the output-path if they exist.

    Returns:
         proc_time (float): Time it took to process the laserscans.
         saved_files (int): Number of successfully processed laserscans.
    """
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    saved_files = 0
    start = time.time()
    for laserscan_file, label_file in zip(laserscan_files, label_files):
        fb_counter.count()
        scan1, scan2, name = split_laserscan(laserscan_file, label_file)
        num_saved_scans = save_split_scans(scan1, scan2, name, output_path, overwrite)
        saved_files += num_saved_scans
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def split_laserscan_files(laserscan_dict: dict, output: str, overwrite: bool, sequential=False):
    for sequence, file_types in laserscan_dict.items():
        laserscan_files = file_types["laserscans"]
        label_files = file_types["labels"]
        output_dir = os.path.join(*[output, SUB_FOLDER, sequence])
        print(f"Splitting Scans in Sequence {sequence}")
        if sequential:
            proc_time, num_saved_files = sequential_split_scans(laserscan_files, label_files,
                                                                output_dir, overwrite)
        else:
            num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
            proc_time, num_saved_files = parallel_split_scans(laserscan_files, label_files,
                                                              output_dir, overwrite, BATCH_SIZE,
                                                              num_of_processes)
        print(f"\nSaved {num_saved_files} scans from {len(laserscan_files)} scans, in {proc_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample vertically the SemanticKITTI data")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to the dataset dir. (in Kitti format). No Default',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to the output directory',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Specify if you want to overwrite existing files at the output location.'
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Specify if you do not want to run in parallel.'
    )

    args = parser.parse_args()
    file_dict = {}
    fpt.add_velodyne_files(file_dict, args.dataset)
    fpt.add_label_files(file_dict, args.dataset)
    split_laserscan_files(file_dict, args.output, args.overwrite, sequential=args.sequential)

