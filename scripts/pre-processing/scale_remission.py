import argparse
import os
import warnings
import numpy as np
import time
import multiprocessing as mp

from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter


SUB_FOLDER = "sequences"
BATCH_SIZE = 500
FREE_PROCESSORS = 3
MAX_REMISSION = 255
SEQUENTIAL_FEEDBACK = 20


def get_dist_scaled_rem(laserscan):
    """Scales the remission based on the squared distance, then standardizes it."""
    new_rem = laserscan.remissions * (np.linalg.norm(laserscan.points, axis=1) ** 2)
    new_rem = new_rem / np.max(new_rem)
    new_rem = (new_rem / np.std(new_rem))
    return new_rem


def get_norm_stand_rem(laserscan):
    """Normalizes and standardizes the remission."""
    standardized = (laserscan.remissions - np.mean(laserscan.remissions)) / np.std(laserscan.remissions)
    new_rem = (standardized - np.min(standardized)) / (np.max(standardized) - np.min(standardized))
    return new_rem


def get_scaled_rem(laserscan):
    """Divides the remission with MAX_REMISSION (255)."""
    new_rem = laserscan.remissions/MAX_REMISSION
    return new_rem


SCALING_TYPES = {"scale": get_scaled_rem,
                 "dist":  get_dist_scaled_rem,
                 "stand": get_norm_stand_rem}



def scale_laserscan(laserscan_file: str, label_file: str, scale_func):
    laserscan, laserscan_num = fpt.read_laserscan(laserscan_file, label_file)
    if any(laserscan.remissions > MAX_REMISSION):
        warnings.warn(f"Found a remission-value bigger than {MAX_REMISSION} in file {laserscan_file}", UserWarning)
    laserscan.remissions = scale_func(laserscan)
    return (laserscan, laserscan_num)


def sequential_scale_rem_sequence(laserscan_files:list, label_files:list, output_path:str, scale_func, overwrite:bool):
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    saved_files = 0
    start = time.time()
    for laserscan_file, label_file in zip(laserscan_files, label_files):
        fb_counter.count()
        new_laserscan, name = scale_laserscan(laserscan_file, label_file, scale_func)
        fpt.save_scan(new_laserscan, name, output_path, overwrite)
        saved_files +=1
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def parallel_scale_rem_sequence(laserscan_files:list, label_files:list, output_path:str, scale_func, overwrite:bool,
                                num_of_processes:int):
    saved_files = 0
    num_of_files = len(laserscan_files)
    start = time.time()
    for batch_num in range(0, num_of_files, BATCH_SIZE):
        pool = mp.Pool(num_of_processes)
        end = min(num_of_files, batch_num + BATCH_SIZE)
        new_laserscans = pool.starmap(scale_laserscan,
                                       [(laserscan_file, label_file, scale_func) for (laserscan_file, label_file) in
                                        zip(laserscan_files[batch_num:end], label_files[batch_num:end])])
        for scan, name in new_laserscans:
            fpt.save_scan(scan, name, output_path, overwrite)
            saved_files += 1
        print(".", end='', flush=True)
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def scale_rem_files(laserscan_dict:dict, output:str, overwrite: bool, scale_func, sequential=False):
    for seq, file_types in laserscan_dict.items():
        laserscan_files = file_types["laserscans"]
        label_files = file_types["labels"]
        output_path = os.path.join(*[output, SUB_FOLDER, seq])
        print(f"Scale remission for sequence number {seq}")
        if sequential:
            proc_time, num_saved_files = sequential_scale_rem_sequence(laserscan_files, label_files, output_path,
                                                                       scale_func, overwrite)
        else:
            num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
            proc_time, num_saved_files = parallel_scale_rem_sequence(laserscan_files, label_files, output_path,
                                                                     scale_func, overwrite, num_of_processes)
        print(f"\nSaved {num_saved_files} scans, in {proc_time:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale the remission so that it is in the interval 0-1.")
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
        help='Specify if you want to overwrite the data in the output-path.'
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Specify if you don\'t want to run in parallel.'
    )

    parser.add_argument(
        '--scale_type', '-t',
        type=str,
        required=False,
        choices=SCALING_TYPES.keys(),
        default="scale",
        help=f"Specify which type of scaling, one of {list(SCALING_TYPES.keys())}. Defaults to \'%(default)s\'"
    )

    args = parser.parse_args()
    scale_func = SCALING_TYPES[args.scale_type]
    file_dict = {}
    fpt.add_velodyne_files(file_dict, args.dataset)
    fpt.add_label_files(file_dict, args.dataset)
    scale_rem_files(file_dict, args.output, args.overwrite, scale_func, sequential=args.sequential)
