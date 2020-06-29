import argparse
import os
import time
import multiprocessing as mp

from common.laserscan import LaserScan
from common import laserscan_handler as lh
from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter

SUB_FOLDER = "sequences"
BATCH_SIZE = 300
FREE_PROCESSORS = 3
SEQUENTIAL_FEEDBACK = 20


def downsample_file(laserscan_file: str, label_file: str):
    laserscan, laser_filename = fpt.read_laserscan(laserscan_file, label_file)
    laserscan.set_laser_info(expected_num_of_lasers=64)
    if len(laserscan.laser_info) != 64:
        print("*"*10 + f" {laserscan_file}: {len(laserscan.laser_info)}")
    dwn_vert_laserscan = lh.vertical_downsample(laserscan, num_desired_lasers=16)
    # To also crop the scan:
    # dwn_laserscan = lh.crop_horizontally(dwn_vert_laserscan, -0.5 * np.pi)
    return (dwn_vert_laserscan, laser_filename)


def sequential_downsample_sequence(laserscan_files: list, label_files: list, output_path: str, overwrite: bool):
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    saved_files = 0
    start = time.time()
    for laserscan_file, label_file in zip(laserscan_files, label_files):
        fb_counter.count()
        dwns_laserscan, name = downsample_file(laserscan_file, label_file)
        fpt.save_scan(dwns_laserscan, name, output_path, overwrite)
        saved_files += 1
    end = time.time()
    fb_counter.done()
    proc_time = end - start
    return proc_time, saved_files


def parallel_downsample_sequence(laserscan_files, label_files, output_path, overwrite, num_of_processes):
    saved_files = 0
    num_of_files = len(laserscan_files)
    start = time.time()
    for batch_num in range(0, num_of_files, BATCH_SIZE):
        pool = mp.Pool(num_of_processes)
        end = min(num_of_files, batch_num + BATCH_SIZE)
        dwns_laserscans = pool.starmap(downsample_file,
                                       [(laserscan_file, label_file) for (laserscan_file, label_file) in
                                        zip(laserscan_files[batch_num:end], label_files[batch_num:end])])
        for scan, name in dwns_laserscans:
            fpt.save_scan(scan, name, output_path, overwrite)
            saved_files += 1
        print(".", end='', flush=True)  # Process indicator
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def downsample_files(laserscan_dict: dict, output: str, overwrite: bool, sequential=False):
    for sequence, file_types in laserscan_dict.items():
        label_files = file_types["labels"]
        laser_files = file_types["laserscans"]
        output_path = os.path.join(*[output, SUB_FOLDER, sequence])
        if sequential:
            proc_time, num_saved_files = sequential_downsample_sequence(laser_files, label_files, output_path,
                                                                        overwrite)
        else:
            num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
            print(f"Using {num_of_processes} processes, num of files: {len(laser_files)}, batch size: {BATCH_SIZE}")
            proc_time, num_saved_files = parallel_downsample_sequence(laser_files, label_files, output_path,
                                                                      overwrite, num_of_processes)
        print(f"\nSaved {num_saved_files} downsampled scans, in {proc_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample the vertical resolution in the SemanticKITTI data (from 64"
                                                 " laser channels to 16 laser channels), by copying "
                                                 "the points from every fourth laser channel and their labels to the "
                                                 "output directory.")
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
        '--sequential',
        action='store_true',
        help='Specify if you do not don not want to run in parallel.'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Specify if you want to overwrite existing files at the output location.'
    )

    args = parser.parse_args()
    print(args.output)
    file_dict = {}
    fpt.add_velodyne_files(file_dict, args.dataset)
    fpt.add_label_files(file_dict, args.dataset)
    downsample_files(file_dict, args.output, args.overwrite, sequential=args.sequential)
