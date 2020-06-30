import argparse
import os
import numpy as np
import time
import multiprocessing as mp

from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter
from common.laserscan import LaserScan
from common import laserscan_handler as lh

FREE_PROCESSORS = 3
INDEX_ORDER = ["X", "Y", "Z", "REMISSION", "RANGE", "NUM_POINTS"]
SEQUENTIAL_FEEDBACK = 50
CONFIG_DEFAULT = "../../config/uav-custom.yaml"
SPLITS = ["train", "valid", "test", "all"]


def sum_mean_values(laserscan_file:str):
    laserscan = LaserScan()
    laserscan.open_scan(laserscan_file)
    return lh.sum_laserscan_properties(laserscan)


def sum_std_values(laserscan_file:str, mean_vector):
    laserscan = LaserScan()
    laserscan.open_scan(laserscan_file)
    xmean = mean_vector[INDEX_ORDER.index("X")]
    ymean = mean_vector[INDEX_ORDER.index("Y")]
    zmean = mean_vector[INDEX_ORDER.index("Z")]
    remmean = mean_vector[INDEX_ORDER.index("REMISSION")]
    rangemean = mean_vector[INDEX_ORDER.index("RANGE")]
    return lh.sum_laserscan_std(laserscan, xmean, ymean, zmean, remmean, rangemean)


def calculate_mean(sum_vector):
    ind = INDEX_ORDER.index("NUM_POINTS")
    num_points = sum_vector[ind]
    mean_vector = sum_vector/num_points
    return mean_vector, num_points


def calculate_std(sum_vector):
    ind = INDEX_ORDER.index("NUM_POINTS")
    num_points = sum_vector[ind]
    std_vector = np.sqrt(sum_vector/(num_points - 1))
    return std_vector, num_points


def print_res(mean_vector, res_type):
    for name, mean in zip(INDEX_ORDER, mean_vector):
        print(f"{name}-{res_type}: {mean: .4f}")
    print(mean_vector)


def parallel_sum_mean_in_sequence(laserscan_files, num_of_processes):
    start = time.time()
    pool = mp.Pool(num_of_processes)
    scan_sums = pool.map(sum_mean_values, laserscan_files)
    pool.close()
    pool.join()
    scan_sums = np.concatenate(scan_sums, axis=0)
    end = time.time()
    proc_time = end - start
    return scan_sums, proc_time


def sequential_sum_mean_in_sequence(laserscan_files):
    start = time.time()
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    scan_sums = []
    for laserscan_file in laserscan_files:
        fb_counter.count()
        scan_sum = sum_mean_values(laserscan_file)
        scan_sums.append(scan_sum)
    end = time.time()
    proc_time = end - start
    fb_counter.done()
    scan_sums = np.concatenate(scan_sums, axis=0)
    return scan_sums, proc_time


def sequential_sum_std_in_sequence(laserscan_files, mean_vector):
    start = time.time()
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    scan_sums = []
    for laserscan_file in laserscan_files:
        fb_counter.count()
        scan_sum = sum_std_values(laserscan_file, mean_vector)
        scan_sums.append(scan_sum)
    end = time.time()
    proc_time = end - start
    fb_counter.done()
    scan_sums = np.concatenate(scan_sums, axis=0)
    return scan_sums, proc_time


def parallel_sum_std_in_sequence(laserscan_files, num_of_processes, mean_vector):
    start = time.time()
    pool = mp.Pool(num_of_processes)
    scan_sums = pool.starmap(sum_std_values, [(laserscan_file, mean_vector) for laserscan_file in laserscan_files])
    pool.close()
    pool.join()
    scan_sums = np.concatenate(scan_sums, axis=0)
    end = time.time()
    proc_time = end - start
    return scan_sums, proc_time


def print_progress(num_files, proc_time, scan_sums, sequence_sum, num_points):
    print(f"Processed {num_files} scans, in {proc_time:.2f} seconds")
    print(f"Shape of sum-vector: {scan_sums.shape}")
    print(f"Summation of sequence: {sequence_sum}")
    print(f"Number of points: {num_points}")


def calculate_dataset_mean(file_dict, sequential: bool, sequences):
    print("\n" + "#"*30 + f" CALCULATING MEAN " + "#"*30)
    sequence_sums = []
    for sequence, laserscan_files in file_dict.items():
        if sequences is None or int(sequence) in sequences:
            scans = laserscan_files["laserscans"]
            print("\n" + "-"*15 + f"Analysing sequence \'{sequence}\' containing {len(scans)} scans." + "-"*15)
            if not sequential:
                num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
                scan_sums, proc_time = parallel_sum_mean_in_sequence(scans, num_of_processes)
            else:
                scan_sums, proc_time = sequential_sum_mean_in_sequence(scans)
            sequence_sum = np.sum(scan_sums, axis=0).reshape(1, -1)
            sequence_sums.append(sequence_sum)
            part_mean_vector, num_points = calculate_mean(sequence_sum[0,:])
            print_progress(len(scans), proc_time, scan_sums, sequence_sum, num_points)
            print_res(part_mean_vector, "mean")
    total_sums = np.sum(np.concatenate(sequence_sums, axis=0), axis=0)
    mean_vector, num_points = calculate_mean(total_sums)
    print("-" * 80 + "\n" + "-" * 32 + "  MEAN RESULT  " + "-" * 33 + "\n" + "-" * 80)
    print(f"Total number of points: {num_points}")
    print_res(mean_vector, "mean")
    return mean_vector, num_points


def calculate_dataset_std(file_dict, sequential: bool, mean_vector, sequences):
    print("\n" + "#"*30 + f" CALCULATING STD " + "#"*30)
    sequence_sums = []
    for sequence, laserscan_files in file_dict.items():
        if sequences is None or int(sequence) in sequences:
            scans = laserscan_files["laserscans"]
            print("\n" + "-"*15 + f"Analysing sequence \'{sequence}\' containing {len(scans)} scans." + "-"*15)
            if not sequential:
                num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
                scan_sums, proc_time = parallel_sum_std_in_sequence(scans, num_of_processes, mean_vector)
            else:
                scan_sums, proc_time = sequential_sum_std_in_sequence(scans, mean_vector)
            sequence_sum = np.sum(scan_sums, axis=0).reshape(1, -1)
            sequence_sums.append(sequence_sum)
            part_std_vector, num_points = calculate_std(sequence_sum[0,:])
            print_progress(len(scans), proc_time, scan_sums, sequence_sum, num_points)
            print_res(part_std_vector, "std")
    total_sums = np.sum(np.concatenate(sequence_sums, axis=0), axis=0)
    std_vector, num_points = calculate_std(total_sums)
    print("-" * 80 + "\n" + "-" * 33 + "  STD RESULT  " + "-" * 33 + "\n" + "-" * 80)
    print(f"Total number of points: {num_points}")
    print_res(std_vector, "std")
    return std_vector, num_points


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Returns mean and standard deviation of the data")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to the semnaticKITTI dataset dir. No Default',
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Specify if you do not want to run in parallel.'
    )

    parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      help='Path to the data configuration file. Defaults to %(default)s',
      default=default_cfg_file,
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=SPLITS,
        default="all",
        help='Split to analyze on one of ' +
             str(SPLITS) + '. Defaults to \'%(default)s\'',
    )

    args = parser.parse_args()
    file_dict = {}
    fpt.add_velodyne_files(file_dict, args.dataset)
    wanted_sequences = fpt.get_sequences(args.data_cfg, args.split)
    print(f"Calculating the mean and std of the sequences: {wanted_sequences}")
    dataset_mean, num_points1 = calculate_dataset_mean(file_dict, args.sequential, wanted_sequences)
    dataset_std, num_points2 = calculate_dataset_std(file_dict, args.sequential, dataset_mean, wanted_sequences)
    assert num_points1 == num_points2, "Bug in code, these should be the same!"
    print("-" * 80 + "\n" + "-" * 32 + "  ALL RESULTS  " + "-" * 33 + "\n" + "-" * 80)
    print_res(dataset_mean, "mean")
    print_res(dataset_std, "std")
