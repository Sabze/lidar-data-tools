import argparse
import os
import multiprocessing as mp
import time

from common import file_processing_tools as fpt

SUB_FOLDER = "sequences"
FREE_PROCESSORS = 3
BATCH_SIZE = 300


def set_entropy(laserscan_file: str, label_file: str, pred_prob_file: str):
    laserscan, laserscan_num = fpt.read_laserscan(laserscan_file, label_file)
    laserscan.remissions = fpt.calc_entropy(pred_prob_file)
    return laserscan, laserscan_num


def set_entropy_sequence(laserscan_files: list, label_files: list, pred_prob_files: list, output_path: str,
                         overwrite: bool, num_of_processes: int):
    saved_files = 0
    num_of_files = len(laserscan_files)
    start = time.time()
    for batch_num in range(0, num_of_files, BATCH_SIZE):
        pool = mp.Pool(num_of_processes)
        end = min(num_of_files, batch_num + BATCH_SIZE)
        new_laserscans = pool.starmap(set_entropy,
                                      [(laserscan_file, label_file, pred_prob_file) for (laserscan_file,
                                                                                         label_file,
                                                                                         pred_prob_file) in
                                       zip(laserscan_files[batch_num:end], label_files[batch_num:end],
                                           pred_prob_files[batch_num:end])])
        for scan, name in new_laserscans:
            fpt.save_scan(scan, name, output_path, overwrite)
            saved_files += 1
        print(".", end='', flush=True)
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def set_entropy_all(file_dict: dict, output: str, overwrite=True):
    for seq, files in file_dict.items():
        laserscan_files = files["laserscans"]
        label_files = files["labels"]
        pred_probs = files["pred_probs"]
        output_path = os.path.join(*[output, SUB_FOLDER, seq])
        print(f"Calc entropy for sequence number {seq}")
        num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
        proc_time, num_saved_files = set_entropy_sequence(laserscan_files, label_files, pred_probs, output_path,
                                                          overwrite, num_of_processes)
        print(f"\nSaved {num_saved_files} scans, in {proc_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copies the velodyne data and the label data to the specified "
                                                 "output dir and overwrites the remission channel of"
                                                 " the velodyne data (in the output dir) with the entropy.")
    parser.add_argument(
        '--probabilities', '-p',
        type=str,
        required=False,
        default=None,
        help='Path to the prediction probabilities. Default is the regular dataset path',
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to the dataset directory (KITTI format). No Default',
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
        help='Specify if you want to overwrite the data at the output-path.'
    )

    args = parser.parse_args()
    if args.probabilities is None:
        probability_dir = args.dataset
    else:
        probability_dir = args.probabilities
    data_dict = {}
    fpt.add_velodyne_files(data_dict, args.dataset)
    fpt.add_label_files(data_dict, args.dataset)
    fpt.add_pred_prob_files(data_dict, probability_dir)
    set_entropy_all(data_dict, args.output, args.overwrite)
    fpt.copy_labeling_tool_files(args.dataset, args.output)
