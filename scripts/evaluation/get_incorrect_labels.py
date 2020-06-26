import argparse
import os
import multiprocessing as mp
import time
import numpy as np

from common import file_processing_tools as fpt

SUB_FOLDER = "sequences"
FREE_PROCESSORS = 3
BATCH_SIZE = 300
INCORRECT_LABEL = 1  # Class outlier (red)
CORRECT_LABEL = 0  # Class Unlabeled (black)
CONFIG_DEFAULT = "../../config/uav-custom.yaml"


def get_accuracy_labels(label_file: str, pred_file: str, learning_map: dict):
    label_filename = os.path.splitext(os.path.basename(label_file))[0]
    labels = fpt.read_label_file(label_file)
    preds = fpt.read_label_file(pred_file)
    accuracy_labels = np.ones(labels.shape, dtype=np.int32)
    for ind in range(0, labels.shape[0]):
        if learning_map[preds[ind]] == learning_map[labels[ind]]:
            accuracy_labels[ind] = CORRECT_LABEL
        else:
            accuracy_labels[ind] = INCORRECT_LABEL
    return accuracy_labels, label_filename


def save_accuracy_labels(label_files: list, pred_prob_files: list, output_path: str, learning_map: dict,
                         overwrite: bool, num_of_processes: int):
    saved_files = 0
    num_of_files = len(label_files)
    start = time.time()
    for batch_num in range(0, num_of_files, BATCH_SIZE):
        pool = mp.Pool(num_of_processes)
        end = min(num_of_files, batch_num + BATCH_SIZE)
        accuracy_labels = pool.starmap(get_accuracy_labels,
                                       [(label_file, pred_file, learning_map) for (label_file, pred_file) in
                                        zip(label_files[batch_num:end],
                                            pred_prob_files[batch_num:end])])
        for labels, name in accuracy_labels:
            fpt.save_label_data(labels, name, output_path, overwrite)
            saved_files += 1
        print(".", end='', flush=True)
        pool.close()
        pool.join()
    end = time.time()
    proc_time = end - start
    return proc_time, saved_files


def get_accuracy_label_files(file_dict: dict, output: str, learning_map: dict, overwrite=False):
    for seq, files in file_dict.items():
        label_files = files["labels"]
        pred_probs = files["predictions"]
        output_path = os.path.join(*[output, SUB_FOLDER, seq])
        print(f"Calc accuracy labels for sequence number {seq}")
        num_of_processes = max(1, mp.cpu_count() - FREE_PROCESSORS)
        proc_time, num_saved_files = save_accuracy_labels(label_files, pred_probs, output_path, learning_map,
                                                          overwrite, num_of_processes)
        print(f"\nSaved {num_saved_files} scans, in {proc_time:.2f} seconds")


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Creates new labels based on the predictions and saves them in "
                                                 "the specified output directory. The points that were "
                                                 "correctly classified  by the predictions will be"
                                                 " labeled as \'Unlabeled\' (no color), "
                                                 "while the incorrectly classified points will get labeled "
                                                 "as \'Outlier\' (red).")
    parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=False,
      default=None,
      help='Path to the predictions. Default is the regular dataset path',
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
      '--data_cfg', '-dc',
      type=str,
      help='Path to the data configuration file. Defaults to %(default)s',
      default=default_cfg_file,
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Specify if you want to overwrite the data at the output-path.'
    )

    args = parser.parse_args()
    class_mapping = fpt.get_learning_mapping(args.data_cfg)
    if args.predictions is None:
        prediction_dir = args.dataset
    else:
        prediction_dir = args.predictions
    data_dict = {}
    fpt.add_label_files(data_dict, args.dataset)
    fpt.add_prediction_files(data_dict, prediction_dir)
    get_accuracy_label_files(data_dict, args.output, class_mapping, args.overwrite)






