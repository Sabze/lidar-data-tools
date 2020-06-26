import argparse
import os
from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter
from common.script_tools import NumberDict

SEQUENTIAL_FEEDBACK = 200
CONFIG_DEFAULT = "../../config/uav-custom.yaml"
SPLITS = ["train", "valid", "test", "all"]


def print_count(count_dict: dict):
    print("Weight | Number of scans | %-scans with this weight | Normalized weight ")
    for weight, info in sorted(count_dict.items()):
        count, freq, norm = info
        print(f"{weight: <12} {count:<17}" + f"({freq:.5f})" + " " * 18 + f"{norm:.6f}")


def count_weights_seq(label_files: list, label_learning_map: dict, label_ignore_map: dict, label_weights: dict):
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    weight_count = {}
    for label_file in label_files:
        fb_counter.count()
        weight = fpt.get_weight(label_file, label_learning_map, label_ignore_map, label_weights)
        weight_count[weight] = weight_count.get(weight, 0) + 1
    fb_counter.done()
    return weight_count


def get_weights_info(all_weights: dict):
    tot_weight = 0
    weight_info = {}
    for weight, count in all_weights.items():
        tot_weight += weight*count
    for weight, count in all_weights.items():
        weight_info[weight] = (count, count*weight/tot_weight, weight/tot_weight)
    return weight_info, tot_weight


def calc_scan_weights(label_dict, wanted_sequences, label_learning_map, label_ignore_map, label_weights):
    weight_count = NumberDict()
    total_num_scans = 0
    for sequence, file_types in label_dict.items():
        if wanted_sequences is None or int(sequence) in wanted_sequences:
            label_files = file_types["labels"]
            print(f"\n" + "-"*15 + f"Analysing sequence \'{sequence}\' containing {len(label_files)} scans." + "-"*15)
            weight_count_seq = count_weights_seq(label_files, label_learning_map, label_ignore_map, label_weights)
            total_num_scans += len(label_files)
            weight_count.update(weight_count_seq)
    print(f"Count dict : {weight_count.number_dict}")
    print("-" * 80 + "\n" + "-" * 35 + "  RESULT  " + "-" * 35 + "\n" + "-" * 80)
    print(f"Total number of scans: {total_num_scans}")
    weight_info, tot_weight = get_weights_info(weight_count.number_dict)
    print(f"Total weight: {tot_weight}")
    print_count(weight_info)


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Calculates the weights of the scans (used for weighted sampling).")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to the dataset dir. (in Kitti format). No Default',
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
    sequences = fpt.get_sequences(args.data_cfg, args.split)
    label_learn_map = fpt.get_learning_mapping(args.data_cfg)
    label_content_map = fpt.get_learning_content(args.data_cfg)
    label_ignore = fpt.get_ignore_mapping(args.data_cfg)
    print(f"Counting number of scans with the different classes in the sequences: {sequences}")
    file_dict = {}
    fpt.add_label_files(file_dict, args.dataset)
    calc_scan_weights(file_dict, sequences, label_learn_map, label_ignore, label_content_map)
