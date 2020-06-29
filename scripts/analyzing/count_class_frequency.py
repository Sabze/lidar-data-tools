import argparse
import time
import os
from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter
from common.script_tools import NumberDict

SEQUENTIAL_FEEDBACK = 500
SPLITS = ["train", "valid", "test", "all"]
CONFIG_DEFAULT = "../../config/labels/uav-custom.yaml"
# def get_mapped_dict(freq_dict, num_labels, learning_map):
#     mapped_dict = {}
#     for label, count in freq_dict.items():
#         mapped_label = learning_map[label]
#         mapped_dict[mapped_label] = mapped_dict.get(mapped_label, 0) + count/num_labels
#     return mapped_dict


def print_freq(count_dict: dict, num_labels: int, label_mapping: dict):
    for label_id, count in sorted(count_dict.items()):
        name = label_mapping.get(label_id, "Unknown")
        print(f"{name} ({label_id}): {count/num_labels:.7f} ({count})")


def print_data_cfg_style(count_dict: dict, num_labels: int, label_mapping: dict):
    for label_id, name in label_mapping.items():
        print(f"{label_id}: {count_dict.get(label_id, 0) / num_labels}")


def sequential_calc_freq_seq(label_files):
    start = time.time()
    freq_dict = NumberDict()
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    total_num_labels = 0
    for label_file in label_files:
        fb_counter.count()
        labels, counts, num_labels = fpt.count_labels(label_file)
        total_num_labels += num_labels
        freq_dict.add(labels, counts)
    end = time.time()
    proc_time = end - start
    fb_counter.done()
    return freq_dict, total_num_labels, proc_time


def calculate_frequencies(file_dict: dict, sequences, label_mapping: dict):
    total_num_labels = 0
    all_class_count = NumberDict()
    for sequence, file_types in file_dict.items():
        label_files = file_types["labels"]
        if sequences is None or int(sequence) in sequences:
            print(f"\n" + "-"*15 + f"Analysing sequence \'{sequence}\' containing {len(label_files)} scans." + "-"*15)
            seq_class_count, num_labels, proc_time = sequential_calc_freq_seq(label_files)
            all_class_count.update(seq_class_count.number_dict)
            total_num_labels += num_labels
            print_freq(seq_class_count.number_dict, total_num_labels, label_mapping)
    print(f"Total labels: {total_num_labels}")
    print("-"*80 + "\n" + "-"*35 + "  RESULT  " + "-"*35 + "\n" + "-"*80)
    print_freq(all_class_count.number_dict, total_num_labels, label_mapping)
    print("-"*50)
    print_data_cfg_style(all_class_count.number_dict, total_num_labels, label_mapping)


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Calculates the frequencies of the different classes (point based).")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to the dataset dir. (in Kitti format). No Default',
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

    parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      help='Path to the data configuration file. Defaults to %(default)s',
      default=default_cfg_file,
    )

    args = parser.parse_args()
    file_dict = {}
    sequences = fpt.get_sequences(args.data_cfg, args.split)
    label_name_map = fpt.get_label_name_mapping(args.data_cfg)
    fpt.add_label_files(file_dict, args.dataset)
    calculate_frequencies(file_dict, sequences, label_name_map)
