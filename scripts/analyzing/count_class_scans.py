import argparse
import os

from common import file_processing_tools as fpt
from common.script_tools import SequentialFeedbackCounter
from common.script_tools import NumberDict

SEQUENTIAL_FEEDBACK = 200
CONFIG_DEFAULT = "../../config/uav-custom.yaml"
SPLITS = ["train", "valid", "test", "all"]


def print_count(count_dict: dict, num_scans: int, label_mapping: dict):
    for label_id, count in sorted(count_dict.items()):
        name = label_mapping.get(label_id, "Unknown")
        print(f"{name} ({label_id}): {count/num_scans:.7f} ({count})")


def count_classes_seq(label_files: list):
    fb_counter = SequentialFeedbackCounter(SEQUENTIAL_FEEDBACK)
    class_count = NumberDict()
    for label_file in label_files:
        fb_counter.count()
        labels, _, num_labels = fpt.count_labels(label_file)
        counts = [1]*len(labels)
        class_count.add(labels, counts)
    fb_counter.done()
    return class_count


def calc_class_count(label_dict, wanted_sequences, label_name_map):
    class_count = NumberDict()
    total_num_scans = 0
    for sequence, file_types in label_dict.items():
        if wanted_sequences is None or int(sequence) in wanted_sequences:
            label_files = file_types["labels"]
            print(f"\n" + "-"*15 + f"Analysing sequence \'{sequence}\' containing {len(label_files)} scans." + "-"*15)
            class_count_seq = count_classes_seq(label_files)
            total_num_scans += len(label_files)
            class_count.update(class_count_seq.number_dict)
    print(f"Count dict : {class_count.number_dict}")
    print("-" * 80 + "\n" + "-" * 35 + "  RESULT  " + "-" * 35 + "\n" + "-" * 80)
    print(f"Total number of scans: {total_num_scans}")
    print_count(class_count.number_dict, total_num_scans, label_name_map)


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Calculates how many scans that contain the different classes, "
                                                 "as well as how many percent of the scans that contain"
                                                 " the different classes.")
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
    label_name_mapping = fpt.get_label_name_mapping(args.data_cfg)
    print(f"Counting number of scans with the different classes in the sequences: {sequences}")
    file_dict = {}
    fpt.add_label_files(file_dict, args.dataset)
    calc_class_count(file_dict, sequences, label_name_mapping)
