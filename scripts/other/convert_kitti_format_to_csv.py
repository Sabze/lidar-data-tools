import argparse
from common.laserscan import LaserScan
import os
import warnings
import numpy as np
import time
from common import file_processing_tools as fpt

SUB_FOLDER = "sequences"
OUTPUT_FOLDER = "x_y_z_rem_label"
INDICE_ORDER = ["X", "Y", "Z", "REMISSION", "RANGE", "NUM_POINTS"]
SEQUENTIAL_FEEDBACK = 50


def save_laserscan_matrix(lasescan_matrix, filename: str, output: str, overwrite=False):
    """ Save lasescan_matrix at output/velodyne/filename.bin
     Args:
         lasescan_matrix: The laserscan matrix to save [N: x,y,z,rem,label]
         filename (str): The name or number for the laserscan.
         output (str): The output directory where the laserscan will be saved.
         overwrite (boolean): If True it will overwrite files at the output-path if they exist.
     """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    laser_dir = os.path.join(output, OUTPUT_FOLDER)
    laser_file = os.path.join(laser_dir, filename + ".csv")
    if not os.path.isdir(laser_dir):
        os.makedirs(laser_dir)
    if not overwrite and (os.path.exists(laser_file)):
        warnings.warn(f"A file with the filename {filename} already exist in the output directory {output}, "
                      f"Aborting save...", UserWarning)
    else:
        np.savetxt(laser_file, lasescan_matrix, delimiter=",")


def get_laserscan_matrix(laserscan_file: str, label_file: str):
    laser_filename = os.path.splitext(os.path.basename(laserscan_file))[0]
    label_filename = os.path.splitext(os.path.basename(label_file))[0]
    if laser_filename != label_filename:
        warnings.warn(UserWarning, f"The laser-file {laser_filename} and label-file {label_filename} are not "
                                   f"the same. Skipping these files...")
        return None, None
    laserscan = LaserScan()
    laserscan.open_scan(laserscan_file)
    laserscan.open_labels(label_file)
    laserscan_matrix = np.concatenate([laserscan.points, laserscan.remissions.reshape(-1, 1),
                                       laserscan.labels.reshape(-1, 1)], axis=1)
    return laserscan_matrix, laser_filename


def convert_kitti(file_dict: dict, output: str):
    print_counter = 0
    for seq, files in file_dict.items():
        start = time.time()
        print(f"Converting sequence number {seq}")
        output_dir = os.path.join(*[output, SUB_FOLDER, seq])
        laserfiles = files["laserscans"]
        labelfiles = files["labels"]
        for laserfile, labelfile in zip(laserfiles, labelfiles):
            if print_counter == SEQUENTIAL_FEEDBACK:
                print(".", end='', flush=True)
                print_counter = 0
            print_counter += 1
            laserscan_matrix, filename = get_laserscan_matrix(laserfile, labelfile)
            save_laserscan_matrix(laserscan_matrix, filename, output_dir)
        end = time.time()
        proc_time = end - start
        print(f"\nSaved files in {output_dir}, in in {proc_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts velodyne and label files in"
                                                 " the SemanticKITTI format to a CSV format.")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to the dataset in KITTI-format. No Default. ',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to the output directory',
    )

    args = parser.parse_args()
    file_dict = {}
    fpt.add_velodyne_files(file_dict, args.dataset)
    fpt.add_label_files(file_dict, args.dataset)
    convert_kitti(file_dict,  args.output)
