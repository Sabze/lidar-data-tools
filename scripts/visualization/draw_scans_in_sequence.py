import argparse
import open3d as o3d
import os

from common import file_processing_tools as fpt
from common import laserscan_visualization as lv
from common.script_tools import SequentialFeedbackCounter

USER_CHECK_COUNT = 5
FEEDBACK_COUNT = 50
CONFIG_DEFAULT = "../../config/uav-custom.yaml"


def view_pcd(pcd_list: list):
    max_scans_to_view = list(range(1, len(pcd_list)))
    scans_to_view = int(input(f"Number of scans to view (1-{len(pcd_list)}): "))
    while scans_to_view not in max_scans_to_view:
        scans_to_view = int(input(f"Number of scans to view (1-{len(pcd_list)}): "))
    print("To the change number of scans to view, press C")
    counter = 0
    start = 0
    while start < len(pcd_list):
        end = min(start+scans_to_view, len(pcd_list))
        print(f"Showing scans {start} to {end}")
        batch = pcd_list[start:end]
        o3d.visualization.draw_geometries(batch)
        counter += 1
        if counter == USER_CHECK_COUNT:
            scans_to_view = check_input(scans_to_view, max_scans_to_view)
            counter = 0
            if scans_to_view is None:
                return None
        start += scans_to_view


def check_input(scans_to_view: int, max_scan_to_view: list):
    stop_com = ["n", "no", "stop", "exit"]
    cont_com = ["y", "c", "yes"]
    next_step = None
    while next_step not in stop_com + cont_com:
        next_step = input("Continue (y/n)?").lower()
        if next_step in stop_com:
            return None
        elif next_step == "c":
            scans_to_view = None
            while scans_to_view not in max_scan_to_view:
                scans_to_view = int(input(f"Number of scans to view (1-{len(pcd_list)}): "))
            print(f"Changing number of scans to view to {scans_to_view}")
            return scans_to_view
        elif next_step in cont_com:
            return scans_to_view


def get_pcd_list(file_dict: dict, cm: dict):
    print("Collecting laserscans")
    laserscans = []
    if len(file_dict) > 1:
        raise Exception("Can only draw one sequence at a time.")
    sequence = list(file_dict.keys())[0]
    files = file_dict[sequence]
    laserfiles = files["laserscans"]
    labelfiles = files["labels"]
    fb_counter = SequentialFeedbackCounter(FEEDBACK_COUNT)
    for laserfile, labelfile in zip(laserfiles, labelfiles):
        fb_counter.count()
        laserscan, _ = fpt.read_laserscan(laserfile, labelfile)
        colors = lv.get_label_colors(laserscan, cm)
        pcd = lv.convert_laserscan_2_open3d(laserscan, colors)
        laserscans.append(pcd)
    fb_counter.done()
    return laserscans


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Draws the laserscans colored by the labels. No registration. ")

    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Path to a sequence folder (in KITTI-format). No Default. ',
    )

    parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      help='Path to the data configuration file. Defaults to %(default)s',
      default=default_cfg_file,
    )

    args = parser.parse_args()
    cm = fpt.get_color_mapping(args.data_cfg)
    data_dict = {}
    fpt.add_velodyne_files(data_dict, args.dataset)
    fpt.add_label_files(data_dict, args.dataset)
    pcd_list = get_pcd_list(data_dict, cm)
    view_pcd(pcd_list)
