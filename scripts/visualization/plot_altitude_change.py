import argparse
import os
import numpy as np
from common import laserscan_handler as lh
from common import utilities as utilities
from common import laserscan_visualization as lv
from common import file_processing_tools as fpt

CONFIG_DEFAULT = "../../config/uav-custom.yaml"


def show_range_images(seq_dir, scan_range, cmap, norm, fov_up, fov_down, size, img_dir, title, save_imgs,
                      name, aspect, xdiff, max_fov_diff):
    laserscans = fpt.get_laserscans(seq_dir, scan_range, name=name, labels=True, fov_up=fov_up,
                                    fov_down=fov_down)
    min_allowed = fov_down - max_fov_diff
    max_allowed = fov_up + max_fov_diff
    for laserscan in laserscans:
        new_laserscan = lh.get_translated_laserscan(laserscan, x_diff=xdiff)
        lv.plot_adjusted_spherical_proj(new_laserscan, min_allowed, max_allowed, cmap, norm, size, aspect=aspect,
                                        save=save_imgs, filedir=img_dir,
                                        title=title, name=laserscan.name, color_bar=False)
        # lv.plot_projection_positions(new_laserscan, aspect=aspect)
    return laserscans[0].name


if __name__ == "__main__":
    head, tail = os.path.split(os.path.abspath(__file__))
    default_cfg_file = os.path.join(head, CONFIG_DEFAULT)
    parser = argparse.ArgumentParser(description="Plot the range images after a change in the x-coord "
                                                 "(used for the UAV-data). ")
    parser.add_argument(
      '--input_path', '-i',
      type=str,
      required=True,
      help='Path to the input directory. No Default. ',
    )

    parser.add_argument(
      '--output_path', '-o',
      type=str,
      help='Path to the output directory. No Default. ',
    )

    parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      help='Path to the data configuration file. Defaults to %(default)s.',
      default=default_cfg_file,
    )

    parser.add_argument(
        '--use_existing_images', '-u',
        action='store_true',
        help='Specify if already generated images should be used.'
    )

    parser.add_argument(
        '--save_images', '-si',
        action='store_true',
        help='Specify if the images should be saved.'
    )

    parser.add_argument(
        '--save_video', '-sv',
        action='store_true',
        help='Specify if a video should be generated and saved.'
    )

    parser.add_argument(
        '--start_frame', '-sf',
        type=int,
        default=1,
        help='The laserscans in the interval [start_frame, end_frame, stride] will be projected to range images.'
             ' Defaults to %(default)s.',
    )

    parser.add_argument(
        '--end_frame', '-ef',
        type=int,
        default=200,
        help='The laserscans in the interval [start_frame, end_frame, stride] will be projected to range images.'
             ' Defaults to %(default)s.'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=4,
        help='The laserscans in the interval [start_frame, end_frame, stride] will be projected to range images.'
             ' Defaults to %(default)s.'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='The fps of the video. Defaults to %(default)s.'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=16,
        help='The height of the range image. Defaults to %(default)s.'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        default=1024,
        help='The width of the range image. Defaults to %(default)s.'
    )

    parser.add_argument(
        '--xdiff', '-x',
        type=int,
        default=3,
        help='The change in height. Defaults to %(default)s.'
    )

    parser.add_argument(
        '--max_fov_diff',
        type=int,
        default=3,
        help='The max allowed change in FoV. Defaults to %(default)s.'
    )

    parser.add_argument(
        '--no_title',
        action='store_true',
        help='No titles will be added to the images.'
    )

    parser.add_argument(
        '--aspect',
        type=int,
        default=7,
        help='The aspect of the range image. Defaults to %(default)s.'
    )

    args = parser.parse_args()
    fov_up = 15
    fov_down = -15
    name= "uav"
    cm = fpt.get_color_mapping(args.data_cfg)
    cmap, norm = lv.get_cmap_labels(cm)
    if (args.save_images or args.save_video) and args.output_path is None:
        raise Exception("Specify an output path!")
    if args.use_existing_images:
        img_dir = args.input_path
        scan_name = "video"
    else:
        if args.save_images:
            img_dir = os.path.join(args.output_path, "images")
        else:
            img_dir = None
        scans = list(range(args.start_frame, args.end_frame, args.stride))
        scan_name = show_range_images(args.input_path, scans, cmap, norm, fov_up, fov_down, (args.height, args.width),
                                      img_dir, not args.no_title, args.save_images,
                                      name, args.aspect, args.xdiff, args.max_fov_diff)
    if args.save_video:
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)
        video_dir = os.path.join(args.output_path, f"{scan_name}.avi")
        utilities.convert_frames_to_video(img_dir, video_dir, args.fps)






