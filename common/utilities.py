import os

import cv2
import numpy as np
from matplotlib import cm


def get_spherical_coords(cart_coords: np.array):
    spherical_coords = np.zeros(cart_coords.shape)
    spherical_coords[:, 0] = np.linalg.norm(cart_coords, axis=1)  # radius [0, inf]
    spherical_coords[:, 1] = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])  # phi [-pi, pi] horizontal angle
    spherical_coords[:, 2] = np.arccos(np.divide(cart_coords[:, 2], spherical_coords[:, 0],
                                                 out=np.zeros(cart_coords.shape[0]),
                                                 where=spherical_coords[:, 0] != 0))  # theta [0, pi] vertical angle
    return spherical_coords


def get_colors(ref_vec, color_map="viridis"):
    col_map = cm.get_cmap(color_map, 100)
    max_val = np.max(ref_vec)
    min_val = np.min(ref_vec)
    norm_values = (ref_vec - np.ones(ref_vec.shape)*min_val)/(max_val-min_val)
    colors = np.array([col_map(val)[0:3] for val in norm_values])
    return colors


def get_trig_angles(angle):
    return np.cos(angle), np.sin(angle)


def is_inbetween_angles(start_angle, end_angle, angle):
    """Return true if ange is in-between start_angle and end_angle
    Args:
        start_angle (float):
        end_angle (float):
        angle (float):
    """
    start_angle = start_angle % (2*np.pi)
    end_angle = end_angle % (2*np.pi)
    angle = angle % (2 * np.pi)
    if start_angle < end_angle:
        if start_angle <= angle < end_angle:
            return True
    elif start_angle > end_angle:
        if angle >= start_angle or angle < end_angle:
            return True
    return False


def print_range(values: np.array, name: str):
    min_val = np.min(values)
    max_val = np.max(values)
    print("%s: min %.2f, max %.2f\n" % (name, min_val, max_val))
    return min_val, max_val


def convert_frames_to_video(frame_path: str, video_path: str, fps: float):
    frame_array = []
    img_files = sorted([f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))])
    for img_file in img_files:
        filename = os.path.join(frame_path, img_file)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        frame_array.append(img)
        size = (width, height)
    if len(img_files) > 0:
        print("Creating video...")
        print(f"Video saved at {video_path}")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
    else:
        print("No images found")
