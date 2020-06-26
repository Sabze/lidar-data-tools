import scipy.stats
from matplotlib import pyplot as plt, colors as clr
import numpy as np
import open3d as o3d
import os


from . import utilities as utilities
from .laserscan import LaserScan
from . import laserscan_handler as lh

NUMBER_COLOR_MAP = {0: [0, 0, 0], 1: [0, 0, 255], 2: [255, 150, 255], 3: [245, 230, 100],
                    4: [250, 80, 100], 5: [50, 120, 255],
                    6: [0, 60, 135], 7: [30, 30, 255], 20: [255, 0, 0]}


def get_bg_color(laserscan: LaserScan, color=(0, 0, 0)):
    return np.zeros((laserscan.size(), 3)) + np.array(color)


def convert_laserscan_2_open3d(pointcloud: LaserScan, colors=None):
    """Return an open3d point cloud given a semantickitti laserscan."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points)
    if np.any(colors):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def convert_open3d_2_laserscan(pcd: 'open3d.open3d.geometry.PointCloud'):
    """Return a semantickitti laserscan given an open3d point cloud."""
    laserscan = LaserScan()
    laserscan.set_points(np.asarray(pcd.points))
    return laserscan


def get_label_colors(laserscan: LaserScan, cm: dict):
    """ Return a array of RGB colors matching the labels for every point in the point cloud."""
    if not any(laserscan.labels):
        raise RuntimeError("Have to set labels before colors")
    try:
        # Colors are saved as bga, so we have to flip them
        colors = [cm[label][::-1] for label in laserscan.labels]
    except KeyError:
        raise KeyError(f"Mapping is insufficient \n Check that all labels are in the color "
                       f"mapping \n Unique Labels: {set(laserscan.labels)}")
    colors = np.array(colors)
    colors = colors.reshape((-1, 3))
    return colors / 255


def draw(laserscan: LaserScan, colors: np.array = None):
    """Draw the points om the laserscan, colors either None or a np.array with RGB values (0-1)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(laserscan.points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def get_pos_color(laserscan: LaserScan, pos: str, color_map="viridis"):
    """ Return the colors based on the position values, pos can for example be x, y, range and intensity """
    color_values = lh.get_scan_values(laserscan, pos.lower())
    return utilities.get_colors(color_values, color_map)


def color_points_by_laser_num(laserscan: LaserScan, laser_nums: list, color=(0, 0, 0), bg_color=None):
    laser_info = laserscan.laser_info
    if bg_color is None:
        bg_color = get_pos_color(laserscan, "range")
    for laser in laser_nums:
        point_num = laser_info[laser]["num_of_points"]
        row = laser_info[laser]["start_index"]
        curr_laser_points = np.array([list(color)] * point_num)  # Color of the laserpoints
        end = row + point_num
        bg_color = np.concatenate((bg_color[:row, :], curr_laser_points, bg_color[end:, :]), axis=0)
    draw(laserscan, bg_color)


def color_points(laserscan: LaserScan, point_inds: list, color: list, bg_colors=None):
    if bg_colors is None:
        bg_colors = get_bg_color(laserscan)
    for point_ind in point_inds:
        bg_colors[point_ind, :] = color
    draw(laserscan, bg_colors)


def plot_azimuthal_angle(laserscan: LaserScan, num_points_to_plot=10000, overlap=1000):

    def get_laser_edge_points(laser_start_point_inds: list, point_inds: list):
        rel_ind = [ind for (ind, val) in enumerate(laser_start_point_inds) if val in point_inds]
        if len(rel_ind) < 1:
            return None, None
        start_s = rel_ind[0]
        end_s = rel_ind[-1] + 1
        if end_s > len(laser_start_point_inds):
            end_s = -1
        return start_s, end_s

    spherical_coords = utilities.get_spherical_coords(laserscan.points)
    phi_values = spherical_coords[:, 1]
    phi_inds = list(range(0, laserscan.size()))
    laser_start_point_inds = [v["start_index"] for (k, v) in laserscan.laser_info.items()]
    laser_start_points = np.take(phi_values, laser_start_point_inds, axis=0)
    start = 0
    while start < laserscan.size():
        end = start + num_points_to_plot
        if end > laserscan.size():
            end = -1
        laser_start, laser_end = get_laser_edge_points(laser_start_point_inds, phi_inds[start:end])
        if laser_start is None:
            print("Found no starting point for lasers")
            break
        plt.figure(1)
        plt.title(f"Indices between {laser_start} - {laser_end}")
        plt.plot(phi_inds[start:end], phi_values[start:end], "r+", markersize=1, zorder=0)
        plt.scatter(laser_start_point_inds[laser_start:laser_end], laser_start_points[laser_start:laser_end], s=4,
                    c="black", alpha=1, zorder=1)
        print(f"\nStart point coords:\n"
              f"{np.array([laser_start_point_inds[laser_start:laser_end], laser_start_points[laser_start:laser_end]]).T}")
        start = end - overlap
        plt.show()
        next_plot = "unknown"
        while next_plot != "y" and next_plot != "n":
            next_plot = input("\ncontinue (y/n)?")
        if next_plot == "n":
            break


def color_angle_interval(laserscan: LaserScan, start_angle: float, end_angle: float, azimuthal_angel=True,
                         angle_color=(1, 0, 0), bg_color=None):
    spherical_coord = utilities.get_spherical_coords(laserscan.points)
    if azimuthal_angel:
        angles = spherical_coord[:, 1]
    else:
        angles = spherical_coord[:, 2]
    inds = []
    for ind, angle in enumerate(angles):
        if utilities.is_inbetween_angles(start_angle, end_angle, angle):
            inds.append(ind)
    color_points(laserscan, inds, list(angle_color), bg_color)
    return


def color_pos_interval(laserscan: LaserScan, pos: str, start_value: float, end_value: float, color=(1, 0, 0),
                       bg_color=None):
    """ Color the points that has pos-values in the interval [start_value, end_value]. pos can be e.g. x, y, range.
     This function should not be used to color angle intervals, instead use color_angle_interval. """
    points = lh.get_scan_values(laserscan, pos)
    inds = []
    for ind, point in enumerate(points):
        if start_value <= point < end_value:
            inds.append(ind)
    color_points(laserscan, inds, list(color), bg_color)
    return


def get_cmap_labels(cm: dict):
    """Return a color map and norm based on the label colors, used for the spherical projection."""
    boundaries = []
    label_colors = []
    for (k, v) in cm.items():
        boundaries.append(k)
        label_colors.append(tuple(np.array(v) / 255)[::-1])  # The colors are saved as bgr
    real_bound = [bound + 0.5 for bound in boundaries]
    real_bound = [0] + real_bound[:-1]
    cmap = clr.ListedColormap(label_colors)
    norm = clr.BoundaryNorm(real_bound, cmap.N, clip=True)
    return cmap, norm


def plot_projection_positions(laserscan: LaserScan, aspect=12):
    """Plots a range image colored by the number of points projected to the same position. Best to do after
    plotting the actual spherical projection."""
    cmap, norm = get_cmap_labels(NUMBER_COLOR_MAP)
    elems = laserscan.count_elems_at_proj_pos()
    plt.figure()
    plt.imshow(elems, cmap=cmap, norm=norm, aspect=aspect)
    plt.colorbar()
    plt.show()


def plot_spherical_proj(laserscan: LaserScan, cmap, norm, proj_shape=None, half_turn=True,
                        aspect='auto', save=False, filedir="imgs", name="", h_flip=False, v_flip=False,
                        title=True, color_bar=True, img_format="png"):
    if proj_shape is not None:
        laserscan.set_projection_var(proj_shape[0], proj_shape[1])
    laserscan.do_range_projection(h_flip=h_flip, v_flip=v_flip, half_turn=half_turn)
    plt.imshow(laserscan.proj_label, cmap=cmap, norm=norm, aspect=aspect)
    if color_bar:
        plt.colorbar()
    if title:
        plt.title(f"{laserscan.name}, {laserscan.proj_H}x{laserscan.proj_W}, labels")
    if save:
        if name == "":
            raise ValueError("No filename specified")
        filename = os.path.join(filedir, f"{name}.{img_format}")
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        elif os.path.exists(filename):
            print(f"Overwriting existing file {filename}")
        plt.savefig(filename, format=img_format, bbox_inches='tight')
    plt.show()


def plot_adjusted_spherical_proj(laserscan: LaserScan, min_allowed: float, max_allowed: float, cmap, norm,
                                 proj_shape=None, half_turn=True, aspect='auto', save=False,
                                 filedir="imgs", name="", h_flip=False, v_flip=False,
                                 title=True, color_bar=True, img_format="png"):
    thetas = lh.get_scan_values(laserscan, "theta")
    min_theta = np.min(thetas)
    max_theta = np.max(thetas)
    print(f"max theta: {max_theta:.3f}, min theta: {min_theta:.3f}")
    angle = np.max([np.min([abs(max_theta), max_allowed]), min_allowed])
    laserscan.proj_fov_down = -angle * 180 / np.pi
    laserscan.proj_fov_up = angle * 180 / np.pi
    plot_spherical_proj(laserscan, cmap, norm, proj_shape, half_turn, aspect, save, filedir, name, h_flip, v_flip,
                        title, color_bar, img_format)
    return min_theta, max_theta


def draw_laserscans(laserscans: list):
    """Draw the laserscans one by one. """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    color = np.array([1, 0, 0])
    pcd_list = []
    for laserscan in laserscans:
        color = np.roll(color, 1)
        colors = np.zeros((laserscan.size(), 3)) + color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(laserscan.points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd)
        o3d.visualization.draw_geometries(pcd_list)
        next = "unknown"
        while next not in ['n', 'c', 's']:
            next = input("Next (n), Clear (c) or Stop (s)?")
        if next == 'c':
            pcd_list = []
        elif next == 's':
            break


def plot_histogram_of_pos(laserscans: list, pos: str, bins=20, show_norm=False, name="", save=False, filepath=""):
    """Plot the histogram of one of (x,y,z, range, remission, phi, theta) specified by 'pos', for the
    laserscans in laserscans."""
    pos = pos.lower()
    data_array = lh.concatenate_scan_values(laserscans, pos)
    print(f"{len(laserscans)} scans, {len(data_array)} points")
    plot_histogram(data_array, len(laserscans), name=f"{name}:{pos}", bins=bins, show_norm=show_norm,
                   save=save, filepath=filepath)
    return data_array


def plot_histogram(data_array: np.array, num_scans: int, name="", bins=20, save=False, filepath="", show_norm=False):
    """ Plot a histogram of data_array. """
    data_mean = np.mean(data_array)
    data_std = np.std(data_array)
    print(f"Mean: {data_mean}, Std: {data_std} ")
    print(f"Min: {np.min(data_array)}, Max: {np.max(data_array)}")
    plt.figure()
    n, bins, patches = plt.hist(data_array, bins=bins)
    plt.title(f"{name}, {num_scans} scans, {len(data_array)} points")
    if save:
        filename = f"{name}:{num_scans}scans:{len(data_array):09}points.png"
        filepath = os.path.join(filepath, filename)
        plt.savefig(filepath, format="png")
        print(f"Saved image {filepath}")
    plt.show()
    if show_norm:
        # Plot the histogram of the normalized values.
        plt.figure()
        plt.title("Normal dist")
        vals = ((data_array-data_mean)/data_std)
        x = np.linspace(np.min(data_array), np.max(data_array), bins + 50)
        y = scipy.stats.norm.pdf(x, data_mean, data_std)
        plt.hist(vals, bins=bins, density=True)
        plt.plot(x, y)
        plt.show()
    return data_array, n, bins, patches


