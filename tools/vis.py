

import open3d as o3d
from pathlib import Path
import torch
import numpy as np

from matplotlib import pyplot as plt
from scipy import stats
import random
import os
from pointcept.datasets.preprocessing.scannet.meta_data.scannetpp_constants import TOP100_COLORS

target_data_dir = '/workspace/3Dcept/data/scannetpp/val100'
pred_dir = '/workspace/3Dcept/exp/scannetpp/ptv3_aidc/result'
out_dir = '/workspace/3Dcept/exp/scannetpp/ptv3_aidc/vis_debug'

_label_to_color_uint8 = TOP100_COLORS


_label_to_color = dict([
    (label, (np.array(color_uint8).astype(np.float64) / 255.0).tolist())
    for label, color_uint8 in _label_to_color_uint8.items()
])

# _name_to_color_uint8 = {
#     "ceiling": [158, 218, 228],  # counter
#     "floor": [151, 223, 137],  # floor
#     "wall": [174, 198, 232],  # wall
#     "beam": [255, 187, 120],  # bed
#     "column": [254, 127, 13],  # refrigerator
#     "window": [196, 176, 213],  # window
#     "door": [213, 39, 40],  # door
#     "chair": [188, 189, 35],  # chair
#     "table": [255, 152, 151],  # table
#     "sofa": [140, 86, 74],  # sofa
#     "bookcase": [196, 156, 147],  # bookshelf
#     "board": [148, 103, 188],  # picture
#     "clutter": [0, 0, 0],  # clutter
# }

# _name_to_color = dict([(name, np.array(color_uint8).astype(np.float64) / 255.0)
#                        for name, color_uint8 in _name_to_color_uint8.items()])


def load_real_data(pth_path):
    """
    Args:
        pth_path: Path to the .pth file.
    Returns:
        points: (N, 3), float64
        colors: (N, 3), float64, 0-1
        labels: (N, ), int64, {1, 2, ..., 36, 39, 255}.
    """
    # - points: (N, 3), float32           -> (N, 3), float64
    # - colors: (N, 3), float32, 0-255    -> (N, 3), float64, 0-1
    # - labels: (N, 1), float64, 0-19,255 -> (N,  ), int64, 0-19,255

    data = torch.load(pth_path)
    points, colors, labels = data['sampled_coords'], data['sampled_colors'], data['sampled_labels']
    points = points.astype(np.float64)
    colors = colors.astype(np.float64) / 255.0
    assert len(points) == len(colors) == len(labels)

    labels = labels.astype(np.int64).squeeze()
    return points, colors, labels


def load_pred_labels(label_path):
    """
    Args:
        label_path: Path to the .txt file.
    Returns:
        labels: (N, ), int64, {1, 2, ..., 36, 39}.
    """
    def read_labels(label_path):
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(int(line.strip()))
        return np.array(labels)

    return np.array(read_labels(label_path))


def render_to_image(pcd, save_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)


def visualize_scene_by_path(scene_path, label_path, out, save_as_image=False):
    # label_dir = Path("data/s3dis/test_epoch79")

    print(f"Visualizing {scene_path}")
    # label_path = label_dir / f"{scene_path.stem}_pred.npy"

    # Load pcd and real labels.
    points, colors, real_labels = load_real_data(scene_path)
    
    os.makedirs(os.path.join(out_dir, scene_path.stem), exist_ok=True)
    # Visualize rgb colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors*255)
    pcd.labels = real_labels
    o3d.io.write_point_cloud(f"{out_dir}/{scene_path.stem}/rgb.ply", pcd)
    # if save_as_image:
    #     render_to_image(pcd, f"{out_dir}/{scene_path.stem}/rgb.png")
    # else:
    #     o3d.visualization.draw_geometries([pcd], window_name="RGB colors")

    # Visualize real labels
    real_label_colors = np.array([_label_to_color[l] for l in real_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(real_label_colors)
    o3d.io.write_point_cloud(f"{out_dir}/{scene_path.stem}/real.ply", pcd)
    # if save_as_image:
    #     render_to_image(pcd, f"{out_dir}/{scene_path.stem}/real.png")
    # else:
    #     o3d.visualization.draw_geometries([pcd], window_name="Real labels")

    # Load predicted labels
    pred_labels = np.load(label_path)
    pred_label_colors = np.array([_label_to_color[l] for l in pred_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pred_label_colors)
    o3d.io.write_point_cloud(f"{out_dir}/{scene_path.stem}/pred.ply", pcd)
    # if save_as_image:
    #     render_to_image(pcd, f"{out_dir}/{scene_path.stem}/pred.png")
    # else:
    #     o3d.visualization.draw_geometries([pcd], window_name="Pred labels")
    ignore_points = np.logical_or(pred_labels == -1, real_labels == -1)
    true_points = (pred_labels == real_labels)
    
    correctness_colors = np.ones_like(real_label_colors) * np.array([[200,50,50]])
    correctness_colors[true_points, :] = np.array([50,200,50])
    correctness_colors[ignore_points, :] = np.array([128,128,128])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(correctness_colors / 255.0)
    o3d.io.write_point_cloud(f"{out_dir}/{scene_path.stem}/correctness.ply", pcd)
    
    


def visualize_scene_by_name(scene_name, save_as_image=False):
    data_root = Path("data") / "s3dis" / "Area_5"
    scene_paths = sorted(list(data_root.glob("*.pth")))

    found = False
    for scene_path in scene_paths:
        if scene_path.stem == scene_name:
            found = True
            visualize_scene_by_path(scene_path, save_as_image=save_as_image)
            break

    if not found:
        raise ValueError(f"Scene {scene_name} not found.")


if __name__ == "__main__":
    files = sorted(os.listdir(pred_dir))
    preds = []
    data = []
    for f in files:
        if os.path.isdir(os.path.join(pred_dir, f)):
            continue
        preds.append(Path(os.path.join(pred_dir, f)))
        data.append(Path(os.path.join(target_data_dir, f"{f.split('_')[0]}.pth")))
    assert len(preds) == len(data)
    # print(preds, data)
    for p,d in zip(preds, data):
        visualize_scene_by_path(d, p, out_dir, False)
        # break
    
    
    # Used in main text
    # hallway_10
    # lobby_1
    # office_27
    # office_30

    # Use in supplementary
    # visualize_scene_by_name("conferenceRoom_2")
    # visualize_scene_by_name("office_35")
    # visualize_scene_by_name("office_18")
    # visualize_scene_by_name("office_5")
    # visualize_scene_by_name("office_28")
    # visualize_scene_by_name("office_3")
    # visualize_scene_by_name("hallway_12")
    # visualize_scene_by_name("office_4")

    # Visualize all scenes
    # data_root = Path("data") / "scannetv2" / "val"
    # scene_paths = sorted(list(data_root.glob("*.pth")))
    # scene_names = [p.stem for p in scene_paths]
    # for scene_name in scene_names:
    #     visualize_scene_by_name(scene_name, save_as_image=True)