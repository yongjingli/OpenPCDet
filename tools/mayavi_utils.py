from mayavi import mlab
import numpy as np
import os
import torch
import json

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def visualize_lidar_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)
    return fig


def read_ladir_data(data_path):
    points = None
    ext = os.path.splitext(data_path)[-1]
    if ext == '.bin':
        points = np.fromfile(data_path, dtype=np.float32)  # data: [x0, y0, z0, f0, x1, y1, z1, f1 ....]
        points = points.reshape(-1, 4)  # points: (N, 3 + C_in), C_in mean reflect intensive
    elif ext == '.npy':
        points = np.load(data_path)
    else:
        raise NotImplementedError
    return points


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)
    return fig


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def visualize_pred_boxes(pred_boxes, pred_scores, pred_labels, fig=None):
    # show pred box in different cls color
    ref_corners3d = boxes_to_corners_3d(pred_boxes)
    for k in range(pred_labels.min(), pred_labels.max() + 1):
        cur_color = tuple(box_colormap[k % len(box_colormap)])
        mask = (pred_labels == k)
        fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=pred_scores[mask], max_num=100)
    # fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=pred_scores, max_num=100)
    return fig


if __name__ == "__main__":
    input_dir = "./lidar_data/kitti_train/"
    data_names = list(filter(lambda x: os.path.splitext(x)[-1] in ['.bin', 'npy'], os.listdir(input_dir)))
    for data_name in data_names:
        print("Process:%s" % data_name)
        data_path = os.path.join(input_dir, data_name)
        lidar_data = read_ladir_data(data_path)

        json_path = os.path.splitext(data_path)[0] + ".json"
        json_load = None
        with open(json_path, 'r') as fp:
            json_load = json.load(fp)

        pred_boxes = np.array(json_load['pred_boxes'], dtype=np.float)
        pred_scores = np.array(json_load['pred_scores'], dtype=np.float)
        pred_labels = np.array(json_load['pred_labels'], dtype=np.int)

        # # pred info
        # pred_boxes = np.array([[8.3662, 13.2544, -0.5237, 3.8060, 1.6872, 1.7878, 4.5859],
        #                          [8.7471, -1.8061, -0.6814, 1.0526, 0.6137, 1.8884, 4.5430],
        #                          [-0.2716, 1.4121, -0.9435, 3.6121, 1.5433, 1.5166, 2.5450],
        #                          [4.7861, 5.9439, -0.7241, 0.8641, 0.5476, 1.7735, 4.4016],
        #                          [2.0455, 2.9136, -0.8119, 0.4879, 0.5821, 1.7135, 6.5862],
        #                          [1.2427, 4.2672, -0.7795, 0.5500, 0.5483, 1.8802, 4.0807],
        #                          [15.4611, -1.7727, -0.3595, 4.3430, 1.7645, 1.9349, 1.4211],
        #                          [0.3924, -1.1081, -0.9817, 4.0029, 1.7315, 1.5281, 4.2831],
        #                          [2.0989, 3.8719, -0.8187, 0.6374, 0.5020, 1.8914, 4.7366],
        #                          [6.1370, -12.0797, -0.4451, 3.8847, 1.5623, 1.4993, 2.1362]], dtype=np.float)
        #
        # pred_scores = np.array([0.8410, 0.7978, 0.7186, 0.6734, 0.6270, 0.4707, 0.4114, 0.2117, 0.1671, 0.1484], dtype=np.float)
        # pred_labels = np.array([1, 2, 1, 2, 2, 2, 1, 1, 2, 1], dtype=np.int)

        # show lidar points
        fig = visualize_lidar_pts(lidar_data, show_intensity=True)

        # show preo boxes
        fig = visualize_pred_boxes(pred_boxes, pred_scores, pred_labels, fig)

        # draw grid
        fig = draw_multi_grid_range(fig, grid_size=10, bv_range=(0, -40, 80, 40))

        # mlab.savefig("mayavi_show.png")
        mlab.show(stop=False)

