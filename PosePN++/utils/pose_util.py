import numpy as np
import math
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as txe
from os import path as osp


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))

    return q


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, pose_max, pose_min


def calibrate_process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, 9:]

    for i in range(len(poses_out)):
        R = poses_in[i, :9].reshape((3, 3))
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, pose_max, pose_min

    
def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted   = pred_p
        groundtruth = gt_p
    else:
        predicted   = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted   = pred_q
        groundtruth = gt_q
    else:
        predicted   = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))

    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1

    # error = 2 * np.arccos(d) * 180 / np.pi0

    d = abs(np.dot(groundtruth, predicted))
    d = min(1.0, max(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error

    
def ds_pc(cloud, target_num):
    if cloud.shape[0] <= target_num:
        # Add in artificial points if necessary
        print('Only %i out of %i required points in raw point cloud. Duplicating...' % (cloud.shape[0], target_num))
        num_to_pad = target_num - cloud.shape[0]
        index      = np.random.choice(cloud.shape[0], size=num_to_pad, replace=True)
        pad_points = cloud[index, :]
        cloud      = np.concatenate((cloud, pad_points), axis=0)

        return cloud
    else:
        index = np.random.choice(cloud.shape[0], size=target_num, replace=True)
        cloud = cloud[index, :]

        return cloud


def filter_overflow_ts(filename, ts_raw):
    file_data = pd.read_csv(filename)
    base_name = osp.basename(filename)

    if base_name.find('vo') > -1:
        ts_key = 'source_timestamp'
    else:
        ts_key = 'timestamp'

    pose_timestamps     = file_data[ts_key].values
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted   = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))
    
    return ts_filted


def position_classification(pose, pose_max, pose_min, num_loc):
    """
    convert location to multi-classes (10 x 10)
    :param pose: [6,]
    :param pose_max: [2,]
    :param pose_min: [2,]
    :param num_grid: 10
    :return: class k
    """
    x = (pose[0] - pose_min[0]) / (pose_max[0] - pose_min[0])
    y = (pose[1] - pose_min[1]) / (pose_max[1] - pose_min[1])
    x = np.maximum(x, 0)
    y = np.maximum(y, 0)
    x = int(np.minimum(x * num_loc, (num_loc - 1)))
    y = int(np.minimum(y * num_loc, (num_loc - 1)))
    class_position = x * num_loc + y
    # class_position = class_position / (num_loc * num_loc)

    return class_position


def orientation_classification(pose, num_ang):
    """
    convert angle to multi-classes (10 x 10)
    :param pose: [6,]
    :param num_ang: 10
    :return: class k
    """
    quat    = qexp(pose[3:])
    z, y, x = txe.quat2euler(quat)
    theta   = math.degrees(x)

    if theta<-180 or theta>180:
        raise ValueError("angle error!")

    class_orientation = (theta - math.degrees(-math.pi)) / (math.degrees(math.pi) - math.degrees(-math.pi))
    class_orientation = int(np.minimum(class_orientation * num_ang, (num_ang - 1)))
    # class_orientation = class_orientation / num_ang

    return class_orientation