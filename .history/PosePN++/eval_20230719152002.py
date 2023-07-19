# pylint: disable=no-member
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from data.OxfordVelodyne_datagenerator import RobotCar
# from data.OxfordVelodyne_refine_datagenerator_lw import RobotCar_refine
from data.NCLT_datagenerator import NCLT
from data.augment import get_augmentations_from_list, Normalize
from models.model_pnplus import PointNetPlusPlusPose
from models.loss import CriterionPose, CriterionlrPose
from utils.pose_util import val_translation, val_rotation, qexp
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpus', action='store_true', default=False,
                    help='if use multi_gpus, default false')
parser.add_argument('--gpu_id', type=int, default=1,
                    help='gpu id for network, only effective when multi_gpus is false')
parser.add_argument('--val_batch_size', type=int, default=100,
                    help='Batch Size during validating [default: pn:100|pn++:200]')
parser.add_argument('--model', type=str, default='PointNetPlusPlusPose',
                    help='PointNetPose or PointNetPlusPlusPose')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='PosePN++-NCLT-step-500/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/data',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='NCLT',
                    help='Oxford or NCLT or 7Scenes or vReLoc')
parser.add_argument('--scene', default='chess',
                    help='scene select: chess, fire, heads, office, pumpkin, redkitchen, stairs')
parser.add_argument('--trajectory', default='loopnew',
                    help='name of figure [default: loop6/7/8/9]')
parser.add_argument('--num_loc', type=int, default=3,
                    help='position classification, default: 10')
parser.add_argument('--num_ang', type=int, default=10,
                    help='orientation classification, default: 8')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for dataloader, default: 4')
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to, default: 4096')
parser.add_argument('--augmentation', type=str, nargs='+', default=[],
                    choices=['Jitter', 'RotateSmall', 'Scale', 'Shift', 'Rotate1D', 'Rotate3D'],
                    help='Data augmentation settings to use during training')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='use normalize or not, default False')
parser.add_argument('--upright_axis', type=int, default=2,
                    help='Will learn invariance along this axis')
parser.add_argument('--skip_val', action='store_true', default=False,
                    help='if skip validation during training, default False')
parser.add_argument('--resume_model', type=str, default='checkpoint_epoch',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--num_grid', type=int, default=12,
                    help='area of the dataset, default: 0-12')

FLAGS = parser.parse_args()
args = vars(FLAGS)
for (k, v) in args.items():
    print('%s: %s' % (str(k), str(v)))
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
TOTAL_ITERATIONS = 0

if not FLAGS.multi_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")

valid_augmentations = []

if FLAGS.normalize:
    if FLAGS.dataset == '7Scenes_pose':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, FLAGS.scene, 'stats.txt')
    elif FLAGS.dataset == 'Oxford':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    elif FLAGS.dataset == 'vReLoc':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    else:
        raise ValueError("dataset error!")

    stats = np.loadtxt(stats_file, dtype=np.float32)
    normalize_aug = Normalize(mean=stats[0], std=np.sqrt(stats[1]))
    valid_augmentations.append(normalize_aug)

valid_kwargs = dict(data_path=FLAGS.dataset_folder,
                    augmentation=valid_augmentations,
                    num_points=FLAGS.num_points,
                    train=False,
                    valid=True)
                    # num_grid=FLAGS.num_grid,
                    # num_loc=FLAGS.num_loc,
                    # num_ang=FLAGS.num_ang)
pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, 'pose_stats.txt')

if FLAGS.dataset == '7Scenes_pose':
    valid_kwargs = dict(scene=FLAGS.scene, **valid_kwargs)
    pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, FLAGS.scene, 'pose_stats.txt')
else:
    pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, 'pose_stats.txt')

pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

if FLAGS.dataset == 'Oxford':
    val_set = RobotCar_refine(**valid_kwargs)
elif FLAGS.dataset == 'NCLT':
    val_set = NCLT(**valid_kwargs)
else:
    raise ValueError("dataset error!")

val_loader = DataLoader(val_set,
                        batch_size=FLAGS.val_batch_size,
                        shuffle=False,
                        num_workers=FLAGS.num_workers,
                        pin_memory=True)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train():
    global TOTAL_ITERATIONS
    setup_seed(FLAGS.seed)
    val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))

    if FLAGS.model == 'PointNetPose':
        model = PointNetPose()
    elif FLAGS.model == 'PointNetPlusPlusPose':
        model = PointNetPlusPlusPose()
    else:
        raise NotImplementedError("%s isn't implemented!" % FLAGS.model)

    model = model.to(device)

    for num in range(51, 52):

        resume_filename = FLAGS.log_dir + FLAGS.resume_model + str(num) + '.tar'
        # resume_filename = FLAGS.log_dir + 'PosePN++.tar'
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(saved_state_dict)

        # if FLAGS.multi_gpus:
        #     model = nn.DataParallel(model)

        LOG_FOUT.write("\n")
        LOG_FOUT.flush()

        log_string('**** EPOCH %03d ****' % starting_epoch)
        sys.stdout.flush()
        valid_one_epoch(model, val_loader, val_writer, device)


def valid_one_epoch(model, val_loader, val_writer, device):
    gt_translation = np.zeros((len(val_set), 3))
    pred_translation = np.zeros((len(val_set), 3))
    gt_rotation = np.zeros((len(val_set), 4))
    pred_rotation = np.zeros((len(val_set), 4))
    error_t = np.zeros(len(val_set))
    error_q = np.zeros(len(val_set))
    time_results = []

    # for step, (val_data, val_pose, val_loc, val_ang) in enumerate(val_loader):
    for step, (val_data, val_pose) in enumerate(val_loader):
        start_idx = step * FLAGS.val_batch_size
        end_idx = min((step + 1) * FLAGS.val_batch_size, len(val_set))
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() * pose_s + pose_m
        gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])
        pcs_tensor = val_data.to(device)  # [B, N, 3]
        gt_t = val_pose[..., :3].to(device, dtype=torch.float32)  # [B, 3]
        gt_q = val_pose[..., 3:].to(device, dtype=torch.float32)  # [B, 3]

        # inference model and time cost
        start = time.time()
        pred_t, pred_q = run_model(model, pcs_tensor, validate=True)
        end = time.time()
        cost_time = (end - start) / FLAGS.val_batch_size
        time_results.append(cost_time)

        # eval of pose regress
        pred_translation[start_idx:end_idx, :] = pred_t.cpu().numpy() * pose_s + pose_m
        pred_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in pred_q.cpu().numpy()])
        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :],
                                                     gt_translation[start_idx:end_idx, :])])
        error_q[start_idx:end_idx] = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                                                    gt_rotation[start_idx:end_idx, :])])

        log_string('MeanTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

    mean_ATE = np.mean(error_t)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)
    mean_time = np.mean(time_results)
    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Cost Time(s): %f' % mean_time)
    val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianATE', median_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianARE', median_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanTime', mean_time, TOTAL_ITERATIONS)

    # trajectory
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=2, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=2, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format(FLAGS.trajectory))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')
    img = cv2.imread(image_filename)
    val_writer.add_image('Trajectory', img, TOTAL_ITERATIONS, dataformats='HWC')

    # save error
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    pred_t_filename = osp.join(FLAGS.log_dir, 'pred_t.txt')
    pred_q_filename = osp.join(FLAGS.log_dir, 'pred_q.txt')
    gt_t_filename = osp.join(FLAGS.log_dir, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')

    # # translation
    # fig = plt.figure()
    # threshold_t  = np.arange(0, 101, 5)
    # cumulative_t = []
    # for i in threshold_t:
    #     t = len(error_t[error_t < i])
    #     p = (t / len(error_t)) * 100
    #     cumulative_t.append(p)
    # plt.plot(threshold_t, cumulative_t, 'b--o', label='error_t')
    # plt.legend()
    # plt.xlabel('Translation Error [m]')
    # plt.ylabel('Percentage of Point Cloud [%]')
    # plt.title('Cumulative distributions of the translation errors (m)')
    # plt.grid(True)
    # image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('translation'))
    # fig.savefig(image_filename, dpi=200, bbox_inches='tight')
    # img = cv2.imread(image_filename)
    # val_writer.add_image('Translation', img, TOTAL_ITERATIONS, dataformats='HWC')

    # # rotation
    # fig = plt.figure()
    # threshold_q  = np.arange(0, 21, 2.5)
    # cumulative_q = []
    # for i in threshold_q:
    #     q = len(error_q[error_q < i])
    #     p = (q / len(error_q)) * 100
    #     cumulative_q.append(p)
    # plt.plot(threshold_q, cumulative_q, 'b--o', label='error_q')
    # plt.legend()
    # plt.xlabel('Rotation Error [degree]')
    # plt.ylabel('Percentage of Point Cloud [%]')
    # plt.title('Cumulative distributions of the rotation errors (degree)')
    # plt.grid(True)
    # image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('rotation'))
    # fig.savefig(image_filename)
    # img = cv2.imread(image_filename)
    # val_writer.add_image('Rotation', img, TOTAL_ITERATIONS, dataformats='HWC')


def run_model(model, PC, validate=False):
    if not validate:
        model.train()
        return model(PC)
    else:
        with torch.no_grad():
            model.eval()
            return model(PC)


if __name__ == "__main__":
    train()
