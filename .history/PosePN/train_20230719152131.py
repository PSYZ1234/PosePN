# pylint: disable=no-member
import argparse
import os
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
# from data.OxfordVelodyne_refine_datagenerator import RobotCar_refine
from data.NCLT_datagenerator import NCLT
from data.augment import get_augmentations_from_list, Normalize
from models.model_pn import PointNetPose
from models.loss import CriterionPose
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
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id for network, only effective when multi_gpus is false')
parser.add_argument('--batch_size', type=int, default=160,
                    help='Batch Size during training [default: 200]')
parser.add_argument('--val_batch_size', type=int, default=160,
                    help='Batch Size during validating [default: 200]')
parser.add_argument('--max_epoch', type=int, default=999,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate', type=float, default=0.001, 
                    help='Initial learning rate [default: 0.001]')
parser.add_argument("--decay_step", type=float, default=500,
                    help="decay step for learning rate, default: 500")
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='PosePN-Oxford-step-500-20220925/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/lw/data',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='Oxford', 
                    help='Oxford or NCLT')
parser.add_argument('--trajectory', default='full6',
                    help='name of figure [default: full6]')
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
parser.add_argument('--real', action='store_true', default=False, 
                    help='if True, load poses from SLAM / integration of VO')
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')

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

train_augmentations = get_augmentations_from_list(FLAGS.augmentation, upright_axis=FLAGS.upright_axis)
valid_augmentations = []

if FLAGS.normalize:
    if FLAGS.dataset == 'Oxford':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    elif FLAGS.dataset == 'NCLT':
        stats_file = osp.join(FLAGS.dataset_folder, FLAGS.dataset, 'stats.txt')
    else:
        raise ValueError("dataset error!")

    stats = np.loadtxt(stats_file, dtype=np.float32)
    normalize_aug = Normalize(mean=stats[0], std=np.sqrt(stats[1]))
    train_augmentations.append(normalize_aug)
    valid_augmentations.append(normalize_aug)

train_kwargs = dict(data_path    = FLAGS.dataset_folder, 
                    augmentation = train_augmentations, 
                    num_points   = FLAGS.num_points, 
                    train        = True, 
                    valid        = False,
                    real         = FLAGS.real)
valid_kwargs = dict(data_path    = FLAGS.dataset_folder, 
                    augmentation = valid_augmentations, 
                    num_points   = FLAGS.num_points, 
                    train        = False, 
                    valid        = True,
                    real         = FLAGS.real)
pose_stats_file = os.path.join(FLAGS.dataset_folder, FLAGS.dataset, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

if FLAGS.dataset == 'Oxford':
    train_set = RobotCar_refine(**train_kwargs)
    val_set = RobotCar_refine(**valid_kwargs)
elif FLAGS.dataset == 'NCLT':
    train_set = NCLT(**train_kwargs)
    val_set = NCLT(**valid_kwargs)
else:
    raise ValueError("dataset error!")

train_loader = DataLoader(train_set, 
                        batch_size=FLAGS.batch_size, 
                        shuffle=True, 
                        num_workers=FLAGS.num_workers, 
                        pin_memory=True)
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
    train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
    model = PointNetPose() 
    loss = CriterionPose() 
    model = model.to(device)
    loss = loss.to(device)

    if FLAGS.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_learning_rate)
    elif FLAGS.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), FLAGS.init_learning_rate)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.decay_step, gamma=0.95)

    if len(FLAGS.resume_model) > 0:
        resume_filename = FLAGS.log_dir + FLAGS.resume_model
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch'] + 1
        TOTAL_ITERATIONS = starting_epoch * len(train_set)
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        starting_epoch = 0

    if FLAGS.multi_gpus:
        model = nn.DataParallel(model)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    for epoch in range(starting_epoch, FLAGS.max_epoch):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        if not FLAGS.skip_val and epoch % 5 == 1:
            valid_one_epoch(model, val_loader, val_writer, device)
            # torch.cuda.empty_cache()
        train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device)
        # torch.cuda.empty_cache()


def train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device):
    global TOTAL_ITERATIONS
    time_results = []

    for step, (train_data, train_pose) in enumerate(train_loader):
        TOTAL_ITERATIONS += 1
        pcs_tensor = train_data.to(device, dtype=torch.float32)  # [B, N, 3]  
        gt_t       = train_pose[..., :3].to(device, dtype=torch.float32)  # [B, 3] 
        gt_q       = train_pose[..., 3:].to(device, dtype=torch.float32)  # [B, 3] 

        # train model
        start          = time.time()   
        scheduler.optimizer.zero_grad()
        pred_t, pred_q = run_model(model, pcs_tensor, validate=False)
        train_loss     = loss(pred_t, pred_q, gt_t, gt_q)    
        train_loss.backward()
        scheduler.optimizer.step()
        scheduler.step()
        end            = time.time()
        cost_time      = (end - start) / FLAGS.batch_size
        time_results.append(cost_time)

        log_string('Loss: %f' % train_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)

    mean_time  = np.mean(time_results)
    log_string('Mean Cost Time(s): %f' % mean_time)

    if epoch % 1 == 0:
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            FLAGS.log_dir+'checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))


def valid_one_epoch(model, val_loader, val_writer, device):
    gt_translation   = np.zeros((len(val_set), 3))
    pred_translation = np.zeros((len(val_set), 3))
    gt_rotation      = np.zeros((len(val_set), 4))
    pred_rotation    = np.zeros((len(val_set), 4))
    error_t          = np.zeros(len(val_set))
    error_q          = np.zeros(len(val_set))
    time_results     = []

    for step, (val_data, val_pose) in enumerate(val_loader):
        start_idx  = step * FLAGS.val_batch_size
        end_idx    = min((step+1)*FLAGS.val_batch_size, len(val_set))       
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() * pose_s + pose_m
        gt_rotation[start_idx:end_idx, :]    = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()]) 
        pcs_tensor = val_data.to(device)  # [B, N, 3]

        # inference model and time cost
        start          = time.time()   
        pred_t, pred_q = run_model(model, pcs_tensor, validate=True)
        end            = time.time()
        cost_time      = (end - start) / FLAGS.val_batch_size
        time_results.append(cost_time)

        # eval of pose regress
        pred_translation[start_idx:end_idx, :] = pred_t.cpu().numpy() * pose_s + pose_m
        pred_rotation[start_idx:end_idx, :]    = np.asarray([qexp(q) for q in pred_q.cpu().numpy()])  
        error_t[start_idx:end_idx]             = np.asarray([val_translation(p, q) for p, q in zip(pred_translation[start_idx:end_idx, :], gt_translation[start_idx:end_idx, :])])
        error_q[start_idx:end_idx]             = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :], gt_rotation[start_idx:end_idx, :])])
        
        log_string('MeanTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

    mean_ATE   = np.mean(error_t)
    mean_ARE   = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)
    mean_time  = np.mean(time_results)
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
    fig       = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose   = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=3, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=3, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('1_trajectory'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # ground truth
    fig     = plt.figure()
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=3, c='black')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('2_gt'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_curve
    fig          = plt.figure()
    threshold_t  = np.arange(0, 42, 2)
    cumulative_t = []
    for i in threshold_t:
        t = len(error_t[error_t < i])
        p = (t / len(error_t)) * 100
        cumulative_t.append(p)
    plt.plot(threshold_t, cumulative_t, 'r--o', label='error_t')
    plt.legend()  
    plt.xlabel('Translation Error [m]')
    plt.ylabel('Percentage of Point Cloud [%]')
    plt.title('Cumulative distributions of the translation errors (m)')
    plt.grid(True)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('3_curve_t'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_curve
    fig          = plt.figure()
    threshold_q  = np.arange(0, 10.5, 0.5)
    cumulative_q = []
    for i in threshold_q:
        q = len(error_q[error_q < i])
        p = (q / len(error_q)) * 100
        cumulative_q.append(p)
    plt.plot(threshold_q, cumulative_q, 'b--o', label='error_q')
    plt.legend()  
    plt.xlabel('Rotation Error [degree]')
    plt.ylabel('Percentage of Point Cloud [%]')
    plt.title('Cumulative distributions of the rotation errors (degree)')
    plt.grid(True)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('4_curve_r'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_distribution
    fig   = plt.figure()
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Data Num')
    plt.ylabel('Error (m)')
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('5_distribution_t'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_distribution
    fig   = plt.figure()
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Data Num')
    plt.ylabel('Error (degree)')
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('6_distribution_q'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    pred_t_filename  = osp.join(FLAGS.log_dir, 'pred_t.txt')
    gt_t_filename    = osp.join(FLAGS.log_dir, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')


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
