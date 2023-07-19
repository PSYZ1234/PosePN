import os
import os.path as osp
import numpy as np
import torch
import pickle
import json
import MinkowskiEngine as ME
from utils.pose_util import process_poses
from torch.utils import data
from data.robotcar_sdk.python.velodyne import load_velodyne_binary


BASE_DIR = osp.dirname(osp.abspath(__file__))
class vReLoc(data.Dataset):
    def __init__(self, data_path, train, valid=False, augmentation=[], num_points=4096, real=False,
                 skip_pcs=False, vo_lib='orbslam', num_grid=0, voxel_size=0.05):
        self.skip_pcs = skip_pcs
        # directories
        data_dir = osp.join(data_path, 'vReLoc')

        # decide which sequences to use
        if train:
            split_file = osp.join(data_dir, 'TrainSplit.txt')
        elif valid:
            split_file = osp.join(data_dir, 'ValidSplit.txt')
        else:
            split_file = osp.join(data_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

        # read poses and collect image names
        pcs_all     = []
        self.pcs    = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats  = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                format(i)), delimiter=',').flatten()[:12] for i in frame_idx]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.bin'.format(i))
                      for i in frame_idx]
            pcs_all.extend(c_imgs)

        #pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        pose_stats_filename = 'pose_stats.txt'
        if train:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t  = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses     = np.empty((0, 6))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))  
        poses_all      = np.empty((0, 6))  
        #pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')
        pose_max_min_filename = 'pose_max_min.txt'
        for seq in seqs:
            pss, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                  align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                  align_s=vo_stats[seq]['s'])
            poses_all      = np.vstack((poses_all, pss)) 
            self.poses_max = np.vstack((self.poses_max, [pss_max]))
            self.poses_min = np.vstack((self.poses_min, [pss_min]))

        if train:
            self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            center_point   = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            center_point   = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 

        # divide the area into four regions
        for i in range(len(poses_all)):
            if num_grid == 0:  # left down   
                if poses_all[i, 0] <= center_point[0] and poses_all[i, 1] <= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])   
            elif num_grid == 1:  # right down -- 1
                if poses_all[i, 0] >= center_point[0] and poses_all[i, 1] <= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])   
            elif num_grid == 2:  # right up -- 2
                if poses_all[i, 0] >= center_point[0] and poses_all[i, 1] >= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])    
            elif num_grid == 3:  # left up -- 3
                if poses_all[i, 0] <= center_point[0] and poses_all[i, 1] >= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])  
            elif num_grid == 4:  # down -- 4=0+1
                if poses_all[i, 1] <= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i]) 
            elif num_grid == 5:  # up -- 5=2+3
                if poses_all[i, 1] >= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])        
            elif num_grid == 6:  # left -- 6=0+3
                if poses_all[i, 0] <= center_point[0]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])        
            elif num_grid == 7:  # right -- 7=1+2
                if poses_all[i, 0] >= center_point[0]:
                    self.poses = np.vstack((self.poses, poses_all[i]))
                    self.pcs.append(pcs_all[i])                
            elif num_grid == 8:  # down + right -- 8=0+1+2    
                if poses_all[i, 0] >= center_point[0] or poses_all[i, 1] <= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])  
            elif num_grid == 9:  # down + left -- 9=0+1+3    
                if poses_all[i, 0] <= center_point[0] or poses_all[i, 1] <= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])   
            elif num_grid == 10:  # up + right -- 10=1+2+3    
                if poses_all[i, 0] >= center_point[0] or poses_all[i, 1] >= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])   
            elif num_grid == 11:  # up + left -- 11=0+2+3    
                if poses_all[i, 0] <= center_point[0] or poses_all[i, 1] >= center_point[1]:
                    self.poses = np.vstack((self.poses, poses_all[i])) 
                    self.pcs.append(pcs_all[i])   
            elif num_grid == 12:  # all -- 12=0+1+2+3    
                self.poses = np.vstack((self.poses, poses_all[i])) 
                self.pcs.append(pcs_all[i]) 
            else:
                raise ValueError("dataset error!")

        self.augmentation = augmentation
        self.num_points   = num_points
        self.voxel_size = voxel_size
        
        if train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        ptcld = load_velodyne_binary(scan_path)  # (4, N)
        scan  = ptcld[:3].transpose()  # (N, 3)
        scan  = np.ascontiguousarray(scan)
        for a in self.augmentation:
            scan = a.apply(scan) 
            
        pose = self.poses[index]  # (6,)  
        coords, feats = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan,
            quantization_size=self.voxel_size)

        return (coords, feats, pose)

    def __len__(self):
        return self.poses.shape[0]