import h5py
import logging
import numpy as np
import os.path as osp
import concurrent.futures
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

from utils.pose_util import process_poses, calibrate_process_poses
from utils.augmentor import Augmentor, AugmentParams

_logger = logging.getLogger(__name__)
BASE_DIR = osp.dirname(osp.abspath(__file__))


def compute_distances_chunk(xyz, centers):
    """
    Calculate the distance between xyz and centers, and return the index of the minimum distance
    """
    dist_mat = np.linalg.norm(xyz[:, np.newaxis, :2] - centers[np.newaxis, :, :2], axis=-1)
    lbl_chunk = np.argmin(dist_mat, axis=1)

    return lbl_chunk


def parallel_kmeans(xyz, centers, n_jobs=8):
    """
    Use concurrent.futures to compute the distance matrix in parallel and return the index of the minimum distance.
    """
    chunk_size = len(xyz) // n_jobs
    xyz_chunks = [xyz[i:i + chunk_size] for i in range(0, len(xyz), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(compute_distances_chunk, xyz_chunks, [centers] * len(xyz_chunks)))

    lbl = np.concatenate(results, axis=0)

    return lbl


class LiDARLocDataset(Dataset):
    """LiDAR localization dataset.

    Access to point clouds, calibration and ground truth data given a dataset directory
    """

    def __init__(self,
                 root_dir,
                 train=True,
                 sample_cls=False,
                 augment=False,
                 aug_rotation=10,
                 aug_translation=1,
                 generate_clusters=False,
                 level_clusters=25,
                 voxel_size=0.25):

        # Only support Oxford and NCLT
        self.root_dir = root_dir

        self.train = train
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.voxel_size = voxel_size

        # which dataset?
        self.scene = osp.split(root_dir)[-1]
        if self.scene == 'Oxford':
            if self.train:
                seqs = ['2019-01-11-14-02-26-radar-oxford-10k', '2019-01-14-12-05-52-radar-oxford-10k',
                        '2019-01-14-14-48-55-radar-oxford-10k', '2019-01-18-15-20-12-radar-oxford-10k']
            else:
                # change test seqs
                seqs = ['2019-01-15-13-06-37-radar-oxford-10k']
                # seqs = ['2019-01-17-13-26-39-radar-oxford-10k']
                # seqs = ['2019-01-17-14-03-00-radar-oxford-10k']
                # seqs = ['2019-01-18-14-14-42-radar-oxford-10k']

            self.scene = 'QEOxford'  # If you use the QEOxford dataset, please uncomment this note

        elif self.scene == 'NCLT':
            if self.train:
                seqs = ['2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11']
            else:
                # seqs = ['2012-02-12']
                # seqs = ['2012-02-19']
                seqs = ['2012-03-31']
                # seqs = ['2012-05-26']
        else:
            raise RuntimeError('Only support Oxford and NCLT!')

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        for seq in seqs:
            seq_dir = osp.join(self.root_dir, seq)
            if self.scene == 'QEOxford':
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'calibrateFalse.h5')
            else:
                h5_path = osp.join(self.root_dir, seq, 'velodyne_left_' + 'False.h5')

            # load h5 file, save pose interpolating time
            print("load " + seq + ' pose from ' + h5_path)
            h5_file = h5py.File(h5_path, 'r')
            ts[seq] = h5_file['valid_timestamps'][5:-5]
            ps[seq] = h5_file['poses'][5:-5]
            self.pcs.extend(
                [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))

        kmeans_pose_stats_filename = osp.join(self.root_dir, self.scene + '_cls_pose_stats.txt')
        mean_pose_stats_filename = osp.join(self.root_dir, self.scene + '_pose_stats.txt')

        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))

        if self.train:
            if self.scene == 'QEOxford':
                # Calculate the mean value of translations in training seqs
                self.mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
                # Save
                np.savetxt(mean_pose_stats_filename, self.mean_t, fmt='%8.7f')
            else:
                # Calculate the mean value of translations in training seqs
                self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
                # Save
                np.savetxt(mean_pose_stats_filename, self.mean_t, fmt='%8.7f')
        else:
            self.mean_t = np.loadtxt(mean_pose_stats_filename)

        for seq in seqs:
            if self.scene == 'QEOxford':
                pss, rotation = calibrate_process_poses(poses_in=ps[seq], mean_t=self.mean_t, align_R=vo_stats[seq]['R'],
                                                        align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            else:
                pss, rotation = process_poses(poses_in=ps[seq], mean_t=self.mean_t, align_R=vo_stats[seq]['R'],
                                              align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        xyz = self.poses[:, :3]

        if self.train:
            if sample_cls and generate_clusters:
                print("generate clusters")
                centers = []
                level = KMeans(n_clusters=level_clusters, random_state=0).fit(xyz)
                for l_cid in range(level_clusters):
                    centers.append(level.cluster_centers_[l_cid])
                centers = np.array(centers)
                np.savetxt(kmeans_pose_stats_filename, centers, fmt='%8.7f')
                self.lbl = parallel_kmeans(xyz[:, :2], centers[:, :2]).reshape(-1, 1)
            else:
                centers = np.loadtxt(kmeans_pose_stats_filename)
                self.lbl = parallel_kmeans(xyz[:, :2], centers[:, :2]).reshape(-1, 1)
        else:
            if sample_cls:
                centers = np.loadtxt(kmeans_pose_stats_filename)
                self.lbl = parallel_kmeans(xyz[:, :2], centers[:, :2]).reshape(-1, 1)
            else:
                centers = np.zeros((level_clusters, 3))
                self.lbl = np.zeros((len(xyz), 1))

        self.centers = centers

        # data augment
        augment_params = AugmentParams()
        if self.augment:
            augment_params.setTranslationParams(
                p_transx=0.5, trans_xmin=-1 * self.aug_translation, trans_xmax=self.aug_translation,
                p_transy=0.5, trans_ymin=-1 * self.aug_translation, trans_ymax=self.aug_translation,
                p_transz=0, trans_zmin=-1 * self.aug_translation, trans_zmax=self.aug_translation)
            augment_params.setRotationParams(
                p_rot_roll=0, rot_rollmin=-1 * self.aug_rotation, rot_rollmax=self.aug_rotation,
                p_rot_pitch=0, rot_pitchmin=-1 * self.aug_rotation, rot_pitchmax=self.aug_rotation,
                p_rot_yaw=0.5, rot_yawmin=-1 * self.aug_rotation, rot_yawmax=self.aug_rotation)
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

        if self.train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # For Classification training
        if type(idx) == list:
            scenes = [0]
            num = 0
            for i in idx:
                scan_path = self.pcs[i]
                ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                # just for z-axis up
                ptcld[:, 2] = -1 * ptcld[:, 2]

                scan = ptcld

                scan = np.ascontiguousarray(scan)

                lbl = self.lbl[i]
                pose = self.poses[i]
                rot = self.rots[i]
                scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)

                if self.train & self.augment:
                    scan = self.augmentor.doAugmentation(scan)  # n, 5

                scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)

                coord, feat = ME.utils.sparse_quantize(
                    coordinates=scan,
                    features=scan_gt_s8,
                    quantization_size=self.voxel_size)

                if num == 0:
                    coords = coord
                    feats = feat
                    lbls = lbl.reshape(-1, 1)
                    poses = pose.reshape(1, 6)
                    scenes.append(len(coord))
                else:
                    coords = np.concatenate((coords, coord))
                    feats = np.concatenate((feats, feat))
                    lbls = np.concatenate((lbls, lbl.reshape(-1, 1)))
                    poses = np.concatenate((poses, pose.reshape(1, 6)))
                    scenes.append(scenes[num] + len(coord))
                num += 1

            return (coords, feats, lbls, poses, scenes)

        else:
            # For Regression training
            scan_path = self.pcs[idx]
            ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            # just for z-axis up
            ptcld[:, 2] = -1 * ptcld[:, 2]
            scan = ptcld

            scan = np.ascontiguousarray(scan)

            lbl = self.lbl[idx]
            pose = self.poses[idx]  # (6,)
            rot = self.rots[idx]  # [3, 3]

            scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)

            if self.train & self.augment:
                scan = self.augmentor.doAugmentation(scan)  # n, 5

            scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)

            coords, feats = ME.utils.sparse_quantize(
                coordinates=scan,
                features=scan_gt_s8,
                quantization_size=self.voxel_size)

        return (coords, feats, lbl, idx, pose)