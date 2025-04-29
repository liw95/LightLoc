"""
This code is designed to implement the following:
1. After reaching a certain epoch, it clips the data based on the median loss.
1. After reaching a certain epoch, it restores the full dataset.

Modified by Infobatch
"""
import math
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset


class RSD(Dataset):
    """
    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.25, delta_start: float = 0.25, windows_radio: float = 0.1,
                 delta_stop: float = 0.85):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.prune_ratio = prune_ratio
        self.num_epochs = num_epochs
        self.delta_start = delta_start
        self.delta_stop = delta_stop
        self.windows = windows_radio    # 0.1
        self.keep_windows_ratio = ((delta_stop - delta_start) - 2 * windows_radio) / 2      # 0.2
        self.weights = torch.ones(len(self.dataset), 1)
        self.select_idx = torch.zeros(len(dataset), dtype=torch.bool)
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, loss, idx, batch_num):
        """
        Scaling loss
        """
        device = loss.device
        weights = self.weights.to(device)[idx][batch_num.long()]
        loss.mul_(weights)

        return loss.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scan_path = self.dataset.pcs[idx]
        if self.dataset.scene == 'NCLT':
            ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()[:, :3]
        ptcld[:, 2] = -1 * ptcld[:, 2]

        scan = ptcld

        scan = np.ascontiguousarray(scan)

        lbl = self.dataset.lbl[idx]
        pose = self.dataset.poses[idx]  # (6,)
        rot = self.dataset.rots[idx]  # [3, 3]
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)

        if self.dataset.train and self.dataset.augment:
            scan = self.dataset.augmentor.doAugmentation(scan)  # n, 5

        scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)

        coords, feats = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan_gt_s8,
            quantization_size=self.dataset.voxel_size)

        return (coords, feats, lbl, idx, pose)

    # First Downsampling
    def cal_std(self, values, labels, label_num):
        # 1. Calculate the variance of each row
        variances = values.var(dim=1)
        # 2. Calculate separately for each cluster
        for i in range(label_num):
            # Obtain the sample indices for this cluster
            label = i
            label_indices = torch.where(labels == label)[0]
            # Sort by variance in descending order and select the top keep_ratio percentage of samples
            label_variances = variances[label_indices]
            num_top_samples = int(self.keep_ratio * len(label_variances))
            # Obtain the indices of the samples with the largest num_top_samples variances
            _, top_indices = torch.topk(label_variances, num_top_samples)
            selected_indices = label_indices[top_indices]
            # Set the mask of these samples to true
            self.select_idx[selected_indices] = True

        remain_idx = torch.where(self.select_idx)[0]
        self.weights[remain_idx] = self.weights[remain_idx] * (1 / self.keep_ratio)

        return len(remain_idx), remain_idx

    # Second Downsampling, update select_idx
    def sec_cal_std(self, values, labels, selected_indices, label_num):
        values = values[selected_indices]
        labels = labels[selected_indices]
        variances = values.var(dim=1)
        self.select_idx[:] = False
        for i in range(label_num):
            label = i
            label_indices = torch.where(labels == label)[0]
            label_variances = variances[label_indices]
            num_top_samples = int(self.keep_ratio * len(label_variances))
            _, top_indices = torch.topk(label_variances, num_top_samples)
            second_selected_indices = label_indices[top_indices]
            self.select_idx[second_selected_indices] = True

        remain_idx = torch.where(self.select_idx)[0]
        self.weights[remain_idx] = self.weights[remain_idx] * (1 / self.keep_ratio)

        return torch.sum(self.select_idx == True)

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        remained_mask = self.select_idx.numpy()
        remained_indices = np.where(remained_mask)[0].tolist()
        np.random.shuffle(remained_indices)

        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def reset_weights(self):
        self.weights[:] = 1

    @property
    def first_prune(self):
        # Round down
        return math.floor(self.num_epochs * self.delta_start)

    @property
    def stop_prune(self):
        # Round down
        return math.floor(self.num_epochs * self.delta_stop)

    @property
    def second_prune(self):
        # Round down
        # ceil(25 * (0.25 + 0.1 + 0.2) = 13.75) = 14
        return math.floor(self.num_epochs * (self.delta_start + self.windows + self.keep_windows_ratio))

    @property
    def win_std(self):
        # Round up
        return math.ceil(self.num_epochs * self.windows)


class IBSampler(object):
    def __init__(self, dataset: RSD):
        self.dataset = dataset
        self.first_prune = dataset.first_prune
        self.stop_prune = dataset.stop_prune
        self.windows = dataset.win_std
        self.iterations = -1
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def reset(self):
        # np.random.seed(self.iterations)
        print("iterations: %d" % self.iterations)
        if self.iterations >= self.stop_prune or self.iterations < (self.first_prune + self.windows):
            print("no pruning")
            if self.iterations == self.stop_prune:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            print("pruning")
            self.sample_indices = self.dataset.prune()
            print(len(self.sample_indices))
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        nxt = next(self.iter_obj)
        return nxt

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self
