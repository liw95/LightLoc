#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import MinkowskiEngine as ME
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from models.lightloc import Regressor
from models.loss import CLS_Criterion, REG_Criterion, RSD_Criterion
from datasets.lidarloc import LiDARLocDataset
from datasets.rsd import RSD
from datasets.base_loader import CollationFunctionFactory

from pathlib import Path
from distutils.util import strtobool
from tqdm import tqdm


_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda')

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Generator for global feature noise
        self.gn_generator = torch.Generator(device=self.device)
        self.gn_generator.manual_seed(self.base_seed + 24601)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = LiDARLocDataset(
            root_dir=self.options.scene,  # which dataset?
            train=True,  # train
            sample_cls=self.options.sample_cls,   # train for sample classification?   True: sample cls; False: reg
            augment=self.options.use_aug,  # use augmentation?
            aug_rotation=self.options.aug_rotation,  # max rotation
            aug_translation=self.options.aug_translation,  # max translation
            generate_clusters=self.options.generate_clusters,   # generate new classification label
            level_clusters=self.options.level_cluster,
            voxel_size=self.options.voxel_size,  # voxel size of point cloud
        )

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        # REG
        if not self.options.sample_cls:
            # RSD
            if self.options.rsd:
                self.dataset = RSD(self.dataset, self.options.epochs, self.options.prune_ratio)
                self.first_start_prune = self.dataset.first_prune
                self.second_start_prune = self.dataset.second_prune
                self.windows = self.dataset.win_std
                # # Setup loss function.
                self.loss = RSD_Criterion(self.first_start_prune, self.second_start_prune, self.windows)
                # Record median loss
                self.values = torch.empty((len(self.dataset), self.windows), dtype=torch.float32, device='cuda')
                self.labels = torch.empty((len(self.dataset), 1), dtype=torch.int64, device='cuda')
                sampler = self.dataset.sampler
                shuffle = False
            else:
                # # Setup loss function.
                self.loss = REG_Criterion()
                sampler = None
                shuffle = True
            # Load classification weights
            classifier_state_dict = torch.load(self.options.classifier_path, map_location="cpu")
            collation_fn = CollationFunctionFactory(collation_type='collate_pair_reg')
            self.dataloader = DataLoader(dataset=self.dataset,
                                         batch_size=self.options.batch_size,
                                         pin_memory=True,
                                         shuffle=shuffle,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         collate_fn=collation_fn,
                                         sampler=sampler,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )
        # CLS
        else:
            classifier_state_dict = None
            # # Setup loss function.
            self.loss = CLS_Criterion()

        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict=encoder_state_dict,
            classifier_state_dict=classifier_state_dict,
            num_head_blocks=self.options.num_head_blocks,
            mlp_ratio=self.options.mlp_ratio,
            level_clusters=self.options.level_cluster,
            sample_cls=self.options.sample_cls,
        )

        self.regressor = self.regressor.to(self.device)

        for param in self.regressor.encoder.parameters():
            param.requires_grad = False

        if not self.options.sample_cls:
            for param in self.regressor.cls_heads.parameters():
                param.requires_grad = False
            steps_per_epoch = len(self.dataset) // self.options.batch_size
        else:
            steps_per_epoch = self.options.training_buffer_size // self.options.batch_size

        train_parameter = filter(lambda p: p.requires_grad, self.regressor.parameters())

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(train_parameter, lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100  # print loss every n iterations, and (optionally) write a visualisation frame

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

    def train(self):
        """
        Main training method.
        """

        creating_buffer_time = 0.
        training_time = 0.

        if self.options.sample_cls:
            # For CLS training
            self.training_start = time.time()

            # Create training buffer.
            buffer_start_time = time.time()
            # print(buffer_start_time)
            self.create_training_buffer()
            buffer_end_time = time.time()
            creating_buffer_time += buffer_end_time - buffer_start_time
            _logger.info(f"Filled training buffer in {(buffer_end_time - buffer_start_time) / 60:.1f} minutes.")

            # Train the regression head.
            for self.epoch in range(self.options.epochs):
                epoch_start_time = time.time()
                self.cls_run_epoch()
                training_time += time.time() - epoch_start_time

            end_time = time.time()
            # Save trained model.
            self.save_model(self.epoch, 'cls')

            _logger.info(f'Done without errors. '
                         f'Creating buffer time: {creating_buffer_time / 60:.1f} minutes. '
                         f'Training time: {training_time / 60 :.1f} minutes. '
                         f'Total time: {(end_time - self.training_start) / 60 :.1f} minutes.')

        else:
            # For REG training
            self.training_start = time.time()
            if self.options.rsd:
                for self.epoch in range(self.options.epochs):
                    if self.epoch == (self.first_start_prune + self.windows):
                        print("First data downsampling")
                        _, selected_indices = self.dataset.cal_std(self.values, self.labels,
                                                                           self.options.level_cluster)
                    elif self.epoch == (self.second_start_prune + self.windows):
                        print("Second data downsampling")
                        _ = self.dataset.sec_cal_std(self.values, self.labels, selected_indices,
                                                             self.options.level_cluster)

                    self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                   max_lr=self.options.learning_rate_max,
                                                                   steps_per_epoch=len(self.dataloader),
                                                                   epochs=self.options.epochs,
                                                                   cycle_momentum=False,
                                                                   last_epoch=self.epoch * len(self.dataloader) - 1)

                    self.reg_run_epoch()
            else:
                for self.epoch in range(self.options.epochs):
                    self.reg_run_epoch()

            end_time = time.time()
            # Save trained model.
            self.save_model(self.epoch, 'reg')

            _logger.info(f'Done without errors. '
                         f'Training time: {(end_time - self.training_start) / 60:.1f} minutes. ')

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=self.options.eval_batch_size, drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).

        collation_fn = CollationFunctionFactory(collation_type='collate_pair_cls')

        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         collate_fn=collation_fn,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )
        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device='cuda'),
            'lbl': torch.empty((self.options.training_buffer_size), dtype=torch.int64, device='cuda'),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()
        cls_pool = ME.MinkowskiGlobalAvgPooling()
        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                # tqdm_loader = tqdm(training_dataloader, total=len(training_dataloader))
                for step, batch in enumerate(training_dataloader):

                    # Copy to device.
                    features = batch['sinput_F'].to(self.device, non_blocking=True)
                    coordinates = batch['sinput_C'].to(self.device, non_blocking=True)
                    pcs_tensor = ME.SparseTensor(features[:, :3], coordinates)
                    lbl = batch['lbl'].to(self.device, non_blocking=True).squeeze().squeeze()

                    batch_size = lbl.size(0)
                    # Compute global features.
                    with autocast(enabled=self.options.use_half):
                        features = self.regressor.get_features(pcs_tensor)
                        features = cls_pool(features).F      # [B, 512]
                    features_to_select = min(batch_size, self.options.training_buffer_size - buffer_idx)

                    if self.options.use_half:
                        features = features.half()
                    else:
                        features = features

                    batch_data = {
                        'features': features,
                        'lbl': lbl,
                    }

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select

                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k][:features_to_select]

                    buffer_idx = buffer_offset
                    # percent = (buffer_idx/self.options.training_buffer_size)*100
                    # print(percent)
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def cls_run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.options.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.options.training_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.options.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.cls_training_step(
                self.training_buffer['features'][random_batch_indices],
                self.training_buffer['lbl'][random_batch_indices]
            )

            self.iteration += 1

    def cls_training_step(self, features_bC, lbl_bC):
        """
        Run one iteration of training, computing the l1 error and minimising it.
        """
        with autocast(enabled=self.options.use_half):
            lbl_pred = self.regressor.get_scene_classification(features_bC)

        # Compute the loss for predictions.
        loss = self.loss(lbl_pred, lbl_bC)

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.4f}, Time: {time_since_start:.2f}s')

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def reg_run_epoch(self):
        # torch.backends.cudnn.benchmark = True
        cls_pool = ME.MinkowskiGlobalAvgPooling()
        reg_pool = ME.MinkowskiAvgPooling(kernel_size=8, stride=8, dimension=3)
        tqdm_loader = tqdm(self.dataloader, total=len(self.dataloader))

        # The number point of sampling
        num_point = 256

        features_bC = torch.empty(
            (self.options.batch_size * num_point, self.regressor.feature_dim + self.options.level_cluster),
            dtype=(torch.float32, torch.float16)[self.options.use_half], device='cuda')
        target_bC = torch.empty((self.options.batch_size * num_point, 3),
                                dtype=torch.float32, device='cuda')

        batch_bC = torch.empty(self.options.batch_size * num_point, dtype=torch.int64, device='cuda')

        for step, batch in enumerate(tqdm_loader):
            features = batch['sinput_F'].to(self.device, non_blocking=True)
            coordinates = batch['sinput_C'].to(self.device, non_blocking=True)
            idx = batch['idx'].to(self.device)
            label = batch['lbl'].to(self.device)
            batch_size = label.size(0)
            # record
            if self.options.rsd and self.epoch == 0:
                self.labels[idx] = label
            pcs_tensor = ME.SparseTensor(features[:, :3], coordinates)
            pcs_tensor_s8 = ME.SparseTensor(features, coordinates)
            self.regressor.eval()
            with torch.no_grad():
                features = self.regressor.get_features(pcs_tensor)
                # Global features
                cls_features = cls_pool(features).F  # [B, 512]
                # Sample probability distribution features
                lbl_1_pred = self.regressor.get_scene_classification(cls_features)
                # Normalization
                cls_label = lbl_1_pred / torch.norm(lbl_1_pred, p=2, dim=1, keepdim=True).half()
                # Ground Truth for scene coordinate regression
                ground_truth = reg_pool(pcs_tensor_s8)
                # [:3] point cloud in the LiDAR coordinate; [3:6] corresponding point cloud in the world coordinate
                gt_sup_point = ground_truth.features_at_coordinates(features.C.float())[:, 3:6]
                # half precision
                featuresF = features.F.half()
                # batch size
                batch_nums = features.C[:, 0]
            #
            random_indices = torch.randperm(len(batch_nums), generator=self.training_generator)[:(batch_size * num_point)]
            features_bC[:(batch_size * num_point), :self.options.level_cluster] = cls_label[features.C[:, 0].long()][
                random_indices]
            features_bC[:(batch_size * num_point), self.options.level_cluster:] = featuresF[random_indices]
            # Add gaussian noise and normalization
            features_bC[:(batch_size * num_point), :self.options.level_cluster] += torch.empty_like(
                features_bC[:(batch_size * num_point), :self.options.level_cluster]).normal_(mean=0, std=0.1,
                                                                                        generator=self.gn_generator)
            target_bC[:(batch_size * num_point)] = gt_sup_point[random_indices]
            batch_bC[:(batch_size * num_point)] = batch_nums[random_indices]

            self.regressor.train()
            with autocast(enabled=self.options.use_half):
                pred_scene = self.regressor.get_scene_coordinates(features_bC)

            if self.options.rsd:
                loss, self.values = self.loss(pred_scene[:(batch_size * num_point)], target_bC[:(batch_size * num_point)],
                                              batch_size, self.epoch, idx, self.values)
            else:
                loss = self.loss(pred_scene, target_bC)

            optimizer_step = self.optimizer._step_count

            # Optimization steps.
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # Clip the gradient.
            torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            torch.cuda.empty_cache()

            if optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
                self.scheduler.step()

            if step % self.iterations_output == 0:
                time_since_start = time.time() - self.training_start
                _logger.info(f'Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                             f'Loss: {loss:.4f}, Time: {time_since_start:.2f}s')


    def save_model(self, epoch, cor):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats.
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        if cor == 'cls':
            state_dict = self.regressor.cls_heads.state_dict()
        else:
            state_dict = self.regressor.reg_heads.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = state_dict[k].half()
        torch.save(state_dict, 'log/' + str(epoch) + '_' + cor + '.pth')
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")


if __name__ == '__main__':

    # CUDA_VISIBLE_DEVICES=4 python train.py --sample_cls=True --generate_clusters=True --batch_size=512 --epochs=50 --level_cluster=25
    # CUDA_VISIBLE_DEVICES=4 python train.py --sample_cls=False --generate_clusters=False --batch_size=256 --epochs=25 --rsd=True --level_cluster=25

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a sample classification network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data path
    parser.add_argument('--scene', type=Path, default='/home/lw/Oxford',
                        help='path to a scene in the dataset folder')

    # Output path
    parser.add_argument('--output_map_file', type=Path, default='log/',
                        help='target file for the trained network')

    # Encoder path
    parser.add_argument('--encoder_path', type=Path, default='log/Backbone.pth',
                        help='file containing pre-trained encoder weights')

    # Classifier path
    parser.add_argument('--classifier_path', type=Path, default='log/49_cls.pth',
                        help='file containing trained classifier weights')

    # Train classification or regression
    parser.add_argument('--sample_cls', type=_strtobool, default=False,
                        help='True: train sample classification module; False: train scene coordinate regression')

    # Number of cluster for sample classification guidance (SCG)
    parser.add_argument('--level_cluster', type=int, default=25,
                        help='the number of first level classification')

    parser.add_argument('--generate_clusters', type=_strtobool, default=False,
                        help='Generate clusters for classification training')

    # Architecture of scene regression head
    parser.add_argument('--num_head_blocks', type=int, default=2,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--mlp_ratio', type=int, default=2.0,
                        help='mlp ratio for res blocks')

    # Redundant sample downsampling (RSD)
    parser.add_argument('--rsd', type=_strtobool, default=True,
                        help='Using RSD for data pruning')

    # Redundant sample downsampling (RSD)
    parser.add_argument('--prune_ratio', type=float, default=0.25,
                        help=' Downsampling ratio of RSD')

    # Learn rate
    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    # Buffer size, only used for training classification
    parser.add_argument('--training_buffer_size', type=int, default=160000,
                        help='number of samples in the training buffer')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=256,
                        help='number of samples for each parameter update. classification: 512; regression: 256')

    # Eval size, only used for generating buffer for training classification
    parser.add_argument('--eval_batch_size', type=int, default=100,
                        help='used to generate buffer')

    # Train epoch
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of runs. classification: 50; regression: 25.')

    # Using half-precision for training
    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    # Using data augmentation
    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    # Max rotation angle (degree)
    parser.add_argument('--aug_rotation', type=int, default=10,
                        help='max rotation angle')

    # Max translation distance (meter)
    parser.add_argument('--aug_translation', type=int, default=1,
                        help='max translation meter')

    # Voxel size for sparse conv
    parser.add_argument('--voxel_size', type=float, default=0.25,
                        help='Oxford 0.25 NCLT 0.30')

    options = parser.parse_args()

    torch.set_num_threads(5)
    trainer = Trainer(options)
    trainer.train()