import os
import os.path as osp
import argparse
import logging
import time
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lightloc import Regressor
from models.sc2pcr import Matcher
from datasets.lidarloc import LiDARLocDataset
from datasets.base_loader import CollationFunctionFactory
from utils.pose_util import val_translation, val_rotation, qexp

import MinkowskiEngine as ME
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == '__main__':
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scene', type=Path, default='/home/lw/Oxford',
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='test batch size')

    parser.add_argument('--classifier_path', type=Path, default='log/Oxford/49_cls.pth',
                        help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--regressor_path', type=Path, default='log/Oxford/24_reg.pth',
                        help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default='log/Backbone.pth',
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--output_path', type=Path, default='log/',
                        help='file used for output')

    parser.add_argument('--voxel_size', type=float, default=0.25)


    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6
    # 5 cpu core
    torch.set_num_threads(7)

    scene_path = Path(opt.scene)
    cls_head_network_path = Path(opt.classifier_path)
    reg_head_network_path = Path(opt.regressor_path)
    encoder_path = Path(opt.encoder_path)
    output_path = Path(opt.output_path)
    # ckpt_path = Path(opt.ckpt_path)
    batch_size = opt.batch_size
    voxel_size = opt.voxel_size

    LOG_FOUT = open(os.path.join(output_path, 'log.txt'), 'w')
    LOG_FOUT.write(str(opt) + '\n')

    # Setup dataset.
    testset = LiDARLocDataset(
        root_dir=scene_path,
        train=False,
        voxel_size=voxel_size
    )


    _logger.info(f'Test point clouds found: {len(testset)}')

    collation_fn = CollationFunctionFactory(collation_type='collate_pair_reg')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6, collate_fn=collation_fn, batch_size=batch_size)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    cls_head_state_dict = torch.load(cls_head_network_path, map_location="cpu")
    _logger.info(f"Loaded classifier head weights from: {cls_head_network_path}")
    reg_head_state_dict = torch.load(reg_head_network_path, map_location="cpu")
    _logger.info(f"Loaded regressor head weights from: {reg_head_network_path}")

    # Load Means
    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, cls_head_state_dict, reg_head_state_dict)
    # network = Regressor.create_from_state_dict(reg_head_state_dict)
    ransac = Matcher(inlier_threshold=2)

    _logger.info(f'#Backbone Model parameters: {sum([x.nelement() for x in network.encoder.parameters()]) / 1e6}')
    _logger.info(f'#Cls Model parameters: {sum([x.nelement() for x in network.cls_heads.parameters()]) / 1e6}')
    _logger.info(f'#Reg Model parameters: {sum([x.nelement() for x in network.reg_heads.parameters()]) / 1e6}')

    cls_pool = ME.MinkowskiGlobalAvgPooling()
    reg_pool = ME.MinkowskiAvgPooling(kernel_size=8, stride=8, dimension=3)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    sn = os.path.split(scene_path)[-1]

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    gt_translation = np.zeros((len(testset), 3))
    pred_translation = np.zeros((len(testset), 3))
    gt_rotation = np.zeros((len(testset), 4))
    pred_rotation = np.zeros((len(testset), 4))

    error_t = np.zeros(len(testset))
    error_txy = np.zeros(len(testset))
    error_q = np.zeros(len(testset))

    correct_loc_results = []
    time_results_network = []

    for step, batch in enumerate(testset_loader):
        val_pose = batch["pose"]
        start_idx = step * batch_size
        end_idx = min((step + 1) * batch_size, len(testset))
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy()
        gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])

        features = batch['sinput_F'].to(device, dtype=torch.float32)
        coordinates = batch['sinput_C'].to(device)
        pcs_tensor = ME.SparseTensor(features[..., :3], coordinates)
        pcs_tensor_s8 = ME.SparseTensor(features, coordinates)

        pose_gt = batch['pose'].to(device, dtype=torch.float32)
        batch_size = pose_gt.size(0)
        pred_t = np.zeros((batch_size, 3))
        pred_q = np.zeros((batch_size, 4))
        index_list = [0]  #

        start = time.time()

        with torch.no_grad():
            features = network.get_features(pcs_tensor)
            cls_features = cls_pool(features).F
            lbl_pred = network.get_scene_classification(cls_features)
            pred_cls = lbl_pred / torch.norm(lbl_pred, p=2, dim=1, keepdim=True)
            ground_truth = reg_pool(pcs_tensor_s8)
            featuresF = torch.cat((pred_cls[features.C[:, 0].long()], features.F), dim=1)
            # featuresF = features.F
            pred_scene = network.get_scene_coordinates(featuresF)

        pred = ME.SparseTensor(
            features=pred_scene,
            coordinates=features.C
        )

        pred_point = pred.F

        ground_truth = ground_truth.features_at_coordinates(features.C.float())
        sup_point = ground_truth[:, :3]

        for i in range(batch_size):
            batch_pred_pcs_tensor = pred.coordinates_at(i).float()
            index_list.append(index_list[i] + len(batch_pred_pcs_tensor))

        gt_point = sup_point
        gt_sup_point = ground_truth[:, 3:6].cpu().numpy()

        for i in range(batch_size):
            # print(start_idx + i)
            a = gt_point[index_list[i]:index_list[i + 1], :]
            b = pred_point[index_list[i]:index_list[i + 1], :]
            c = gt_sup_point[index_list[i]:index_list[i + 1], :]

            batch_pred_t, batch_pred_q, _ = ransac.estimator(
                a.unsqueeze(0), b.unsqueeze(0))


            pred_t[i, :] = batch_pred_t
            pred_q[i, :] = batch_pred_q

        end = time.time()
        cost_time = (end - start) / batch_size
        time_results_network.append(cost_time)

        pred_translation[start_idx:end_idx, :] = pred_t
        pred_rotation[start_idx:end_idx, :] = pred_q

        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :],
                                                     gt_translation[start_idx:end_idx, :])])
        error_txy[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                   zip(pred_translation[start_idx:end_idx, :2],
                                                       gt_translation[start_idx:end_idx, :2])])

        error_q[start_idx:end_idx] = np.asarray(
            [val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                gt_rotation[start_idx:end_idx, :])])

        # log_string('ValLoss(m): %f' % float(val_loss))
        log_string('MeanXYZTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanXYTE(m): %f' % np.mean(error_txy[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

        torch.cuda.empty_cache()

    mean_ATE = np.mean(error_t)
    mean_xyATE = np.mean(error_txy)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_xyATE = np.median(error_txy)
    median_ARE = np.median(error_q)
    mean_time_network = np.mean(time_results_network)
    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean XY Position Error(m): %f' % mean_xyATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median XY Position Error(m): %f' % median_xyATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Network Cost Time(s): %f' % mean_time_network)

    # save error
    error_t_filename = osp.join(output_path, 'error_t.txt')
    error_q_filename = osp.join(output_path, 'error_q.txt')
    loss_ap_filename = osp.join(output_path, 'loss_ap.txt')
    loss_tp_filename = osp.join(output_path, 'loss_tp.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')

    # trajectory
    fig = plt.figure()
    real_pose = pred_translation
    gt_pose = gt_translation
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=1, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=1, c='red')
    plt.plot(real_pose[:, 1], real_pose[:, 0], linewidth=1, color='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(output_path),
                                  '{:s}.png'.format('trajectory'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_distribution
    fig = plt.figure()
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Data Num')
    plt.ylabel('Error (m)')
    image_filename = os.path.join(os.path.expanduser(output_path),
                                  '{:s}.png'.format('distribution_t'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_distribution
    fig = plt.figure()
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Data Num')
    plt.ylabel('Error (degree)')
    image_filename = os.path.join(os.path.expanduser(output_path),
                                  '{:s}.png'.format('distribution_q'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(output_path, 'error_t.txt')
    error_q_filename = osp.join(output_path, 'error_q.txt')
    pred_q_filename = osp.join(output_path, 'pred_q.txt')
    pred_t_filename = osp.join(output_path, 'pred_t.txt')
    gt_t_filename = osp.join(output_path, 'gt_t.txt')
    gt_q_filename = osp.join(output_path, 'gt_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_q_filename, gt_rotation, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')