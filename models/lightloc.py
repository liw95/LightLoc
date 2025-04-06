#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import re
import torch
import logging
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


def Norm(norm_type, num_feats, bn_momentum=0.1, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class Conv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 dimension=3):
        super(Conv, self).__init__()

        self.net = nn.Sequential(ME.MinkowskiConvolution(inplanes,
                                                         planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         dilation=dilation,
                                                         bias=bias,
                                                         dimension=dimension),)

    def forward(self, x):
        return self.net(x)


class Encoder(ME.MinkowskiNetwork):
    """
    FCN encoder, used to extract features from the input point clouds.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels, norm_type, D=3):
        super(Encoder, self).__init__(D)

        self.in_channels = 3
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.conv_planes = [32, 64, 128, 256, 256, 256, 256, 512, 512]

        # in_channels, conv_planes, kernel_size, stride  dilation  bias
        self.conv1 = Conv(self.in_channels, self.conv_planes[0], 3, 1, 1, True)
        self.conv2 = Conv(self.conv_planes[0], self.conv_planes[1], 3, 2, bias=True)
        self.conv3 = Conv(self.conv_planes[1], self.conv_planes[2], 3, 2, bias=True)
        self.conv4 = Conv(self.conv_planes[2], self.conv_planes[3], 3, 2, bias=True)

        self.res1_conv1 = Conv(self.conv_planes[3], self.conv_planes[4], 3, 1, bias=True)
        # 1
        self.res1_conv2 = Conv(self.conv_planes[4], self.conv_planes[5], 1, 1, bias=True)
        self.res1_conv3 = Conv(self.conv_planes[5], self.conv_planes[6], 3, 1, bias=True)

        self.res2_conv1 = Conv(self.conv_planes[6], self.conv_planes[7], 3, 1, bias=True)
        # 2
        self.res2_conv2 = Conv(self.conv_planes[7], self.conv_planes[8], 1, 1, bias=True)
        self.res2_conv3 = Conv(self.conv_planes[8], self.out_channels, 3, 1, bias=True)

        self.res2_skip = Conv(self.conv_planes[6], self.out_channels, 1, 1, bias=True)

    def forward(self, x):

        x = MEF.relu(self.conv1(x))
        x = MEF.relu(self.conv2(x))
        x = MEF.relu(self.conv3(x))
        res = MEF.relu(self.conv4(x))

        x = MEF.relu(self.res1_conv1(res))
        x = MEF.relu(self.res1_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res1_conv3(x))

        res = res + x

        x = MEF.relu(self.res2_conv1(res))
        x = MEF.relu(self.res2_conv2(x))
        x._F = x.F.to(torch.float32)
        x = MEF.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


def one_hot(x, N):
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1), x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


class CondLayer(nn.Module):
    """
    pixel-wise feature modulation.
    """
    def __init__(self, in_channels):
        super(CondLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x, gammas, betas):
        return F.relu(self.bn((gammas * x) + betas))


class Cls_Head(nn.Module):
    """
    Classification Head
    """
    def __init__(self, in_channels=512, level_cluster=25):
        super(Cls_Head, self).__init__()

        channels_c = [512, 256, level_cluster]

        # level network.
        self.conv1_l1 = nn.Linear(in_channels, channels_c[0])
        self.norm1_l1 = nn.BatchNorm1d(channels_c[0])
        self.conv2_l1 = nn.Linear(channels_c[0], channels_c[1])
        self.norm2_l1 = nn.BatchNorm1d(channels_c[1])
        self.conv3_l1 = nn.Linear(channels_c[1], channels_c[2])
        self.dp1_l1 = nn.Dropout(0.5)
        self.dp2_l1 = nn.Dropout(0.5)


    def forward(self, res):

        x1 = self.dp1_l1(F.relu(self.norm1_l1(self.conv1_l1(res))))
        x1 = self.dp2_l1(F.relu(self.norm2_l1(self.conv2_l1(x1))))
        # output the classification probability.
        out_lbl_1 = self.conv3_l1(x1)

        return out_lbl_1


class Reg_Head(nn.Module):
    """
    nn.Linear版
    """
    def __init__(self, num_head_blocks, in_channels=512, mlp_ratio=1.0):
        super(Reg_Head, self).__init__()
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = in_channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels \
            else nn.Linear(self.in_channels, self.head_channels)

        block_channels = int(self.head_channels * mlp_ratio)
        self.res3_conv1 = nn.Linear(self.in_channels, self.head_channels)
        self.res3_conv2 = nn.Linear(self.head_channels, block_channels)
        self.res3_conv3 = nn.Linear(block_channels, self.head_channels)

        self.res_blocks = []
        self.norm_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Linear(self.head_channels, self.head_channels),
                nn.Linear(self.head_channels, block_channels),
                nn.Linear(block_channels, self.head_channels),
            ))

            super(Reg_Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Reg_Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Reg_Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, block_channels)
        self.fc3 = nn.Linear(block_channels, 3)

    def forward(self, res):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:

            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        return sc


class Regressor(ME.MinkowskiNetwork):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, num_head_blocks, num_encoder_features, level_clusters=25,
                 mlp_ratio=1.0, sample_cls=False, D=3):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__(D)

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim, norm_type='BN')
        self.cls_heads = Cls_Head(in_channels=self.feature_dim, level_cluster=level_clusters)
        if not sample_cls:
            self.reg_heads = Reg_Head(num_head_blocks=num_head_blocks, in_channels=self.feature_dim + level_clusters,
                                      mlp_ratio=mlp_ratio)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, classifier_state_dict=None,
                            num_head_blocks=None, level_clusters=25, mlp_ratio=1.0, sample_cls=False):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        classifier_state_dict: trained classifier state dictionary
        num_head_blocks: How many extra residual blocks to use in the head.
        level_cluster: How many classification categories.
        mlp_ratio: Channel expansion ratio.
        sample_cls: training for classifier
        """

        num_encoder_features = encoder_state_dict['res2_conv3.net.0.bias'].shape[1]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(num_head_blocks, num_encoder_features, level_clusters, mlp_ratio, sample_cls)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        if classifier_state_dict!=None:
            regressor.cls_heads.load_state_dict(classifier_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^reg_heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.net.0.bias'].shape[1]
        num_decoder_features = state_dict['cls_heads.conv1_l1.weight'].shape[1]
        head_channels = state_dict['cls_heads.conv1_l1.weight'].shape[0]
        level_clusters = state_dict['cls_heads.conv3_l1.weight'].shape[0]
        reg = any(key.startswith("reg_heads") for key in state_dict)
        if reg:
            mlp_ratio = state_dict['reg_heads.res3_conv2.weight'].shape[0] / \
                        state_dict['reg_heads.res3_conv2.weight'].shape[1]
        else:
            mlp_ratio = 1

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tEncoder feature size: {num_encoder_features}"
                     f"\n\tDecoder feature size: {num_decoder_features}"
                     f"\n\tHead channels: {head_channels}"
                     f"\n\tMLP ratio: {mlp_ratio}")
        regressor = cls(num_head_blocks, num_encoder_features, mlp_ratio=mlp_ratio, level_clusters=level_clusters)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, cls_head_state_dict, reg_head_state_dict=None):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in cls_head_state_dict.items():
            merged_state_dict[f"cls_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        if reg_head_state_dict != None:
            for k, v in reg_head_state_dict.items():
                merged_state_dict[f"reg_heads.{k}"] = v.squeeze(-1).squeeze(-1)

        return cls.create_from_state_dict(merged_state_dict)


    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        out = self.reg_heads(features)
        return out

    def get_scene_classification(self, features):
        out = self.cls_heads(features)

        return out

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.encoder(inputs)
        out = self.get_scene_coordinates(features.F)
        out = ME.SparseTensor(
            features=out,
            coordinates=features.C,
        )

        return {'pred': out}