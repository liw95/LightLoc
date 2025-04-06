import torch
from torch import nn


class RSD_Criterion(nn.Module):
    def __init__(self, first_prune_epoch, second_prune_epoch, windows):
        super(RSD_Criterion, self).__init__()
        self.first_prune_epoch = first_prune_epoch
        self.second_prune_epoch = second_prune_epoch
        self.windows = windows

    def forward(self, pred_point, gt_point, batch_size, epoch_nums, idx, values):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1)  # [B*N]
        loss = loss_map.detach().clone().view(batch_size, -1)   # [B, N]
        if self.first_prune_epoch <= epoch_nums < (self.first_prune_epoch + self.windows):
            current_epoch = epoch_nums - self.first_prune_epoch
            values[idx, current_epoch] = torch.median(loss, dim=1).values
        elif self.second_prune_epoch <= epoch_nums < (self.second_prune_epoch + self.windows):
            current_epoch = epoch_nums - self.second_prune_epoch
            values[idx, current_epoch] = torch.median(loss, dim=1).values
        loss_map = torch.mean(loss_map)

        # loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        # if self.first_prune_epoch <= epoch_nums < (self.first_prune_epoch + self.windows):
        #     loss_lw = loss_map.detach().clone()
        #     current_epoch = epoch_nums - self.first_prune_epoch
        #     # print(batch_nums.shape)
        #     # print(loss_map.shape)
        #     for i in range(batch_size):
        #         # 取出当前序列中的第N个batch数据
        #         mask = batch_nums == i
        #         values[idx[i], current_epoch] = torch.median(loss_lw[mask])  # 将其求中值，以忽略噪点影响
        # elif self.second_prune_epoch <= epoch_nums < (self.second_prune_epoch + self.windows):
        #     loss_lw = loss_map.detach().clone()
        #     current_epoch = epoch_nums - self.second_prune_epoch
        #     for i in range(batch_size):
        #         # 取出当前序列中的第N个batch数据
        #         mask = batch_nums == i
        #         values[idx[i], current_epoch] = torch.median(loss_lw[mask])  # 将其求中值，以忽略噪点影响

        # loss_map = torch.mean(loss_map)

        return loss_map, values


class REG_Criterion(nn.Module):
    def __init__(self):
        super(REG_Criterion, self).__init__()

    def forward(self, pred_point, gt_point):
        loss_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = torch.mean(loss_map)

        return loss_map


class CLS_Criterion(nn.Module):
    def __init__(self):
        super(CLS_Criterion, self).__init__()
        self.loc_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred_loc, gt_loc):
        loss = self.loc_loss(pred_loc, gt_loc.long())

        return loss