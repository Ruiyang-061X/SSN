import torch
from torch.autograd import Function
from torch import nn


class OHEMHingeLoss(Function):

    @staticmethod
    def forward(ctx, predicted_label, label, is_positive, ohem_ratio, group_size):
        n_sample = predicted_label.size()[0]
        loss = torch.zeros(n_sample)
        slope = torch.zeros(n_sample)
        for i in range(n_sample):
            loss[i] = max(0, 1 - is_positive * predicted_label[i, label[i] - 1])
            slope[i] = -is_positive if loss[i] != 0 else 0
        loss = loss.view(-1, group_size).contiguous()
        sorted_loss, indice = torch.sort(loss, dim=1, descending=True)
        n_keep = int(ohem_ratio * group_size)
        loss_ = torch.zeros(1)
        loss_ = loss_.cuda()
        for i in range(loss.size(0)):
            loss_ += sorted_loss[i, : n_keep].sum()
        ctx.loss_indice = indice[ : , : n_keep]
        ctx.label = label
        ctx.slope = slope
        ctx.shape = predicted_label.size()
        ctx.group_size = group_size
        ctx.n_group = loss.size(0)

        return loss_

    @staticmethod
    def backward(ctx, dout):
        label = ctx.label
        slope = ctx.slope
        dx = torch.zeros(ctx.shape)
        for i in range(ctx.n_group):
            for j in ctx.loss_indice[i]:
                indice = j + i * ctx.group_size
                dx[indice, label[indice] - 1] = slope[indice] * dout.data[0]
        dx = dx.cuda()

        return dx, None, None, None, None

class CompletenessLoss(nn.Module):

    def __init__(self, ohem_ratio=0.17):
        super(CompletenessLoss, self).__init__()
        self.ohem_ratio = ohem_ratio
        self.sigmoid = nn.Sigmoid()

    def forward(self, predicted_label, label, sample_split, sample_group_size):
        predicted_dim = predicted_label.size()[1]
        predicted_label = predicted_label.view(-1, sample_group_size, predicted_dim)
        label = label.view(-1, sample_group_size)
        positive_group_size = sample_split
        negative_group_size = sample_group_size - sample_split
        positive_probability = predicted_label[ : , : sample_split, : ].contiguous().view(-1, predicted_dim)
        negative_probability = predicted_label[ : , sample_split : , : ].contiguous().view(-1, predicted_dim)
        positive_loss = OHEMHingeLoss.apply(positive_probability, label[ : , : sample_split].contiguous().view(-1), 1, 1.0, positive_group_size)
        negative_loss = OHEMHingeLoss.apply(negative_probability, label[ : , sample_split : ].contiguous().view(-1), -1, self.ohem_ratio, negative_group_size)
        positive_count = positive_probability.size()[0]
        negative_count = int(negative_probability.size()[0] * self.ohem_ratio)

        return positive_loss / float(positive_count + negative_count) + negative_loss / float(positive_count + negative_count)

class ClasswiseRegressionLoss(nn.Module):

    def __init__(self):
        super(ClasswiseRegressionLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, predicted_label, label, regression_label):
        indice = label.data - 1
        predicted_label = predicted_label[ : , indice, : ]
        class_predicted_label = torch.cat((torch.diag(predicted_label[ : , : , 0]).view(-1, 1), torch.diag(predicted_label[ : , : , 1]).view(-1, 1)), dim=1)
        loss = self.smooth_l1_loss(class_predicted_label.view(-1), regression_label.view(-1)) * 2

        return loss