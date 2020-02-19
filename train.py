import argparse
import os
import time
import numpy as np
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import model_zoo
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import SGD
import torch
from dataset import SSNDataset
from ssn import SSN
from transform import *
from util import CompletenessLoss, ClasswiseRegressionLoss


parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, choices=['RGB', 'RGBDiff', 'Flow'])
parser.add_argument('--base_model', type=str, default='BNInception')
parser.add_argument('--n_body_segment', type=int, default=5)
parser.add_argument('--n_augmentation_segment', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--epoch', type=int, default=45)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_step', type=int, default=[20, 40], nargs='+')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--clip_gradient', type=float, default=None)
parser.add_argument('--bn_mode', type=str, default='frozen')
parser.add_argument('--completeness_loss_weight', type=float, default=0.1)
parser.add_argument('--regression_loss_weight', type=float, default=0.1)
parser.add_argument('--print_every', type=int, default=20)
parser.add_argument('--validation_every', type=int, default=1)
args = parser.parse_args()

cudnn.benchmark = True

if not os.path.exists('trained_model'):
    os.mkdir('trained_model')

if args.modality == 'RGB':
    new_length = 1
if args.modality in ['RGBDiff', 'Flow']:
    new_length = 5
if 'vgg' in args.base_model or 'resnet' in args.base_model:
    input_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    if args.modality == 'Flow':
        input_mean = [0.5]
        input_std = [np.mean(input_std)]
    if args.modality == 'RGBDiff':
        input_mean = input_mean + [0] * 3 * new_length
        input_std = input_std + [np.mean(input_std) * 2] * 3 * new_length
elif args.base_model == 'BNInception':
    input_size = 224
    input_mean = [104, 117, 128]
    input_std = [1]
    if args.modality == 'Flow':
        input_mean = [128]
    if args.modality == 'RGBDiff':
        input_mean = input_mean * (new_length + 1)
elif args.base_model == 'InceptionV3':
    input_size = 299
    input_mean = [104, 117, 128]
    input_std = 1
    if args.modality == 'RGBDiff':
        input_mean = input_mean * (new_length + 1)
    if args.modality == 'Flow':
        input_mean = [128]
elif 'inception' in args.base_model:
    input_size = 299
    input_mean = [0.5]
    input_std = [0.5]
if args.modality == 'RGB':
    augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75, 0.66]), GroupRandomHorizontalFlip(is_flow=False)])
if args.modality == 'RGBDiff':
    augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)])
if args.modality == 'Flow':
    augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)])
if args.modality != 'RGBDiff':
    normalize = GroupNormalize(input_mean, input_std)
else:
    normalize = IdentityTransform()
crop_size = input_size
scale_size = input_size * 256 // 224
trainset_video_record_list_path = 'dataset/thumos14_tag_val_proposal_list.txt'
trainset_transform = transforms.Compose([augmentation, Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])), ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])), normalize])
trainset = SSNDataset(video_record_list_path=trainset_video_record_list_path, n_body_segment=args.n_body_segment, n_augmentation_segment=args.n_augmentation_segment, new_length=new_length, modality=args.modality, transform=trainset_transform, proposal_per_video=8, fg_ratio=1, bg_ratio=1, incomplete_ratio=6, fg_iou_thresh=0.7, bg_iou_thresh=0.01, incomplete_iou_thresh=0.3, bg_coverage_thresh=0.02, incomplete_overlap_thresh=0.7)
trainset_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
validationset_video_record_list_path = 'dataset/thumos14_tag_test_proposal_list.txt'
validationset_transform = transforms.Compose([GroupScale(scale_size), GroupCenterCrop(crop_size), Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])), ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])), normalize])
validationset =  SSNDataset(video_record_list_path=validationset_video_record_list_path, n_body_segment=args.n_body_segment, n_augmentation_segment=args.n_augmentation_segment, new_length=new_length, modality=args.modality, transform=validationset_transform, random_shift=False, proposal_per_video=8, fg_ratio=1, bg_ratio=1, incomplete_ratio=6, fg_iou_thresh=0.7, bg_iou_thresh=0.01, incomplete_iou_thresh=0.3, bg_coverage_thresh=0.02, incomplete_overlap_thresh=0.7, regression_constant=trainset.regression_constant)
validationset_loader = DataLoader(dataset=validationset, batch_size=args.batch_size, shuffle=False, drop_last=True)

n_class = 20
stpp_cfg = (1, 1, 1)
ssn = SSN(base_model=args.base_model, n_class=n_class, dropout=args.dropout, stpp_cfg=stpp_cfg, bn_mode=args.bn_mode, modality=args.modality, n_body_segment=args.n_body_segment, n_augmentation_segment=args.n_augmentation_segment, new_length=new_length)
if args.modality == 'Flow':
    if args.base_model == 'BNInception':
        url = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_thumos_flow_init-89dfeaf803e.pth'
    if args.base_model == 'InceptionV3':
        url = 'https://yjxiong.blob.core.windows.net/ssn-models/inceptionv3_thumos_flow_init-0527856bcec6.pth'
    ssn.base_model.load_state_dict(model_zoo.load_url(url)['state_dict'])
ssn = ssn.cuda()

activity_loss_function = CrossEntropyLoss()
activity_loss_function = activity_loss_function.cuda()
completeness_loss_function = CompletenessLoss()
completeness_loss_function = completeness_loss_function.cuda()
regression_loss_function = ClasswiseRegressionLoss()
regression_loss_function = regression_loss_function.cuda()

first_conv_weight = []
first_conv_bias = []
normal_weight = []
normal_bias = []
conv_count = 0
for m in ssn.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        ps = list(m.parameters())
        conv_count += 1
        if conv_count == 1:
            first_conv_weight += [ps[0]]
            if len(ps) == 2:
                first_conv_bias += [ps[1]]
        else:
            normal_weight += [ps[0]]
            if len(ps) == 2:
                normal_bias += [ps[1]]
    if isinstance(m, nn.Linear):
        ps = list(m.parameters())
        normal_weight += [ps[0]]
        if len(ps) == 2:
            normal_bias += [ps[1]]
optimize_policy = [
    {'name': 'first_conv_weight', 'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1},
    {'name': 'first_conv_bias', 'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0},
    {'name': 'normal_weight', 'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1},
    {'name': 'normal_bias', 'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0},
]
for i in optimize_policy:
    print('group {} has {} parameters, lr_mult: {}, decay_mult: {}'.format(i['name'], len(i['params']), i['lr_mult'], i['decay_mult']))
optimizer = SGD(optimize_policy, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

class AverageMeter():

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(predicted_label, label, topk=(1, )):
    maxk = max(topk)
    _, predicted_label = predicted_label.topk(maxk, 1, True, True)
    predicted_label = predicted_label.t()
    correct = predicted_label.eq(label.view(1, -1).expand_as(predicted_label))

    result = []
    for i in topk:
        correct_i = correct[ : i].view(-1).float().sum(0)
        result += [correct_i / args.batch_size * 100.0]

    return result

def adjust_lr(optimizer, epoch, lr_step):
    decay = 0.1 ** (sum(epoch >= np.array(lr_step)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']

@torch.no_grad()
def validation():
    print('start validation...')
    loss_avg = AverageMeter()
    activity_loss_avg = AverageMeter()
    completeness_loss_avg = AverageMeter()
    regression_loss_avg = AverageMeter()
    activity_accuracy_avg = AverageMeter()
    fg_accuracy_avg = AverageMeter()
    bg_accuracy_avg = AverageMeter()

    ssn.eval()
    for i, (frame_out, label_out, regression_label_out, length_out, scale_out, stage_split_out, proposal_type_out) in enumerate(validationset_loader):
        frame_out = frame_out.cuda()
        label_out = label_out.cuda()
        regression_label_out = regression_label_out.cuda()
        scale_out = scale_out.cuda()
        proposal_type_out = proposal_type_out.cuda()
        activity_predicted_label, activity_label, completeness_predicted_label, completeness_label, regression_predicted_label, label, regression_label = ssn(x=frame_out, augmentation_scale=scale_out, label=label_out, regression_label=regression_label_out, proposal_type=proposal_type_out)
        activity_loss = activity_loss_function(activity_predicted_label, activity_label)
        completeness_loss = completeness_loss_function(completeness_predicted_label, completeness_label, trainset.fg_per_video, trainset.fg_per_video + trainset.incomplete_per_video)
        regression_loss = regression_loss_function(regression_predicted_label, label, regression_label)
        loss = activity_loss + args.completeness_loss_weight * completeness_loss + args.regression_loss_weight * regression_loss
        fg_accuracy = accuracy(activity_predicted_label.view(-1, 2, activity_predicted_label.size(1))[ : , 0, : ].contiguous(), activity_label.view(-1, 2)[ : , 0].contiguous())
        bg_accuracy = accuracy(activity_predicted_label.view(-1, 2, activity_predicted_label.size(1))[ : , 1, : ].contiguous(), activity_label.view(-1, 2)[ : , 1].contiguous())
        activity_accuracy = accuracy(activity_predicted_label, activity_label)
        activity_loss_avg.update(activity_loss.item(), args.batch_size)
        completeness_loss_avg.update(completeness_loss.item(), args.batch_size)
        regression_loss_avg.update(regression_loss.item(), args.batch_size)
        loss_avg.update(loss.item(), args.batch_size)
        fg_accuracy_avg.update(fg_accuracy[0].item(), activity_predicted_label.size(0) // 2)
        bg_accuracy_avg.update(bg_accuracy[0].item(), activity_predicted_label.size(0) // 2)
        activity_accuracy_avg.update(activity_accuracy[0].item(), activity_predicted_label.size(0))
    print('Validation result: Activity loss {:.3f} Completeness loss {:.3f} Regression loss {:.3f} Loss {:.3f} fg accuracy {:.3f} bg accuracy {:.3f} Activity accuracy {:3f}'.format(activity_loss_avg.avg, completeness_loss_avg.avg, regression_loss_avg.avg, loss_avg.avg, fg_accuracy_avg.avg, bg_accuracy_avg.avg, activity_accuracy_avg.avg))
    ssn.train()
    loss_ = loss_avg.avg
    accuracy_ = activity_accuracy_avg.avg

    return loss_, accuracy_

print('start training...')
for i in range(args.epoch):
    adjust_lr(optimizer, i, args.lr_step)

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    activity_loss_avg = AverageMeter()
    completeness_loss_avg = AverageMeter()
    regression_loss_avg = AverageMeter()
    activity_accuracy_avg = AverageMeter()
    fg_accuracy_avg = AverageMeter()
    bg_accuracy_avg = AverageMeter()

    end = time.time()
    for j, (frame_out, label_out, regression_label_out, length_out, scale_out, stage_split_out, proposal_type_out) in enumerate(trainset_loader):
        data_time.update(time.time() - end)

        ssn.zero_grad()
        frame_out = frame_out.cuda()
        label_out = label_out.cuda()
        regression_label_out = regression_label_out.cuda()
        scale_out = scale_out.cuda()
        proposal_type_out = proposal_type_out.cuda()
        activity_predicted_label, activity_label, completeness_predicted_label, completeness_label, regression_predicted_label, label, regression_label = ssn(x=frame_out, augmentation_scale=scale_out, label=label_out, regression_label=regression_label_out, proposal_type=proposal_type_out)
        activity_loss = activity_loss_function(activity_predicted_label, activity_label)
        completeness_loss = completeness_loss_function(completeness_predicted_label, completeness_label, trainset.fg_per_video, trainset.fg_per_video + trainset.incomplete_per_video)
        regression_loss = regression_loss_function(regression_predicted_label, label, regression_label)
        loss = activity_loss + args.completeness_loss_weight * completeness_loss + args.regression_loss_weight * regression_loss
        fg_accuracy = accuracy(activity_predicted_label.view(-1, 2, activity_predicted_label.size(1))[ : , 0, : ].contiguous(), activity_label.view(-1, 2)[ : , 0].contiguous())
        bg_accuracy = accuracy(activity_predicted_label.view(-1, 2, activity_predicted_label.size(1))[ : , 1, : ].contiguous(), activity_label.view(-1, 2)[ : , 1].contiguous())
        activity_accuracy = accuracy(activity_predicted_label, activity_label)
        activity_loss_avg.update(activity_loss.item(), args.batch_size)
        completeness_loss_avg.update(completeness_loss.item(), args.batch_size)
        regression_loss_avg.update(regression_loss.item(), args.batch_size)
        loss_avg.update(loss.item(), args.batch_size)
        fg_accuracy_avg.update(fg_accuracy[0].item(), activity_predicted_label.size(0) // 2)
        bg_accuracy_avg.update(bg_accuracy[0].item(), activity_predicted_label.size(0) // 2)
        activity_accuracy_avg.update(activity_accuracy[0].item(), activity_predicted_label.size(0))
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = nn.utils.clip_grad_norm_(ssn.parameters(), args.clip_gradient)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if j % args.print_every == 0:
            print('Epoch: [{}/{}][{}/{}], lr {:.5f} Data {:.3f} ({:.3f}) Batch {:.3f} ({:.3f}) Activity loss {:.3f} ({:.3f}) Completeness loss {:.3f} ({:.3f}) Regression loss {:.3f} ({:.3f}) Loss {:.3f} ({:.3f}) fg accuracy {:.3f} ({:.3f}) bg accuracy {:.3f} ({:.3f}) Activity accuracy {:.3f} ({:.3f})'.format(i, args.epoch, j, len(trainset_loader), optimizer.param_groups[0]['lr'], data_time.val, data_time.avg, batch_time.val, batch_time.avg, activity_loss_avg.val, activity_loss_avg.avg, completeness_loss_avg.val, completeness_loss_avg.avg, regression_loss_avg.val, regression_loss_avg.avg, loss_avg.val, loss_avg.avg, fg_accuracy_avg.val, fg_accuracy_avg.avg, bg_accuracy_avg.val, bg_accuracy_avg.avg, activity_accuracy_avg.val, activity_accuracy_avg.avg))

    if (i + 1) % args.validation_every == 0 or i == args.epoch - 1:
        loss_, accuracy_ = validation()
        trained_model_name = '{}_{}_{}_{:.3f}_{:.3f}.pth'.format(args.base_model, args.modality, i, loss_, accuracy_)
        trained_model_path = 'trained_model/' + trained_model_name
        torch.save(ssn.state_dict(), trained_model_path)