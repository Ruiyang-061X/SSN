import os
import math
from itertools import groupby
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_video_record_list(video_record_list_path):
    line = open(video_record_list_path).readlines()
    groups = groupby(line, lambda i : i.startswith('#'))
    groups = [[k.strip() for k in list(j)] for i, j in groups if not i]

    def parse_group(group):
        offset = 0
        video_name = group[offset]
        offset += 1
        n_frame = int(float(group[1]) * float(group[2]))
        n_gt = int(group[3])
        offset = 4
        gt = [i.split() for i in group[offset : offset + n_gt]]
        offset += n_gt
        n_proposal = int(group[offset])
        offset += 1
        proposal = [i.split() for i in group[offset : offset + n_proposal]]

        return video_name, n_frame, gt, proposal

    return [parse_group(i) for i in groups]

def temporal_iou(span_A, span_B):
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    intersection = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])
    if intersection[0] > intersection[1]:

        return 0
    else:

        return float(intersection[1] - intersection[0]) / float(union[1] - union[0])

class SSNInstance:

    def __init__(self, start_frame, end_frame, n_frame, fps=1, label=None, best_iou=None, overlap=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, n_frame)
        self.fps = fps
        self.label = label
        self.best_iou = best_iou
        self.overlap = overlap
        self.coverage = (end_frame - start_frame) / n_frame
        self.start_time = start_frame / fps
        self.end_time = end_frame / fps

    def compute_regression_label(self, gt):
        ious = [temporal_iou((self.start_frame, self.end_frame), (i.start_frame, i.end_frame)) for i in gt]
        best_gt_indice = np.argmax(ious)
        best_gt = gt[best_gt_indice]
        proposal_center = (self.start_frame + self.end_frame) / 2
        best_gt_center = (best_gt.start_frame + best_gt.end_frame) / 2
        proposal_length = self.end_frame - self.start_frame + 1
        best_gt_length = best_gt.end_frame - best_gt.start_frame + 1
        location_regression_label = (best_gt_center - proposal_center) / proposal_length
        length_regression_label = math.log(best_gt_length / proposal_length)
        self.regression_label = [location_regression_label, length_regression_label]

class SSNVideoRecord:

    def __init__(self, video_record):
        self.path = video_record[0]
        self.n_frame = int(video_record[1])
        self.gt = [SSNInstance(start_frame=int(i[1]), end_frame=int(i[2]), n_frame=self.n_frame, label=int(i[0]), best_iou=1.0) for i in video_record[2] if int(i[2]) > int(i[1])]
        self.gt = list(filter(lambda i : i.start_frame < self.n_frame, self.gt))
        self.proposal = [SSNInstance(start_frame=int(i[3]), end_frame=int(i[4]), n_frame=self.n_frame, label=int(i[0]), best_iou=float(i[1]), overlap=float(i[2])) for i in video_record[3] if int(i[4]) > int(i[3])]
        self.proposal = list(filter(lambda i : i.start_frame < self.n_frame, self.proposal))

    def get_fg(self, fg_iou_thresh, with_gt=True):
        fg = [i for i in self.proposal if i.best_iou > fg_iou_thresh]
        if with_gt:
            fg += self.gt
        for i in fg:
            i.compute_regression_label(gt=self.gt)

        return fg

    def get_negative(self, incomplete_iou_thresh, bg_iou_thresh, incomplete_overlap_thresh=0.7, bg_coverage_thresh=0.01):
        incomplete_proposal = []
        bg_proposal = []
        for i in range(len(self.proposal)):
            if self.proposal[i].best_iou < incomplete_iou_thresh and self.proposal[i].overlap > incomplete_overlap_thresh:
                incomplete_proposal += [self.proposal[i]]
            elif self.proposal[i].best_iou < bg_iou_thresh and self.proposal[i].coverage > bg_coverage_thresh:
                bg_proposal += [self.proposal[i]]

        return incomplete_proposal, bg_proposal

class SSNDataset(Dataset):

    def __init__(self, video_record_list_path=None, n_body_segment=5, n_augmentation_segment=2, new_length=1, modality='RGB', transform=None, random_shift=True, test_mode=False, proposal_per_video=8, fg_ratio=1, bg_ratio=1, incomplete_ratio=6, fg_iou_thresh=0.7, bg_iou_thresh=0.01, incomplete_iou_thresh=0.3, bg_coverage_thresh=0.02, incomplete_overlap_thresh=0.7, with_gt=True, regression_constant=None, test_interval=6):
        self.video_record_list_path = video_record_list_path
        self.n_body_segment = n_body_segment
        self.n_augmentation_segment = n_augmentation_segment
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.proposal_per_video = proposal_per_video
        self.fg_ratio = fg_ratio
        self.incomplete_ratio = incomplete_ratio
        self.bg_ratio = bg_ratio
        self.fg_iou_thresh = fg_iou_thresh
        self.incomplete_iou_thresh = incomplete_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.incomplete_overlap_thresh = incomplete_overlap_thresh
        self.bg_coverage_thresh = bg_coverage_thresh
        self.with_gt = with_gt
        self.regression_constant = regression_constant
        self.test_interval = test_interval
        self.starting_ratio = 0.5
        self.ending_ratio = 0.5
        total = fg_ratio + bg_ratio + incomplete_ratio
        self.fg_per_video = int(proposal_per_video * (fg_ratio / total))
        self.bg_per_video = int(proposal_per_video * (bg_ratio / total))
        self.incomplete_per_video = proposal_per_video - self.fg_per_video - self.bg_per_video
        self._parse_video_record_list(regression_constant=regression_constant)

    def _load_image(self, path, indice):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':

            return [Image.open(os.path.join(path, 'img_{:05d}.jpg'.format(indice))).convert('RGB')]
        elif self.modality == 'Flow':
            flow_x = Image.open(os.path.join(path, 'flow_x_{:05d}.jpg'.format(indice))).convert('L')
            flow_y = Image.open(os.path.join(path, 'flow_y_{:05d}.jpg'.format(indice))).convert('L')

            return [flow_x, flow_y]

    def _compute_regression_constant(self):
        regression_labels = []
        for i in self.video_record_list:
            fg = i.get_fg(fg_iou_thresh=self.fg_iou_thresh, with_gt=False)
            for j in fg:
                regression_labels += [list(j.regression_label)]
        self.regression_constant = np.array([np.mean(regression_labels, axis=0), np.std(regression_labels, axis=0)])

    def _parse_video_record_list(self, regression_constant=None):
        video_record_list = load_video_record_list(video_record_list_path=self.video_record_list_path)
        self.video_record_list = [SSNVideoRecord(i) for i in video_record_list]
        self.video_record_list = list(filter(lambda i : len(i.gt) > 0, self.video_record_list))
        self.video_record_dict = {i.path : i for i in self.video_record_list}
        self.fg_pool = []
        self.incomplete_pool = []
        self.bg_pool = []
        for i in self.video_record_list:
            fg = i.get_fg(fg_iou_thresh=self.fg_iou_thresh, with_gt=self.with_gt)
            self.fg_pool += [(i.path, j) for j in fg]
            incomplete, bg = i.get_negative(incomplete_iou_thresh=self.incomplete_iou_thresh, bg_iou_thresh=self.bg_iou_thresh, incomplete_overlap_thresh=self.incomplete_overlap_thresh, bg_coverage_thresh=self.bg_coverage_thresh)
            self.incomplete_pool += [(i.path, j) for j in incomplete]
            self.bg_pool += [(i.path, j) for j in bg]
        if regression_constant is None:
            self._compute_regression_constant()
        else:
            self.regression_constant = regression_constant

        print('''
        SSNDataset: proposal file {} parsed
        
        There are {} usable proposals from {} videos
        {} fg proposals
        {} incomplete proposals
        {} bg proposals
        
        Sampling config:
        fg: {}
        incomplete: {}
        bg: {}
                    
        Regression constant:
        location: mean {:.4f} std {:.4f}   
        duration: mean {:.4f} std {:.4f}
        '''.format(self.video_record_list_path, len(self.fg_pool) + len(self.incomplete_pool) + len(self.bg_pool), len(self.video_record_list), len(self.fg_pool), len(self.incomplete_pool), len(self.bg_pool), self.fg_per_video, self.incomplete_per_video, self.bg_per_video, self.regression_constant[0][0], self.regression_constant[0][1], self.regression_constant[1][0], self.regression_constant[1][1]))

    def _video_centric_sample(self, video_record):
        fg = video_record.get_fg(fg_iou_thresh=self.fg_iou_thresh, with_gt=self.with_gt)
        incomplete, bg = video_record.get_negative(self.incomplete_iou_thresh, self.bg_iou_thresh, self.incomplete_overlap_thresh, self.bg_coverage_thresh)

        def sample_proposal(proposal_type, path, proposal, n_requested, pool):
            if len(proposal) == 0:
                indice = np.random.choice(len(pool), n_requested, replace=False)

                return [(pool[i], proposal_type) for i in indice]
            else:
                indice = np.random.choice(len(proposal), n_requested, replace=len(proposal) < n_requested)

                return [((path, proposal[i]), proposal_type) for i in indice]

        proposal_out = []
        proposal_out += sample_proposal(proposal_type=0, path=video_record.path, proposal=fg, n_requested=self.fg_per_video, pool=self.fg_pool)
        proposal_out += sample_proposal(proposal_type=1, path=video_record.path, proposal=incomplete, n_requested=self.incomplete_per_video, pool=self.incomplete_pool)
        proposal_out += sample_proposal(proposal_type=2, path=video_record.path, proposal=bg, n_requested=self.bg_per_video, pool= self.bg_pool)

        return proposal_out

    def _get_random_shift_segment_indice(self, valid_length, n_segment):
        average_length = (valid_length + 1) // n_segment
        if average_length > 0:
            indice = np.multiply(range(n_segment), average_length) + np.random.randint(average_length, size=n_segment)
        elif valid_length > n_segment:
            indice = np.sort(np.random.randint(valid_length, size=n_segment))
        else:
            indice = np.zeros((n_segment, ))

        return indice

    def _get_validation_segment_indice(self, valid_length, n_segment):
        if valid_length > n_segment:
            stride = valid_length / float(n_segment)
            indice = np.array([int(i * stride + stride / 2.0) for i in range(n_segment)])
        else:
            indice = np.zeros((n_segment, ))

        return indice

    def _sample_ssn_indice(self, proposal, n_frame):
        start_frame = proposal.start_frame + 1
        end_frame = proposal.end_frame
        length = end_frame - start_frame + 1
        valid_length = length - self.new_length
        valid_start = max(1, start_frame - int(length * self.starting_ratio))
        valid_end = min(n_frame - self.new_length + 1, end_frame + int(length * self.ending_ratio))
        valid_start_length = start_frame - valid_start - self.new_length + 1
        valid_end_length = valid_end - end_frame - self.new_length + 1
        start_scale = (valid_start_length + self.new_length - 1) / (length * self.starting_ratio)
        end_scale = (valid_end_length + self.new_length - 1) / (length * self.ending_ratio)
        start_indice = (self._get_random_shift_segment_indice(valid_length=valid_start_length, n_segment=self.n_augmentation_segment) if self.random_shift else self._get_validation_segment_indice(valid_length=valid_start_length, n_segment=self.n_augmentation_segment)) + valid_start
        course_indice = (self._get_random_shift_segment_indice(valid_length=valid_length, n_segment=self.n_body_segment) if self.random_shift else self._get_validation_segment_indice(valid_length=valid_length, n_segment=self.n_body_segment)) + start_frame
        end_indice = (self._get_random_shift_segment_indice(valid_length=valid_end_length, n_segment=self.n_augmentation_segment) if self.random_shift else self._get_validation_segment_indice(valid_length=valid_length, n_segment=self.n_augmentation_segment)) + end_frame
        indice = np.concatenate((start_indice, course_indice, end_indice))
        stage_split = [self.n_augmentation_segment, self.n_augmentation_segment + self.n_body_segment, 2 * self.n_augmentation_segment + self.n_body_segment]

        return indice, start_scale, end_scale, stage_split

    def _load_proposal(self, proposal):
        n_frame = self.video_record_dict[proposal[0][0]].n_frame
        indice, start_scale, end_scale, stage_split = self._sample_ssn_indice(proposal=proposal[0][1], n_frame=n_frame)
        if proposal[1] == 0:
            label = proposal[0][1].label
        if proposal[1] == 1:
            label = proposal[0][1].label
        if proposal[1] == 2:
            label = 0
        frame = []
        for i, j in enumerate(indice):
            j = int(j)
            for k in range(self.new_length):
                frame += self._load_image(path=proposal[0][0], indice=min(j + k, n_frame))
        if proposal[1] == 0:
            regression_label = proposal[0][1].regression_label
            regression_label = ((regression_label[0] - self.regression_constant[0][0]) / self.regression_constant[1][0], (regression_label[1] - self.regression_constant[0][1]) / self.regression_constant[1][1])
        else:
            regression_label = (0.0, 0.0)

        return frame, label, regression_label, start_scale, end_scale, stage_split, proposal[1]

    def get_test_data(self, video_record, test_interval, batch_size=4):
        path = video_record.path
        n_frame = video_record.n_frame
        proposal = video_record.proposal
        frame_indice = np.arange(0, n_frame - self.new_length, test_interval, dtype=np.int) + 1
        n_frame_sampled = len(frame_indice)
        if len(proposal) == 0:
            proposal += [SSNInstance(start_frame=0, end_frame=n_frame - 1, n_frame=n_frame, label=-1)]
        rel_proposal_out = []
        proposal_indice_out = []
        scale_out = []
        for i in proposal:
            rel_proposal = (i.start_frame / n_frame, i.end_frame / n_frame)
            length = rel_proposal[1] - rel_proposal[0]
            starting_length = length * self.starting_ratio
            ending_length = length * self.ending_ratio
            start = rel_proposal[0] - starting_length
            end = rel_proposal[1] + ending_length
            start = max(0.0, start)
            end = min(1.0, end)
            proposal_indice = (int(start * n_frame_sampled), int(rel_proposal[0] * n_frame_sampled), int(rel_proposal[1] * n_frame_sampled), int(end * n_frame_sampled))
            starting_scale = (rel_proposal[0] - start) / starting_length
            ending_scale = (end - rel_proposal[1]) / ending_length
            scale = (starting_scale, ending_scale)
            rel_proposal_out += [rel_proposal]
            proposal_indice_out += [proposal_indice]
            scale_out += [scale]

        def frame_generate(batch_size):
            frame = []
            count = 0
            for i, j in enumerate(frame_indice):
                j = int(j)
                for k in range(self.new_length):
                    frame += [self._load_image(path, min(j + k, n_frame))]
                count += 1
                if count % batch_size == 0:
                    frame = self.transform(frame)
                    yield frame
                    frame = []
            if len(frame):
                frame = self.transform(frame)
                yield frame

        return frame_generate(batch_size), n_frame_sampled, torch.from_numpy(np.array(rel_proposal_out)), torch.from_numpy(np.array(proposal_indice_out)), torch.from_numpy(np.array(scale_out))

    def get_train_data(self, indice):
        video_record = self.video_record_list[indice]
        proposal = self._video_centric_sample(video_record=video_record)
        frame_out = []
        label_out = []
        regression_label_out = []
        length_out = []
        scale_out = []
        stage_split_out = []
        proposal_type_out = []
        for i, j in enumerate(proposal):
            frame, label, regression_label, start_scale, end_scale, stage_split, proposal_type = self._load_proposal(j)
            frame = self.transform(frame)
            frame_out += [frame]
            label_out += [label]
            regression_label_out += [regression_label]
            length_out += [2 * self.n_augmentation_segment + self.n_body_segment]
            scale_out += [[start_scale, end_scale]]
            stage_split_out += [stage_split]
            proposal_type_out += [proposal_type]
        frame_out = torch.cat(frame_out)
        label_out = torch.from_numpy(np.array(label_out))
        regression_label_out = torch.from_numpy(np.array(regression_label_out, dtype=np.float32))
        length_out = torch.from_numpy(np.array(length_out))
        scale_out = torch.from_numpy(np.array(scale_out, dtype=np.float32))
        stage_split_out = torch.from_numpy(np.array(stage_split_out))
        proposal_type_out = torch.from_numpy(np.array(proposal_type_out))

        return frame_out, label_out, regression_label_out, length_out, scale_out, stage_split_out, proposal_type_out

    def get_gt_pool(self):
        gt_pool = []
        for i in self.video_record_list:
            path = i.path
            gt_pool += [[path, j.label - 1, j.start_frame / i.n_frame, j.end_frame / i.n_frame] for j in i.gt]

        return gt_pool

    def __getitem__(self, indice):
        indice = indice % len(self.video_record_list)
        if self.test_mode:

            return self.get_test_data(video_record=self.video_record_list[indice], test_interval=self.test_interval)
        else:

            return self.get_train_data(indice=indice)

    def __len__(self):

        return len(self.video_record_list) * 10