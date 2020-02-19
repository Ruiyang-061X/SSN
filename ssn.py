import torch
from torch import nn
import torchvision
import model_zoo

class Identity(nn.Module):

    def forward(self, x):

        return x

class StructuredTemporalPyramidPooling(nn.Module):

    def __init__(self, feature_dim, stand_along_classifier=False, stpp_cfg=(1, (1, 2), 1)):
        super(StructuredTemporalPyramidPooling, self).__init__()
        self.feature_dim = feature_dim
        self.stand_along_classifier = stand_along_classifier
        self.stpp_cfg = stpp_cfg
        starting_part, starting_multiplier = self._parse_stpp_cfg(stpp_cfg[0])
        course_part, course_multiplier = self._parse_stpp_cfg(stpp_cfg[1])
        ending_part, ending_multiplier = self._parse_stpp_cfg(stpp_cfg[2])
        self.part = (starting_part, course_part, ending_part)
        self.multiplier = (starting_multiplier, course_multiplier, ending_multiplier)
        self.feature_multiplier = starting_multiplier + course_multiplier + ending_multiplier
        if self.stand_along_classifier:
            self.activity_in_feature = feature_dim
        else:
            self.activity_in_feature = feature_dim * self.feature_multiplier
        self.completeness_in_feature = feature_dim * self.feature_multiplier

    def _parse_stpp_cfg(self, stpp_cfg):
        if isinstance(stpp_cfg, int):

            return (stpp_cfg, ), stpp_cfg
        if isinstance(stpp_cfg, tuple):

            return stpp_cfg, sum(stpp_cfg)

    def forward(self, x, scale, segment_split):
        x1 = segment_split[0]
        x2 = segment_split[1]
        n_segment = segment_split[2]
        x_dim = x.size()[1]
        x = x.view(-1, n_segment, x_dim)
        n_sample = x.size()[0]
        scale = scale.view(-1, 2)

        def get_stage_stpp(x, part, multiplier, scale):
            stage_stpp = []
            stage_length = x.size(1)
            for i in part:
                indice = torch.arange(0, stage_length + 1e-5, stage_length / i)
                for j in range(i):
                    part_x = x[ : , int(indice[j]) : int(indice[j + 1]), : ].mean(dim=1) / multiplier
                    if scale is not None:
                        part_x = part_x * scale.resize(n_sample, 1)
                    stage_stpp += [part_x]

            return stage_stpp

        feature_part = []
        feature_part += get_stage_stpp(x[ : , : x1, : ], self.part[0], self.multiplier[0], scale[ : , 0])
        feature_part += get_stage_stpp(x[ : , x1 : x2, : ], self.part[1], self.multiplier[1], None)
        feature_part += get_stage_stpp(x[ : , x2 : , : ], self.part[2], self.multiplier[2], scale[ : , 1])
        stpp_x = torch.cat(feature_part, dim=1)
        if not self.stand_along_classifier:

            return stpp_x, stpp_x

        course_x = x[ : , x1 : x2, : ].mean(dim=1)

        return course_x, stpp_x

class SSN(nn.Module):

    def __init__(self, base_model='BNInception', n_class=20, dropout=0.8, n_crop=1, stpp_cfg=(1, (1, 2), 1), bn_mode='frozen', with_regression=True, modality='RGB', n_body_segment=5, n_augmentation_segment=2, new_length=1, test_mode=False):
        super(SSN, self).__init__()
        self.base_model = base_model
        self.n_class = n_class
        self.dropout = dropout
        self.n_crop = n_crop
        self.stpp_cfg = stpp_cfg
        self.bn_mode = bn_mode
        self.with_regression = with_regression
        self.modality = modality
        self.n_body_segment = n_body_segment
        self.n_augmentation_segment = n_augmentation_segment
        self.new_length = new_length
        self.test_mode = test_mode
        self.n_segment = n_body_segment + 2 * n_augmentation_segment

        print('''
        Initializing SSN with base model: {}
        SSN Configuration:
            modality:               {}
            n_body_segment:         {}
            n_augmentation_segment: {}
            n_segment:              {}
            new_length:             {}
            dropout:                {}
            with_regression:        {}
            bn_mode:                {}
            stpp_cfg:               {}
        '''.format(self.base_model, self.modality, self.n_body_segment, self.n_augmentation_segment, self.n_segment, self.new_length, self.dropout, self.with_regression, self.bn_mode, self.stpp_cfg))

        self._prepare_bn()
        self._prepare_base_model()
        self._prepare_ssn()
        if self.modality == 'RGBDiff':
            print('Converting the imagenet model to RGBDiff init model')
            self._construct_rgbdiff_model()
            print('Done')
        if self.modality == 'Flow':
            print('Converting the imagenet model to flow init model')
            self._construct_flow_model()
            print('Done')
        if self.test_mode:
            self._prepare_test_linear()

    def _prepare_bn(self):
        if self.bn_mode == 'partial':
            self.freeze_count = 2
        if self.bn_mode == 'frozen':
            self.freeze_count = 1
        if self.bn_mode == 'full':
            self.freeze_count = None

    def _prepare_base_model(self):
        if 'vgg' in self.base_model or 'resnet' in self.base_model:
            self.base_model = getattr(torchvision.models, self.base_model)(True)
            self.base_model.last_layer_name = 'fc'
            return
        if self.base_model == 'BNInception':
            self.base_model = getattr(model_zoo, self.base_model)()
            self.base_model.last_layer_name = 'fc'
            return
        if self.base_model == 'InceptionV3':
            self.base_model = getattr(model_zoo, self.base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
        if 'inception' in self.base_model:
            self.base_model = getattr(model_zoo, self.base_model)()
            self.base_model.last_layer_name = 'classif'
            return

    def _prepare_ssn(self):
        in_feature = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.stpp = StructuredTemporalPyramidPooling(in_feature, True, self.stpp_cfg)
        self.activity_linear = nn.Linear(in_features=self.stpp.activity_in_feature, out_features=self.n_class + 1)
        self.completeness_linear = nn.Linear(in_features=self.stpp.completeness_in_feature, out_features=self.n_class)
        nn.init.normal_(self.activity_linear.weight.data, 0, 0.001)
        nn.init.constant_(self.activity_linear.bias.data, 0)
        nn.init.normal_(self.completeness_linear.weight.data, 0, 0.001)
        nn.init.constant_(self.completeness_linear.bias.data, 0)
        if self.with_regression:
            self.regression_linear = nn.Linear(in_features=self.stpp.completeness_in_feature, out_features=2 * self.n_class)
            nn.init.normal_(self.regression_linear.weight.data, 0, 0.001)
            nn.init.constant_(self.regression_linear.bias.data, 0)

    def _construct_rgbdiff_model(self):
        modules = list(self.base_model.modules())
        first_conv_index = list(filter(lambda i: isinstance(modules[i], nn.Conv2d), range(len(modules))))[0]
        conv_layer = modules[first_conv_index]
        container = modules[first_conv_index - 1]

        ps = [i.clone() for i in conv_layer.parameters()]
        kernel_size = ps[0].size()
        new_kernel_size=  kernel_size[ : 1] + (3 * self.new_length, ) + kernel_size[2 : ]
        new_kernel = ps[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv_layer = nn.Conv2d(in_channels=new_kernel_size[1], out_channels=conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=True if len(ps) == 2 else False)
        new_conv_layer.weight.data = new_kernel
        if len(ps) == 2:
            new_conv_layer.bias.data = ps[1].data
        layer_name = list(container.state_dict())[0][ : -7]
        setattr(container, layer_name, new_conv_layer)

    def _construct_flow_model(self):
        modules = list(self.base_model.modules())
        first_conv_index = list(filter(lambda i: isinstance(modules[i], nn.Conv2d), range(len(modules))))[0]
        conv_layer = modules[first_conv_index]
        container = modules[first_conv_index - 1]

        ps = [i.clone() for i in conv_layer.parameters()]
        kernel_size = ps[0].size()
        new_kernel_size = kernel_size[: 1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernel = ps[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv_layer = nn.Conv2d(in_channels=2 * self.new_length, out_channels=conv_layer.out_channels, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding, bias=True if len(ps) == 2 else False)
        new_conv_layer.weight.data = new_kernel
        if len(ps) == 2:
            new_conv_layer.bias.data = ps[1].data
        layer_name = list(container.state_dict())[0][: -7]
        setattr(container, layer_name, new_conv_layer)

    def _prepare_test_linear(self):
        self.test_linear = nn.Linear(in_features=self.activity_linear.in_features, out_features=self.activity_linear.out_features + self.completeness_linear.out_features * self.stpp.feature_multiplier + (self.regression_linear.out_features * self.stpp.feature_multiplier if self.with_regression else 0))
        completeness_linear_weight_resized = self.completeness_linear.weight.data.view(self.completeness_linear.out_features, self.stpp.feature_multiplier, self.activity_linear.in_features).transpose(0, 1).contiguous().view(-1, self.activity_linear.in_features)
        completeness_linear_bias_resized = self.completeness_linear.bias.view(1, -1).expand(self.stpp.feature_multiplier, self.completeness_linear.out_features).contiguous().view(-1) / self.stpp.feature_multiplier
        weight = torch.cat((self.activity_linear.weight.data, completeness_linear_weight_resized))
        bias = torch.cat((self.activity_linear.bias.data, completeness_linear_bias_resized))
        if self.with_regression:
            regression_linear_weight_resized = self.regression_linear.weight.data.view(self.regression_linear.out_features, self.stpp.feature_multiplier, self.activity_linear.in_features).transpose(0, 1).contiguous().view(-1, self.activity_linear.in_features)
            regression_linear_bias_resized = self.regression_linear.bias.view(1, -1).expand(self.stpp.feature_multiplier, self.regression_linear.out_features).contiguous().view(-1) / self.stpp.feature_multiplier
            weight = torch.cat((weight, regression_linear_weight_resized))
            bias = torch.cat((bias, regression_linear_bias_resized))
        self.test_linear.weight.data = weight
        self.test_linear.bias.data = bias

    def train(self, mode=True):
        super(SSN, self).train(mode=mode)
        if self.freeze_count is None:
            return
        count = 0
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= self.freeze_count:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def _get_diff(self, x):
        x = x.view((-1, self.n_segment, self.new_length + 1, 3) + x.size()[2 : ])
        new_x = x[ : , : , 1 : , : , : , : ].clone()
        for i in reversed(range(1, self.new_length + 1, 1)):
            new_x[ : , : , i - 1, : , : , : ] = x[ : , : , i , : , : , : ] - x[ : , : , i - 1 , : , : , : ]

        return new_x

    def train_forward(self, x, augmentation_scale, label, regression_label, proposal_type):
        if self.modality in ['RGB', 'RGBDiff']:
            sample_length = 3 * self.new_length
        if self.modality == 'Flow':
            sample_length = 2 * self.new_length
        if self.modality == 'RGBDiff':
            x = self._get_diff(x)
        x = self.base_model(x.view((-1, sample_length) + x.size()[-2 : ]))
        activity_x, completeness_x = self.stpp(x, augmentation_scale, [self.n_augmentation_segment, self.n_augmentation_segment + self.n_body_segment, 2 * self.n_augmentation_segment + self.n_body_segment])
        activity_x = self.activity_linear(activity_x)
        completeness_x_ = completeness_x
        completeness_x = self.completeness_linear(completeness_x)
        proposal_type = proposal_type.view(-1).data
        activity_indice = ((proposal_type == 0) + (proposal_type == 2)).nonzero().squeeze()
        completeness_indice = ((proposal_type == 0) + (proposal_type == 1)).nonzero().squeeze()
        label = label.view(-1)
        if self.with_regression:
            regression_x = self.regression_linear(completeness_x_).view(-1, self.completeness_linear.out_features, 2)
            regression_indice = (proposal_type == 0).nonzero().squeeze()
            regression_label = regression_label.view(-1, 2)

            return activity_x[activity_indice, : ], label[activity_indice], completeness_x[completeness_indice, : ], label[completeness_indice], regression_x[regression_indice, : ], label[regression_indice], regression_label[regression_indice]

        return activity_x[activity_indice, : ], label[activity_indice], completeness_x[completeness_indice, : ], label[completeness_indice]

    def test_forward(self, x):
        if self.modality in ['RGB', 'RGBDiff']:
            sample_length = 3 * self.new_length
        if self.modality == 'Flow':
            sample_length = 2 * self.new_length
        if self.modality == 'RGBDiff':
            x = self._get_diff(x)
        x = self.base_model(x.view((-1, sample_length) + x.size()[-2 : ]))
        test_x = self.test_linear(x)

        return test_x, x

    def forward(self, x, augmentation_scale, label, regression_label, proposal_type):
        if not self.test_mode:

            return self.train_forward(x, augmentation_scale, label, regression_label, proposal_type)

        return self.test_forward(x)