import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone

import numpy as np
import time


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class Scale_selection(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3)):
        super(Scale_selection, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 4, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)
        # self.batchnorm = nn.BatchNorm2d(256)


    def __make_weight(self,feature,scale_feature):
        weight_feature = feature * scale_feature
        return weight_feature

    def _make_scale(self, features, size):
        conv = nn.Conv2d(features, features, kernel_size=3, padding=size, bias=False, dilation=size)
        return conv

    def forward(self, feats):
        multi_scales = [stage(feats) for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = torch.cat((multi_scales[0], multi_scales[1], multi_scales[2]), 1)
        overall_weight = self.softmax(torch.cat((weights[0], weights[1], weights[2]), 1))
        # overall_features = torch.cat((multi_scales[0], multi_scales[1], multi_scales[2], multi_scales[3]), 1)
        # overall_weight = self.softmax(torch.cat((weights[0], weights[1], weights[2], weights[3]), 1))
        output_features = overall_features * overall_weight
        bottles = self.bottleneck(torch.cat((output_features, feats), 1))
        return self.relu(bottles)


class Attention_Module(nn.Module):
    def __init__(self, kernel_size=1):
        super(Attention_Module, self).__init__()

        self.convc = nn.Conv2d(512, 1, kernel_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convc(x)
        return self.sigmoid(x)


# the defenition of the P2PNet model
class Network(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.context = Scale_selection(512, 512)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.backend_feat = ['U', 512, 512, 'U', 256, 'U', 128, 'U', 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.att = Attention_Module()

    def forward(self, samples: NestedTensor):

        features = self.backbone(samples)
        features_ssn = self.context(features[3])
        features_att = self.att(features)
        cnn_out = (features_ssn * features_att) + features
        features_de = self.backend(cnn_out)
        out = self.output_layer(features_de)

        return out

class SetCriterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.den_loss = nn.MSELoss(size_average=False)

    def forward(self, outputs, targets):
        losses = 0
        output = outputs.squeeze()
        batch_size = output.size(0)
        for i in range(batch_size):
            loss = self.loss_density(output[i], targets[i])
            losses += loss
        # loss_den = losses/(batch_size)
        loss_den = losses / (2 * batch_size)
        return loss_den



# create the P2PNet model
def build_model(args, training):

    backbone = build_backbone(args)
    model = Network(backbone, args.row, args.line)
    if not training:
        return model

    criterion = SetCriterion()
    return model, criterion