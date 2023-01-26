import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import opt
import math
import torch.nn.utils.weight_norm as weightNorm
import numbers
from einops import rearrange
from torchvision import models

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:{}'.format(opt['gpu'])) if use_cuda else torch.device('cpu')

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNetEncoder(nn.Module):
    def __init__(self, output_dim=256, type='org'):
        super(LeNetEncoder, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )

        self.in_features = 50*4*4
        self.output_dim = output_dim
        self.type = type

        self.bottleneck = nn.Linear(self.in_features, output_dim)
        self.bottleneck.apply(init_weights)
        self.bn = nn.BatchNorm1d(output_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class DTNEncoder(nn.Module):
    def __init__(self, output_dim=256, hidden_feature_dim=256*4*4, type='org'):
        super(DTNEncoder, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
        self.in_features = hidden_feature_dim

        self.output_dim = output_dim
        self.type = type

        self.bottleneck = nn.Linear(self.in_features, output_dim)
        self.bottleneck.apply(init_weights)
        self.bn = nn.BatchNorm1d(output_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

''' 
class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x
'''
class feat_classifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=10, type="linear"):
        super(feat_classifier, self).__init__()
        self.dim = feature_dim
        if type == "linear":
            self.fc = nn.Linear(feature_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(feature_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def init(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        x = self.fc(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):

    def __init__(self, image_channels):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 512
        # self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        # self.fc = model_resnet.fc
        self.output_dim = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # y = self.fc(x)
        return x


class WeightedResidualUnit(nn.Module):
    def __init__(self, dim, num=2, type='bn'):
        super(WeightedResidualUnit, self).__init__()
        self.fc = nn.Linear(dim * num, dim)
        self.w = nn.Parameter(torch.ones(num+1))
        self.weight_init()
        self.num = num
        self.bn = nn.BatchNorm1d(dim, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)


    def weight_init(self):
        n, w = self.fc.weight.size()
        bias = torch.zeros(n).to(device)
        weights = torch.zeros(n, w).to(device)
        weights[:, :n] = torch.eye(n, n).to(device)*0.5
        weights[:, n:] = torch.eye(n, n).to(device)*0.5
        self.fc.weight = torch.nn.Parameter(weights)
        self.fc.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        w = [torch.exp(self.w[i]) / torch.sum(torch.exp(self.w)) for i in range(len(self.w))]
        x_merge = self.fc(torch.cat(x, 1))
        out = x_merge*w[self.num]

        for i in range(self.num):
            out += x[i] * w[i]

        if self.type == "bn":
            out = self.bn(out)
            out = self.dropout(out)
            out = self.relu(out)

        return out



class Cosine_feature_merge_with_skip_connection(nn.Module):
    def __init__(self, num_models=8, input_features=64, type='bn'):
        super(Cosine_feature_merge_with_skip_connection, self).__init__()
        # self.layer_number = layers
        self.type = type
        self.layers = self.get_layer_node_number(num_models)
        self.layer_number = len(self.layers)

        self.fcs = {}
        for l, n in self.layers.items():
            self.fcs[l] = []
            [self.fcs[l].append(WeightedResidualUnit(input_features, num=2, type=type)) for i in range(n)]

        self.groups = {}
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_layer_node_number(self, num_models=8):
        layers={}
        n = num_models
        l = 0
        while n>1:
            n = math.ceil(n/2)
            # n = math.floor(n / 2)
            layers[l] = n
            l += 1
        return layers


    def forward(self, input):
        output = input.copy()
        if len(self.groups.keys()) == 0:
            for l in range(self.layer_number):
                pairs, output = self.group(output, l)
                self.groups[l] = pairs

        output = input.copy()
        dist_value = 0
        all_features = []
        for l in range(self.layer_number):
            output, dist = self.merge_simple(output, l)
            all_features.extend(output)
            if l > 0:
                dist_value += dist
        all_features[-1] = all_features[-1] + input[-1]
        return dist_value, all_features

    def group(self, input, layer):
        d = len(input)
        # M = torch.zeros(d, d)
        M = torch.ones(d, d)
        for i in range(len(input)):
            for j in range(i+1, len(input)):
                simi = self.cos(input[i], input[j])
                # dist = (1-simi)/2.0
                dist = 1-simi**2
                dist = torch.mean(dist)
                M[i, j] = dist
                M[j, i] = dist

        pairs = []
        merged_ids=set()
        for p in range(int(d/2)):
            # idx = torch.argmax(M)
            idx = torch.argmin(M)
            i = int(idx/d)
            j = int(idx%d)
            pairs.append((i, j))
            merged_ids.add(i)
            merged_ids.add(j)
            M[i,:] = 1.0
            M[:,j] = 1.0
            M[j,:] = 1.0
            M[:,i] = 1.0
        all_ids = {i for i in range(d)}
        single_node = list(all_ids - merged_ids)
        if len(single_node) > 0:
            pairs.append((single_node[0], single_node[0]))


        outputs = []
        for id in range(len(pairs)):
            i, j = pairs[id]
            if i==j:
                out = input[i]
            else:
                out = self.fcs[layer][id]([input[i], input[j]])
            outputs.append(out)

        return pairs, outputs

    def merge_simple(self, input, layer):
        dist_value = torch.zeros(1).to(device)
        outputs = []
        for i in range(len(self.groups[layer])):
            f1_id, f2_id = self.groups[layer][i]
            simi = self.cos(input[f1_id], input[f2_id])
            # dist = (1 - simi) / 2.0
            dist = (1 - simi**2)
            dist = torch.mean(dist)
            dist_value += dist   # similarity: 相同时为1，正交时为0，相反时为-1
            if f1_id == f2_id:
                out = input[f1_id]
            else:
                out = self.fcs[layer][i]([input[f1_id], input[f2_id]])

            outputs.append(out)

        return outputs, dist_value
