from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import torch.nn.functional as F




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def cosine_distance(x, y):
    if x.ndim == 1:
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
    elif x.ndim == 2:
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    np.seterr(divide='ignore', invalid='ignore')
    s = np.dot(x, y.T)/(x_norm*y_norm)
    s *= -1
    dist = s + 1
    dist = np.clip(dist, 0, 2)
    if x is y or y is None:
        dist[np.diag_indices_from(dist)] = 0.0
    if np.any(np.isnan(dist)):
        if x.ndim == 1:
            dist = 1.
        else:
            dist[np.isnan(dist)] = 1.
    return dist

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # define margin cnn
        self.conv1_d = nn.Conv1d(101, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 614/2 64
        self.bn1_d = nn.BatchNorm1d(64)
        # 150 64
        self.layer1_d = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # output 75 128
        self.layer2_d = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # conv replacing pooling
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        # output 37 256
        self.layer3_d = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # conv replacing pooling
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        # output 20 512
        self.layer4_d = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # conv replacing pooling
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool_d = nn.AdaptiveAvgPool1d(1)
        self.fc_d = nn.Linear(512, 101)  # the number of margin_l=101

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def margin_long_tail(self, x):
        x = self.conv1_d(x)
        x = self.bn1_d(x)
        x = self.relu(x)
        x = self.layer1_d(x)
        x = self.layer2_d(x)
        x = self.layer3_d(x)
        x = self.layer4_d(x)
        x = self.avgpool_d(x)
        x = torch.flatten(x, 1)
        x = self.fc_d(x)
        return x

    def margin_miu(self, x):
        x = self.conv1_d(x)
        x = self.bn1_d(x)
        x = self.relu(x)
        x = self.layer1_d(x)
        x = self.layer2_d(x)
        x = self.layer3_d(x)
        x = self.layer4_d(x)
        x = self.avgpool_d(x)
        x = torch.flatten(x, 1)
        x = self.fc_d(x)
        x = (x-torch.min(x))/(torch.max(x)-torch.min(x))*101
        return x

    def margin_sigma(self, x):
        x = self.conv1_d(x)
        x = self.bn1_d(x)
        x = self.relu(x)
        x = self.layer1_d(x)
        x = self.layer2_d(x)
        x = self.layer3_d(x)
        x = self.layer4_d(x)
        x = self.avgpool_d(x)
        x = torch.flatten(x, 1)
        x = self.fc_d(x)
        return x

    @staticmethod
    def gaussian(z, age, m_p_miu, m_p_sigma):  # Calculate the Gaussian distribution for each age
        x = torch.linspace(0, 100, 101).expand([z.shape[0], -1]).cuda()
        pi = torch.Tensor([3.1415926]).cuda()
        u = m_p_miu.T[age]
        sig = m_p_sigma.T[age]
        m_p = torch.exp(-torch.pow((x - u), 2)/(2 * torch.pow(sig, 2))) / (torch.sqrt(2 * pi) * sig)
        return m_p

    @staticmethod
    def distributed_softmax(x, margin):
        a = torch.ones(x.shape[1]).cuda()
        mask = torch.diag(a).cuda()  # 1D vector  Output a 2D square matrix with input as diagonal elements
        mask = (1 - mask).expand((x.shape[0], -1, -1))  # diagonal is 0, the others are 1

        b = x.expand([x.shape[1], -1, -1]).permute(1, 0, 2)  # batch_size, multi, score
        b = b * mask
        b = torch.sum(torch.exp(b), dim=2) - 1  # eliminate exp(0) sum of the negative score
        y = torch.exp(x - margin) / (b + torch.exp(x - margin))
        return y

    def _forward_impl(self, x, age, pro, intra, inter):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # propotype update
        z = x.cpu().clone().detach().numpy()
        age = age.cpu().clone().detach().numpy()
        pro_t = pro[0].copy()
        intra_t = intra.copy()
        inter_t = inter.copy()
        for i in range(z.shape[0]):
            temp = pro[0][age[i], :].copy()  # shallow copy
            pro[0][age[i]] = pro[0][age[i], :] + (z[i] - pro[0][age[i], :]) / (pro[1][age[i]] + 1)  # update Prototype
            pro[1][age[i]] += 1  # update instance number
            # cosine_distances 0~2 Divisor 0 is set to 1 for unknown relationship
            intra[age[i]] = intra[age[i]] + cosine_distance(z[i], temp) * cosine_distance(z[i], pro[0][age[i]])
        # Calculate distance between two matrices and multi-row vectors
        inter = cosine_distance(pro[0], pro[0])
        delta_pro = np.concatenate((pro[0]-pro_t, intra-intra_t, inter-inter_t), axis=1)[np.newaxis, :]
        pro_input = np.concatenate((pro[0], intra), axis=1)[np.newaxis, :]
        m_l = self.margin_long_tail(torch.from_numpy(delta_pro).cuda())
        m_p_miu = self.margin_miu(torch.from_numpy(pro_input).cuda())
        m_p_sigma = self.margin_sigma(torch.from_numpy(pro_input).cuda())
        m_p = self.gaussian(z, age, m_p_miu, m_p_sigma)
        margin = m_p + 0.1*m_l
        # The margin is normalized to 0 as the mean and 1 as the variance
        margin = (margin-torch.mean(margin, dim=1).expand([101, -1]).permute(1, 0))/torch.std(margin, dim=1).expand([101, -1]).permute(1, 0)
        margin = margin*0.01
        if not False: #cfg.model.margin: ?????? ?????????? False ?? ?????? ??????????
            margin = margin*0
        x = self.fc(x)
        if margin.requires_grad:
            x = F.softmax(x-margin, dim=1)  # make a baseline
            # x = F.softmax(x-margin, dim=1)
        else:
            x =F.softmax(x, dim=1)
        # L1 normalize
        # x = F.normalize(x-margin, p=1, dim=1)  # negative logarithm possible
        return x, pro, intra, inter

    def forward(self, x, age, proc, intra, inter):
        return self._forward_impl(x, age, proc, intra, inter)



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # partial load pretrained model
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

