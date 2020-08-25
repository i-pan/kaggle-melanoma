import pretrainedmodels 
import pretrainedmodels.utils
import numpy as np
import torch
import torch.nn as nn
import copy
import timm.models
import re

from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.senet import SEResNeXtBottleneck
from .efficientnet import EfficientNet
from .resnext_wsl import (
    resnext101_32x8d_wsl  as rx101_32x8, 
    resnext101_32x16d_wsl as rx101_32x16, 
    resnext101_32x32d_wsl as rx101_32x32,
    resnext101_32x48d_wsl as rx101_32x48
)
from .inception_i3d import InceptionV1_I3D
from .mmdet_resnext import ResNeXt as mmdet_resnext
from .mmdet_resnet import ResNet as mmdet_resnet, ConvWS2d
from .torchvision_densenet import densenet121 as tv_densenet121
from .big_transfer import ResNetV2, load_npz_from_url


def skresnet34(pretrained=True, num_input_channels=3):
    model = timm.models.skresnet34(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def mixnet_s(pretrained=True, num_input_channels=3):
    model = timm.models.mixnet_s(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats


def mixnet_m(pretrained=True, num_input_channels=3):
    model = timm.models.mixnet_m(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats


def mixnet_l(pretrained=True, num_input_channels=3):
    model = timm.models.mixnet_l(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def mixnet_xl(pretrained=True, num_input_channels=3):
    model = timm.models.mixnet_xl(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def bit_resnet26x1(pretrained='bitm'):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    head_size = 1000 if pretrained == 'bits' else 21843
    model = ResNetV2([3,4,6,3], 1, head_size=head_size)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz')      
    if weights: model.load_from(weights)
    for i in range(1, 5):
        blocks = getattr(model.body, f'block{i}')[:2]
        setattr(model.body, f'block{i}', blocks)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def bit_resnet50x1(pretrained='bitm'):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    head_size = 1000 if pretrained == 'bits' else 21843
    model = ResNetV2([3,4,6,3], 1, head_size=head_size)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz')      
    if weights: model.load_from(weights)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def bit_resnet50x3(pretrained='bitm', stride_down=False):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    model = ResNetV2([3,4,6,3], 3)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R50x3.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R50x3.npz')      
    if weights: model.load_from(weights)
    if stride_down:
        model.body.block1[0].conv2.stride = (2,2)
        model.body.block1[0].downsample.stride = (2,2)
        # model.body.block4[0].conv2.stride = (1,1)
        # model.body.block4[0].downsample.stride = (1,1)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def bit_resnet101x1(pretrained='bitm'):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    model = ResNetV2([3,4,23,3], 1)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz')      
    if weights: model.load_from(weights)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def bit_resnet101x3(pretrained='bitm'):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    model = ResNetV2([3,4,23,3], 3)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R101x3.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R101x3.npz')      
    if weights: model.load_from(weights)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def bit_resnet152x2(pretrained='bitm'):
    assert pretrained in ('bitm', 'bits', None, False), f'`pretrained` must be one of [`bitm`, `bits`, None] but got {pretrained}'
    model = ResNetV2([3,8,36,3], 2)
    weights = None
    if pretrained == 'bitm':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz')
    elif pretrained == 'bits':
        weights = load_npz_from_url('https://storage.googleapis.com/bit_models/BiT-S-R152x2.npz')      
    if weights: model.load_from(weights)
    dim_feats = model.fc.in_channels
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def res2net(name, pretrained=True, stride_down=False):
    model = getattr(timm.models.res2net, name)(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def res2net50_26w_4s(pretrained=True, stride_down=False):
    return res2net('res2net50_26w_4s', pretrained, stride_down)


def resnest14(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest14d(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnest26(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest26d(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnest50(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest50d(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnest101(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest101e(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnest200(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest200e(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnest269(pretrained=True, stride_down=False):
    model = timm.models.resnest.resnest269e(pretrained=pretrained)
    if stride_down:
        #model.conv1[3].stride = (2, 2)
        model.layer1[0].conv1.stride = (2,2)
        model.layer1[0].downsample[0] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def ecaresnetlight(pretrained=True):
    model = timm.models.resnet.ecaresnetlight(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def ecaresnet50d_pruned(pretrained=True):
    model = timm.models.resnet.ecaresnet50d_pruned(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def i3d(pretrained=True):
    model = InceptionV1_I3D()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics400_se_rgb_inception_v1_seg1_f64s1_imagenet_deepmind-9b8e02b3.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    dim_feats = 1024
    return model, dim_feats


def densenet121(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet121')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet161(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet161')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet169(pretrained='imagenet', num_input_channels=3):
    model = getattr(pretrainedmodels, 'densenet169')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet121(pretrained=True, memory_efficient=True):
    model = tv_densenet121(pretrained=pretrained, memory_efficient=True)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats


def generic(name, pretrained, num_input_channels=3):
    model = getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnet18(pretrained='imagenet', num_input_channels=3):
    return generic('resnet18', pretrained=pretrained, num_input_channels=num_input_channels)


def resnet34(pretrained='imagenet', num_input_channels=3):
    return generic('resnet34', pretrained=pretrained, num_input_channels=num_input_channels)


def resnet50(pretrained='imagenet', num_input_channels=3, stride_down=False):
    model, dim_feats = generic('resnet50', pretrained=pretrained, num_input_channels=num_input_channels)
    if stride_down:
        # To reduce memory consumption:
        # 1) Layer 1 stride = 2
        model.layer1[0].conv2.stride = (2, 2)
        model.layer1[0].downsample[0].stride = (2, 2)
        # 2) Layer 4 stride = 1
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
    return model, dim_feats


def resnet101(pretrained='imagenet', num_input_channels=3):
    return generic('resnet101', pretrained=pretrained, num_input_channels=num_input_channels)


def resnet152(pretrained='imagenet', num_input_channels=3):
    return generic('resnet152', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnet50(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet50', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnet101(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet101', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnet152(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnet152', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnext50(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnext50_32x4d', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnext101(pretrained='imagenet', num_input_channels=3):
    return generic('se_resnext101_32x4d', pretrained=pretrained, num_input_channels=num_input_channels)


def inceptionv3(pretrained='imagenet', num_input_channels=3):
    model, dim_feats = generic('inceptionv3', pretrained=pretrained, num_input_channels=num_input_channels)
    model.aux_logits = False
    return model, dim_feats


def inceptionv4(pretrained='imagenet', num_input_channels=3):
    return generic('inceptionv4', pretrained=pretrained, num_input_channels=num_input_channels)


def inceptionresnetv2(pretrained='imagenet', num_input_channels=3):
    return generic('inceptionresnetv2', pretrained=pretrained, num_input_channels=num_input_channels)


def resnext101_wsl(d, pretrained='instagram', num_input_channels=3):
    model = eval('rx101_32x{}'.format(d))(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnext101_32x8d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(8, pretrained=pretrained, num_input_channels=num_input_channels)
    

def resnext101_32x16d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(16, pretrained=pretrained, num_input_channels=num_input_channels)


def resnext101_32x32d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(32, pretrained=pretrained, num_input_channels=num_input_channels)


def resnext101_32x48d_wsl(pretrained='instagram', num_input_channels=3):
    return resnext101_wsl(48, pretrained=pretrained, num_input_channels=num_input_channels)


def efficientnet_b0(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b0(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b1_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b1(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b2_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b2(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b3_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_b3(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b4(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b4(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b5(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b5(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b6(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b6_ns(pretrained=True, stride_down=False, **kwargs):
    model = timm.models.tf_efficientnet_b6_ns(pretrained=pretrained, **kwargs)
    if stride_down:
        print('Reducing stride from (2,2) to (1,1) ...')
        model.conv_stem.stride = (1,1)
        #model.blocks[1][0].conv_dw.stride = (1,1)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b6_ns_features(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6_ns(pretrained=pretrained, features_only=True, **kwargs)
    return model, [32,40,72,200,576]

def tf_efficientnet_b7(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b7(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def tf_efficientnet_b8(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b8(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_l2(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_l2_ns_475(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

def efficientnet_l2_abbrev(pretrained=True, truncate=1):
    model = timm.models.tf_efficientnet_l2_ns_475(pretrained=pretrained)
    model.blocks = model.blocks[:-truncate]
    model.conv_head = pretrainedmodels.utils.Identity()
    model.bn2 = pretrainedmodels.utils.Identity()
    dim_feats = model.blocks[-1][-1].bn3.num_features
    model.classifier = pretrainedmodels.utils.Identity()
    return model, dim_feats

# def efficientnet(b, pretrained, num_input_channels=3):
#     if pretrained == 'imagenet':
#         model = EfficientNet.from_pretrained('efficientnet-{}'.format(b))
#     elif pretrained is None:
#         model = EfficientNet.from_name('efficientnet-{}'.format(b))
#     if num_input_channels != 3:
#         first_layer_weights = model.state_dict()['_conv_stem.weight']
#         layer_params = {'in_channels' : num_input_channels,
#                         'out_channels': model._conv_stem.out_channels,
#                         'kernel_size' : model._conv_stem.kernel_size,
#                         'stride':  model._conv_stem.stride,
#                         'padding': model._conv_stem.padding,
#                         'bias': model._conv_stem.bias}
#         model._conv_stem = nn.Conv2d(**layer_params)
#         first_layer_weights = np.sum(first_layer_weights.cpu().numpy(), axis=1) / num_input_channels
#         first_layer_weights = np.repeat(np.expand_dims(first_layer_weights, axis=1), num_input_channels, axis=1)
#         model.state_dict()['_conv_stem.weight'].data.copy_(torch.from_numpy(first_layer_weights))
#     dim_feats = model._fc.in_features
#     model._dropout = pretrainedmodels.utils.Identity()
#     model._fc = pretrainedmodels.utils.Identity()
#     return model, dim_feats


# def efficientnet_b0(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b0', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b1(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b1', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b2(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b2', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b3(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b3', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b4(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b4', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b5(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b5', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b6(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b6', pretrained=pretrained, num_input_channels=num_input_channels)


# def efficientnet_b7(pretrained='imagenet', num_input_channels=3):
#     return efficientnet('b7', pretrained=pretrained, num_input_channels=num_input_channels)


def se_resnext50_slim(pretrained='imagenet'):
    model, dim_feats = generic('se_resnext50_32x4d', pretrained=pretrained)
    slim_model = copy.deepcopy(model)
    slim_model.inplanes = 512
    slim_model.layer3 = slim_model._make_layer(
            SEResNeXtBottleneck,
            planes=128,
            blocks=6,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=1,
            downsample_padding=0
    )
    slim_model.layer4 = slim_model._make_layer(
            SEResNeXtBottleneck,
            planes=256,
            blocks=3,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=1,
            downsample_padding=0
    )
    return slim_model, 1024


def resnet50_slim(pretrained='imagenet'):
    model, dim_feats = generic('resnet50', pretrained=pretrained)
    slim_model = copy.deepcopy(model)
    slim_model.inplanes = 512
    slim_model.layer3 = slim_model._make_layer(
            Bottleneck,
            planes=128,
            blocks=6,
            stride=2,
            dilate=False
    )
    slim_model.layer4 = slim_model._make_layer(
            Bottleneck,
            planes=128,
            blocks=3,
            stride=2,
            dilate=False
    )
    return slim_model, 512


def resnext101_32x8d_wsl_slim(pretrained='instagram', num_input_channels=3):
    model, dim_feats = resnext101_wsl(8, pretrained=pretrained, num_input_channels=num_input_channels)
    slim_model = copy.deepcopy(model)
    slim_model.inplanes = 512
    slim_model.layer3 = slim_model._make_layer(
        Bottleneck,
        128, 23, stride=2, dilate=False)
    slim_model.layer4 = slim_model._make_layer(
        Bottleneck,
        256, 3, stride=2, dilate=False)
    return slim_model, 1024


def resnext50_32x8d_wsl_slim(pretrained='instagram', num_input_channels=3):
    model, dim_feats = resnext101_wsl(8, pretrained=pretrained, num_input_channels=num_input_channels)
    slim_model = copy.deepcopy(model)
    slim_model.inplanes = 512
    slim_model.layer3 = slim_model._make_layer(
        Bottleneck,
        128, 6, stride=2, dilate=False)
    slim_model.layer4 = slim_model._make_layer(
        Bottleneck,
        256, 3, stride=2, dilate=False)
    return slim_model, 1024


def resnext26_gn_ws(pretrained=True):
    model = mmdet_resnext(depth=50,
                          groups=32,
                          norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True},
                          conv_cfg={'type': 'ConvWS'})
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth')
        weights = weights['state_dict']
        weights = {k:v for k,v in weights.items() if not re.search(r'^fc', k)}
        model.load_state_dict(weights)
    for i in range(1, 5):
        layers = getattr(model, f'layer{i}')[:2]
        setattr(model, f'layer{i}', layers)
    return model, 2048


def resnext50_gn_ws(pretrained=True):
    model = mmdet_resnext(depth=50,
                          groups=32,
                          norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True},
                          conv_cfg={'type': 'ConvWS'})
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth')
        weights = weights['state_dict']
        weights = {k:v for k,v in weights.items() if not re.search(r'^fc', k)}
        model.load_state_dict(weights)
    return model, 2048


def resnext101_gn_ws(pretrained=True):
    model = mmdet_resnext(depth=101,
                          groups=32,
                          norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True},
                          conv_cfg={'type': 'ConvWS'})
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth')
        weights = weights['state_dict']
        weights = {k:v for k,v in weights.items() if not re.search(r'^fc', k)}
        model.load_state_dict(weights)
    return model, 2048


def resnet50_gn_ws(pretrained=True, stride_down=False, atrous=False):
    model = mmdet_resnet(depth=50,
                         norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True},
                         conv_cfg={'type': 'ConvWS'})
    dim_feats = 2048
    if stride_down:
        # To reduce memory consumption:
        # 1) Layer 1 stride = 2
        model.layer1[0].conv2.stride = (2, 2)
        model.layer1[0].downsample[0].stride = (2, 2)
        # 2) Layer 2 stride = 1
        model.layer2[0].conv2.stride = (1, 1)
        model.layer2[0].downsample[0].stride = (1, 1)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth')
        weights = weights['state_dict']
        weights = {k:v for k,v in weights.items() if not re.search(r'^fc', k)}
        model.load_state_dict(weights)
    if atrous: 
        model.layer4[0].conv2.dilation = (2, 2)
        model.layer4[0].conv2.padding  = (2, 2)
        model.layer4[1].conv2.dilation = (4, 4)
        model.layer4[1].conv2.padding  = (4, 4)
        model.layer4[2].conv2.dilation = (8, 8)
        model.layer4[2].conv2.padding  = (8, 8)
    return model, dim_feats


def deeper_resnet50_gn_ws(pretrained=True, atrous=False):
    model = mmdet_resnet(depth=50,
                         norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True},
                         conv_cfg={'type': 'ConvWS'})
    dim_feats = 2048
    # Create layer 5 from layer 4
    layer5 = copy.deepcopy(model.layer4)
    # Need to alter input weights to accept 2048-D 
    input_weights = layer5.state_dict()['0.conv1.weight']
    layer5[0].conv1 = ConvWS2d(2048, 512, kernel_size=1, stride=1, bias=False)
    input_weights = torch.cat([input_weights]*2, dim=1)
    layer5.state_dict()['0.conv1.weight'].data.copy_(input_weights)
    # Need to alter downsample weights to accept 2048-D
    ds_weights = layer5.state_dict()['0.downsample.0.weight']
    layer5[0].downsample[0] = ConvWS2d(2048, 2048, kernel_size=1, stride=2, bias=False)
    ds_weights = torch.cat([ds_weights]*2, dim=1)
    layer5.state_dict()['0.downsample.0.weight'].data.copy_(ds_weights)
    # Layer 6 is copy of layer 5
    layer6 = copy.deepcopy(layer5)
    layers = [copy.deepcopy(model.layer4), layer5, layer6]
    if atrous:
        grid = [1, 2, 4]
        base_rate = 2
        for i in range(len(layers)):
            r = base_rate ** (i+1)
            layers[i][0].conv2.stride = 1
            layers[i][0].downsample[0].stride = 1
            for j in range(len(grid)):
                layers[i][j].conv2.dilation = (r*grid[i], r*grid[i])
                layers[i][j].conv2.padding  = (r*grid[i], r*grid[i])
    model.layer4 = nn.Sequential(*layers)
    return model, dim_feats

