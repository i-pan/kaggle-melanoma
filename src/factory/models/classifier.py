import logging
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels.utils
import math

global AMP_AVAIL
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAIL = True
except ImportError:
    print('Automatic mixed precision not available !')
    AMP_AVAIL = False

try:
    from mmdet.models.necks import NASFPN
except:
    print('Unable to import NASFPN from mmdet')


from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
from .arc import *
from .backbones import *
from .constants import AVGPOOL
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveConcatPool3d
from .mmdet_resnet import ConvWS2d


class Net2D(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 multisample_dropout=True,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem',
                 arc=False):
        super().__init__()
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, **backbone_params)

        # Change pooling layer, if specified
        if pool == 'gem':
            setattr(self.backbone, AVGPOOL[backbone], GeM())
        elif pool == 'concat':
            setattr(self.backbone, AVGPOOL[backbone], AdaptiveConcatPool2d())
            dim_feats *= 2

        self.multisample_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

        if arc:
            self.arc = ArcMarginProduct(in_features=dim_feats, out_features=num_classes)


    def forward_base(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        if self.multisample_dropout:
            x = torch.mean(
                torch.stack(
                    [self.fc(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.fc(self.dropout(features))

        x = x if self.fc.out_features > 1 else x[:,0]

        return x


    def forward(self, x):
        square_tta = False
        if x.ndim == 5 and x.size(0) == 1:
            # shape = (1, num_crops, C, H, W)
            x = x.squeeze(0)
            square_tta = True
        if hasattr(self, '_autocast') and self._autocast:
            with autocast(): x = self.forward_base(x)
        else:
            x = self.forward_base(x)

        if square_tta:
            # shape = (N, num_classes)
            x = x.mean(0).unsqueeze(0)

        return x


class NetMeta2D(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 multisample_dropout=True,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem',
                 meta_dim=32,
                 arc=False,
                 norm_eval=False):
        super().__init__()
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, **backbone_params)

        # Change pooling layer, if specified
        if pool == 'gem':
            setattr(self.backbone, AVGPOOL[backbone], GeM())
        elif pool == 'concat':
            setattr(self.backbone, AVGPOOL[backbone], AdaptiveConcatPool2d())
            dim_feats *= 2

        self.age_embed = nn.Embedding(10, meta_dim)
        self.sex_embed = nn.Embedding( 2, meta_dim)
        self.ant_embed = nn.Embedding( 6, meta_dim)

        self.multisample_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim_feats+meta_dim*3, num_classes)

        if arc:
            self.arc = ArcMarginProduct(in_features=dim_feats, out_features=num_classes)

        self.norm_eval = norm_eval


    def forward_base(self, x):
        features = self.backbone(x['img'])
        age_feat = self.age_embed(x['age'])
        sex_feat = self.sex_embed(x['sex'])
        ant_feat = self.ant_embed(x['ant'])

        if hasattr(self, 'arc') and self.training:
            arcmargin = self.arc(features)

        features = torch.cat([features, age_feat, sex_feat, ant_feat], dim=1)

        if self.multisample_dropout:
            x = torch.mean(
                torch.stack(
                    [self.fc(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.fc(self.dropout(features))

        x = x if self.fc.out_features > 1 else x[:,0]

        if hasattr(self, 'arc') and self.training:
            return x, arcmargin
            
        return x


    def forward(self, x):
        square_tta = False
        if x['img'].ndim == 5 and x['img'].size(0) == 1:
            # shape = (1, num_crops, C, H, W)
            x = {k : v.squeeze(0) for k,v in x.items()}
            square_tta = True

        if hasattr(self, '_autocast') and self._autocast:
            with autocast(): x = self.forward_base(x)
        else:
            x = self.forward_base(x)

        if square_tta:
            # shape = (N, num_classes)
            x = x.mean(0).unsqueeze(0)

        return x

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self


class NetMetaFPN2D(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 multisample_dropout=True,
                 dropout=0.2,
                 out_channels=196,
                 stack_times=3,
                 backbone_params={},
                 pool='concat',
                 meta_dim=32):
        super().__init__()
        self.backbone, in_channels = eval(backbone)(pretrained=pretrained, **backbone_params)
        self.fpn = NASFPN(in_channels=in_channels, out_channels=out_channels, stack_times=stack_times, num_outs=5, norm_cfg=dict(type='BN'))
        dim_feats = out_channels*5

        # Change pooling layer, if specified
        if pool == 'gem':
            self.pool = GeM()
        elif pool == 'concat':
            self.pool = AdaptiveConcatPool2d()
            dim_feats *= 2

        self.age_embed = nn.Embedding(10, meta_dim)
        self.sex_embed = nn.Embedding( 2, meta_dim)
        self.ant_embed = nn.Embedding( 6, meta_dim)

        self.multisample_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim_feats+meta_dim*3, num_classes)


    def forward_base(self, x):
        B = x['img'].size(0)
        pyramid  = self.backbone(x['img'])
        age_feat = self.age_embed(x['age'])
        sex_feat = self.sex_embed(x['sex'])
        ant_feat = self.ant_embed(x['ant'])
        features = self.fpn(pyramid)
        features = torch.cat([self.pool(i).view(B, -1) for i in features], dim=1)
        features = torch.cat([features, age_feat, sex_feat, ant_feat], dim=1)

        if self.multisample_dropout:
            x = torch.mean(
                torch.stack(
                    [self.fc(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.fc(self.dropout(features))

        x = x if self.fc.out_features > 1 else x[:,0]

        if hasattr(self, 'arc') and self.training:
            return x, arcmargin
            
        return x


    def forward(self, x):
        square_tta = False
        if x['img'].ndim == 5 and x['img'].size(0) == 1:
            # shape = (1, num_crops, C, H, W)
            x = {k : v.squeeze(0) for k,v in x.items()}
            square_tta = True

        if hasattr(self, '_autocast') and self._autocast:
            with autocast(): x = self.forward_base(x)
        else:
            x = self.forward_base(x)

        if square_tta:
            # shape = (N, num_classes)
            x = x.mean(0).unsqueeze(0)

        return x


class MultiRes(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 num_backbones=3,
                 scale_factor=None,
                 input_sizes=[384,256],
                 multisample_dropout=True,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem',
                 arc=False):
        super().__init__()

        self.num_backbones = num_backbones
        self.scale_factor = scale_factor
        self.input_sizes = input_sizes
        if type(input_sizes) != type(None):
            assert len(input_sizes) == num_backbones-1

        dim_feats = 0
        for i in range(num_backbones):
            bb, df = eval(backbone)(pretrained=pretrained, **backbone_params)
            
            if pool == 'gem':
                setattr(bb, AVGPOOL[backbone], GeM())
            elif pool == 'concat':
                setattr(bb, AVGPOOL[backbone], AdaptiveConcatPool2d())
                df *= 2

            setattr(self, f'backbone{i}', bb)
            dim_feats += df

        # Change pooling layer, if specified
        self.multisample_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim_feats, num_classes)

        if arc:
            self.arc = ArcMarginProduct(in_features=dim_feats, out_features=num_classes)


    def forward_base(self, x):
        if x.ndim == 5:
            # Paired images
            B, N, C, H, W = x.size()
            assert self.training and N==2
            x = x.view(-1, C, H, W)
        features = []
        for i in range(self.num_backbones):
            if i > 0:
                if type(self.input_sizes) != None:
                    x = F.interpolate(x, size=(self.input_sizes[i-1], self.input_sizes[i-1]), mode='bilinear')
                else:
                    x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
            features += [getattr(self, f'backbone{i}')(x)]
        features = torch.cat(features, dim=1)

        if self.multisample_dropout:
            x = torch.mean(
                torch.stack(
                    [self.fc(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.fc(self.dropout(features))

        x = x if self.fc.out_features > 1 else x[:,0]

        return x


    def forward_tta(self, x):
        x_hflip = torch.flip(x, dims=(-1,))
        x_vflip = torch.flip(x, dims=(-2,))
        x_trans = x.transpose(-1, -2)
        return torch.mean(torch.stack([
                   self.forward_base(x),
                   self.forward_base(x_hflip),
                   self.forward_base(x_vflip),
                   self.forward_base(x_trans)
                ], dim=0), dim=0)


    def forward(self, x):
        square_tta = False
        if x.ndim == 5 and x.size(0) == 1:
            # shape = (1, num_crops, C, H, W)
            x = x.squeeze(0)
            square_tta = True

        if hasattr(self, 'tta') and self.tta:
            x = self.forward_tta(x)
        else:
            if hasattr(self, '_autocast') and self._autocast:
                with autocast(): x = self.forward_base(x)
            else:
                x = self.forward_base(x)

        if square_tta:
            # shape = (N, num_classes)
            x = x.mean(0).unsqueeze(0)

        return x


class ArcNet(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem'):
        super().__init__()
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, **backbone_params)

        # Change pooling layer, if specified
        if pool == 'gem':
            setattr(self.backbone, AVGPOOL[backbone], GeM())
        elif pool == 'concat':
            setattr(self.backbone, AVGPOOL[backbone], AdaptiveConcatPool2d())
            dim_feats *= 2

        self.dropout = nn.Dropout(p=dropout)
        self.arc = ArcMarginProduct(in_features=dim_feats, out_features=num_classes)


    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.arc(features) if self.training else F.normalize(features)


class SiameseNet(nn.Module):


    def __init__(self,
                 backbone,
                 pretrained,
                 dropout=0.2,
                 backbone_params={},
                 pool='gem'):
        super().__init__()
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, **backbone_params)

        # Change pooling layer, if specified
        if pool == 'gem':
            setattr(self.backbone, AVGPOOL[backbone], GeM())
        elif pool == 'concat':
            setattr(self.backbone, AVGPOOL[backbone], AdaptiveConcatPool2d())
            dim_feats *= 2

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(dim_feats * 4, 1)


    def extract_features(self, x):
        return self.dropout(self.backbone(x))


    def forward_head(self, x, y):
        a = x+y
        d = x-y
        s = (x-y)**2
        t = x*y
        z = torch.cat([a,d,s,t], dim=1)
        return self.head(z)[:,0]

    
    def forward(self, x):
        if self.training:
            assert x.ndim == 5 # (N, 2, C, H, W)
            assert x.size(1) == 2
            feat1 = self.extract_features(x[:,0])
            feat2 = self.extract_features(x[:,1])
            return self.forward_head(feat1,feat2)
        else:
            raise NotImplementedError('For inference, use `extract_features` and `forward_head` methods')






