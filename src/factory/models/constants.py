AVGPOOL = {
    'avgpool': [
        'resnet18', 
        'resnet34', 
        'resnet50', 
        'resnet101'
    ],
    'avg_pool': [
        'se_resnet50', 
        'se_resnext50', 
        'se_resnet101', 
        'se_resnext101',
        'resnet50_gn_ws',
        'bit_resnet101x3',
        'bit_resnet152x2'
    ],
    'global_pool': [
        'resnest14',
        'resnest26',
        'resnest50',
        'resnest101',
        'resnest200',
        'resnest269',
        'efficientnet_b0',
        'efficientnet_b1_pruned',
        'efficientnet_b1',
        'efficientnet_b2_pruned',
        'efficientnet_b2',
        'efficientnet_b3_pruned',
        'efficientnet_b3',
        'tf_efficientnet_b4',
        'tf_efficientnet_b5',
        'tf_efficientnet_b6_ns',
        'tf_efficientnet_b6',
        'tf_efficientnet_b7',
        'tf_efficientnet_b8',
        'efficientnet_l2',
        'efficientnet_l2_abbrev',
        'mixnet_s',
        'mixnet_m',
        'mixnet_l',
        'mixnet_xl'
    ]
}

AVGPOOL = {vi : k for k,v in AVGPOOL.items() for vi in v}