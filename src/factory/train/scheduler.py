from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR
)
from .torch_lr_scheduler import OneCycleLR
from .onecycle import CustomOneCycleLR