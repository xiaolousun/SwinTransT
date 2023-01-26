from torch.optim.lr_scheduler import _LRScheduler
from typing import Callable, List, Optional, Union
from math import cos, pi

class CosineAnnealingLr(_LRScheduler):

    """
     CosineAnnealing LR scheduler.
    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 optimizer,
                 max_epoch:Optional[int] = 200,
                 warmup_epoch:Optional[int] = 10,
                 warmup_ratio:Optional[float] = 0.1,
                 min_lr: Optional[float] = None,
                 min_lr_ratio: Optional[float] = None,
                 **kwargs) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.warmup_epoch = warmup_epoch
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, **kwargs)

    def get_lr(self):

        if self.last_epoch < self.warmup_epoch:
            # lr = lr_max * current_epoch / warmup_epoch
            k = (1 - self.last_epoch / self.warmup_epoch) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lrs]
            return warmup_lr
        else:
            cos_lr = [self.min_lr + (_lr-self.min_lr)*(1 + cos(pi * (self.last_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch))) / 2 for _lr in self.base_lrs]
            return cos_lr
        
  
# def annealing_cos(start: float,
#                   end: float,
#                   factor: float,
#                   weight: float = 1.) -> float:
#     """Calculate annealing cos learning rate.
#     Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
#     percentage goes from 0.0 to 1.0.
#     Args:
#         start (float): The starting learning rate of the cosine annealing.
#         end (float): The ending learing rate of the cosine annealing.
#         factor (float): The coefficient of `pi` when calculating the current
#             percentage. Range from 0.0 to 1.0.
#         weight (float, optional): The combination factor of `start` and `end`
#             when calculating the actual starting learning rate. Default to 1.
#     """
#     cos_out = cos(pi * factor) + 1
#     return end + 0.5 * weight * (start - end) * cos_out

