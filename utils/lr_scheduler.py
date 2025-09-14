import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        init_lr (float): Initial learning rate for warmup.
        base_lr (float): Base learning rate after warmup.
        final_lr (float): Final learning rate at the end of schedule.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of training steps.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        base_lr,
        final_lr,
        warmup_steps,
        total_steps,
        last_epoch=-1,
    ):
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / float(self.warmup_steps)
            lr_mult = self.init_lr + (self.base_lr - self.init_lr) * alpha
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            progress = min(1.0, progress)  # Ensure progress doesn't exceed 1
            lr_mult = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
                1 + math.cos(math.pi * progress)
            )

        return [lr_mult for _ in self.base_lrs]
