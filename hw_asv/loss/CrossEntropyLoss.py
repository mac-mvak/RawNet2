import torch
import torch.nn as nn

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: list = None, size_average=None,
                  ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        if weight is not None:
            weight = torch.tensor(weight)
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, logits, label, **kwargs):
        return super().forward(logits, label)
