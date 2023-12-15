import torch
from hw_asr.base.base_metric import BaseMetric
from .utils import compute_eer
import torch.nn.functional as F


class EER(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, logits, label, **kwargs):
        probs = F.softmax(logits, dim=-1)[:, -1]
        mask = label.bool()
        return compute_eer(probs[mask].detach().cpu().numpy(),
                            probs[~mask].detach().cpu().numpy())[0]






