import logging
import torch
import torch.nn as nn
from typing import List

logger = logging.getLogger(__name__)

MAX_LEN = 64600


def adder(vec, v):
    v = v[..., :MAX_LEN]
    if vec is None:
        vec = v
    else:
        size_1, size_2 = vec.shape[-1], v.shape[-1]
        pad = size_1 - size_2
        vec = nn.functional.pad(vec, (0, max(-pad, 0)))
        v = nn.functional.pad(v, (0, max(pad, 0)))
        vec = torch.cat([vec, v])
    return vec



def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio = None
    label = []

    for item in dataset_items:
        audio = adder(audio, item['audio'])
        label.append(item['label'])
    result_batch = {'audio' : audio, 
                    'label' : torch.tensor(label, dtype=torch.long)
                    }
    return result_batch

