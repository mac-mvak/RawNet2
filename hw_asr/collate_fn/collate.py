import logging
import torch
import torch.nn as nn
from typing import List

logger = logging.getLogger(__name__)

MAX_LEN = 64600

def pad_f(x, max_len=64600):
    x = x.squeeze()
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len].unsqueeze(0)
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = torch.tile(x, (1, num_repeats))[:, :max_len][0]
    padded_x = padded_x.unsqueeze(0)
    return padded_x 

def adder(vec, v):
    v = pad_f(v)
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

