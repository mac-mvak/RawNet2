import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asv.base.base_dataset import BaseDataset
from hw_asv.utils import ROOT_PATH

logger = logging.getLogger(__name__)

INDEX_NAMES = {
    'train': 'ASVspoof2019.LA.cm.train.trn.txt',
    'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
    'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
}


class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in INDEX_NAMES
        data_dir = ROOT_PATH / "data" / "datasets" / "LA"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_index(self, part):
        index_path = self._data_dir / 'ASVspoof2019_LA_cm_protocols'
        index_path = index_path / INDEX_NAMES[part]
        index = []
        with open(index_path) as f:
            while True:
                line = f.readline()
                if line is None or line=='':
                    break
                split_line = line.split()
                file_name = split_line[1] + '.flac'
                label = 1 if split_line[-1] == 'bonafide' else 0
                path = self._data_dir / f'ASVspoof2019_LA_{part}' / 'flac' / file_name
                index.append({'path':path, 'label':label})
        return index


