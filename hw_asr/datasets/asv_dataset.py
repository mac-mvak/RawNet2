import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):

        data_dir = ROOT_PATH / "data" / "datasets" / "LA"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_index(self, part):
        index_path = self._data_dir / 'ASVspoof2019_LA_cm_protocols'
        index_path = index_path / f'ASVspoof2019.LA.cm.{part}.trn.txt'
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


