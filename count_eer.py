import torchaudio
import torch
import torch.nn.functional as F
from hw_asv.datasets import ASVDataset
from hw_asv.metric import EER
from hw_asv.collate_fn.collate import collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hw_asv.model import RawNet2

device = torch.device('cuda:0')

checkpoint = torch.load('final_data/model.pth')

aa = checkpoint['config']['arch']
cfg = checkpoint['config']
dataset = ASVDataset('eval', config_parser=cfg)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, drop_last=False)
model = RawNet2(**aa['args'])
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
metr = EER()
all_logits = []
all_tars = []

for batch in tqdm(loader):
    audio = batch['audio'].to(device)
    logits = model(audio=audio)
    logits = logits['logits']
    all_logits.append(logits.detach())
    all_tars.append(batch['label'].detach())

all_tars = torch.cat(all_tars)
all_logits = torch.cat(all_logits)

eer_l = metr(all_logits, all_tars)

print(f'EER={eer_l}')




