import torchaudio
import torch
import torch.nn.functional as F
from hw_asv.model import RawNet2

target_sr = 16000
max_len = 64600
audios_names = [
    'audio_1.wav',
    'audio_2.wav',
    'audio_1.flac',
    'audio_2.flac',
    'audio_3.flac'
]

device = torch.device('cuda:0')

checkpoint = torch.load('final_data/model.pth')

aa = checkpoint['config']['arch']

model = RawNet2(**aa['args'])
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

model.eval()

for name in audios_names:
    path = 'test_data/audio/' + name
    aud, sr = torchaudio.load(path)
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    audio_tensor = audio_tensor[..., :max_len]
    audio_tensor = audio_tensor.to(device)
    preds = model(audio=audio_tensor)
    preds = preds['logits'].squeeze()
    probs = F.softmax(preds)
    print(name, probs.detach().cpu().tolist())
    