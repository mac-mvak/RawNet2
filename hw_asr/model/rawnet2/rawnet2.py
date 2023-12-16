import torch
import torch.nn as nn
import torch.nn.functional as F
from .sinc_conv import SincConv_fast
from .blocks import Residual_block



class RawNet2(nn.Module):
    def __init__(self, sinc_channels, sinc_kernel,
                 res_channels_1,
                 res_channels_2,
                 gru_hidden, abs=True) -> None:
        super().__init__()
        self.abs = abs
        self.sinc_block = SincConv_fast(sinc_channels, sinc_kernel)
        self.after_sinc = nn.Sequential(
            nn.MaxPool1d(3),
            nn.BatchNorm1d(sinc_channels),
            nn.LeakyReLU(0.3)
        )
        resblocks = [Residual_block(sinc_channels, res_channels_1, first=True),
                     Residual_block(res_channels_1, res_channels_2)] 
        resblocks = resblocks + [Residual_block(res_channels_2, res_channels_2) for _ in range(4)]
                    
        self.resblock = nn.Sequential(*resblocks)

        self.before_gru = nn.Sequential(nn.BatchNorm1d(res_channels_2),
                                        nn.LeakyReLU(0.3))
        self.gru = nn.GRU(res_channels_2, gru_hidden, num_layers=3, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 2)


    def forward(self, audio, **kwargs):
        out = self.sinc_block(audio.unsqueeze(1))
        if self.abs:
            out = torch.abs(out)
        out = self.after_sinc(out)
        out = self.resblock(out)
        out = self.before_gru(out)
        out = out.transpose(1, 2)
        gru_out, _ = self.gru(out)
        gru_out = gru_out[:, -1, :]
        logits = self.fc(gru_out)
        return {"logits": logits}





















