import torch.nn as nn
import torch.nn.functional as F


class FMS(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=n_features, out_features=n_features)

    def forward(self, x):
        pooled = self.avg(x).squeeze(-1)
        pooled = self.fc(pooled)
        pooled = F.sigmoid(pooled).unsqueeze(-1)
        return x * pooled + pooled


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, first = False):
        super(Residual_block, self).__init__()
        self.blocks = nn.Sequential()
        self.first = first
        
        if not self.first:
            self.blocks.append(nn.BatchNorm1d(num_features = in_channels))
            self.blocks.append(nn.LeakyReLU(negative_slope=0.3))
        
        self.blocks.append(nn.Conv1d(in_channels = in_channels,
			out_channels = out_channels,
			kernel_size = 3,
			padding = 1,
			stride = 1))
        
        self.blocks.append(nn.BatchNorm1d(num_features = out_channels))
        self.blocks.append(nn.LeakyReLU(negative_slope=0.3))
        self.blocks.append(nn.Conv1d(in_channels = out_channels,
			out_channels = out_channels,
			padding = 1,
			kernel_size = 3,
			stride = 1))
        
        if in_channels != out_channels:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = in_channels,
				out_channels = out_channels,
				kernel_size = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        self.fms = FMS(out_channels)
        
    def forward(self, x):
        out = self.blocks(x)
        if self.downsample:
            x = self.conv_downsample(x)
            
        out = out + x
        out = self.mp(out)
        out = self.fms(out)
        return out




