import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Building blocks
# ---------------------------------------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, 3, padding=1)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

# ---------------------------------------------------------
# UNet + ConvLSTM Model
# ---------------------------------------------------------
class OceanPredictorUNet(nn.Module):
    def __init__(self, input_channels, base_channels=32, out_channels=1):
        super().__init__()

        self.inc = conv_block(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        self.bottleneck = conv_block(base_channels * 8, base_channels * 8)
        self.convlstm = ConvLSTMCell(base_channels * 8, base_channels * 8)

        self.up3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = Up(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        skips = []
        for t in range(T):
            xt = x[:, t]
            s0 = self.inc(xt)
            s1 = self.down1(s0)
            s2 = self.down2(s1)
            s3 = self.down3(s2)
            skips.append((s0, s1, s2, s3))

        device = x.device
        _, _, Hb, Wb = skips[0][-1].shape
        h = torch.zeros(B, skips[0][-1].shape[1], Hb, Wb, device=device)
        c = torch.zeros_like(h)

        for t in range(T):
            b = self.bottleneck(skips[t][-1])
            h, c = self.convlstm(b, h, c)

        s0, s1, s2, _ = skips[-1]
        x = self.up3(h, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)

        return self.out_conv(x)

