import torch
import torch.nn as nn
from src.model import Conv2LSTM   # reuse your working model :contentReference[oaicite:0]{index=0}


class OceanConv2DAsymXPU(nn.Module):
    def __init__(self, cfg, input_channels):
        super().__init__()

        print("🚀 USING VECTORIZED ASYMMETRIC Conv2DLSTM (XPU)")

        self.channels_per_depth_stack = input_channels
        self.max_below = cfg.model.max_below
        self.max_above = cfg.model.max_above
        self.max_window = self.max_below + self.max_above + 1
        self.hidden_dim = cfg.model.hidden_dim


        # Reduce large channel explosion
        self.input_reduce = nn.Conv2d(
            in_channels=self.channels_per_depth_stack * self.max_window,
            out_channels=self.hidden_dim,
            kernel_size=1
        )

        self.model2d = Conv2LSTM(
            input_channels=self.hidden_dim,
            hidden_dim=self.hidden_dim
        )

    def forward(self, x):
        """
        x: (B, T, C, Z, H, W)
        """

        B, T, C, Z, H, W = x.shape
        device = x.device

        # ----------------------------------
        # STEP 1: build index matrix (Z, window)
        # ----------------------------------
        z_idx = torch.arange(Z, device=device)

        window_indices = []
        for z in range(Z):
            below = min(self.max_below, z)
            above = min(self.max_above, Z - z - 1)

            z_min = z - below
            z_max = z + above + 1

            idx = torch.arange(z_min, z_max, device=device)

            # pad to max_window
            if len(idx) < self.max_window:
                pad = idx[-1].repeat(self.max_window - len(idx))
                idx = torch.cat([idx, pad])

            window_indices.append(idx)

        window_indices = torch.stack(window_indices)  # (Z, window)

        # ----------------------------------
        # STEP 2: gather depth windows
        # ----------------------------------
        # x: (B, T, C, Z, H, W)
        # -> expand indices for gather

        x = x.permute(0, 1, 3, 2, 4, 5)  # (B,T,Z,C,H,W)

        # expand indices
        idx = window_indices[None, None, :, :, None, None]  # (1,1,Z,window,1,1)
        idx = idx.expand(B, T, Z, self.max_window, H, W)

        x = x.unsqueeze(3).expand(B, T, Z, self.max_window, C, H, W)

        # gather along Z dimension
        x = torch.gather(x, 2, idx.unsqueeze(4).expand(-1,-1,-1,-1,C,-1,-1))

        # result: (B,T,Z,window,C,H,W)

        # ----------------------------------
        # STEP 3: reshape
        # ----------------------------------
        x = x.permute(0, 2, 1, 4, 3, 5, 6)  # (B,Z,T,C,window,H,W)
        x = x.reshape(B * Z, T, C * self.max_window, H, W)

        # ----------------------------------
        # STEP 4: channel reduction (VERY IMPORTANT)
        # ----------------------------------
        x = torch.stack([
            self.input_reduce(x[:, t])
            for t in range(T)
        ], dim=1)

        # ----------------------------------
        # STEP 5: Conv2DLSTM (parallel!)
        # ----------------------------------
        out = self.model2d(x)  # (B*Z,1,H,W)

        # ----------------------------------
        # STEP 6: reshape back
        # ----------------------------------
        out = out.reshape(B, Z, 1, H, W)
        out = out.permute(0, 2, 1, 3, 4)  # (B,1,Z,H,W)

        z_center = out.shape[2] // 2
        return out[:, :, z_center]
