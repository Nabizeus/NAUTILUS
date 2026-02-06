import torch
import torch.nn as nn


class ConvLSTMHybrid(nn.Module):
    """
    CNN + LSTM hybrid for spatiotemporal 4D data (B, T, C, Z, Y, X)
    """
    def __init__(self, input_channels=4, hidden_dim=64, kernel_size=3, depth_channels=75):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((depth_channels // 5, 8, 8))  # compress space
        )

        # Much smaller input to LSTM now
        lstm_input_dim = hidden_dim * (depth_channels // 5) * 8 * 8
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, depth_channels * 32 * 32),  # modest upsample
        )

        self.depth_channels = depth_channels

    def forward(self, x):
        B, T, C, Z, Y, X = x.shape
        feats = []
        for t in range(T):
            ft = self.conv3d(x[:, t])       # (B, hidden_dim, Z', Y', X')
            ft = ft.flatten(1)              # (B, reduced_features)
            feats.append(ft)
        feats = torch.stack(feats, dim=1)   # (B, T, features)

        lstm_out, _ = self.lstm(feats)      # (B, T, hidden_dim)
        lstm_last = lstm_out[:, -1, :]      # (B, hidden_dim)

        y = self.decoder(lstm_last)         # (B, depth_channels * 32 * 32)
        y = y.view(B, 1, self.depth_channels, 32, 32)
        return y



# ================================================================
# 🔹 Wrapper for full model training pipeline
# ================================================================
class OceanPredictor(nn.Module):
    """
    Wrapper for the ocean tracer forecast model.
    Uses ConvLSTMHybrid as the core backbone.
    """
    def __init__(self, cfg):
        super().__init__()
        input_channels = 4 #len(cfg.data.neighbors.velocity_paths)  # U, V, W → 3
        hidden_dim = cfg.model.hidden_dim if "model" in cfg and "hidden_dim" in cfg.model else 64
        kernel_size = cfg.model.kernel_size if "model" in cfg and "kernel_size" in cfg.model else 3
        depth_channels = cfg.model.depth_channels if "model" in cfg and "depth_channels" in cfg.model else 75

        self.model = ConvLSTMHybrid(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            depth_channels=depth_channels,
        )

    def forward(self, x):
        return self.model(x)
