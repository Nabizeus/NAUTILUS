import torch
import torch.nn as nn


class ConvLSTMHybrid(nn.Module):
    """
    CNN + LSTM hybrid for spatiotemporal 4D data (T, C, Z, Y, X)
    """
    def __init__(self, input_channels=4, hidden_dim=64, kernel_size=3, depth_channels=75):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim * depth_channels,
            hidden_size=hidden_dim * depth_channels,
            batch_first=True,
        )
        self.decoder = nn.Conv3d(hidden_dim, 1, 1)

    def forward(self, x):
        # x: (B, T, C, Z, Y, X)
        B, T, C, Z, Y, X = x.shape
        x = x.view(B * T, C, Z, Y, X)
        feats = self.conv3d(x)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        lstm_last = lstm_out[:, -1, :]
        lstm_last = lstm_last.view(B, -1, Z, Y, X)
        y = self.decoder(lstm_last)
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
