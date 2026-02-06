import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv2LSTM(nn.Module):
    """
    CNN + LSTM hybrid for spatiotemporal 4D data (B, T, C, H, W)
    """
    def __init__(self, input_channels, hidden_dim=64, kernel_size=3, H=None, W=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        print(f"Inside class Conv2LSTM(nn.Module)__init_: {input_channels}")

        #CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # reduce spatial resolution by 2
        )

        
    
       
        # If H and W are provided, compute the flattened feature size after encoding
        if H is not None and W is not None:
            self.flattened_size = hidden_dim * (H // 2) * (W // 2)
        else:
            self.flattened_size = None  # can infer at runtime from a sample batch


        # LSTM — input_size determined later if not given
        self.lstm = None # will be initialized dynamically if needed

        # Decoder (to map LSTM output back to spatial map)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, debug=False):
        """
        Forward pass for Conv2LSTM.
        Logs all major shape transformations for debugging.
        """
        
        # --- NaN/Inf safety check on input ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            if debug:
                print("⚠️ [Conv2LSTM] NaN/Inf detected in input! Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        
        if debug:
            print(f"🟦 [Conv2LSTM] Input x shape: {x.shape}")  # (B, T, C, H, W)
        # x: (B, Ti, C, H, W) (batch, time, channels, height, width)
    
        
    
    

        B, T, C, H, W = x.shape
        self.Hgt = H
        self.Wdt = W


        encoded_seq = []

        # Run encoder over each time step
        for t in range(T):
            feats = self.encoder(x[:, t])       # (B, hidden_dim, H_enc, W_enc)
            if t == 0:
                H_enc, W_enc = feats.shape[-2:] # remember spatial size after encoding
                if debug:
                    print(f"🟪 [Conv2LSTM] Encoded spatial size: H_enc={H_enc}, W_enc={W_enc}")

            feats_flat = feats.flatten(1)       # (B, hidden_dim * H_enc * W_enc)
            encoded_seq.append(feats_flat)

        # stack into temporal sequence
        encoded = torch.stack(encoded_seq, dim=1)   # (B, T, F)
        F_dim = encoded.size(-1)
        if debug:
            print(f"🟩 [Conv2LSTM] Encoded sequence shape: {encoded.shape}")

        # Initialize LSTM lazily (first forward pass)
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=F_dim,
                hidden_size=self.hidden_dim,
                batch_first=True
            ).to(x.device)

        # temporal modeling
        lstm_out, _ = self.lstm(encoded)            # (B, T, hidden_dim)
        if debug:
            print(f"🟧 [Conv2LSTM] LSTM output shape: {lstm_out.shape}")
        last_hidden = lstm_out[:, -1, :]            # (B, hidden_dim)

        # project back to 2D feature map
        spatial = last_hidden.view(B, self.hidden_dim, 1, 1)
        spatial = spatial.expand(-1, -1, H_enc, W_enc)  # broadcast spatially
        if debug:
            print(f"🟥 [Conv2LSTM] Spatial projection shape: {spatial.shape}")

        # Decode and upscale
        out = self.decoder(spatial)                     # (B, 1, H, W)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        if debug:
            print(f"✅ [Conv2LSTM] Output shape: {out.shape}")



        # --- NaN/Inf safety check on output ---
        if torch.isnan(out).any() or torch.isinf(out).any():
            if debug:
                print("⚠️ [Conv2LSTM] NaN/Inf detected in output! Clamping.")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)

        return out

class OceanPredictor(nn.Module):
    """
    Wrapper for the ocean tracer forecast model.
    Uses Conv2LSTM as the core backbone.
    """
    def __init__(self, cfg):
        super().__init__()

        input_channels = len(cfg.data.neighbors.velocity_paths) #if "data" in cfg and "neighbors" in cfg.data else 4
        if hasattr(cfg.data.tracers, "variables"):
            input_channels += len(cfg.data.tracers.variables)
        hidden_dim = getattr(cfg.model, "hidden_dim", 64)
        kernel_size = getattr(cfg.model, "kernel_size", 3)

        

        #dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)

        #H = cfg.data.spatial_crop[1]  # e.g., 292 latitude
        #W = cfg.data.spatial_crop[3]   # e.g., 362 longitude


        
        H = 20 
        W = 20 


        
        
        print(f"H {H}")
        print(f"W {W}")



        self.model = Conv2LSTM(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            H=H,
            W=W
        )

    def forward(self, x):
        return self.model(x)

