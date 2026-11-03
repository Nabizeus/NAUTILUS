import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next



class Conv2LSTM(nn.Module):
    """
    Spatiotemporal convolutional LSTM-style network for 4D data (B, T, C, H, W).

    Compared to the old version, this one:
      ✅ Keeps spatial structure (no flattening)
      ✅ Updates hidden state via convolution (no global averaging)
      ✅ Allows velocities (U,V,W) to drive local advection-like evolution
    """

    def __init__(self,  input_channels, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Encoder to extract local spatial features
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
        )
        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        # ConvLSTM cell (single-layer recurrent conv)
        # Input = encoded features + hidden state
       

        # Decoder maps hidden state to final tracer output
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Identity() #If tracer is normalized 0..1. For -1..1 use nn.Tanh()
        )

    def forward(self, x, debug=False):
        """
        Forward pass
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, 1, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=device)
        c = torch.zeros_like(h)

        for t in range(T):
            # Encode current frame (spatial features)
            f_t = self.encoder(x[:, t])  # (B, hidden_dim, H, W)

            skip = f_t

            h, c = self.cell(f_t, h, c)


            if debug and t == T - 1:
                print(f"[Conv2LSTM] Step {t}: hidden shape {h.shape}")

        out = self.decoder(h+skip)
        return out


class OceanPredictor(nn.Module):
    """
    Wrapper around Conv2LSTM for predicting tracer evolution.
    Takes multiple input channels (e.g. tracer, U, V, W).
    """

    def __init__(self, cfg, input_channels, H=None, W=None):
        super().__init__()
        self.input_channels = input_channels
        self.H = H
        self.W = W

        print(f"[OceanPredictor] Using dynamic H={self.H}, W={self.W}")

        # Infer number of channels: tracer(s) + velocity components

        forcing_vars = [
            cfg.data.forcing.u_var,
            cfg.data.forcing.v_var,
            cfg.data.forcing.w_var,
        ]

        if hasattr(cfg.data.forcing, "dif_var"):
            forcing_vars.append(cfg.data.forcing.dif_var)

        if hasattr(cfg.data.forcing, "mld_var"):
            forcing_vars.append(cfg.data.forcing.mld_var)



        LEVELS = getattr(cfg.data, "vertical_levels", 1) 
        n_vel = len(cfg.data.forcing.velocity_paths)
        n_tracer = len(cfg.data.tracers.variables)


        input_channels = LEVELS * (
            len(cfg.data.tracers.variables)
            + len(forcing_vars)
        )



        #input_channels = LEVELS * (n_vel + n_tracer)
        print(f"[OceanPredictor] LEVELS={LEVELS}")
        print(f"[OceanPredictor] Velocity vars={n_vel}, Tracer vars={n_tracer}")
        print(f"[OceanPredictor] Total Input Channels={input_channels}")


        hidden_dim = getattr(cfg.model, "hidden_dim", 64)
        kernel_size = getattr(cfg.model, "kernel_size", 3)

        self.model = Conv2LSTM(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )

    def forward(self, x, debug=False):
        return self.model(x, debug=debug)

