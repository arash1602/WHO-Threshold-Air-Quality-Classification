import torch
import torch.nn as nn

class AQM_CNNGRU_Fusion_Model(nn.Module):
    """
    Gated CNN + GRU fusion classifier for SenseurCity AQM.

    Input  : [batch, input_window, input_channels]  (batch-first, same as the
             GRU/TCN/MLP models)
    Output : [batch, output_channels]               (1 raw logit for the
             harmful-air classification task)

    The two branches see the *same* window and process it independently:

        * a causal 1-D convolution captures short, local shape (a sharp ramp in
          PM over a few hours);
        * a GRU carries longer-range state across the whole window.

    Their features are projected to a common width ``fusion_dim`` and combined
    by a learned gate rather than a plain concatenation:

        g     = sigmoid(W [f_cnn ; f_gru])
        fused = g * f_cnn + (1 - g) * f_gru

    ``g`` is a per-unit weight in [0, 1], so the model can lean on either branch
    per feature and per sample instead of committing to a fixed split. Reading
    the mean gate after training says which branch the model actually used.

    The convolution is *causal*: left-padding by ``kernel_size - 1`` with no right
    padding means output position t depends only on input positions <= t. ``padding='same'`` would pad on the right and mix
    a zero-padding artifact into the final timestep, which is exactly the
    position both branches are read from.
    """

    def __init__(
        self,
        input_channels: int = 5,
        cnn_channels: int = 32,
        kernel_size: int = 3,
        gru_hidden: int = 32,
        num_layers: int = 1,
        fusion_dim: int = 32,
        dropout: float = 0.1,
        output_channels: int = 2,
    ):
        super().__init__()

        # --- CNN branch: causal conv over time -----------------------------
        # Conv1d wants [batch, channels, time], so forward() permutes; the
        # left-pad + chomp pair is what keeps it causal.
        self.conv = nn.Sequential(
            nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
            nn.Conv1d(input_channels, cnn_channels, kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cnn_proj = nn.Linear(cnn_channels, fusion_dim)

        # --- GRU branch ----------------------------------------------------
        # nn.GRU's own dropout only applies *between* stacked layers, so it is
        # inert at num_layers=1; the explicit Dropout below always applies.
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_drop = nn.Dropout(dropout)
        self.gru_proj = nn.Linear(gru_hidden, fusion_dim)

        # --- Gated fusion ---------------------------------------------------
        self.gate = nn.Linear(2 * fusion_dim, fusion_dim)
        self.head = nn.Linear(fusion_dim, output_channels)

        print(
            f"AQM_CNNGRU_Fusion_Model | cnn_channels={cnn_channels} "
            f"| kernel_size={kernel_size} | gru_hidden={gru_hidden} "
            f"| num_layers={num_layers} | fusion_dim={fusion_dim} "
            f"| dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_channels]
        c = self.conv(x.permute(0, 2, 1))     # -> [batch, cnn_channels, seq_len]
        f_cnn = self.cnn_proj(c[:, :, -1])    # last time step -> [batch, fusion_dim]

        out, _ = self.gru(x)                  # -> [batch, seq_len, gru_hidden]
        f_gru = self.gru_proj(self.gru_drop(out[:, -1]))

        g = torch.sigmoid(self.gate(torch.cat([f_cnn, f_gru], dim=-1)))
        fused = g * f_cnn + (1.0 - g) * f_gru
        return self.head(fused)               # -> [batch, output_channels]

    @torch.no_grad()
    def branch_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Mean gate value per sample: ~1 leans on the CNN branch, ~0 on the GRU.

        Not used in training — an interpretability hook for reporting which
        branch the fused model actually relies on.
        """
        c = self.conv(x.permute(0, 2, 1))
        f_cnn = self.cnn_proj(c[:, :, -1])
        out, _ = self.gru(x)
        f_gru = self.gru_proj(out[:, -1])
        return torch.sigmoid(self.gate(torch.cat([f_cnn, f_gru], dim=-1))).mean(dim=-1)
