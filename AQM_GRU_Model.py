import torch
import torch.nn as nn


class AQM_GRU_Model(nn.Module):
    """
    GRU recurrent network for SenseurCity AQM.

    Input  : [batch, input_window, input_channels]  (batch_first)
    Output : [batch, output_channels]               (PM10, PM2.5)

    The final hidden state of the last GRU layer is passed through a
    linear head to produce the predictions.
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_channels: int = 2,
    ):
        super().__init__()

        # nn.GRU dropout only applies between stacked layers (not after the last),
        # and requires num_layers > 1 to have any effect.
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, output_channels)

        print(
            f"AQM_GRU_Model | hidden_size={hidden_size} "
            f"| num_layers={num_layers} | dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_channels]
        out, _ = self.gru(x)          # out: [batch, seq_len, hidden_size]
        return self.head(out[:, -1])  # last time step -> [batch, output_channels]
