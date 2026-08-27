"""Export trained LSTM models to TorchScript for use with tch-rs in Rust."""

import torch
import torch.nn as nn


class TimeSeriesLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=3, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def export_model(checkpoint_path, output_path, input_size, hidden_size, num_layers, seq_len=12):
    """Load a .pth checkpoint and export as TorchScript on GPU (or CPU fallback)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Exporting on {device}")

    model = TimeSeriesLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Trace with a dummy input on the same device
    dummy_input = torch.randn(1, seq_len, input_size, device=device)
    traced = torch.jit.trace(model, dummy_input)

    traced.save(output_path)
    print(f"Exported {checkpoint_path} -> {output_path}")
    print(f"  Input shape: (1, {seq_len}, {input_size}) on {device}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 'unknown'):.4f}")


if __name__ == "__main__":
    # Model1: full feature set (336 features), 4 layers, hidden 336, seq_len 1
    # seq_len=1 matches inference (single timestep, negative indexing in forward)
    export_model(
        checkpoint_path="checkpoints/LSTM_A1_720_100.pth",
        output_path="checkpoints/LSTM_A1_720_100.pt",
        input_size=336,
        hidden_size=336,
        num_layers=4,
        seq_len=1,
    )

    # Model2: columnB features (96 features), 4 layers, hidden 96, seq_len 1
    export_model(
        checkpoint_path="checkpoints/LSTM_A2_720_100.pth",
        output_path="checkpoints/LSTM_A2_720_100.pt",
        input_size=96,
        hidden_size=96,
        num_layers=4,
        seq_len=1,
    )

    print("Done.")
