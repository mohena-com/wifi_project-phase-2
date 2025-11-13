import torch
import torch.nn as nn

class CSILSTMNet(nn.Module):
    def __init__(self, csi_input_size, meta_input_size, window_size,
                 num_classes, hidden_size=64, num_layers=2, dropout_p=0.3):
        super().__init__()

        # Add dropout inside LSTM (between layers)
        self.csi_lstm = nn.LSTM(
            csi_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        self.meta_lstm = nn.LSTM(
            meta_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        # Dropout before classifier
        self.dropout = nn.Dropout(p=dropout_p)

        # Final classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, csi_seq, meta_seq):
        _, (h_csi, _) = self.csi_lstm(csi_seq)
        _, (h_meta, _) = self.meta_lstm(meta_seq)

        h_csi_last = h_csi[-1]
        h_meta_last = h_meta[-1]

        combined = torch.cat([h_csi_last, h_meta_last], dim=1)
        combined = self.dropout(combined)  # dropout active in train()

        out = self.fc(combined)
        return out
