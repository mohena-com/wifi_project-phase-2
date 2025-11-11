import torch
import torch.nn as nn

# MobileNetV3-like block (1D) — same API as before
class MobileNetV3Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, exp_factor=4, se_ratio=0.25):
        super().__init__()
        mid_channels = max(1, in_channels * exp_factor)
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand_conv = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm1d(mid_channels)
        self.expand_act = nn.SiLU()

        self.depthwise_conv = nn.Conv1d(mid_channels, mid_channels, kernel_size, stride, padding=kernel_size//2,
                                        groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm1d(mid_channels)
        self.depthwise_act = nn.SiLU()

        se_channels = max(1, int(mid_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(mid_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv1d(se_channels, mid_channels, 1),
            nn.Sigmoid()
        )

        self.project_conv = nn.Conv1d(mid_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.expand_act(self.expand_bn(self.expand_conv(x)))
        out = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(out)))
        se_weight = self.se(out)
        out = out * se_weight
        out = self.project_bn(self.project_conv(out))

        if self.use_residual:
            return x + out
        return out


class MobileNetV3_1D_LSTM(nn.Module):
    """
    Bigger MobileNetV3-like 1D feature extractor + stronger LSTM head for metadata.
    Use this when you want more capacity for gait identification.
    """
    def __init__(self, csi_channels, meta_feature_dim, num_classes,
                 width_mult: float = 1.0,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 3,
                 dropout: float = 0.4):
        super().__init__()

        # channel config (you can tune width_mult to scale model)
        def w(x): return max(8, int(x * width_mult))

        self.stem = nn.Sequential(
            nn.Conv1d(csi_channels, w(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(w(32)),
            nn.SiLU()
        )

        # deeper stack of MBConv-like blocks
        self.block1 = MobileNetV3Block1D(w(32), w(48), stride=1, exp_factor=4)
        self.block2 = MobileNetV3Block1D(w(48), w(64), stride=2, exp_factor=4)
        self.block3 = MobileNetV3Block1D(w(64), w(96), stride=2, exp_factor=3)
        self.block4 = MobileNetV3Block1D(w(96), w(160), stride=2, exp_factor=3)

        # projection to a compact vector
        self.head_conv = nn.Sequential(
            nn.Conv1d(w(160), w(256), kernel_size=1, bias=False),
            nn.BatchNorm1d(w(256)),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)   # (B, w(256), 1)
        )

        # LSTM for metadata (make it stronger)
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=meta_feature_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        # classifier head: combine conv features and lstm hidden states
        combined_feat_dim = w(256) + (lstm_hidden * 2)  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(combined_feat_dim, max(256, combined_feat_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(256, combined_feat_dim // 2), num_classes)
        )

    def forward(self, csi_seq, meta_seq):
        # csi_seq: (B, W, C) -> conv expects (B, C, W)
        x = csi_seq.permute(0, 2, 1).contiguous()
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head_conv(x).squeeze(-1)   # (B, w(256))

        # metadata LSTM
        _, (h_n, _) = self.lstm(meta_seq)  # h_n: (num_layers * num_directions, B, hidden_size)
        # take last forward+backward
        if h_n.shape[0] >= 2:
            h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, hidden_size*2)
        else:
            h_cat = h_n[-1]
        combined = torch.cat([x, h_cat], dim=1)

        out = self.classifier(combined)
        return out
