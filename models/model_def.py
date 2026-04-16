import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, pool=False):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class IntelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Stage 1: Input 150x150 -> 75x75
        self.stage1 = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64, pool=True),
        )
        # Stage 2: 75x75 -> 37x37
        self.stage2 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128, pool=True),
        )
        # Stage 3: 37x37 -> 18x18
        self.stage3 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256, pool=True),
        )
        # Stage 4: 18x18 -> 9x9
        self.stage4 = nn.Sequential(
            conv_block(256, 512),
            conv_block(512, 512, pool=True),
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)