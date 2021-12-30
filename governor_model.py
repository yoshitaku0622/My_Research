from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import csv
from pathlib import Path

class gov_model(nn.Module):
    def __init__(self):
        super(gov_model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, False),
            
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=1024, bias=True)
        )

    def forward(self, input):
        output = self.features(input)
        output = self.avgpool(output)
        output = output.view(1, -1) #1x4000
        output = self.classifier(output) #1x1024
        
        return output #D画像の予測