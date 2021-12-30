from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models

#img = Image.open('../')
#image_array = np.array(img, dtype='uint8')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #base = models.resnet18(pretrained=True)
        #self.base = nn.Sequential(*list(base.children())[:-2])
        #self.feature_size = base.fc.in_features

        self.base = models.mobilenet_v2(pretrained=True)
        self.feature_size = self.base.last_channel

    def forward(self, x):
        feat = self.base.features(x) #中間表現
        print("エンコーダ(feat):",feat.shape)
        z = feat.mean(2)
        print("エンコーダ(z):", z.shape)
        #return feat.mean(2)
        return z


def deconv_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
        )

class Decoder(nn.Module):
    def __init__(self, features_size):
        super(Decoder, self).__init__()
        self.feature_size = features_size
        self.base = nn.Sequential(
            
                deconv_bn_relu(self.feature_size, 512), #batch_size, 512, 7
                deconv_bn_relu(512, 512), #batch_size, 512, 7
                deconv_bn_relu(512, 512), #512*7   #batch_size, 512, 7
                nn.Upsample(scale_factor = 7), #512*49 #batch_size, 512, 49
                deconv_bn_relu(512, 256), #batch_size, 256, 49
                deconv_bn_relu(256, 256), #batch_size, 256, 49
                deconv_bn_relu(256, 256), #batch_size, 256, 49
                nn.Upsample(scale_factor = 2),  #batch_size, 256, 98
                deconv_bn_relu(256, 128), #batch_size, 128, 98
                deconv_bn_relu(128, 128), #batch_size, 128, 98
                deconv_bn_relu(128, 128), #batch_size, 128, 98
                nn.Upsample(scale_factor = 2), #512*196    #batch_size, 128, 196
                deconv_bn_relu(128, 64), #batch_size, 64, 196
                deconv_bn_relu(64, 64), #batch_size, 64, 196
                nn.Conv1d(64, 3, 1), #batch_size, 3, 196
                nn.Sigmoid(),
            
        )
    
    def forward(self, x):
        out  = self.base(x).unsqueeze(2)
        #return self.base(x).unsqueeze(2)
        print("デコーダout:", out.shape)
        return out

class GovModel(nn.Module):
    def __init__(self):
        super(GovModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(self.encoder.feature_size)

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)