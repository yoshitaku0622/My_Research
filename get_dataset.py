import numpy as np
import torch.utils.data as data
from PIL import Image, ImageCms
from torchvision import transforms

class Image_dataset(data.Dataset):
    def __init__(self, df):
        self.df = df
        #D画像transform
        self.d_transform = transforms.Compose([
            transforms.Resize((1, 196)),
            transforms.ToTensor(),
        ])

        #P画像tranform
        self.p_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile  = ImageCms.createProfile("LAB")

        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        p_path = self.df["pathological_path"].iloc[index]
        p_img = Image.open(p_path).convert("RGB")
        p_img = self.p_transform(p_img)

        d_path = self.df["dermoscopy_path"].iloc[index]
        d_img = Image.open(d_path).convert("RGB")
        #Dimgmをrgbからlabに変換

        d_img = ImageCms.applyTransform(d_img, self.rgb2lab_transform)
        d_img = self.d_transform(d_img)

        return p_img, d_img