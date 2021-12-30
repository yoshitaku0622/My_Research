import glob
import logging
import os.path as osp
import pathlib
import pickle
from pathlib import Path
from posixpath import pathsep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from get_dataset import Image_dataset
from governor_model import gov_model
from numpy.core.arrayprint import set_string_function
from PIL import Image, ImageCms
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from gov_model import GovModel

def main():
    result_dir = Path("result")/"6th"
    result_dir.mkdir(exist_ok=True)
    df = pd.read_csv("/home/governor/My_Research/governer_data.csv")
    #df.shape = (102, 3)
    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.1, n_splits=2, random_state=7).split(df, groups=df["patient_id"]))
    #print(train_inds.size)
    train, valid = df.iloc[train_inds], df.iloc[test_inds]
    train_dataset = Image_dataset(train)
    val_dataset = Image_dataset(valid)

    # Dataloaderの作成
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # GPU初期設定
    # ネットワークモデルをimport
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ネットワークをGPUへ
    # tetsu modelに変更
    net = GovModel()
    net.to(device)

    # 損失関数の設定
    criterion = nn.L1Loss()

    # 最適化手法の設定
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 損失値を保持するリスト
    train_loss_list = []
    val_loss_list = []
    
    num_epochs = 3

    # 学習、検証
    for epoch in range(num_epochs):
        print("Epoch {} / {} ".format(epoch + 1, num_epochs))
        print("----------")
        # 学習
        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0
        net.train()  # trainモード

        for inputs, labels in train_dataloader:
            # GPU初期設定
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # GPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # ネットワークにデータ入力
            outputs = net(inputs)
            # 損失値の計算
            loss = criterion(outputs, labels)
            # バックプロパゲーションで重み更新
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()  # epoch_lossに追加

        # epochごとのlossを計算
        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        train_loss_list.append(epoch_train_loss)  # train_loss_listの格納
        print("Train Loss: {:.4f}".format(epoch_train_loss))

        # 検証
        net.eval()  # valモード
        # GPU初期設定
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                # GPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)
                # ネットワークにデータ入力
                outputs = net(inputs)
                # 損失値の計算
                loss = criterion(outputs, labels)
                #print("output_size", outputs.size())
                #print("labels_size", labels.size())

                epoch_valid_loss += loss.item()
            # epochごとのlossの計算
            epoch_valid_loss = epoch_valid_loss / len(val_dataloader)
            print("Val Loss: {:.4f}".format(epoch_valid_loss))
            val_loss_list.append(epoch_valid_loss)  # val_loss_listの格納

    # テスト
    gen_dir = result_dir / "gen_images"
    gen_dir.mkdir(exist_ok=True)
    output_list = []

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            # GPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)
            # ネットワークにデータ入力
            outputs = net(inputs)
            output_list.append(outputs.cpu().numpy())
    outputs = np.concatenate(output_list, axis=0)

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")

    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")

    for i, path in enumerate(valid["pathological_path"]):
        name = path.split("/")[-1].replace("tif", "jpg")
        img = Image.fromarray((outputs[i].transpose(1, 2, 0)*255).astype(np.uint8), mode = "LAB").resize((196, 30))
        #print(img.size): (196, 30)
        img = ImageCms.applyTransform(img, lab2rgb_transform)
        img.save(str(gen_dir / name))


    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, num_epochs + 1, 1), train_loss_list, label="train_loss")
    ax_loss.plot(range(1, num_epochs + 1, 1), val_loss_list, label="val_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    fig_loss.savefig(result_dir / "loss.png")

if __name__ == "__main__":
    main()