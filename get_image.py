from PIL import Image
import numpy as np
import pickle
import torch
import pathlib
import glob
from PIL import Image
from get_path_img import get_Pimg, get_Dimg, get_dir
from sklearn.model_selection import train_test_split

    
dir_list = get_dir() #ディレクトリのパスのリスト取得
train_dir_list, val_dir_list = train_test_split(dir_list) #listをtrain,valに分ける

#PとDのtrainのリスト取ってくる
P_train_path_list = []
D_train_path_list = []

for dir in train_dir_list:
    dir_path = pathlib.Path(dir)
    
    D_tmp = '' #D画像のコピーを定義
    for file in dir_path.glob('*.jpg'):
        D_tmp = file #D画像のコピー作成
    
    for file in dir_path.glob('*.tif'):
        P_train_path_list.append(file)
        D_train_path_list.append(D_tmp)

#PとDのvalのリスト取ってくる
P_val_path_list = []
D_val_path_list = []

for dir in val_dir_list:
    dir_path = pathlib.Path(dir)
    
    D_tmp = ''
    for file in dir_path.glob('*.jpg'):
        D_tmp = file #D画像のコピー作成
    
    for file in dir_path.glob('*.tif'):
        P_val_path_list.append(file)
        D_val_path_list.append(D_tmp)

P_train = get_Pimg(P_train_path_list)
P_val = get_Pimg(P_val_path_list)
D_train = get_Dimg(D_train_path_list)
D_val = get_Dimg(D_val_path_list)

#変換処理


with open("P_train_transformed.pkl", "wb") as f:
    pickle.dump(P_train, f)

with open("P_val_transformed.pkl", "wb") as f:
    pickle.dump(P_val, f)

with open("D_train_transformed.pkl", "wb") as f:
    pickle.dump(D_train, f)

with open("D_val_transformed.pkl", "wb") as f:
    pickle.dump(D_val, f)