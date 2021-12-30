from PIL import Image
import numpy as np
from numpy.core.arrayprint import set_string_function
from torchvision import models
import pathlib
from natsort import natsorted
import os.path as osp
import glob
from posixpath import pathsep
import re
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_P_pathlist(): #P画像のパスのlist取得

    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent #resolveは絶対パス

    # データセットのルートパス
    rootpath = current_dir / 'img_data'

    P_path_list = []
    for p in sorted(rootpath.glob('*/*/*.tif')):
        P_path_list.append(p)
    
    P_path_list = natsorted(P_path_list)

    return P_path_list # パスのリストを返す

def get_D_pathlist(): #D画像のパスのlist取得

    # 現在のディレクトリを取得(Pathological-Image)
    current_dir = pathlib.Path(__file__).resolve().parent #resolveは絶対パス
   
    # データセットのルートパス
    rootpath = current_dir / 'img_data'

    D_path_list = []
    for p in rootpath.glob('*/*/*_d.jpg'):
        if 'アノテーション' not in str(p):
            D_path_list.append(p)

    D_path_list = natsorted(D_path_list)

    return D_path_list

def get_dir(): #患者のディレクトリを取得

    # 現在のディレクトリを取得(Pathological-Image)
    current_dir = pathlib.Path(__file__).resolve().parent #resolveは絶対パス
   
    # データセットのルートパス
    rootpath = current_dir / 'img_data'

    dir_list = [] #患者のディレクトリのリスト
    for p in sorted(rootpath.glob('*/*/**/')):
        dir_list.append(p)

    return dir_list # 患者のディレクトリのリストを返す

def get_Pimg(P_paths): #病理画像を取得
    P_list = [] #病理画像のリスト

    for x in P_paths:
        img = Image.open(x)
        P_list.append(img) #リストに追加

    return P_list

def get_Dimg(D_img_pahts): #D画像の切片を取得
    Dimage_list = [] #病理画像のリスト

    for x in D_img_pahts:
        img = Image.open(x)
        imgarray = np.array(img) #tiffをnumpyに変換
        Dimage_list.append(imgarray) #リストに追加

    return Dimage_list

if __name__ == "__main__":
    P_paths = get_P_pathlist()
    D_paths = get_D_pathlist()
    P_list = get_Pimg(P_paths)
    print('病理枚数', len(P_paths))
    print('D枚数', len(D_paths))