from PIL import Image
import numpy as np
import pathlib
from natsort import natsorted
Image.MAX_IMAGE_PIXELS = 933120000

width = 500
height = 600

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

def get_Pimg(P_paths): #病理画像を取得
    P_list = [] #病理画像のリスト

    for x in P_paths:
        img = Image.open(x)
        P_list.append(img) #リストに追加

    return P_list #P画像のリスト返す

def resize_Pimg(P_path_list):
    for img_filename in P_path_list:
        print(img_filename)
        img = Image.open(img_filename)
        img_resize = img.resize((width, height))
        img_resize.save(f"./governor_data/{img_filename.name}")

if __name__ == "__main__":
    P_paths = get_P_pathlist()
    resize_Pimg(P_paths)
    print('病理枚数', len(P_paths))