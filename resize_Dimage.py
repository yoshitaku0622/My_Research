from PIL import Image
import numpy as np
import pathlib
from natsort import natsorted
Image.MAX_IMAGE_PIXELS = 933120000




def makeDimage():
    # 現在のディレクトリを取得
    current_dir = pathlib.Path(__file__).resolve().parent #resolveは絶対パス

    # データセットのルートパス
    rootpath = current_dir / 'img_data'

    D_list = []
    for p in sorted(rootpath.glob('*/*/*_d.jpg')):
        D_list.append(p)
    
    D_list = natsorted(D_list)


    for D_path in D_list:
        # TODO 画像を読み込む
        img = Image.open(D_path)
        print(f"画像サイズ{img}")
        img = np.array(img)
        heights, widths, _ = img.shape

        # 保存する画像は1 x widthsの画像サイズ
        new_img_ndarray = np.zeros((1, widths, 3), np.uint8)
        print(new_img_ndarray)
        # 左から順に見ていく
        for width in range(widths):
            col = img[:,width]
            col_R = col[:,0]
            col_G = col[:,1]
            col_B = col[:,2]

            R_avg = int(np.average(col_R))
            G_avg = int(np.average(col_G))
            B_avg = int(np.average(col_B))
            new_img_ndarray[:,width] = [R_avg,G_avg,B_avg]
            new_img = Image.fromarray(new_img_ndarray)

        new_img.save(f"./governor_Ddata/{D_path.name}")

makeDimage()