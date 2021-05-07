import cv2
import os
import random
import glob

import numpy as np

class datacreate:
    def __init__(self):
        self.num = 0
        self.mag = 4

#任意のフレーム数を切り出すプログラム
    def datacreate(self,
                img_path,     #切り取る動画が入ったファイルのpath
                data_number,  #データセットの生成数
                cut_frame,    #1枚の画像から生成するデータセットの数
                HR_height,    #HRの保存サイズ
                HR_width,
                ext='jpg'):

        LR_height = HR_height  #低解像度画像のsize = 高解像度のsize
        LR_width = HR_width 

        low_data_list = []
        high_data_list = []

        path = img_path + "/*"
        files = glob.glob(path)

        while self.num < data_number:
            photo_num = random.randint(0, len(files) - 1)
            img = cv2.imread(files[photo_num])
            height, width = img.shape[:2]

            if HR_height > height or HR_width > width:
                break
                
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            gray = color_img[:, :, 0]      
            bicubic_img = cv2.resize(gray , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)
            bicubic_img = cv2.resize(bicubic_img , (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

            for i in range(cut_frame):
                ram_h = random.randint(0, height - LR_height)
                ram_w = random.randint(0, width - LR_width)

                LR_img = bicubic_img[ram_h : ram_h + LR_height, ram_w: ram_w + LR_width]
                high_img = gray[ram_h : ram_h + HR_height, ram_w: ram_w + HR_width]

                low_data_list.append(LR_img)
                high_data_list.append(high_img)

                self.num += 1

                if self.num == data_number:
                    break

        return low_data_list, high_data_list
