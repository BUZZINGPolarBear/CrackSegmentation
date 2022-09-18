import os
import cv2 as cv
import numpy as np
import glob
import torch
from torch.utils import data
from PIL import Image
import math


class CrackDataSet(data.Dataset):
    """
    크랙 DataSet 클래스. 파이토치의 Dataset클래스 상속

    Args:
        img_list: 학습 데이터 명
        anno_list: annotation 데이터 명
        type: train데이터인지, test데이터인지 설정
    """
    train_set_ratio = 0.8
    valid_set_ratio = 0.2

    def __init__(self, data_root, type='train'):
        train_img_list = os.listdir(data_root + '/data/image/image')
        train_anno_list = os.listdir(data_root + '/data/image/seg')
        self.type = type

        if(self.type == 'train'):
            # ! 파일의 이름 list 앞에 전체 경로를 붙여줍니다.
            for i in range(math.floor(len(train_img_list) * self.train_set_ratio)):
                train_img_list[i] = data_root + '/data/image/image/' + train_img_list[i]

            for i in range(math.floor(len(train_anno_list) * self.train_set_ratio)):
                train_anno_list[i] = data_root + '/data/image/seg/' + train_anno_list[i]
            self.img_list = train_img_list
            self.anno_list = train_anno_list
        elif (self.type == 'val'):
            # ! 파일의 이름 list 앞에 전체 경로를 붙여줍니다.
            for i in range(math.floor(len(train_img_list) * self.train_set_ratio), math.floor(len(train_img_list) * self.valid_set_ratio)):
                train_img_list[i] = data_root + '/data/image/image/' + train_img_list[i]

            for i in range(math.floor(len(train_img_list) * self.train_set_ratio), math.floor(len(train_anno_list) * self.valid_set_ratio)):
                train_anno_list[i] = data_root + '/data/image/seg/' + train_anno_list[i]
            self.img_list = train_img_list
            self.anno_list = train_anno_list



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_data_path = self.img_list[index]
        img = Image.open(img_data_path)

        label = self.anno_list[index]

        # img, label = transform(img, label)
        return img, label
