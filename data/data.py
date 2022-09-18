import os
import cv2 as cv
import numpy as np
import glob
import torch
from torch.utils import data
from PIL import Image


class CrackDataSet(data.Dataset):
    """
    크랙 DataSet 클래스. 파이토치의 Dataset클래스 상속

    Args:
        img_list: 학습 데이터 명
        anno_list: annotation 데이터 명
        type: train데이터인지, test데이터인지 설정
    """

    def __init__(self, data_root, type='train'):
        train_img_list = os.listdir(data_root + '/data/image/image')
        train_anno_list = os.listdir(data_root + '/data/image/seg')

        # ! 파일의 이름 list 앞에 전체 경로를 붙여줍니다.
        for i in range(len(train_img_list)):
            train_img_list[i] = data_root + '/data/image/image/' + train_img_list[i]

        for i in range(len(train_anno_list)):
            train_anno_list[i] = data_root + '/data/image/seg/' + train_anno_list[i]
        self.img_list = train_img_list
        self.anno_list = train_anno_list

        self.type = type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_data_path = self.img_list[index]
        img = Image.open(img_data_path)

        anno_data_path = self.anno_list[index]
        label = Image.open(anno_data_path)

        # img, label = transform(img, label)
        return img, label
