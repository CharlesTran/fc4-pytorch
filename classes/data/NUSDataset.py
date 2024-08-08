import os
from typing import Tuple

import numpy as np
import torch
import yaml
import random
import torch.utils.data as data

from auxiliary.utils import *
from auxiliary.settings import *
from classes.data.DataAugmenter import DataAugmenter

BASE = "/home/ubuntu/Desktop/public/colorconstancy/dataset/NUS8"

class NUSDataset(data.Dataset):
    def __init__(self, train: bool = True,):
        self.__train = train
        self.__da = DataAugmenter()

        path_to_trainval = os.path.join(BASE, "trainval.yaml")
        self.__path_to_data = os.path.join(BASE, 'Canon EOS 600D', 'preprocessed', 'numpy_data')
        self.__path_to_label = os.path.join(BASE, 'Canon EOS 600D', 'preprocessed', 'numpy_labels')

        if not os.path.exists(path_to_trainval):
            files = os.listdir(self.__path_to_data)
            random.shuffle(files)
            trainval = {}
            train_split = []
            val_split = []
            for i in range(len(files)):
                if i < 0.8*len(files):
                    train_split.append(files[i])
                else:
                    val_split.append(files[i])
            trainval['train'] = train_split
            trainval['val'] = val_split
            with open(path_to_trainval,'w') as f:
                yaml.dump(data=trainval, stream=f)
        with open(path_to_trainval, 'r') as f:
            folds = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.file_names = folds["train" if self.__train else "val"]

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name)))
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name)))

        if self.__train:
            img, illuminant = self.__da.augment(img, illuminant)
        else:
            img = self.__da.crop(img)

        img = torch.from_numpy(hwc_to_chw(bgr_to_rgb(img)).copy()).to(DEVICE)
        illuminant = torch.from_numpy(illuminant.copy()).to(DEVICE)
        histogram = torch.zeros(2, 64, 64)
        valid_chroma_rgb, valid_colors_rgb = get_hist_colors(
                img, rgb_to_uv)
        histogram[0, :, :] = compute_histogram(
                valid_chroma_rgb, get_hist_boundary(), 64,
                rgb_input=valid_colors_rgb)
        edge_img = compute_edges(img)
        valid_chroma_edges, valid_colors_edges = get_hist_colors(
                edge_img, rgb_to_uv)

            
        histogram[1, :, :] = compute_histogram(
                valid_chroma_edges, get_hist_boundary(), 64,
                rgb_input=valid_colors_edges)
        
        additional_histogram = histogram.to(DEVICE)
        u_coord, v_coord = get_uv_coord(64, normalize=True)
        additional_histogram = torch.cat([additional_histogram, u_coord],
                                          dim=0)
        additional_histogram = torch.cat([additional_histogram, v_coord],
                                            dim=0)
        # additional_histogram = torch.unsqueeze(additional_histogram, axis=0)
        # additional_histogram = to_tensor(additional_histogram, dims=4)
        
        img = linear_to_nonlinear(normalize(img))

        if not self.__train:
            img = img.type(torch.FloatTensor)

        return img, illuminant, file_name, additional_histogram

    def __len__(self) -> int:
        return len(self.__fold_data)

        

