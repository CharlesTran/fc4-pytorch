import os
from typing import Tuple

import numpy as np
import scipy.io
import torch
import torch.utils.data as data

from auxiliary.utils import *
from auxiliary.settings import *
from classes.data.DataAugmenter import DataAugmenter

BASE = "/home/ubuntu/Desktop/public/colorconstancy/dataset/gehler"

class ColorCheckerDataset(data.Dataset):

    def __init__(self, train: bool = True, folds_num: int = 1):

        self.__train = train
        self.__da = DataAugmenter()

        path_to_folds = os.path.join(BASE, "folds.mat")
        path_to_metadata = os.path.join(BASE, "metadata.txt")
        self.__path_to_data = os.path.join(BASE, "preprocessed", "numpy_data")
        self.__path_to_label = os.path.join(BASE, "preprocessed", "numpy_labels")
        self.__path_to_hist = os.path.join(BASE, "preprocessed", "hist")
        
        folds = scipy.io.loadmat(path_to_folds)
        img_idx = folds["tr_split" if self.__train else "te_split"][0][folds_num][0]

        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    def __getitem__(self, index: int) -> Tuple:
        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='float32')
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')

        
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
