import os
import sys
from glob import glob
sys.path.append("./")
import cv2
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
import scipy.io
from auxiliary.utils import *
from classes.data.DataAugmenter import DataAugmenter

PA = "/home/ubuntu/Desktop/public/colorconstancy/dataset/NUS8/Canon EOS 600D"
PATH_TO_IMAGES = os.path.join(PA,"png")
PATH_TO_COORDINATES = os.path.join(PA,"CHECKER")
PATH_TO_CC_METADATA = os.path.join(PA,"Canon600D_gt.mat")

BASE_PATH = "preprocessed"
PATH_TO_NUMPY_DATA = os.path.join(PA, BASE_PATH, "numpy_data")
PATH_TO_NUMPY_LABELS = os.path.join(PA, BASE_PATH, "numpy_labels")
PATH_TO_NONLINEAR_IMAGES = os.path.join(PA, BASE_PATH, "nonlinear_images")
PATH_TO_GT_CORRECTED = os.path.join(PA, BASE_PATH, "gt_corrected")

def convert_to_8bit(arr, clip_percentile):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile, keepdims=True)), 0, 255)
    return arr.astype(np.uint8)


def mat_illum(path):
    index_list = glob(path + '\\real_illum\*.mat')
    mat = []
    for i in range(len(index_list)):
        mat_load = scipy.io.loadmat(index_list[i],  squeeze_me=True, struct_as_record = False)
        mat.append(mat_load['real_illum'].real_illum_by_cc19_PNG)
    return mat

def white_balance_image(img, filename, Gain_R, Gain_G, Gain_B):    
    img = img * 100.0; #12bit data
    image = convert_to_8bit(img, 2.5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(PATH_TO_NONLINEAR_IMAGES, filename+'.png'), image)
    image[:, :, 0] = np.minimum(image[:, :, 0] * Gain_R,255)
    image[:, :, 1] = np.minimum(image[:, :, 1] * Gain_G,255)
    image[:, :, 2] = np.minimum(image[:, :, 2] * Gain_B,255)
            
    gamma = 1/2.2
    image = pow(image, gamma) * (255.0/pow(255,gamma))
    image = np.array(image,dtype=np.uint8)
    image8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image8
def get_mcc_coord(file_name: str) -> np.ndarray:
    """ Computes the relative MCC coordinates for the given image """
    lines = open(os.path.join(PATH_TO_COORDINATES, file_name.split('.')[0] + "_mask.txt"), 'r').readlines()
    roi = list(map(float, lines[0].strip().split(',')))
    return roi

def load_image_without_mcc(file_name: str, roi: list):
    img12 = cv2.imread(os.path.join(PATH_TO_IMAGES, file_name+'.PNG'), cv2.IMREAD_UNCHANGED).astype(np.float32)
    img12 = np.maximum(0.,img12 - 2048.)
    img = np.clip(img12/img12.max(), 0, 1)*(2**12-1)
    x, y, w, h = map(int, roi)
    img[y:y+h, x:x+w] = 1e-5
    return img

def main():
    print("\n=================================================\n")
    print("\t Masking MCC charts")
    print("\n=================================================\n")
    print("Paths: \n"
          "\t - Numpy data generated at ..... : {} \n"
          "\t - Numpy labels generated at ... : {} \n"
          "\t - Images fetched from ......... : {} \n"
          "\t - Coordinates fetched from .... : {} \n"
          .format(PATH_TO_NUMPY_DATA, PATH_TO_NUMPY_LABELS, PATH_TO_IMAGES, PATH_TO_COORDINATES))

    os.makedirs(PATH_TO_NUMPY_DATA, exist_ok=True)
    os.makedirs(PATH_TO_NUMPY_LABELS, exist_ok=True)
    os.makedirs(PATH_TO_GT_CORRECTED, exist_ok=True)
    os.makedirs(PATH_TO_NONLINEAR_IMAGES, exist_ok=True)

    mat = scipy.io.loadmat(PATH_TO_CC_METADATA,  squeeze_me=True, struct_as_record = False)
    flist = glob(PATH_TO_IMAGES + '/*.PNG')

    for i in tqdm(range(len(flist)), desc="Preprocessing images"):
        filename = mat['all_image_names'][i]

        img_without_mcc=load_image_without_mcc(filename, get_mcc_coord(filename))
        np.save(os.path.join(PATH_TO_NUMPY_DATA, filename), img_without_mcc)

        illuminant = [float(mat['groundtruth_illuminants'][i][0]), float(mat['groundtruth_illuminants'][i][1]), float(mat['groundtruth_illuminants'][i][2])]
        np.save(os.path.join(PATH_TO_NUMPY_LABELS, mat['all_image_names'][i]), illuminant)

        Gain_R= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][0]))
        Gain_G= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][1]))
        Gain_B= float(np.max(mat['groundtruth_illuminants'][i]))/float((mat['groundtruth_illuminants'][i][2]))
        
        image8 = white_balance_image(img_without_mcc, filename, Gain_R, Gain_G, Gain_B)
        save_dir = os.path.join(PATH_TO_GT_CORRECTED, filename+'.png')
        cv2.imwrite(save_dir, image8)

if __name__ == "__main__":
    main()