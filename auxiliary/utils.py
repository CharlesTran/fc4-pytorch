import math
import os
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.functional import interpolate
import cv2
from auxiliary.settings import DEVICE
import itertools

EPS = 1e-9
PI = 22.0 / 7.0

def log_metrics(train_loss: float, val_loss: float, current_metrics: dict, best_metrics: dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    header = log_data.keys() if not os.path.exists(path_to_log) else False
    log_data.to_csv(path_to_log, mode='a', header=header, index=False)


def print_metrics(current_metrics: dict, best_metrics: dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, Tensor]) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / 2.2))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / 2.2)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def normalize(img):
    max_int = 65535.0
    if isinstance(img, np.ndarray):
        return np.clip(img, 0.0, max_int) * (1.0 / max_int)
    else:
        return torch.clamp(img, 0.0, max_int) * (1.0 / max_int)


def rgb_to_bgr(x):
    if torch.is_tensor(x):
        return x.flip(-1)
    else:
        return x[..., ::-1]



def bgr_to_rgb(x):
    if torch.is_tensor(x):
        return x[..., [2, 1, 0]]
    else:
        return x[..., ::-1]


def hwc_to_chw(x):
    """ Converts an image from height x width x channels to channels x height x width """
    if torch.is_tensor(x):
        return x.permute(2, 0, 1)
    else:
        return x.transpose(2, 0, 1)


def scale(x: Tensor) -> Tensor:
    """ Scales all values of a tensor between 0 and 1 """
    x = x - x.min()
    x = x / x.max()
    return x


def rescale(x: Tensor, size: Tuple) -> Tensor:
    """ Rescale tensor to image size for better visualization """
    return interpolate(x, size, mode='bilinear')


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()


def tvd(pred: Tensor, label: Tensor) -> Tensor:
    """
    Total Variation Distance (TVD) is a distance measure for probability distributions
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return (Tensor([0.5]) * torch.abs(pred - label)).sum()


def jsd(p: List, q: List) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2

def get_hist_boundary():
    """ Returns histogram boundary values.
    Returns:
    bounardy_values: a list of boundary values.
    """
    boundary_values = [-2.85, 2.85]
    assert (boundary_values[0] == -boundary_values[1])
    return boundary_values

def get_hist_colors(img, from_rgb):
    """ Gets valid chroma and color values for histogram computation.

    Args:
        img: input image as an ndarray in the format (height x width x channel).
        from_rgb: a function to convert from rgb to chroma.

    Returns:
        valid_chroma: valid chroma values.
        valid_colors: valid rgb color values.
    """
    
    img_r = img.reshape(3, -1)
    img_chroma = from_rgb(img_r)
    valid_pixels = torch.sum(img_r, axis=0) > EPS  # exclude any zero pixels
    valid_chroma = img_chroma[:, valid_pixels]
    valid_colors = img_r[:, valid_pixels]
    return valid_chroma, valid_colors

def rgb_to_uv(rgb):
    """ Converts RGB to log-chroma space.

    Args:
        rgb: input color(s) in rgb space.
        tensor: boolean flag for input torch tensor; default is false.

    Returns:
        color(s) in chroma log-chroma space.
    """

    if isinstance(rgb, Tensor): # rgb
        log_rgb = torch.log(rgb + EPS)
        u = log_rgb[0, :] - log_rgb[1, :] 
        v = log_rgb[2, :] - log_rgb[1, :]
        return torch.stack([u, v], dim=0)
    else:
        log_rgb = np.log(rgb + EPS)# bgr
        u = log_rgb[:, 1] - log_rgb[:, 0] # g/b
        v = log_rgb[:, 1] - log_rgb[:, 2] # g/r
        return np.stack([u, v], axis=-1)
    
def compute_histogram(chroma_input, hist_boundary, nbins, rgb_input=None):
    """ Computes log-chroma histogram of a given log-chroma values.

    Args:
        chroma_input: k x 2 array of log-chroma values; k is the total number of
        pixels and 2 is for the U and V values.
        hist_boundary: histogram boundaries obtained from the 'get_hist_boundary'
        function.
        nbins: number of histogram bins.
        rgb_input: k x 3 array of rgb colors; k is the totanl number of pixels and
        3 is for the rgb vectors. This is an optional argument, if it is
        omitted, the computed histogram will not consider the overall
        brightness value in Eq. 3 in the paper.

    Returns:
        N: nbins x nbins log-chroma histogram.
    """
    hist_boundary = torch.tensor(hist_boundary)
    eps = torch.sum(torch.abs(hist_boundary)) / (nbins - 1)
    hist_boundary = torch.sort(hist_boundary)[0]
    A_u = torch.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps).to(DEVICE)
    A_v = torch.flip(A_u, [0])
    
    if rgb_input is None:
        Iy = torch.ones(chroma_input.shape[0]).to(DEVICE)
    else:
        Iy = torch.sqrt(torch.sum(rgb_input ** 2, dim=0))
    
    # differences in log_U space
    diff_u = torch.abs(chroma_input[0, :].unsqueeze(1) - A_u.unsqueeze(0))

    # differences in log_V space
    diff_v = torch.abs(chroma_input[1, :].unsqueeze(1) - A_v.unsqueeze(0))

    # counts only U values that are higher than the threshold value
    diff_u[diff_u > eps] = 0
    diff_u[diff_u != 0] = 1

    # counts only V values that are higher than the threshold value
    diff_v[diff_v > eps] = 0
    diff_v[diff_v != 0] = 1

    Iy_diff_v = Iy.unsqueeze(1) * diff_v
    N = torch.matmul(Iy_diff_v.t(), diff_u)
    
    norm_factor = torch.sum(N) + EPS
    N = torch.sqrt(N / norm_factor)  # normalization
    
    return N

def compute_edges(im):
    """ Computes gradient intensities of a given image; this is used to
        generate the edge histogram N_1, as described in the paper.

    Args:
        im: image as an ndarray (float).

    Returns:
        gradient intensities as ndarray with the same dimensions of im (float).
    """

    assert (len(im.shape) == 3)  # should be a 3D tensor
    assert (im.shape[0] == 3)  # should be 3-channel color image
    edge_img = torch.zeros(im.shape).to(DEVICE)
    img_pad = torch.nn.functional.pad(im,( 1, 1, 1, 1), 'reflect').to(DEVICE)
    offsets = [-1, 0, 1]
    for filter_index, (dx, dy) in enumerate(itertools.product(offsets, repeat=2)):
        if dx == 0 and dy == 0:
            continue
        edge_img[:, :, :] = edge_img[:, :, :] + (
            torch.abs(im[:, :, :] - img_pad[:, 1 + dx:im.shape[1] + 1 + dx,
                            1 + dy:im.shape[2] + 1 + dy]))
    edge_img = edge_img / 8
    return edge_img

def to_tensor(im, dims=3):
    """ Converts a given ndarray image to torch tensor image.

    Args:
        im: ndarray image (height x width x channel x [sample]).
        dims: dimension number of the given image. If dims = 3, the image should
        be in (height x width x channel) format; while if dims = 4, the image
        should be in (height x width x channel x sample) format; default is 3.

    Returns:
        torch tensor in the format (channel x height x width)  or (sample x
        channel x height x width).
    """

    assert (dims == 3 or dims == 4)
    if dims == 3:
        im = im.transpose((2, 0, 1))
    elif dims == 4:
        im = im.transpose((3, 2, 0, 1))
    else:
        raise NotImplementedError
    return torch.from_numpy(im)

def get_uv_coord(hist_size, tensor=True, normalize=False, device='cuda'):
    """ Gets uv-coordinate extra channels to augment each histogram as
        mentioned in the paper.

    Args:
        hist_size: histogram dimension (scalar).
        tensor: boolean flag for input torch tensor; default is true.
        normalize: boolean flag to normalize each coordinate channel; default
        is false.
        device: output tensor allocation ('cuda' or 'cpu'); default is 'cuda'.

    Returns:
        u_coord: extra channel of the u coordinate values; if tensor arg is True,
        the returned tensor will be in (1 x height x width) format; otherwise,
        it will be in (height x width) format.
        v_coord: extra channel of the v coordinate values. The format is the same
        as for u_coord.
    """

    u_coord, v_coord = np.meshgrid(
        np.arange(-(hist_size - 1) / 2, ((hist_size - 1) / 2) + 1),
        np.arange((hist_size - 1) / 2, (-(hist_size - 1) / 2) - 1, -1))
    if normalize:
        u_coord = (u_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
        v_coord = (v_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
    if tensor:
        u_coord = torch.from_numpy(u_coord).to(device=device, dtype=torch.float32)
        u_coord = torch.unsqueeze(u_coord, dim=0)
        u_coord.requires_grad = False
        v_coord = torch.from_numpy(v_coord).to(device=device, dtype=torch.float32)
        v_coord = torch.unsqueeze(v_coord, dim=0)
        v_coord.requires_grad = False
    return u_coord, v_coord

