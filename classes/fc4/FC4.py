from typing import Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.settings import USE_CONFIDENCE_WEIGHTED_POOLING
from classes.fc4.squeezenet.SqueezeNetLoader import SqueezeNetLoader
from classes.fc4.vit.ViTLoader import ViTLoader

"""
FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling
* Original code: https://github.com/yuanming-hu/fc4
* Paper: https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/
"""


class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        vit = ViTLoader().load(pretrained=True)
        self.backbone = vit

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(1024, 512, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4 if USE_CONFIDENCE_WEIGHTED_POOLING else 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        bs = x.shape[0]
        x = self.backbone(x)

        # n_pixel = x.shape[-1] * x.shape[-2]
        # image_pred = x.flatten(1)
        # image_pred, _ = torch.sort(image_pred, dim=1)
        # tmp = []
        # for b in range(bs):
        #     num_otsu_sel = get_otsu_k(image_pred[b, ...], sorted=True)
        #     num_otsu_sel = max(num_otsu_sel, n_pixel // 2 + 1)
        #     tpk = int(max(1, (n_pixel - num_otsu_sel) * self.otsu_portion))
        #     topk_output = torch.topk(image_pred[b, ...], k=tpk, dim=0)[0]
        #     tmp.append(topk_output.mean())
        # image_pred = torch.stack(tmp)
            
        out = self.final_convs(x)

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            # Per-patch color estimates (first 3 dimensions)
            rgb = normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]

            # Confidence-weighted pooling
            pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

            return pred, rgb, confidence

        # Summation pooling
        pred = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)

        return pred

def get_otsu_k(attention, return_value=False, sorted=False):
    def _get_weighted_var(seq, pivot: int):
        # seq is of shape [t], in ascending order
        length = seq.shape[0]
        wb = pivot / length
        vb = seq[:pivot].var()
        wf = 1 - pivot / length
        vf = seq[pivot:].var()
        return wb * vb + wf * vf

    # attention shape: t
    # TODO use half
    length = attention.shape[0]
    if length == 1:
        return 0
    elif length == 2:
        return 1
    if not sorted:
        attention, _ = torch.sort(attention)
    optimal_i = length // 2
    min_intra_class_var = _get_weighted_var(attention, optimal_i)

    # for i in range(1, length):
    #     intra_class_var = _get_weighted_var(attention, i)
    #     if intra_class_var < min_intra_class_var:
    #         min_intra_class_var = intra_class_var
    #         optimal_i = i

    got_it = False
    # look left
    for i in range(optimal_i - 1, 0, -1):
        intra_class_var = _get_weighted_var(attention, i)
        if intra_class_var > min_intra_class_var:
            break
        else:
            min_intra_class_var = intra_class_var
            optimal_i = i
            got_it = True
    # look right
    if not got_it:
        for i in range(optimal_i + 1, length):
            intra_class_var = _get_weighted_var(attention, i)
            if intra_class_var > min_intra_class_var:
                break
            else:
                min_intra_class_var = intra_class_var
                optimal_i = i

    if return_value:
        return attention[optimal_i]
    else:
        return optimal_i