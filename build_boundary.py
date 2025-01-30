import os

import cv2
import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


class SurfaceLoss():
    def __init__(self, idc = [1,2,3]):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = idc  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)
        loss = multipled.mean()
        return loss

def bound_loss(probs, dist_maps):
    assert simplex(probs)
    assert not one_hot(dist_maps)
    b, _, _, _ = probs.shape
    pc1 = probs[:, 1, ...].type(torch.float32)
    dc1 = dist_maps[:, 1, ...].type(torch.float32)
    pc2 = probs[:, 2, ...].type(torch.float32)
    dc2 = dist_maps[:, 2, ...].type(torch.float32)
    pc3 = probs[:, 3, ...].type(torch.float32)
    dc3 = dist_maps[:, 3, ...].type(torch.float32)

    multipled1 = einsum("bwh,bwh->bwh", pc1, dc1).sum()
    multipled2 = einsum("bwh,bwh->bwh", pc2, dc2).sum()
    multipled3 = einsum("bwh,bwh->bwh", pc3, dc3).sum() # /1e4

    return multipled1/b, multipled2/b, multipled3/b


if __name__ == "__main__":
    '''
    Loss = SurfaceLoss(idc = [1,2,3])
    # 读取图片
    dist_lb = np.array([[0, 1, 1, 1, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0, 3, 3],
                          [0, 0, 0, 0, 0, 3, 3]])
    dist_lb = torch.tensor(dist_lb)
    dist_lb = class2one_hot(dist_lb, 4)[0].numpy() # (w,h) -> (c,w,h)
    dist_lb = one_hot2dist(dist_lb)  # (c,w,h) -> (c,w,h)
    # 保存为numpy

    # 读取numpy
    dist_lb = torch.tensor(dist_lb).unsqueeze(dim=0)
    logits = torch.tensor([[[0, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 3, 3],
                            [0, 0, 0, 0, 0, 3, 3]]])
    # 相当于softmax之后
    logits = class2one_hot(logits, 4)

    dist1 = dist_lb[0,1,:,:].numpy()
    pred1 = logits[0,1,:,:].numpy()

    res = Loss(logits, dist_lb)
    print('loss:', res)

    res1 = bound_loss(logits, dist_lb)
    print('loss:', res1)
    '''
    sou_dir = 'E:/challenge/NEU_Seg-main/annotations/training'
    tar_dir = 'E:/challenge/NEU_Seg-main/annotations/boundary'
    for file in os.listdir(sou_dir):
        dist_lb = cv2.imread(sou_dir + '/' + file, 0)
        dist_lb = torch.tensor(dist_lb)
        dist_lb = class2one_hot(dist_lb, 4)[0].numpy()  # (w,h) -> (c,w,h)
        dist_lb = one_hot2dist(dist_lb)  # (c,w,h) -> (c,w,h)
        np.save(tar_dir + '/' + file.replace('.png','.npy'), dist_lb)
