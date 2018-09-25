
import numpy as np

def bbox_overlaps(pred, gt):
    '''
    args:
    2 numpy  array
    pred: (N, 4)
    gt:   (M, 4)
    
    returns:
    overlaps: (N, M)
    '''
    N, _ = pred.shape
    M, _ = gt.shape
    overlaps = np.zeros((N, M)).astype('float')
    for n in range(N):
        for m in range(M):
            overlaps[n, m] = iou(pred[n, :], gt[m, :])
    
    return overlaps


def iou(pbox, gbox):
    '''
    compute intersection over union between pred and gt.
    return: a number.
    '''
    xl = max(pbox[0], gbox[0])
    yl = max(pbox[1], gbox[1])
    xr = min(pbox[2], gbox[2])
    yr = min(pbox[3], gbox[3])
    
    w = xr - xl
    h = yr - yl
    
    w = max(0, w)
    h = max(0, h)
    
    area = w * h
    u = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) + (gbox[2] - gbox[0]) * (gbox[3] - gbox[1])
    
    iOU = area / u
    return iOU    