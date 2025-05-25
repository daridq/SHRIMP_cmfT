import torch
import numpy as np
import pysteps.verification.spatialscores as pvs

class Hilburn_Loss():
    @staticmethod
    def loss(y_pred, y_true, b=5, c=3):
        mse = torch.mean((y_pred - y_true) ** 2)
        weight = torch.exp(b * torch.pow(y_true, c))
        return weight * mse
    
def avg_fss(pred, target, thr=0.00001, scale=5):
    Fss = pvs.fss_init(thr, scale)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    for p, t in zip(pred, target):
        pvs.fss_accum(Fss, np.squeeze(p), np.squeeze(t))
    return pvs.fss_compute(Fss)


