import numpy as np
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()

        self.omega = omega
        self.epsilon = epsilon

    def forward(self, prediction, target):
        C = (self.omega - self.omega * np.log(1.0+self.omega/self.epsilon))

        diff_abs = torch.abs(prediction-target)
        loss = torch.where(diff_abs < self.omega,
                           self.omega * torch.log(1.0+diff_abs/self.epsilon),
                           diff_abs - C
        )

class AWingLoss(nn.Module):
    def __init__(self, omega=14.0, epsilon=1.0, theta=0.5, alpha=2.1, weighted_map=True):
        super(AWingLoss, self).__init__()

        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        self.alpha = alpha

        self.weighted_map = weighted_map

    def forward(self, prediction, target):
        A = self.omega * (1.0/(1.0+torch.pow(self.theta/self.epsilon, self.alpha-target))) * (self.alpha-target) * torch.pow(self.theta/self.epsilon, self.alpha-target-1.0) * (1.0/self.epsilon)
        C = (self.theta*A - self.omega*torch.log(1.0+torch.pow(self.theta/self.epsilon, self.alpha-target)))

        diff_abs = torch.abs(prediction-target)
        loss = torch.where(diff_abs < self.theta,
                           self.omega * torch.log(1.0+torch.pow(diff_abs/self.epsilon, self.alpha-target)),
                           A * diff_abs - C
        )

        if self.weighted_map:
            loss *= self.generate_loss_map_mask(target)

        return loss.mean()
    
    def generate_loss_map_mask(self, target, W=10.0, k_size=3, threshold=0.2):
        target_array = target.cpu().numpy()
        mask = np.zeros_like(target_array)

        for batch in range(mask.shape[0]):
            for loc in range(mask.shape[1]):
                H_d = ndimage.grey_dilation(target_array[batch, loc], size=(k_size, k_size))
                mask[batch, loc, H_d > threshold] = W
        
        return torch.Tensor(mask+1).to(target.device)