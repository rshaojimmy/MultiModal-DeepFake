import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

        
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        # p = F.softmax(logits, dim=1)
        p = F.sigmoid(logits)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        # Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # Yg = torch.gather(p, 1, targets)
        # modify gradient of cross entropy
        loss_weight = (p.detach()**self.q)*self.q
        if np.isnan(p.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none') * loss_weight

        return loss
