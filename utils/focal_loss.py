import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, nr_classes = 8, size_average=True, ignore_index=-1, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.nr_classes = nr_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        outputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)

        log_probs = torch.log(outputs + self.smooth)
        ce_loss = -targets_one_hot * log_probs
        
        # Compute focal loss
        focal_weight = self.alpha * targets_one_hot * (1 - outputs) ** self.gamma
        loss = focal_weight * ce_loss
        
        # Average the loss over all pixels and batches
        return loss.sum(dim=(1, 2, 3)).mean()


        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()