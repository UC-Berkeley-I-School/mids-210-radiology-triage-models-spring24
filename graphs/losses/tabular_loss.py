"""
Custom loss function for the tabular model (following multi-header binary classification)
"""
import torch.nn as nn

class MultiHeadBCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        o1, o2, o3, o4, o5, o6 = outputs
        t1, t2, t3, t4, t5, t6 = targets
        l1 = nn.BCELoss()(o1, t1)
        l2 = nn.BCELoss()(o2, t2)
        l3 = nn.BCELoss()(o3, t3)
        l4 = nn.BCELoss()(o4, t4)
        l5 = nn.BCELoss()(o5, t5)
        l6 = nn.BCELoss()(o6, t6)
    
        return (l1 + l2 + l3 + l4 + l5 + l6) / 6 # just take average for initial model setup