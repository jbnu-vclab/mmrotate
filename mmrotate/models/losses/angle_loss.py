import torch
import torch.nn as nn

from mmrotate.models.builder import ROTATED_LOSSES
from mmrotate.core import obb2poly
from mmdet.models.losses.utils import weighted_loss

def get_up_vec_from_obb(obb):
    poly = obb2poly(obb, version='le90')

    upper_seg_mid_x = (poly[:, 0] + poly[:, 2]) / 2
    upper_seg_mid_y = (poly[:, 1] + poly[:, 3]) / 2
    upper_seg_mid = torch.stack((upper_seg_mid_x, upper_seg_mid_y), dim=1)

    direction = upper_seg_mid - obb[:, :2]
    direction = direction / torch.norm(direction, dim=1, keepdim=True)

    return direction

def get_cos_sim_loss(pred, target):
    direction_p = get_up_vec_from_obb(pred)
    direction_t = get_up_vec_from_obb(target)

    cos = nn.CosineSimilarity(dim=1)

    cosine_similarity = cos(direction_p, direction_t)

    cos_sim_loss = 1 - cosine_similarity

    return cos_sim_loss

@weighted_loss
def angle_loss(pred, target, beta=1.0, cos_weight=1.0):
    """Angle loss.

    Computing the Angle loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as cosine similarity.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert pred.size() == target.size() and target.numel() > 0

    # Smooth-L1 norm
    diff = torch.abs(pred - target)
    
    adaptive_lambda = torch.tensor( \
        #* [1.0,  1.0,  1.0,  1.0,  1.0], \
        #* [0.95, 0.95, 0.95, 0.95, 1.2], \
        #* [0.90, 0.90, 0.90, 0.90, 1.4], \
        #* [0.85, 0.85, 0.85, 0.85, 1.6], \
        #* [0.80, 0.80, 0.80, 0.80, 1.8], \
        #* [0.75, 0.75, 0.75, 0.75, 2.0], \
        [0.70, 0.70, 0.70, 0.70, 2.2], \
        dtype=torch.float32, device=diff.device)

    diff *= adaptive_lambda.unsqueeze(0)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    
    loss = xy_loss
    
    return loss

@ROTATED_LOSSES.register_module()
class AngleLoss(nn.Module):

    def __init__(self, reduction='mean', beta=1.0, loss_weight=1.0):
        super(AngleLoss, self).__init__()
        self.reduction = reduction
        self.beta = 1.0
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * angle_loss(pred, target, beta=self.beta)
        
        return loss_bbox