import torch
import torch.nn as nn

from mmrotate.models.builder import ROTATED_LOSSES
from mmrotate.core import obb2poly
from mmdet.models.losses.utils import weighted_loss

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

    xy_p = pred[:, :2]
    xy_t = target[:, :2]

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    
    # Angle norm (cosine similarity between two direction vectors)
    poly_p = obb2poly(pred, version='le90')
    poly_t = obb2poly(target, version='le90')

    upper_seg_mid_p_x = (poly_p[:, 0] + poly_p[:, 2]) / 2
    upper_seg_mid_p_y = (poly_p[:, 1] + poly_p[:, 3]) / 2
    upper_seg_mid_t_x = (poly_t[:, 0] + poly_t[:, 2]) / 2
    upper_seg_mid_t_y = (poly_t[:, 1] + poly_t[:, 3]) / 2

    upper_seg_mid_p = torch.stack((upper_seg_mid_p_x, 
                                                upper_seg_mid_p_y), 
                                                dim=1)
    upper_seg_mid_t = torch.stack((upper_seg_mid_t_x, 
                                                upper_seg_mid_t_y), 
                                                dim=1)
    
    direction_p = upper_seg_mid_p - xy_p
    direction_t = upper_seg_mid_t - xy_t

    cos = nn.CosineSimilarity(dim=1)
    
    cosine_similarity = cos(direction_p, direction_t)
    
    cos_sim_loss = 1 - cosine_similarity

    loss = xy_loss + cos_sim_loss * cos_weight
    
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