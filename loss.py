import torch
import torch.nn.functional as F

class ContrastLoss(torch.nn.Module):
    """
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.15):
        super(ContrastLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        print('euclidean_distance: ', euclidean_distance)
        print('torch.clamp: ', torch.clamp(self.margin - euclidean_distance, min=0.0))
        loss_contrastive = torch.mean(((1-label) * euclidean_distance) +
                                      (label * torch.clamp(self.margin - euclidean_distance, min=0.0)))

        return loss_contrastive


# class TripletLoss(torch.nn.Module):
#
#     def __init__(self, margin=2.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, output3):
#         dist_pos = F.pairwise_distance(output1, output2, 2)
#         dist_neg = F.pairwise_distance(output2, output3, 2)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#
#
#         return loss_contrastive