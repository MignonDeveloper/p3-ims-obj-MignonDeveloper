# https://github.com/CoinCheung/pytorch-loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim


    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# name과 criterion(loss function) class를 묶는 entrypoint
_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'label_smoothing': LabelSmoothingLoss,
}


# criterion(loss function) 이름을 가지는 class return
def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


# criterion(loss function) list에 전달된 이름이 존재하는지 판단
def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name): # 정의한 criterion list내에 이름이 있으면
        create_fn = criterion_entrypoint(criterion_name) # class를 가져와서
        criterion = create_fn(**kwargs) # 전달된 인자를 사용해 선언
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name) # 없을 시, Error raise
        
    return criterion # defined criterion return


    