import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()


class Criterion(_Loss):
    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return

class SymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        loss = loss * self.alpha
        return loss

class JSCriterion(Criterion):
    def __init__(self, alpha=1.0, name='JS Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + \
            F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
        loss = loss * self.alpha
        return loss

class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, logits_1, logits_2):
        m = F.softmax(logits_1, dim=-1) + F.softmax(logits_2, dim=-1)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(logits_1, dim=-1), m, reduction='batchmean') + F.kl_div(F.log_softmax(logits_2, dim=-1), m, reduction='batchmean')
        return 0.5 * loss