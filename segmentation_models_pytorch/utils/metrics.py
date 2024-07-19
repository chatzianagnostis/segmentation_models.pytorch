from . import base
from . import functional as F
from ..base.modules import Activation


class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(base.Metric):
    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

import torch
class PanopticQuality(base.Metric):
    __name__ = "panoptic_quality"

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.reset()

    def reset(self):
        self.true_positives = []
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = y_pr.argmax(dim=1)  # Convert from logits to predicted class

        # Convert to integer to ensure calculations are done correctly
        y_pr = y_pr.int()
        y_gt = y_gt.int()

        # Calculate TP, FP, FN
        for class_id in torch.unique(y_gt):
            if class_id == 0:  # Skip background
                continue
            tp = ((y_pr == class_id) & (y_gt == class_id)).sum().item()
            fp = ((y_pr == class_id) & (y_gt != class_id)).sum().item()
            fn = ((y_pr != class_id) & (y_gt == class_id)).sum().item()

            iou = tp / (tp + fp + fn + self.eps)  # Add small value to avoid division by zero
            self.true_positives.append(iou)
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self):
        sq = self.compute_sq(self.true_positives)
        rq = self.compute_rq(sum(self.true_positives), self.false_positives, self.false_negatives)
        pq = self.compute_pq(sq, rq)
        return pq

    def forward(self, y_pr, y_gt):
        self.reset()
        self.update(y_pr, y_gt)
        pq = self.compute()
        return torch.tensor(pq, device=y_pr.device)

    def compute_sq(self, true_positives):
        if not true_positives:
            return 0.0
        return sum(true_positives) / len(true_positives)

    def compute_rq(self, true_positives, false_positives, false_negatives):
        if true_positives == 0:
            return 0.0
        precision = true_positives / (true_positives + false_positives + self.eps)
        recall = true_positives / (true_positives + false_negatives + self.eps)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute_pq(self, sq, rq):
        return sq * rq