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
# Panoptic quality works for ground trouth masks are 1 channel  
class SegmentationQuality(base.Metric):
    __name__ = "segmentation_quality"

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

        # Flatten the tensors to handle batch and spatial dimensions together
        y_pr = y_pr.view(-1)
        y_gt = y_gt.view(-1)

        # Calculate IoU for each class
        for class_id in torch.unique(y_gt):
            if class_id == 0:  # Skip background
                continue
            tp = ((y_pr == class_id) & (y_gt == class_id)).sum().item()
            fp = ((y_pr == class_id) & (y_gt != class_id)).sum().item()
            fn = ((y_pr != class_id) & (y_gt == class_id)).sum().item()

            iou = tp / (tp + fp + fn + self.eps)  # Add small value to avoid division by zero
            self.true_positives.append(iou)

    def compute(self):
        if not self.true_positives:
            return torch.tensor(0.0)
        sq = sum(self.true_positives) / len(self.true_positives)
        return torch.tensor(sq)

    def forward(self, y_pr, y_gt):
        self.reset()
        self.update(y_pr, y_gt)
        return self.compute()


class RecognitionQuality(base.Metric):
    __name__ = "recognition_quality"

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = y_pr.argmax(dim=1)  # Convert from logits to predicted class

        # Convert to integer to ensure calculations are done correctly
        y_pr = y_pr.int()
        y_gt = y_gt.int()

        # Flatten the tensors to handle batch and spatial dimensions together
        y_pr = y_pr.view(-1)
        y_gt = y_gt.view(-1)

        # Calculate TP, FP, FN for each class
        for class_id in torch.unique(y_gt):
            if class_id == 0:  # Skip background
                continue
            tp = ((y_pr == class_id) & (y_gt == class_id)).sum().item()
            fp = ((y_pr == class_id) & (y_gt != class_id)).sum().item()
            fn = ((y_pr != class_id) & (y_gt == class_id)).sum().item()

            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self):
        if self.true_positives == 0:
            return torch.tensor(0.0)
        precision = self.true_positives / (self.true_positives + self.false_positives + self.eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + self.eps)
        if precision + recall == 0:
            return torch.tensor(0.0)
        rq = 2 * (precision * recall) / (precision + recall)
        return torch.tensor(rq)

    def forward(self, y_pr, y_gt):
        self.reset()
        self.update(y_pr, y_gt)
        return self.compute()

class PanopticQuality(base.Metric):
    __name__ = "panoptic_quality"

    def __init__(self, sq_metric, rq_metric, **kwargs):
        super().__init__(**kwargs)
        self.sq_metric = sq_metric
        self.rq_metric = rq_metric

    def forward(self, y_pr, y_gt):
        sq = self.sq_metric(y_pr, y_gt)
        rq = self.rq_metric(y_pr, y_gt)
        pq = sq * rq
        return pq
