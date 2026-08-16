import numpy as np
import torch

from src.metrics.segmentation_metrics import SegmentationMetrics, fast_miou


def test_perfect_prediction_gives_miou_1():
    m = SegmentationMetrics(num_classes=3)
    target = torch.randint(0, 3, (4, 16, 16))
    m.update(target, target)
    assert abs(m.mean_iou() - 1.0) < 1e-9
    assert abs(m.pixel_accuracy() - 1.0) < 1e-9


def test_completely_wrong_prediction_gives_miou_0():
    # binary case: predict the opposite class everywhere
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    pred = torch.ones((1, 8, 8), dtype=torch.long)
    m = SegmentationMetrics(num_classes=2)
    m.update(pred, target)
    assert m.mean_iou() == 0.0


def test_logits_are_argmaxed():
    m = SegmentationMetrics(num_classes=2)
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    logits = torch.zeros((1, 2, 4, 4))
    logits[:, 0] = 10.0  # class 0 always wins
    m.update(logits, target)
    assert abs(m.mean_iou() - 1.0) < 1e-9


def test_accumulates_across_batches():
    m = SegmentationMetrics(num_classes=2)
    t1 = torch.zeros((1, 4, 4), dtype=torch.long)
    p1 = torch.zeros((1, 4, 4), dtype=torch.long)
    m.update(p1, t1)
    t2 = torch.ones((1, 4, 4), dtype=torch.long)
    p2 = torch.zeros((1, 4, 4), dtype=torch.long)  # all wrong on batch 2
    m.update(p2, t2)
    # class 0: intersection 16, union 16+16=32 -> iou 0.5 ; class1: iou 0
    assert m.confusion.sum() == 32
    summary = m.summary(["a", "b"])
    assert "mIoU" in summary and "per_class_iou" in summary


def test_fast_miou_matches_full_workflow():
    target = torch.randint(0, 4, (2, 10, 10))
    val = fast_miou(target, target, num_classes=4)
    assert abs(val - 1.0) < 1e-9
