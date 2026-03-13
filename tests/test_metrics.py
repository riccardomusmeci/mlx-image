import mlx.core as mx
import numpy as np

from mlxim.metrics.classification import Accuracy, accuracy_at_k


def test_accuracy_at_k_basic(sample_logits, sample_targets):
    acc = accuracy_at_k(sample_logits, sample_targets, top_k=(1, 5))
    assert acc[1] == 1.0, f"Expected top-1 accuracy 1.0, got {acc[1]}"
    assert acc[5] == 1.0, f"Expected top-5 accuracy 1.0, got {acc[5]}"


def test_accuracy_at_k_perfect():
    logits = mx.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = mx.array([2, 0, 1])
    acc = accuracy_at_k(logits, targets, top_k=(1,))
    assert acc[1] == 1.0


def test_accuracy_at_k_zero():
    # Predictions: class 9, class 9, class 9. Targets: class 0, 1, 2 — all wrong.
    logits = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    targets = mx.array([0, 1, 2])  # all wrong — none predict 0/1/2
    acc = accuracy_at_k(logits, targets, top_k=(1,))
    assert acc[1] == 0.0


def test_accuracy_at_k_no_cross_contamination():
    """Predictions matching OTHER samples' targets must not count as correct.

    Sample 0: predicts class 1, target is class 0 → wrong
    Sample 1: predicts class 0, target is class 1 → wrong

    The old np.isin bug would mark both correct since {0,1} are both in the
    flattened target array. With the fix, both are correctly marked wrong.
    """
    logits = mx.array(
        [
            [0.0, 1.0, 0.0],  # predicts class 1
            [1.0, 0.0, 0.0],  # predicts class 0
        ]
    )
    targets = mx.array([0, 1])  # sample 0 wants 0, sample 1 wants 1
    acc = accuracy_at_k(logits, targets, top_k=(1,))
    assert acc[1] == 0.0, f"Cross-contamination detected: expected 0.0, got {acc[1]}"


def test_accuracy_update_and_compute():
    acc = Accuracy(top_k=(1,))
    # Batch 1: 2/2 correct — preds [9, 8], targets [9, 8]
    logits1 = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    targets1 = mx.array([9, 8])
    acc.update(logits1, targets1)

    # Batch 2: 1/2 correct — preds [9, 0], targets [9, 5]
    logits2 = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets2 = mx.array([9, 5])
    acc.update(logits2, targets2)

    result = acc.compute()
    # 3 correct out of 4 total = 0.75
    assert np.isclose(result["acc@1"], 0.75), f"Expected 0.75, got {result['acc@1']}"


def test_accuracy_unequal_batch_sizes():
    """Unequal batch sizes must be weighted by sample count, not averaged per-batch."""
    acc = Accuracy(top_k=(1,))

    # Batch 1: 1 sample, correct
    logits1 = mx.array([[0.0, 0.0, 1.0]])
    targets1 = mx.array([2])
    acc.update(logits1, targets1)

    # Batch 2: 3 samples, all wrong
    logits2 = mx.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    targets2 = mx.array([0, 0, 0])
    acc.update(logits2, targets2)

    result = acc.compute()
    # 1 correct out of 4 total = 0.25
    # Old bug: mean of [1.0, 0.0] = 0.5
    assert np.isclose(result["acc@1"], 0.25), f"Expected 0.25, got {result['acc@1']}"


def test_accuracy_reset():
    acc = Accuracy(top_k=(1,))
    logits = mx.array([[0.0, 1.0]])
    targets = mx.array([1])
    acc.update(logits, targets)
    assert acc.total == 1
    assert acc.correct[1] == 1
    acc.reset()
    assert acc.total == 0
    assert acc.correct[1] == 0


def test_accuracy_as_dict():
    acc = Accuracy(top_k=(1, 5))
    logits = mx.array([[0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = mx.array([2])
    acc.update(logits, targets)
    result = acc.as_dict()
    assert "acc@1" in result, f"Expected 'acc@1' key, got {list(result.keys())}"
    assert "acc@5" in result, f"Expected 'acc@5' key, got {list(result.keys())}"
    # No double-prefix
    assert "acc@acc@1" not in result, "Double-prefix bug: 'acc@acc@1' found"


def test_accuracy_compute_empty():
    acc = Accuracy(top_k=(1, 5))
    result = acc.compute()
    assert result["acc@1"] == 0.0
    assert result["acc@5"] == 0.0
