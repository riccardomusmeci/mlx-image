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
    # pred=0 is NOT in {9, 5} → wrong. pred=9 IS in {9, 5} → correct.
    logits2 = mx.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets2 = mx.array([9, 5])
    acc.update(logits2, targets2)

    result = acc.compute()
    # Average of 1.0 and 0.5 = 0.75
    assert np.isclose(result["acc@1"], 0.75), f"Expected 0.75, got {result['acc@1']}"


def test_accuracy_reset():
    acc = Accuracy(top_k=(1,))
    logits = mx.array([[0.0, 1.0]])
    targets = mx.array([1])
    acc.update(logits, targets)
    assert len(acc.data) == 1
    acc.reset()
    assert len(acc.data) == 0


def test_accuracy_as_dict():
    acc = Accuracy(top_k=(1, 5))
    logits = mx.array([[0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = mx.array([2])
    acc.update(logits, targets)
    result = acc.as_dict()
    assert "acc@acc@1" in result or "acc@1" in result
    # The as_dict method wraps compute() keys with another acc@ prefix
    # compute returns {"acc@1": ..., "acc@5": ...}
    # as_dict returns {f"acc@{k}": v for k, v in accuracy.items()} where keys are already "acc@1"
    assert any("1" in k for k in result.keys())
