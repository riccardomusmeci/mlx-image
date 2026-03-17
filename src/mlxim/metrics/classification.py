import mlx.core as mx
import numpy as np


def accuracy_at_k(logits: mx.array, targets: mx.array, top_k: tuple[int, ...] = (1, 5)) -> dict:
    """Compute the accuracy at k.

    Args:
        logits (mx.array): model output
        targets (mx.array): ground truth
        top_k (Tuple[int], optional): top k accuracy. Defaults to (1, 5).

    Returns:
        Dict: accuracy at k
    """
    max_k = max(top_k)
    preds_top_k = mx.argsort(logits, axis=1)[:, -max_k:]
    acc = {}
    for k in top_k:
        preds_at_k = np.array(preds_top_k[:, -k:])
        targets_arr = np.expand_dims(np.array(targets), axis=1)
        correct_predictions = np.any(preds_at_k == targets_arr, axis=1)
        acc[k] = np.mean(correct_predictions)

    return acc


class Accuracy:
    """Compute the accuracy of a classifier."""

    def __init__(self, top_k: tuple[int, ...] = (1, 5)) -> None:
        self.top_k = top_k
        self.correct = dict.fromkeys(top_k, 0)
        self.total = 0

    def update(self, logits: mx.array, targets: mx.array) -> None:
        """Update the state with a batch of logits and targets.

        Args:
            logits (mx.array): the predicted logits.
            targets (mx.array): the ground truth labels.
        """
        batch_size = targets.shape[0]
        max_k = max(self.top_k)
        preds_top_k = mx.argsort(logits, axis=1)[:, -max_k:]
        for k in self.top_k:
            preds_at_k = np.array(preds_top_k[:, -k:])
            targets_arr = np.expand_dims(np.array(targets), axis=1)
            correct = int(np.sum(np.any(preds_at_k == targets_arr, axis=1)))
            self.correct[k] += correct
        self.total += batch_size

    def compute(self) -> dict[str, float]:
        """Compute the accuracy over all batches.

        Returns:
            Dict[str, float]: the accuracy at k.
        """
        if self.total == 0:
            return {f"acc@{k}": 0.0 for k in self.top_k}
        return {f"acc@{k}": self.correct[k] / self.total for k in self.top_k}

    def __repr__(self) -> str:
        accuracy = self.compute()
        acc_repr = ""
        for k, v in accuracy.items():
            acc_repr += f"> {k}={v:.4f}\n"
        return acc_repr

    def reset(self) -> None:
        """Reset the state of the accuracy metric."""
        self.correct = dict.fromkeys(self.top_k, 0)
        self.total = 0

    def as_dict(self) -> dict:
        """Return a dict with accuracy at k.

        Returns:
            Dict: the accuracy at k.
        """
        return self.compute()
