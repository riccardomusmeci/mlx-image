from typing import Dict, Tuple, Union

import mlx.core as mx
import numpy as np


def accuracy_at_k(logits: mx.array, targets: mx.array, top_k: Tuple[int, int] = (1, 5)) -> Dict:
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
        targets_at_k = np.expand_dims(np.array(targets), axis=1)
        correct_predictions = np.any(np.isin(np.array(preds_at_k), targets_at_k), axis=1)
        acc[k] = np.mean(correct_predictions)

    return acc


class Accuracy:
    """Compute the accuracy of a classifier."""

    def __init__(self, top_k: Tuple[int, int] = (1, 5)) -> None:
        self.top_k = top_k
        self.data = []  # type: ignore

    def update(self, logits: mx.array, targets: mx.array) -> None:
        """
        Update the state with a batch of logits and targets.

        Args:
            logits (torch.Tensor): The predicted labels.
            targets (torch.Tensor): The ground truth labels.
        """
        self.data.append(accuracy_at_k(logits, targets, top_k=self.top_k))

    def compute(self) -> Dict[str, float]:
        """
        Compute the accuracy over all batches.

        Returns:
            Dict[str, float]: the accuracy at k.
        """
        accuracy = {f"acc@{k}": np.mean([x[k] for x in self.data]) for k in self.top_k}
        return accuracy

    def __repr__(self) -> str:
        accuracy = self.compute()
        if isinstance(accuracy, float):
            return f"acc@{self.top_k[0]}={accuracy:.4f}"
        else:
            acc_repr = ""
            for k, v in accuracy.items():
                acc_repr += f"> acc@{k}={v:.4f}\n"
            return acc_repr

    def reset(self) -> None:
        """
        Reset the state of the accuracy metric.
        """
        self.data = []

    def as_dict(self) -> Dict:
        """Return a dict with accuracy at k.

        Returns:
            Dict: the accuracy at k.
        """
        accuracy = self.compute()
        return {f"acc@{k}": v for k, v in accuracy.items()}
