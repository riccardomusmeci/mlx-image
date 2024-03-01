import mlx.core as mx


class Accuracy:
    """Compute the accuracy of a classifier."""

    def __init__(self) -> None:
        self.correct = 0  # To track the number of correct predictions
        self.total = 0  # To track the total number of predictions

    def update(self, logits: mx.array, targets: mx.array) -> None:
        """
        Update the state with a batch of logits and targets.

        Args:
            logits (torch.Tensor): The predicted labels.
            targets (torch.Tensor): The ground truth labels.
        """
        # Convert probabilities to predicted class labels
        preds = logits.argmax(axis=1)
        # Check if the predictions are correct
        correct = mx.equal(preds, targets).sum().item()
        # Update correct and total count
        self.correct += correct
        self.total += targets.size

    def compute(self) -> float:
        """
        Compute the accuracy over all batches.

        Returns:
            float: The accuracy as a percentage.
        """
        accuracy = self.correct / self.total
        return accuracy

    def reset(self) -> None:
        """
        Reset the state of the accuracy metric.
        """
        self.correct = 0
        self.total = 0
