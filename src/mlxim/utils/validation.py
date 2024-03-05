import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..model._registry import MODEL_CONFIG


class ValidationResults:
    """Class to store and update validation results for models.

    Args:
        path (str, optional): path to the csv file to store the results. Defaults to "results/results-imagenet-1k.csv".
    """

    def __init__(self, path: str = "results/results-imagenet-1k.csv") -> None:
        self.path = path
        self.results = pd.read_csv(self.path, index_col=None)

    def update(
        self,
        model_name: str,
        acc_1: float,
        acc_5: float,
        throughput: float,
        param_count: float,
        img_size: int,
        crop_pct: float,
        interpolation: str,
        apple_silicon: str,
        hf_weights: Optional[str] = None,
    ) -> None:
        """Update the results dataframe with new results.

        Args:
            model_name (str): model name
            acc_1 (float): accuracy@1
            acc_5 (float): accuracy@5
            throughput (float): model throughput
            param_count (float): number of model parameters
            img_size (int): image size
            crop_pct (float): crop percentage
            interpolation (str): interpolation method
            apple_silicon (str): Apple Silicon name (e.g. m1_pro, m1_max, etc.)
            hf_weights (str, optional): if None, it will be fetched from the model config. Defaults to None.
        """

        if hf_weights is None:
            hf_weights = os.path.join(
                MODEL_CONFIG[model_name].weights.repo_id, MODEL_CONFIG[model_name].weights.filename
            )

        new_row = {
            "model": [model_name],
            "acc@1": [acc_1],
            "acc@5": [acc_5],
            "hf_weights": [hf_weights],
            "throughput": [throughput],
            "param_count": [param_count],
            "img_size": [img_size],
            "crop_pct": [crop_pct],
            "interpolation": [interpolation],
            "apple_silicon": [apple_silicon],
        }
        new_data = pd.DataFrame(new_row)
        self.results = pd.concat([self.results, new_data], ignore_index=True)

    def save(self) -> None:
        """Save the results to a csv file."""
        print(f"Saving csv results to {self.path}")
        self.results.to_csv(self.path, index=False)