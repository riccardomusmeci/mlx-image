import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


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
        param_count: float,
        img_size: int,
        crop_pct: float,
        interpolation: str,
        engine: str,
        hf_weights: Optional[str] = None,
    ) -> None:
        """Update the results dataframe with new results.

        Args:
            model_name (str): model name
            acc_1 (float): accuracy@1
            acc_5 (float): accuracy@5
            param_count (float): number of model parameters
            img_size (int): image size
            crop_pct (float): crop percentage
            interpolation (str): interpolation method
            engine (str): engine used for the dataset to load the images
            hf_weights (str, optional): if None, it will be fetched from the model config. Defaults to None.
        """

        new_row = {
            "model": [model_name],
            "acc@1": [round(acc_1, 5)],
            "acc@5": [round(acc_5, 5)],
            "param_count": [param_count],
            "img_size": [img_size],
            "crop_pct": [crop_pct],
            "interpolation": [interpolation],
            "engine": [engine],
        }
        new_data = pd.DataFrame(new_row)
        self.results = pd.concat([self.results, new_data], ignore_index=True)

    def save(self) -> None:
        """Save the results to a csv file."""
        print(f"Saving csv results to {self.path}")
        self.results = self.results.sort_values(by="acc@1", ascending=False)
        self.results.to_csv(self.path, index=False)
