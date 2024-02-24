from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Metrics:
    dataset: str
    accuracy_at_1: float
    accuracy_at_5: float


@dataclass
class HFWeights:
    repo_id: str
    filename: str


@dataclass
class Transform:
    crop: int
    resize: int = 256
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    # TODO: check if choice over selected items is possible
    interpolation: str = "bilinear"


@dataclass
class ModelConfig:
    metrics: Metrics
    transform: Transform
    weights: HFWeights
