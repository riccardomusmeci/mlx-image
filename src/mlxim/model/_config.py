from dataclasses import dataclass


@dataclass
class Metrics:
    dataset: str
    accuracy_at_1: float | None = None
    accuracy_at_5: float | None = None


@dataclass
class HFWeights:
    repo_id: str
    filename: str


@dataclass
class Transform:
    img_size: int
    crop_pct: float = 1.0
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"


@dataclass
class ModelConfig:
    metrics: Metrics
    transform: Transform
    weights: HFWeights
