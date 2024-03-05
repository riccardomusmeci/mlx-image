import argparse
import json
import os
import time
from shutil import copy2
from typing import Dict, Union

from tqdm import tqdm

from mlxim.data import DataLoader, LabelFolderDataset
from mlxim.io import load_config
from mlxim.metrics.classification import Accuracy
from mlxim.model import create_model, num_params
from mlxim.transform import ImageNetTransform
from mlxim.utils.time import now
from mlxim.utils.validation import ValidationResults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation script")

    parser.add_argument("--config", type=str, default="config/validation.yaml")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    dataset = LabelFolderDataset(transform=ImageNetTransform(**config["transform"]), **config["dataset"])

    loader = DataLoader(dataset=dataset, **config["loader"])

    model = create_model(num_classes=len(dataset.class_map), **config["model"])
    model.eval()

    accuracy = Accuracy(**config["metric"])
    val_time = []
    throughput = []
    for _i, batch in tqdm(enumerate(loader), total=len(loader)):
        tic = time.time()
        x, target = batch
        logits = model(x)
        toc = time.time()
        accuracy.update(logits, target)
        val_time.append(toc - tic)
        throughput.append(x.shape[0] / (toc - tic))

    acc = accuracy.compute()
    avg_time_batch = sum(val_time) / len(val_time)
    avg_throughput = sum(throughput) / len(throughput)

    print("Validation result:")
    print(f"Avg. time per batch: {avg_time_batch:.4f} s")
    print(f"Avg. throughput: {avg_throughput:.4f} images/s")
    print(accuracy)

    output_dir = os.path.join(config["output"], config["model"]["model_name"], now())

    results = ValidationResults("results/results-imagenet-1k.csv")
    results.update(
        model_name=config["model"]["model_name"],
        acc_1=acc["acc@1"],
        acc_5=acc["acc@5"],
        throughput=avg_throughput,
        param_count=num_params(model),
        img_size=config["transform"]["img_size"],
        crop_pct=config["transform"]["crop_pct"],
        interpolation=config["transform"]["interpolation"],
        apple_silicon=config["apple_silicon"],
    )
    results.save()

    print(f"Saving output files to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # save accuracy dict to file
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(accuracy.as_dict(), f, indent=4)

    with open(os.path.join(output_dir, "stats.txt"), "w") as f:
        f.write(f"average_time_per_batch (s): {avg_time_batch}\n")
        f.write(f"average_throughput (images/s): {avg_throughput}\n")

    # copy config file
    copy2(args.config, output_dir)
