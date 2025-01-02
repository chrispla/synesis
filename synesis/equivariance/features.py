"""Methods for training and evaluating a model to predict the
transformed feature given an original feature and
a transformation parameter.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.equivariance.features import configs as task_configs
from config.features import configs as feature_configs
from config.transforms import configs as transform_configs
from synesis.datasets.dataset_utils import SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.probes import get_probe
from synesis.transforms.transform_utils import get_transform
from synesis.utils import deep_update


def preprocess_batch(
    batch_raw_data, transform_obj, transform, feature_extractor, device
):
    batch_raw_data = batch_raw_data.to(device)

    transformed_raw_data = transform_obj(batch_raw_data)
    # assert shape is the same after transformation
    assert batch_raw_data.shape == transformed_raw_data.shape
    # get transformation parameters that were actually applied to batch
    # they will be of shape [batch, channel, 1], and on device
    transform_params = transform_obj.transform_parameters[
        f"{transform.lower()}_factors"
    ]

    # combine original and transformed data
    combined_raw_data = torch.cat([batch_raw_data, transformed_raw_data], dim=0)

    with torch.no_grad():
        combined_features = feature_extractor(combined_raw_data)
        if combined_features.dim() == 2:
            combined_features = combined_features.unsqueeze(1)

    # split the combined features back into original and transformed features
    original_features, transformed_features = torch.split(
        combined_features, batch_raw_data.shape[0], dim=0
    )

    return original_features, transformed_features, transform_params


def train(
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = True,
):
    """Train a model to predict the transformed feature given
    an original feature and a transformation parameter. Does
    feature extraction on-the-fly.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform (factor of variation).
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        device: Device to use for training (defaults to "cuda" if available).
        logging: Whether to log the model to wandb.
    Returns:
        If logging is True, returns the wandb run path to the model artifact.
        Otherwise, returns trained model.
    """
    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    if logging:
        run_name = f"EQUI_FEAT_{transform}_{dataset}_{feature}"
        wandb.init(
            project="synesis",
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
            },
        )
        artifact = wandb.Artifact(run_name, type="model", metadata={"task": task})

    if task_config["training"].get("feature_aggregation") or task_config[
        "evaluation"
    ].get("feature_aggregation"):
        raise NotImplementedError(
            "Feature aggregation is not currently implemented for transform prediction."
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="train",
        download=False,
        item_format="raw",
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="validation",
        download=False,
        item_format="raw",
    )

    # If dataset returns subitems per item, need to wrap it
    if train_dataset[0][0].dim() == 3:
        wrapped_train = SubitemDataset(train_dataset)
        wrapped_val = SubitemDataset(val_dataset)
        del train_dataset, val_dataset
        train_dataset = wrapped_train
        val_dataset = wrapped_val

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    transform_obj = get_transform(transform_config)

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=task_config["training"]["batch_size"]
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=task_config["training"]["batch_size"]
    )

    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=feature_config["feature_dim"] + 1,  # 1 transform parameter
        n_outputs=feature_config["feature_dim"],
        **task_config["model"]["params"],
    ).to(device)

    criterion = task_configs[task]["training"]["criterion"]()
    optimizer_class = task_configs[task]["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(),
        **task_configs[task]["training"]["optimizer"]["params"],
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    num_epochs = task_config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_raw_data, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            # prepare data for equivariance training
            original_features, transformed_features, transform_params = (
                preprocess_batch(
                    batch_raw_data, transform_obj, transform, feature_extractor, device
                )
            )

            # add parameter to original features - currently both are (b, c, 1)
            concat_features = torch.cat([original_features, transform_params], dim=2)

            optimizer.zero_grad()
            predicted_features = model(concat_features)
            loss = criterion(original_features, predicted_features)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if logging:
                wandb.log({"train/loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_l2_distance = 0
        total_cosine_distance = 0
        with torch.no_grad():
            for batch_raw_data, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                # prepare data for equivariance training
                original_features, transformed_features, transform_params = (
                    preprocess_batch(
                        batch_raw_data,
                        transform_obj,
                        transform,
                        feature_extractor,
                        device,
                    )
                )

                # add parameter to original features - currently both are (b, c, 1)
                concat_features = torch.cat(
                    [original_features, transform_params], dim=2
                )

                predicted_features = model(concat_features)
                loss = criterion(original_features, predicted_features)

                total_val_loss += loss.item()

                # Compute L2 distance
                l2_distance = F.mse_loss(
                    predicted_features, transformed_features, reduction="sum"
                )
                total_l2_distance += l2_distance.item()

                # Compute cosine distance
                cosine_distance = (
                    1
                    - F.cosine_similarity(
                        predicted_features, transformed_features, dim=1
                    )
                    .sum()
                    .item()
                )
                total_cosine_distance += cosine_distance

            avg_val_loss = total_val_loss / len(val_loader)
            avg_l2_distance = total_l2_distance / len(val_loader.dataset)
            avg_cosine_distance = total_cosine_distance / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - "
            + f"Val Loss: {avg_val_loss:.4f}"
            + f" - L2 Distance: {avg_l2_distance:.4f}"
            + f" - Cosine Distance: {avg_cosine_distance:.4f}"
        )

        if logging:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "l2_distance": avg_l2_distance,
                    "cosine_distance": avg_cosine_distance,
                }
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= task_config["training"]["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_path = Path("ckpt") / "EQUI" / "FEAT" / transform / dataset / f"{feature}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    if logging:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb_path = wandb.run.path + "/" + artifact.name
        wandb.finish()
        return wandb_path

    return model


def evaluate(
    model: Union[nn.Module, str],
    feature: str,
    dataset: str,
    transform: str,
    task: str,
    task_config: Optional[dict] = None,
    device: Optional[str] = None,
    logging: bool = True,
):
    """Evaluate a model to predict the transformed feature given
    an original feature and a transformation parameter. Does
    feature extraction on-the-fly.

    Args:
        model: Model to evaluate.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        transform: Name of the transform (factor of variation).
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        logging: Whether to log to wandb.
    Returns:
        Dictionary of evaluation metrics.
    """

    if isinstance(model, str):
        # resume wandb run
        entity, project, run_id, model_name = model.split("/")
        wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    feature_config = feature_configs.get(feature)
    transform_config = transform_configs.get(transform)
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format="raw",
    )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model, str):
        # Load model from wandb artifact
        model_wandb_path = f"{entity}/{project}/{model_name}"
        artifact = wandb.use_artifact(f"{model_wandb_path}:latest")
        artifact_dir = artifact.download()
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=feature_config["feature_dim"] * 2,
            n_outputs=1,  # currently only predicting one parameter
            **task_config["model"]["params"],
        )
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    model.to(device)

    if task_config["training"].get("feature_aggregation") or task_config[
        "evaluation"
    ].get("feature_aggregation"):
        raise NotImplementedError(
            "Feature aggregation is not currently implemented for transform prediction."
        )

    # If dataset returns subitems per item, need to wrap it
    if test_dataset[0][0].dim() == 3:
        wrapped_test = SubitemDataset(test_dataset)
        del test_dataset
        test_dataset = wrapped_test

    feature_extractor = get_feature_extractor(feature)
    feature_extractor = feature_extractor.to(device)
    transform_obj = get_transform(transform_config)

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=task_config["evaluation"]["batch_size"]
    )

    model.eval()
    total_loss = 0
    total_l2_distance = 0
    total_cosine_distance = 0

    criterion = task_configs[task]["training"]["criterion"]()

    with torch.no_grad():
        for batch_raw_data, _ in tqdm(test_loader, desc="Evaluating"):
            # prepare data for equivariance training
            original_features, transformed_features, transform_params = (
                preprocess_batch(
                    batch_raw_data,
                    transform_obj,
                    transform,
                    feature_extractor,
                    device,
                )
            )

            # add parameter to original features - currently both are (b, c, 1)
            concat_features = torch.cat([original_features, transform_params], dim=2)

            predicted_features = model(concat_features)
            loss = criterion(original_features, predicted_features)

            total_loss += loss.item()

            # Compute L2 distance
            l2_distance = F.mse_loss(
                predicted_features, transformed_features, reduction="sum"
            )
            total_l2_distance += l2_distance.item()

            # Compute cosine distance
            cosine_distance = (
                1
                - F.cosine_similarity(predicted_features, transformed_features, dim=1)
                .sum()
                .item()
            )
            total_cosine_distance += cosine_distance

    avg_loss = total_loss / len(test_loader)
    avg_l2_distance = total_l2_distance / len(test_loader.dataset)
    avg_cosine_distance = total_cosine_distance / len(test_loader.dataset)

    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Average L2 distance: {avg_l2_distance:.4f}")
    print(f"Average cosine distance: {avg_cosine_distance:.4f}")

    if logging:
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Loss", avg_loss)
        metrics_table.add_data("Average L2 Distance", avg_l2_distance)
        metrics_table.add_data("Average Cosine Distance", avg_cosine_distance)

        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return {
        "avg_loss": avg_loss,
        "avg_l2_distance": avg_l2_distance,
        "avg_cosine_distance": avg_cosine_distance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to predict transformed features."
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Feature name.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--transform",
        "-tf",
        type=str,
        required=True,
        help="Data transform name.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=False,
        default="default",
        help="Task name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Do not log to wandb.",
    )

    args = parser.parse_args()

    model = train(
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )

    results = evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        transform=args.transform,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )
