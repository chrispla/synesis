"""Methods for training and evaluating downstream models."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.features import configs as feature_configs
from config.informativeness.downstream import configs as task_configs
from synesis.datasets.dataset_utils import AggregateDataset, SubitemDataset, get_dataset
from synesis.features.feature_utils import get_feature_extractor
from synesis.metrics import instantiate_metrics
from synesis.probes import get_probe
from synesis.utils import deep_update


def train(
    feature: str,
    dataset: str,
    task: str,
    task_config: Optional[dict] = None,
    item_format: str = "feature",
    device: Optional[str] = None,
    logging: bool = False,
):
    """
    Train a downstream model.

    Args:
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task.
        task_config: Override certain values of the task configuration.
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        device: Device to use for training (defaults to "cuda" if available).
        logging: Whether to log to wandb.

    Returns:
        If logging is True, returns the wandb run path to the model artifact.
        Otherwise, returns the trained model.
    """
    feature_config = feature_configs[feature]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)), task_config
    )
    # Set up logging
    if logging:
        run_name = f"INFO_DOWN_{task}_{dataset}_{feature}"
        wandb.init(
            project="synesis",
            name=run_name,
            config={
                "feature": feature,
                "feature_config": feature_config,
                "dataset": dataset,
                "task": task,
                "task_config": task_config,
                "item_format": item_format,
            },
        )
        artifact = wandb.Artifact(run_name, type="model", metadata={"task": task})

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="train",
        download=False,
        item_format=item_format,
    )
    val_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="validation",
        download=False,
        item_format=item_format,
    )

    if train_dataset[0][0].dim() == 3:
        # If item is 3D, this is a dataset that returns items with subitems
        # (e.g. for audio).
        if task_config["training"]["feature_aggregation"]:
            # If feature_aggreation, we'll wrap the dataset so that it returns
            # aggregated features
            aggregated_train = AggregateDataset(
                train_dataset, feature_extractor_name=feature
            )
            aggregated_val = AggregateDataset(
                val_dataset, feature_extractor_name=feature
            )
            del train_dataset, val_dataset
            train_dataset = aggregated_train
            val_dataset = aggregated_val
        else:
            # If not feature_aggregation, we'll wrap the dataset so that it behaves
            # as a subitem dataset
            wrapped_train = SubitemDataset(train_dataset)
            wrapped_val = SubitemDataset(val_dataset)
            del train_dataset, val_dataset
            train_dataset = wrapped_train
            val_dataset = wrapped_val

    dataloader = DataLoader(
        train_dataset,
        batch_size=task_config["training"]["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=task_config["training"]["batch_size"],
        shuffle=False,
    )

    # if raw_data  (e.g. audio) is being returned from dataset,
    # extract features on-the-fly
    # (the AggregateDatset wrapper also computes features)
    if item_format == "raw" and not task_config["training"]["feature_aggregation"]:
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    # train setup
    model = get_probe(
        model_type=task_config["model"]["type"],
        in_features=feature_config["feature_dim"],
        n_outputs=len(train_dataset.label_encoder.classes_),
        **task_config["model"]["params"],
    ).to(device)
    criterion = task_config["training"]["criterion"]()
    optimizer_class = task_config["training"]["optimizer"]["class"]
    optimizer = optimizer_class(
        model.parameters(), **task_config["training"]["optimizer"]["params"]
    )

    val_metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=len(train_dataset.label_encoder.classes_),
    )

    # train and validation loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    num_epochs = task_config["training"]["num_epochs"]
    patience = task_config["training"]["patience"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for item, target in progress_bar:
            item = item.to(device)
            target = target.to(device)

            if (
                item_format == "raw"
                and not task_config["training"]["feature_aggregation"]
            ):
                with torch.no_grad():
                    item = extractor(item)
                    # if channels eaten up, unsqueeze
                    if item.dim() == 2:
                        item = item.unsqueeze(1)
            optimizer.zero_grad()
            output = model(item)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)

            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if logging:
                # Log training metrics
                wandb.log({"train/loss": loss.item()})

        model.eval()
        val_loss = 0
        val_outputs = []
        val_targets = []
        with torch.no_grad():
            for item, target in val_dataloader:
                item = item.to(device)
                target = target.to(device)

                if (
                    item_format == "raw"
                    and not task_config["training"]["feature_aggregation"]
                ):
                    with torch.no_grad():
                        item = extractor(item)
                        # if channels eaten up, unsqueeze
                        if item.dim() == 2:
                            item = item.unsqueeze(1)

                val_output = model(item)
                val_loss += criterion(val_output, target).item()

                # Store outputs and targets for metric calculation
                val_outputs.append(val_output)
                val_targets.append(target)

        # Concatenate all outputs and targets
        val_outputs = torch.cat(val_outputs, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        # Calculate metrics
        val_metric_results = {}
        for metric_cfg, metric in zip(
            task_config["evaluation"]["metrics"], val_metrics
        ):
            metric = metric.to(device)
            val_metric_results[metric_cfg["name"]] = metric(val_outputs, val_targets)

        avg_val_loss = val_loss / len(val_dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} -",
            f"Avg train loss: {avg_loss:.4f},",
            f"Avg val loss: {avg_val_loss:.4f}",
        )
        for name, value in val_metric_results.items():
            print(f"{name}: {value:.4f}")

        if logging:
            # Log validation metrics
            wandb.log(
                {
                    "val/loss": val_loss,
                    **{
                        f"val/{name}": value
                        for name, value in val_metric_results.items()
                    },
                }
            )

        # Check if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model
    save_path = Path("ckpt") / "INFO" / "DOWN" / task / dataset / f"{feature}.pt"
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
    task: str,
    task_config: Optional[dict] = None,
    item_format: str = "feature",
    device: Optional[str] = None,
    logging: bool = False,
):
    """
    Evaluate a given trained downstream model.

    Args:
        model: Trained downstream model, or wandb artifact path to model.
               If str is provided, the configs saved online are used, and
               the local ones are ignored.
        feature: Name of the feature/embedding model.
        dataset: Name of the dataset.
        task: Name of the downstream task.
        item_format: Format of the input data: ["raw", "feature"].
                     Defaults to "feature". If raw, feature is
                     extracted on-the-fly.
        task_config: Override certain values of the task configuration.
        device: Device to use for evaluation (defaults to "cuda" if available).
        logging: Whether to log to wandb.

    Returns:
        Dictionary of evaluation metrics.
    """

    if isinstance(model, str):
        # Resume wandb run
        entity, project, run_id, model_name = model.split("/")
        wandb.init(project=project, entity=entity, id=run_id, resume="allow")

    feature_config = feature_configs[feature]
    task_config = deep_update(
        deep_update(task_configs["default"], task_configs.get(task, None)),
        task_config,
    )

    test_dataset = get_dataset(
        name=dataset,
        feature=feature,
        split="test",
        download=False,
        item_format=item_format,
    )

    if isinstance(model, str):
        # Load model from wandb artifact
        model_wandb_path = f"{entity}/{project}/{model_name}"
        artifact = wandb.use_artifact(f"{model_wandb_path}:latest")
        artifact_dir = artifact.download()
        model = get_probe(
            model_type=task_config["model"]["type"],
            in_features=feature_config["feature_dim"],
            n_outputs=len(test_dataset.label_encoder.classes_),
            **task_config["model"]["params"],
        )
        model.load_state_dict(torch.load(Path(artifact_dir) / f"{feature}.pt"))
        os.remove(Path(artifact_dir) / f"{feature}.pt")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    metrics = instantiate_metrics(
        metric_configs=task_config["evaluation"]["metrics"],
        num_classes=len(test_dataset.label_encoder.classes_),
    )

    if test_dataset[0][0].dim() == 3:
        # If item is 3D, this is a dataset that returns items with subitems
        # (e.g. for audio).
        if task_config["evaluation"]["feature_aggregation"]:
            # If feature_aggreation, we'll wrap the dataset so that it returns
            # aggregated features
            aggregated_test = AggregateDataset(
                test_dataset, feature_extractor_name=feature
            )
            del test_dataset
            test_dataset = aggregated_test
        else:
            # If not feature_aggregation, we'll wrap the dataset so that it behaves
            # as a subitem dataset
            wrapped_test = SubitemDataset(test_dataset)
            del test_dataset
            test_dataset = wrapped_test

    dataloader = DataLoader(
        test_dataset,
        batch_size=task_config["evaluation"]["batch_size"],
        shuffle=False,
    )

    # if raw_data  (e.g. audio) is being returned from dataset,
    # extract features on-the-fly
    # (the AggregateDatset wrapper also computes features)
    if item_format == "raw" and not task_config["evaluation"]["feature_aggregation"]:
        extractor = get_feature_extractor(feature)
        extractor.to(device)

    model.eval()
    total_loss = 0
    test_outputs = []
    test_targets = []
    criterion = task_config["evaluation"]["criterion"]()

    with torch.no_grad():
        for item, target in dataloader:
            item = item.to(device)
            target = target.to(device)

            if (
                item_format == "raw"
                and not task_config["evaluation"]["feature_aggregation"]
            ):
                with torch.no_grad():
                    item = extractor(item)
                    # if channels eaten up, unsqueeze
                    if item.dim() == 2:
                        item = item.unsqueeze(1)

            output = model(item)
            total_loss += criterion(output, target).item()

            # Store outputs and targets for metric calculation
            test_outputs.append(output)
            test_targets.append(target)

    # Concatenate all outputs and targets
    test_outputs = torch.cat(test_outputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    # Calculate metrics
    test_metric_results = {}
    for metric_cfg, metric in zip(task_config["evaluation"]["metrics"], metrics):
        metric = metric.to(device)
        test_metric_results[metric_cfg["name"]] = metric(
            test_outputs, test_targets
        ).item()

    avg_loss = total_loss / len(dataloader)
    print(f"Avg test loss: {avg_loss:.4f}")

    for name, value in test_metric_results.items():
        print(f"{name}: {value:.4f}")

    if logging:
        # Create a table for the evaluation metrics
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Average Loss", avg_loss)
        for name, value in test_metric_results.items():
            metrics_table.add_data(name, value)

        # Log the table to wandb
        wandb.log({"evaluation_metrics": metrics_table})
        wandb.finish()

    return test_metric_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a downstream model.")
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
        "--task",
        "-t",
        type=str,
        required=True,
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
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )

    results = evaluate(
        model=model,
        feature=args.feature,
        dataset=args.dataset,
        task=args.task,
        device=args.device,
        logging=not args.nolog,
    )
