import warnings

import pytest
import torch
from torch import nn

from synesis.informativeness.downstream import evaluate, train


@pytest.fixture(params=["VGGishMTAT"])
def mock_feature_name(request):
    return request.param


@pytest.fixture(params=["TinySOL"])
def dataset_name(request):
    return request.param


@pytest.fixture(params=["pitch_classification"])
def task_name(request):
    return request.param


@pytest.fixture(params=[True, False])
def feature_aggregation(request):
    return request.param


@pytest.fixture(params=["raw", "feature"])
def item_format(request):
    return request.param


def test_train_model(
    dataset_name,
    task_name,
    item_format,
    mock_feature_name,
    feature_aggregation,
    tmp_path,
):
    if item_format == "raw" and feature_aggregation:
        warnings.warn(
            "Currently, using raw data and feature aggregation is too slow"
            + " Skipping this test..."
        )
        return

    # Override config with minimal settings
    task_config = {
        "training": {
            "batch_size": 16,
            "num_epochs": 2,
            "feature_aggregation": feature_aggregation,
        }
    }

    # Train model
    model = train(
        feature=mock_feature_name,
        dataset=dataset_name,
        task="test",
        task_config=task_config,
        item_format=item_format,
        logging=False,
    )

    # Basic assertions
    assert isinstance(model, nn.Module), "Model should be a PyTorch module"
    assert not model.training, "Model should be in eval mode after training"

    # Check if model weights exist and are not all zero
    has_nonzero_params = False
    for param in model.parameters():
        if torch.any(param != 0):
            has_nonzero_params = True
            break
    assert has_nonzero_params, "Model parameters should not all be zero"


def test_evaluate_model(
    dataset_name,
    task_name,
    item_format,
    feature_aggregation,
    mock_feature_name,
):
    if item_format == "raw" and feature_aggregation:
        warnings.warn(
            "Currently, using raw data and feature aggregation is too slow"
            + " Skipping this test..."
        )
        return

    # Configure minimal evaluation settings
    task_config = {
        "training": {
            "batch_size": 16,
            "num_epochs": 1,
            "feature_aggregation": feature_aggregation,
        },
        "evaluation": {
            "batch_size": 16,
        },
    }

    # First train a model with minimal epochs
    model = train(
        feature=mock_feature_name,
        dataset=dataset_name,
        task=task_name,
        task_config=task_config,
        item_format=item_format,
        logging=False,
    )

    # Evaluate the model
    results = evaluate(
        model=model,
        feature=mock_feature_name,
        dataset=dataset_name,
        task=task_name,
        task_config=task_config,
        item_format=item_format,
        logging=False,
    )

    # Check evaluation results
    assert isinstance(results, dict), "Evaluation should return a dictionary of metrics"
    assert len(results) > 0, "Should have at least one metric result"

    # Check that metric values are reasonable
    for metric_name, value in results.items():
        assert isinstance(
            value, (float, int)
        ), f"Metric {metric_name} should be numeric"
        assert not torch.isnan(
            torch.tensor(value)
        ), f"Metric {metric_name} should not be NaN"


if __name__ == "__main__":
    pytest.main()
