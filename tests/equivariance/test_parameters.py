import pytest
import torch
from torch import nn

from synesis.equivariance.parameters import evaluate, train


@pytest.fixture(params=["VGGishMTAT"])
def feature_name(request):
    return request.param


@pytest.fixture(params=["TinySOL"])
def dataset_name(request):
    return request.param


@pytest.fixture(params=["default"])
def task_name(request):
    return request.param


@pytest.fixture(params=["Gain"])
def transform_name(request):
    return request.param


def test_train(
    feature_name,
    dataset_name,
    task_name,
    transform_name,
):
    model = train(
        feature=feature_name,
        dataset=dataset_name,
        transform=transform_name,
        task=task_name,
        task_config={"training": {"num_epochs": 2}},
        # logging=False,
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


def test_evaluate(
    feature_name,
    dataset_name,
    task_name,
    transform_name,
):
    model = train(
        feature=feature_name,
        dataset=dataset_name,
        transform=transform_name,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        # logging=False,
    )

    # Evaluate the model
    results = evaluate(
        model=model,
        feature=feature_name,
        dataset=dataset_name,
        transform=transform_name,
        task=task_name,
        task_config={"training": {"num_epochs": 1}},
        # logging=False,
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
