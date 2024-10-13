from pathlib import Path

import numpy as np
import pytest
import torch

from config.features import feature_config
from synesis.datasets.magnatagatune import MagnaTagATune
from synesis.datasets.mtgjamendo import MTGJamendo
from synesis.features.feature_utils import dynamic_batch_extractor, get_pretrained_model


@pytest.fixture(params=[MagnaTagATune, MTGJamendo])
def dataset_class(request):
    return request.param


@pytest.fixture(params=feature_config.keys())
def feature_name(request):
    return request.param


def test_feature_extraction(dataset_class, feature_name, tmp_path):
    # Set up dataset with the correct feature
    dataset = dataset_class(
        feature=feature_name,
        root=f"data/{dataset_class.__name__}",
        item_format="audio",
        split=None,
    )

    # Take a small subset of paths for testing
    subset_size = min(5, len(dataset))
    dataset.paths = dataset.paths[:subset_size]
    dataset.labels = dataset.labels[:subset_size]

    # Set up temporary output directory
    output_dir = tmp_path / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update feature_paths to use the temporary directory
    dataset.feature_paths = [
        output_dir / f"{Path(path).stem}.npy" for path in dataset.paths
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the pretrained model
    model = get_pretrained_model(feature_name)

    # Test if model is in eval mode
    assert not model.training, f"Model {feature_name} should be in eval mode"

    # Get feature config
    config = feature_config[feature_name]
    item_len_samples = int(config["item_len_sec"] * config["sample_rate"])

    # Perform feature extraction
    dynamic_batch_extractor(
        dataset,
        model,
        item_len=item_len_samples,
        padding="repeat",
        batch_size=2,
        device=device,
    )

    # Check if features were extracted and saved correctly
    for feature_path in dataset.feature_paths:
        assert feature_path.exists(), f"Feature file {feature_path} not created"

        # Load the extracted feature
        feature = np.load(feature_path)

        # Check feature shape
        assert feature.ndim == 2, f"Feature {feature_path} should be 2-dimensional"

        # If feature_dim is specified in the config, check it
        if "feature_dim" in config:
            assert feature.shape[1] == config["feature_dim"], (
                f"Feature {feature_path} should have {config['feature_dim']} dims, "
                f"but has {feature.shape[1]}"
            )

        # Check that not all entries are zero
        assert np.any(feature), f"All entries in feature {feature_path} are zero"


if __name__ == "__main__":
    pytest.main()
