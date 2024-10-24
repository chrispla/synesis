import numpy as np
import pytest
import torch

from synesis.datasets.magnatagatune import MagnaTagATune
from synesis.datasets.mtgjamendo import MTGJamendo

DATASETS = [
    (
        MagnaTagATune,
        {
            "root": "data/MagnaTagATune",
            "splits": [None, "train", "test", "validation"],
            "item_format": "audio",
        },
    ),
    (
        MTGJamendo,
        {
            "root": "data/MTGJamendo",
            "splits": [None, "train", "test", "validation"],
            "subsets": [None, "top50tags", "genre", "instrument", "moodtheme"],
            "item_format": "audio",
        },
    ),
]


@pytest.fixture(params=DATASETS)
def dataset_config(request):
    return request.param


def test_dataset_loading(dataset_config):
    DatasetClass, config = dataset_config
    for split in config["splits"]:
        dataset = DatasetClass(
            feature="vggish_mtat",
            root=config["root"],
            item_format=config["item_format"],
            split=split,
        )

        # Test paths and labels are loaded
        assert len(dataset.paths) > 0
        assert len(dataset.labels) > 0
        assert len(dataset.paths) == len(dataset.labels)

        # Test labels are encoded
        assert isinstance(dataset.labels, np.ndarray)
        assert dataset.labels.dtype == np.int64

        # Test random items are 2D tensors
        for _ in range(5):
            idx = np.random.randint(0, len(dataset))
            item, label = dataset[idx]
            assert len(item.shape) == 2
            assert torch.is_tensor(item)
            assert isinstance(label, np.ndarray)


def test_mtgjamendo_subsets():
    config = next(conf for cls, conf in DATASETS if cls == MTGJamendo)
    for subset in config["subsets"]:
        for split in config["splits"]:
            dataset = MTGJamendo(
                feature="vggish_mtat",
                root=config["root"],
                item_format=config["item_format"][0],
                split=split,
                subset=subset,
            )
            assert (
                len(dataset) > 0
            ), f"Empty dataset for subset: {subset}, split: {split}"
            if hasattr(dataset, "metadata_path"):
                if subset:
                    assert str(subset) in str(
                        dataset.metadata_path
                    ), f"Subset {subset} not in metadata path for split {split}"
                if split:
                    assert str(split) in str(
                        dataset.metadata_path
                    ), f"Split {split} not in metadata path for subset {subset}"


if __name__ == "__main__":
    pytest.main()
