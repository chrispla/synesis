import numpy as np
import pytest
import torch
from torch import tensor
from torch.utils.data import Dataset

from synesis.datasets.dataset_utils import AggregateDataset, SubitemDataset
from synesis.datasets.magnatagatune import MagnaTagATune
from synesis.datasets.mtgjamendo import MTGJamendo
from synesis.datasets.giantsteps_key import GiantstepsKey
from synesis.datasets.tinysol import TinySOL

DATASETS = [
    (
        MagnaTagATune,
        {
            "root": "data/MagnaTagATune",
            "splits": [None, "train", "test", "validation"],
            "item_format": "raw",
        },
    ),
    (
        MTGJamendo,
        {
            "root": "data/MTGJamendo",
            "data_path": "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3",
            "splits": [None, "train", "test", "validation"],
            "subsets": [None, "top50tags", "genre", "instrument", "moodtheme"],
            "item_format": "raw",
        },
    ),
    (
        TinySOL,
        {
            "root": "data/TinySOL",
            "splits": [None, "train", "test", "validation"],
            "item_format": "raw",
        },
    ),
    (
        TinySOL,
        {
            "root": "data/TinySOL",
            "splits": [None, "train", "test", "validation"],
            "item_format": "raw",
        },
    ),
    (
        GiantstepsKey,
        {
            "root": "data/GiantStepsKey",
            "splits": [None, "train", "test", "validation"],
            "item_format": "raw",
        },
    ),
]


class MockDataset(Dataset):
    # Create a simple dataset with a few items
    def __init__(self):
        self.items = [
            torch.stack(
                [
                    tensor([0, 0, 0]).unsqueeze(0).float(),
                    tensor([1, 1, 1]).unsqueeze(0).float(),
                    tensor([2, 2, 2]).unsqueeze(0).float(),
                ]
            ),
            torch.stack(
                [
                    tensor([3, 3, 3]).unsqueeze(0).float(),
                    tensor([4, 4, 4]).unsqueeze(0).float(),
                ]
            ),
            torch.stack(
                [
                    tensor([5, 5, 5]).unsqueeze(0).float(),
                ]
            ),
        ]
        self.item_format = "feature"
        self.label_encoder = None
        self.raw_data_paths = None
        self.feature_paths = None
        self.paths = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx], "mock_label"


@pytest.fixture(params=DATASETS)
def dataset_config(request):
    return request.param


@pytest.fixture(params=[True, False])
def itemization(request):
    return request.param


@pytest.fixture(params=["raw"])
def item_format(request):
    return request.param


def test_subitem_wrapper():
    dataset = MockDataset()
    subitem_dataset = SubitemDataset(dataset)

    dataloader = torch.utils.data.DataLoader(subitem_dataset, batch_size=4)

    for i, (item, label) in enumerate(dataloader):
        assert item.shape[0] <= 4, "Batch size should not exceed 4"
        assert len(item.shape) == 3, "Feature dimension should be 3"
        assert label[0] == "mock_label", "Label should be the same for all items"
        if i == 0:
            assert torch.equal(
                item,
                torch.tensor(
                    [
                        [[0, 0, 0]],
                        [[1, 1, 1]],
                        [[2, 2, 2]],
                        [[3, 3, 3]],
                    ]
                ),
            )
        elif i == 1:
            assert torch.equal(
                item,
                torch.tensor(
                    [
                        [[4, 4, 4]],
                        [[5, 5, 5]],
                    ]
                ),
            )


def test_aggregate_wrapper():
    dataset = MockDataset()
    aggregate_dataset = AggregateDataset(dataset)

    dataloader = torch.utils.data.DataLoader(
        aggregate_dataset, batch_size=2, shuffle=False
    )

    item, label = next(iter(dataloader))
    assert item.shape[0] <= 2, "Batch size should not exceed 2"
    assert len(item.shape) == 3, "(b, c, len)"
    assert item[0][0][0] == 1.0, "First item should be (0+1+2)/3"
    assert item[1][0][1] == 3.5, "Second item should be (3+4)/2"


def test_dataset_loading(dataset_config, itemization, item_format):
    DatasetClass, config = dataset_config
    for split in config["splits"]:
        dataset = DatasetClass(
            feature="VGGishMTAT",
            root=config["root"],
            item_format=item_format,
            itemization=itemization,
            split=split,
        )

        # Test paths and labels are loaded
        assert len(dataset.paths) > 0
        assert len(dataset.raw_data_paths) > 0
        assert len(dataset.feature_paths) > 0
        assert len(dataset.labels) > 0
        assert len(dataset.paths) == len(dataset.labels)
        assert dataset.labels.dtype == torch.long

        # Test random items
        for _ in range(5):
            idx = np.random.randint(0, len(dataset))
            item, label = dataset[idx]
            assert torch.is_tensor(item)
            assert torch.is_tensor(label)
            assert label.dtype == torch.long

            if itemization:
                # each item in batch will have a channel dim
                assert len(item.shape) == 3
                # each item will be the same length
                assert all(subitem.shape[1] == item[0].shape[1] for subitem in item)
                # all items have a channel dim of 1
                assert all(subitem.shape[0] == 1 for subitem in item)
            else:
                assert len(item.shape) == 2
                assert item.shape[0] == 1


def test_mtgjamendo_subsets():
    config = next(conf for cls, conf in DATASETS if cls == MTGJamendo)
    for subset in config["subsets"]:
        for split in config["splits"]:
            dataset = MTGJamendo(
                feature="VGGishMTAT",
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
