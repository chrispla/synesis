import hashlib
import os
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
import wget
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track
from synesis.utils import download_github_dir, download_github_file


def download_files(files_dir, audio_dir, mtg=False):
    FILES_DIR = files_dir
    BASE_URL = "http://geo-samples.beatport.com/lofi/"
    BACKUP_BASE_URL = (
        "http://www.cp.jku.at/datasets/giantsteps/backup/"
        if not mtg
        else "http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/"
    )
    AUDIO_PATH = audio_dir

    errors = 0
    successful = 0
    backup = 0
    total_count = 0

    os.makedirs(AUDIO_PATH, exist_ok=True)

    def md5_for(file_path):
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def download_file_wget(url, output_path):
        try:
            wget.download(url, out=output_path, bar=None)
            return True
        except Exception:
            return False

    def process_md5_file(md5_file):
        nonlocal successful, backup, errors, total_count
        total_count += 1
        filename = os.path.splitext(md5_file)[0]
        mp3_filename = f"{filename}.mp3"
        mp3_url = f"{BASE_URL}{mp3_filename}"
        mp3_backup_url = f"{BACKUP_BASE_URL}{mp3_filename}"
        audio_file_path = os.path.join(AUDIO_PATH, mp3_filename)
        md5_file_path = os.path.join(FILES_DIR, md5_file)

        if download_file_wget(mp3_url, audio_file_path):
            md5_value = md5_for(audio_file_path)
        else:
            md5_value = None

        with open(md5_file_path, "r") as f:
            expected_md5 = f.read().strip()

        if md5_value == expected_md5:
            successful += 1
        else:
            if download_file_wget(mp3_backup_url, audio_file_path):
                md5_value = md5_for(audio_file_path)
            else:
                md5_value = None

            if md5_value == expected_md5:
                successful += 1
                backup += 1
            else:
                errors += 1
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)

    md5_list = os.listdir(FILES_DIR)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_md5_file, md5_file) for md5_file in md5_list]
        with tqdm(total=len(md5_list), desc="Downloading files") as pbar:
            for _ in as_completed(futures):
                pbar.update(1)

    print(f"\nFiles successfully downloaded: {successful}/{total_count}")
    print(f"Files from backup location: {backup}/{successful}")
    print(f"Errors: {errors}")


class GiantstepsKey(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/GiantstepsKey",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = "mp3",
        item_format: str = "feature",
        itemization: bool = True,
        seed: int = 42,
    ) -> None:
        """Giantsteps dataset implementation.

        The implementation is the same as MARBLE, with giantsteps used as test set
        and Giantsteps-Jamendo as train set.

        Args:
            feature: If split is None, prepare dataset for this feature extractor.
                     If split is not None, load these extracted features.
            root: Root directory of the dataset. Defaults to "data/MagnaTagATune".
            split: Split of the dataset to use: ["train", "test", "validation", None],
                     where None uses the full dataset (e.g. for feature extraction).
            download: Whether to download the dataset if it doesn't exist.
            feature_config: Configuration for the feature extractor.
            audio_format: Format of the audio files: ["mp3", "wav", "ogg"].
            item_format: Format of the items to return: ["raw", "feature"].
            itemization: For datasets with variable-length items, whether to return them
                      as a list of equal-length items (True) or as a single item.
            seed: Random seed for reproducibility.
        """
        self.tasks = ["key_estimation"]
        self.fvs = ["pitch", "tempo", "eq"]

        root = Path(root)
        self.root = root
        if split not in [None, "train", "test", "validation"]:
            raise ValueError(
                f"Invalid split: {split} "
                + "Options: None, 'train', 'test', 'validation'"
            )
        self.split = split
        self.item_format = item_format
        self.itemization = itemization
        self.audio_format = audio_format
        self.feature = feature
        self.label_encoder = LabelEncoder()

        if not feature_config:
            # load default feature config
            feature_config = feature_configs[feature]
        self.feature_config = feature_config

        self.seed = seed
        if download:
            self._download()

        self._load_metadata()

    def _download(self) -> None:
        # make data dir if it doesn't exist or if it exists but is empty
        if (
            os.path.exists(
                os.path.join(self.root, "giantsteps-mtg-key-dataset/annotations")
            )
            and (
                len(
                    os.listdir(
                        os.path.join(
                            self.root, "giantsteps-mtg-key-dataset/annotations"
                        )
                    )
                )
            )
            != 0
        ):
            import warnings

            warnings.warn(
                f"annotations already exists in '{self.root}'."
                + "Skipping annotations download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "giantsteps-mtg-key-dataset/annotations").mkdir(
                parents=True, exist_ok=True
            )

            download_github_dir(
                owner="GiantSteps",
                repo="giantsteps-mtg-key-dataset",
                path="annotations",
                save_dir=str(self.root / "giantsteps-mtg-key-dataset/annotations"),
            )

        if os.path.exists(
            os.path.join(self.root, "giantsteps-key-dataset/annotations")
        ) and (
            len(
                os.listdir(
                    os.path.join(self.root, "giantsteps-key-dataset/annotations")
                )
            )
            != 0
        ):
            import warnings

            warnings.warn(
                f"annotations already exists in '{self.root}'."
                + "Skipping annotations download.",
                stacklevel=2,
            )
        else:
            (Path(self.root) / "giantsteps-key-dataset/annotations").mkdir(
                parents=True, exist_ok=True
            )
            download_github_dir(
                owner="GiantSteps",
                repo="giantsteps-key-dataset",
                path="annotations",
                save_dir=str(self.root / "giantsteps-key-dataset/annotations"),
            )

        if os.path.exists(os.path.join(self.root, "mp3")) and (
            len(os.listdir(os.path.join(self.root, "mp3"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Dataset '{self.__class__.__name__}' already exists in '{self.root}'."
                + "Skipping audio download.",
                stacklevel=2,
            )
        else:
            (Path(self.root) / "mp3").mkdir(parents=True, exist_ok=True)

            print(f"Downloading Giansteps to {self.root}...")

            if (
                os.path.exists(os.path.join(self.root, "giantsteps-key-dataset/md5"))
                and (
                    len(
                        os.listdir(
                            os.path.join(self.root, "giantsteps-key-dataset/md5")
                        )
                    )
                )
                != 0
            ):
                import warnings

                warnings.warn(
                    f"md5 already exists in '{self.root}'." + "Skipping md5 download.",
                    stacklevel=2,
                )
            else:
                (Path(self.root) / "giantsteps-key-dataset/md5").mkdir(
                    parents=True, exist_ok=True
                )
                download_github_dir(
                    owner="GiantSteps",
                    repo="giantsteps-key-dataset",
                    path="md5",
                    save_dir=str(self.root / "giantsteps-key-dataset/md5"),
                )

            if (
                os.path.exists(os.path.join(self.root, "giantsteps-key-dataset/audio"))
                and (
                    len(
                        os.listdir(
                            os.path.join(self.root, "giantsteps-key-dataset/audio")
                        )
                    )
                )
                != 0
            ):
                import warnings

                warnings.warn(
                    f"audio already exists in '{self.root}'."
                    + "Skipping audio download.",
                    stacklevel=2,
                )
            else:
                (Path(self.root) / "giantsteps-key-dataset/audio").mkdir(
                    parents=True, exist_ok=True
                )
                download_github_file(
                    owner="GiantSteps",
                    repo="giantsteps-key-dataset",
                    file_path="audio_dl.sh",
                    save_dir=str(self.root / "giantsteps-key-dataset"),
                )

            if (
                os.path.exists(
                    os.path.join(self.root, "giantsteps-mtg-key-dataset/md5")
                )
                and (
                    len(
                        os.listdir(
                            os.path.join(self.root, "giantsteps-mtg-key-dataset/md5")
                        )
                    )
                )
                != 0
            ):
                import warnings

                warnings.warn(
                    f"md5 already exists in '{self.root}'." + "Skipping md5 download.",
                    stacklevel=2,
                )

            else:
                (Path(self.root) / "giantsteps-mtg-key-dataset/md5").mkdir(
                    parents=True, exist_ok=True
                )

                download_github_dir(
                    owner="GiantSteps",
                    repo="giantsteps-mtg-key-dataset",
                    path="md5",
                    save_dir=str(self.root / "giantsteps-mtg-key-dataset/md5"),
                )

            if (
                os.path.exists(
                    os.path.join(self.root, "giantsteps-mtg-key-dataset/audio")
                )
                and (
                    len(
                        os.listdir(
                            os.path.join(self.root, "giantsteps-mtg-key-dataset/audio")
                        )
                    )
                )
                != 0
            ):
                import warnings

                warnings.warn(
                    f"audio already exists in '{self.root}'."
                    + "Skipping audio download.",
                    stacklevel=2,
                )

            else:
                (Path(self.root) / "giantsteps-mtg-key-dataset/audio").mkdir(
                    parents=True, exist_ok=True
                )
                download_github_file(
                    owner="GiantSteps",
                    repo="giantsteps-mtg-key-dataset",
                    file_path="audio_dl.sh",
                    save_dir=str(self.root / "giantsteps-mtg-key-dataset"),
                )

            download_files(
                files_dir=os.path.join(self.root, "giantsteps-key-dataset/md5"),
                audio_dir=os.path.join(self.root, "mp3"),
            )

            download_files(
                files_dir=os.path.join(self.root, "giantsteps-mtg-key-dataset/md5"),
                audio_dir=os.path.join(self.root, "mp3"),
                mtg=True,
            )

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
        # First load all annotations
        test_annotations_path = self.root / "giantsteps-key-dataset/annotations/key"
        train_annotations_txt = (
            self.root / "giantsteps-mtg-key-dataset/annotations/annotations.txt"
        )
        audio_path = self.root / "mp3"

        # Get all files ending in .key in the test annotations dir
        test_annotations = [
            f
            for f in os.listdir(test_annotations_path)
            if f.endswith(".key")
            and os.path.isfile(os.path.join(test_annotations_path, f))
        ]
        # create df with file path and key
        test_annotations = pd.DataFrame(
            [
                (Path(f).stem, open(test_annotations_path / f, "r").read().strip())
                for f in test_annotations
            ],
            columns=["file_path", "key"],
        )
        test_annotations["file_path"] = audio_path / (
            test_annotations["file_path"].astype(str) + ".mp3"
        )
        # clean up
        test_annotations = test_annotations[~test_annotations["key"].str.contains("/")]
        test_annotations = test_annotations[test_annotations["key"].notna()]
        test_annotations = test_annotations[test_annotations["key"] != "-"]

        # Load and process train annotations
        train_annotations = pd.read_csv(train_annotations_txt, sep="\t")
        train_annotations = train_annotations.iloc[:, :3]
        train_annotations.columns = ["file_path", "key", "confidence"]
        train_annotations = train_annotations[train_annotations["confidence"] == 2]
        train_annotations["file_path"] = audio_path / (
            train_annotations["file_path"].astype(str) + ".LOFI.mp3"
        )

        # Clean annotations
        train_annotations = train_annotations[
            ~train_annotations["key"].str.contains("/")
        ]
        train_annotations = train_annotations[train_annotations["key"].notna()]
        train_annotations = train_annotations[train_annotations["key"] != "-"]

        # Create train/validation split
        train_fold, val_fold = train_test_split(
            train_annotations, test_size=0.1, random_state=self.seed
        )
        train_fold["split"] = "train"
        val_fold["split"] = "validation"
        test_annotations["split"] = "test"

        # Combine all annotations
        annotations = pd.concat([train_fold, val_fold, test_annotations])

        # Normalize key names
        enharmonic = {
            "C#": "Db",
            "D#": "Eb",
            "F#": "Gb",
            "G#": "Ab",
            "A#": "Bb",
        }
        annotations["key"] = annotations["key"].replace(enharmonic, regex=True)
        annotations["key"] = annotations["key"].apply(lambda x: x.strip())

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(annotations["key"])

        # Filter by requested split
        if self.split:
            mask = annotations["split"] == self.split
            paths = annotations.loc[mask, "file_path"].tolist()
            encoded_labels = encoded_labels[mask]
        else:
            paths = annotations["file_path"].tolist()

        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

        # Store paths and labels
        self.raw_data_paths = paths
        self.labels = encoded_labels

        # Generate feature paths
        self.feature_paths = [
            str(path)
            .replace(f".{self.audio_format}", ".pt")
            .replace("mp3", self.feature)
            for path in paths
        ]

        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

        # Remove non-existent paths
        idx_to_remove = [
            idx for idx, path in enumerate(self.paths) if not os.path.exists(path)
        ]

        if idx_to_remove:
            self.raw_data_paths = [
                p for i, p in enumerate(self.raw_data_paths) if i not in idx_to_remove
            ]
            self.feature_paths = [
                p for i, p in enumerate(self.feature_paths) if i not in idx_to_remove
            ]
            self.labels = self.labels[
                [i for i in range(len(self.labels)) if i not in idx_to_remove]
            ]
            self.paths = (
                self.raw_data_paths if self.item_format == "raw" else self.feature_paths
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        path = (
            self.raw_data_paths[idx]
            if self.item_format == "raw"
            else self.feature_paths[idx]
        )
        label = self.labels[idx]

        track = load_track(
            path=path,
            item_format=self.item_format,
            itemization=self.itemization,
            item_len_sec=self.feature_config["item_len_sec"],
            sample_rate=self.feature_config["sample_rate"],
        )

        return track, label
