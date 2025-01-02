import os
import os.path
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset, random_split

from config.features import configs as feature_configs
from synesis.datasets.dataset_utils import load_track
from synesis.utils import download_github_dir


class AllTempo(Dataset):
    def __init__(
        self,
        feature: str,
        root: Union[str, Path] = "data/GiantsetpsKey",
        split: Optional[str] = None,
        download: bool = False,
        feature_config: Optional[dict] = None,
        audio_format: str = ["mp3", "wav"],
        item_format: str = "feature",
        itemization: bool = True,
        seed: int = 42,
        datasets=["gtzan", "hainsworth", "acmm", "giantsteps", "xballroom"],
        test_dataset="gtzan",
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
        self.tasks = ["tempo_estimation"]
        self.fvs = ["pitch", "tempo", "eq"]
        self.datasets = datasets
        self.test_dataset = test_dataset

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
        if (
            os.path.exists(os.path.join(self.root, "acm_mirum"))
            and (len(os.listdir(os.path.join(self.root, "acm_mirum")))) != 0
        ):
            import warnings

            warnings.warn(
                f"acm_mirum already exists in '{self.root}'." + "Skipping download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "acm_mirum").mkdir(parents=True, exist_ok=True)

            download_github_dir(
                owner="Pliploop",
                repo="AllTempo",
                path="acm_mirum",
                save_dir=str(self.root / "acm_mirum"),
            )

        # same for extended_ballroom, giansteps, gtzan, hainsworth
        if os.path.exists(os.path.join(self.root, "extended_ballroom")) and (
            len(os.listdir(os.path.join(self.root, "extended_ballroom"))) != 0
        ):
            import warnings

            warnings.warn(
                f"extended_ballroom already exists in '{self.root}'."
                + "Skipping download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "extended_ballroom").mkdir(parents=True, exist_ok=True)

            download_github_dir(
                owner="Pliploop",
                repo="AllTempo",
                path="extended_ballroom",
                save_dir=str(self.root / "extended_ballroom"),
            )

        if os.path.exists(os.path.join(self.root, "giantsteps")) and (
            len(os.listdir(os.path.join(self.root, "giantsteps"))) != 0
        ):
            import warnings

            warnings.warn(
                f"giantsteps already exists in '{self.root}'." + "Skipping download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "giantsteps").mkdir(parents=True, exist_ok=True)

            download_github_dir(
                owner="Pliploop",
                repo="AllTempo",
                path="giantsteps",
                save_dir=str(self.root / "giantsteps"),
            )

        if os.path.exists(os.path.join(self.root, "gtzan")) and (
            len(os.listdir(os.path.join(self.root, "gtzan"))) != 0
        ):
            import warnings

            warnings.warn(
                f"gtzan already exists in '{self.root}'." + "Skipping download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "gtzan").mkdir(parents=True, exist_ok=True)

            download_github_dir(
                owner="Pliploop",
                repo="AllTempo",
                path="gtzan",
                save_dir=str(self.root / "gtzan"),
            )

        if os.path.exists(os.path.join(self.root, "hainsworth")) and (
            len(os.listdir(os.path.join(self.root, "hainsworth"))) != 0
        ):
            import warnings

            warnings.warn(
                f"hainsworth already exists in '{self.root}'." + "Skipping download.",
                stacklevel=2,
            )

        else:
            (Path(self.root) / "hainsworth").mkdir(parents=True, exist_ok=True)

            download_github_dir(
                owner="Pliploop",
                repo="AllTempo",
                path="hainsworth",
                save_dir=str(self.root / "hainsworth"),
            )

    def _load_metadata(self) -> Tuple[list, torch.Tensor]:
        gtzan_annotations, gtzan_idx2class = self.get_gtzan_tempo_annotations(
            audio_path=os.path.join(self.root, "gtzan/audio"),
            tempo_annotations_folder=os.path.join(
                self.root, "gtzan/gtzan_tempo_beat/tempo"
            ),
            split_path=os.path.join(self.root, "gtzan/music_dataset_split"),
        )

        hainsworth_annotations, _ = self.get_hainsworth_tempo_annotations(
            hainsworth_audio_path=os.path.join(self.root, "hainsworth/audio"),
            hainsworth_annotations_path=os.path.join(
                self.root, "hainsworth/beat_and_downbeat_annotations"
            ),
        )

        acmm_annotations, _ = self.get_acmm_tempo_annotations(
            acmm_annotation_path=os.path.join(
                self.root, "acm_mirum/acm-mirum/Annotations/acm_mirum_tempos.mf"
            ),
            audio_path=os.path.join(self.root, "acm_mirum/Audio"),
        )

        giantsteps_annotations, _ = self.get_giantsteps_tempo_annotations(
            giansteps_annotations_path=os.path.join(
                self.root, "giantsteps/annotations_v2/tempo"
            ),
            giansteps_audio_path=os.path.join(self.root, "giantsteps/audio"),
        )

        xballroom_annotations, _ = self.get_xballroom_tempo_annotations(
            xml_path=os.path.join(
                self.root,
                "extended_ballroom/ballroom_extended_2016/extendedballroom_v1.1.xml",
            )
        )

        datasets = {
            "gtzan": gtzan_annotations,
            "hainsworth": hainsworth_annotations,
            "acmm": acmm_annotations,
            "giantsteps": giantsteps_annotations,
            "xballroom": xballroom_annotations,
        }

        annotations = pd.concat(
            [datasets[dataset_] for dataset_ in self.datasets if dataset_ in datasets]
        )

        # task is the testing set, the rest is split using val_split
        task = self.test_dataset
        val_split = 0.1

        test_annotations = annotations[annotations["task"] == task]
        train_val_annotations = annotations[annotations["task"] != task]

        train_len = int(len(train_val_annotations) * (1 - val_split))

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_annotations, val_annotations = random_split(
            train_val_annotations,
            [train_len, len(train_val_annotations) - train_len],
            generator=generator,
        )

        train_annotations = train_val_annotations.iloc[train_annotations.indices]
        val_annotations = train_val_annotations.iloc[val_annotations.indices]

        train_annotations.loc[:, "split"] = "train"
        val_annotations.loc[:, "split"] = "val"
        test_annotations.loc[:, "split"] = "test"

        annotations = pd.concat([train_annotations, val_annotations, test_annotations])

        self.labels = annotations["labels"].tolist()
        encoded_labels = torch.tensor(self.labels)
        print(encoded_labels)

        if self.split:
            # keep these indices in path and labels
            indices = annotations["split"] == self.split

            raw_data_paths = annotations["file_path"][indices].tolist()
            encoded_labels = encoded_labels[indices]

        else:
            raw_data_paths = annotations["file_path"].tolist()
            encoded_labels = encoded_labels

        self.raw_data_paths, self.labels = raw_data_paths, encoded_labels

        self.feature_paths = []
        for path in raw_data_paths:
            path = str(path)
            for fmt_ in self.audio_format:
                path = path.replace(f".{fmt_}", self.feature)
            path = path.replace("mp3", self.feature)
            self.feature_paths.append(path)

        self.paths = (
            self.raw_data_paths if self.item_format == "raw" else self.feature_paths
        )

        # remove paths that don't exist
        idx_to_remove = [
            idx for idx, path in enumerate(self.paths) if not os.path.exists(path)
        ]

        self.raw_data_paths = [
            path
            for idx, path in enumerate(self.raw_data_paths)
            if idx not in idx_to_remove
        ]

        self.feature_paths = [
            path
            for idx, path in enumerate(self.feature_paths)
            if idx not in idx_to_remove
        ]

        self.labels = self.labels[
            [idx for idx in range(len(self.labels)) if idx not in idx_to_remove]
        ]

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

    def tempo_to_dummies(self, tempo_series):
        dummies = []
        new_tempo_series = []
        for tempo in tempo_series:
            dummy = [0] * 300
            if tempo < 300:
                dummy[tempo] = 1
                new_tempo_series.append(tempo)
            else:
                new_tempo_series.append(None)
            dummies.append(dummy)
        return dummies, new_tempo_series

    def get_acmm_tempo_annotations(self, acmm_annotation_path, audio_path):
        tempo_annotations = pd.read_csv(acmm_annotation_path, sep="\t", header=None)
        tempo_annotations.columns = ["file_path", "tempo"]
        tempo_annotations.tempo = tempo_annotations.tempo.apply(lambda x: int(x))
        tempo_annotations["split"] = "train"
        tempo_annotations["labels"] = None

        dummies, tempi = self.tempo_to_dummies(tempo_annotations["tempo"])
        tempo_annotations["labels"] = dummies
        tempo_annotations["tempo"] = tempi
        tempo_annotations["file_name"] = tempo_annotations["file_path"].apply(
            lambda x: x.split("/")[-1].replace(".wav", "")
        )
        tempo_annotations["file_path"] = (
            audio_path + "/" + tempo_annotations["file_name"] + ".mp3"
        )
        tempo_annotations["task"] = "acmm"

        self.n_classes = 300

        idx2class = {i: c for i, c in enumerate(range(300))}

        print(tempo_annotations)

        return tempo_annotations, idx2class

    def get_gtzan_tempo_annotations(
        self, audio_path, tempo_annotations_folder, split_path
    ):
        train_annotations = pd.read_csv(
            f"{split_path}/GTZAN_split/train_filtered.txt", sep=" ", header=None
        )

        val_annotations = pd.read_csv(
            f"{split_path}/GTZAN_split/valid_filtered.txt", sep=" ", header=None
        )

        test_annotations = pd.read_csv(
            f"{split_path}/GTZAN_split/test_filtered.txt", sep=" ", header=None
        )

        train_annotations["split"] = "train"
        val_annotations["split"] = "val"
        test_annotations["split"] = "test"
        annotations = pd.concat([train_annotations, val_annotations, test_annotations])
        annotations.columns = ["file_path", "split"]

        annotations["tempo"] = None

        dummies = []
        tempi = []

        annotations["tempo_file"] = annotations["file_path"].apply(
            lambda x: (
                f"{tempo_annotations_folder}/gtzan_"
                + f"{x.split('/')[1][:-4].replace('.','_')}.bpm"
            )
        )

        for idx, row in annotations.iterrows():
            with open(row.tempo_file, "r") as f:
                # read the first line to an int
                tempo = int(eval(f.readline().replace("\n", "")))
                tempi.append(tempo)

        # labels
        dummies, tempi = self.tempo_to_dummies(tempi)

        annotations["labels"] = dummies
        annotations["tempo"] = tempi

        annotations = annotations[annotations["tempo"].notna()]

        self.n_classes = 300
        annotations["file_path"] = audio_path + "/" + annotations["file_path"]
        annotations["task"] = "gtzan"

        idx2class = {i: c for i, c in enumerate(range(300))}

        return annotations, idx2class

    def get_hainsworth_tempo_annotations(
        self, hainsworth_audio_path, hainsworth_annotations_path
    ):
        annotations = {}

        for root, dirs, files in os.walk(hainsworth_audio_path):
            for file in files:
                if file.endswith(".wav"):
                    file_name = file.split(".")[0]
                    annotation_path = os.path.join(
                        hainsworth_annotations_path, file_name + ".beats"
                    )
                    # open the file with pandas with tab delimiter
                    annotation = pd.read_csv(annotation_path, sep="\t", header=None)
                    # the first column is the beat times
                    # use to get the average tempo
                    tempo = int(60 / np.mean(np.diff(annotation[0])))
                    annotations[file] = tempo

        annotations = pd.DataFrame(annotations.items(), columns=["file_path", "tempo"])
        annotations["split"] = "train"
        annotations["labels"] = None
        dummies, tempi = self.tempo_to_dummies(annotations["tempo"])
        annotations["labels"] = dummies
        annotations["tempo"] = tempi
        annotations["file_path"] = hainsworth_audio_path + annotations["file_path"]
        annotations = annotations[annotations["tempo"].notna()]
        self.n_classes = 300
        annotations["task"] = "hainsworth"
        idx2class = {i: c for i, c in enumerate(range(300))}

        return annotations, idx2class

    def get_xballroom_tempo_annotations(self, xml_path):
        import xml.etree.ElementTree as ET

        def parse_xml_to_dataframe(xml_path):
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            xml_parent_dir = os.path.dirname(xml_path)

            # Prepare a list to store song information
            songs_data = []

            # Iterate through all song elements in the XML
            for genre in root:
                genre_name = genre.tag  # Get the genre name from the XML tag
                for song in genre.findall("song"):
                    bpm = song.get("bpm")
                    song_id = song.get("id")

                    # Construct the file path based on genre folder and song ID
                    file_path = os.path.join(
                        xml_parent_dir, genre_name, f"{song_id}.mp3"
                    )

                    # Append the extracted data to the list
                    songs_data.append({"tempo": int(bpm), "file_path": file_path})

            # Convert the list to a DataFrame
            df = pd.DataFrame(songs_data)
            return df

        annotations = parse_xml_to_dataframe(xml_path)

        annotations["split"] = "train"
        annotations["labels"] = None
        dummies, tempi = self.tempo_to_dummies(annotations["tempo"])
        annotations["labels"] = dummies
        annotations["tempo"] = tempi
        annotations["task"] = "xballroom"
        self.n_classes = 300
        idx2class = {i: c for i, c in enumerate(range(300))}

        return annotations, idx2class

    def get_giantsteps_tempo_annotations(
        self, giansteps_annotations_path, giansteps_audio_path
    ):
        annotations = {}
        for root, dirs, files in os.walk(giansteps_annotations_path):
            for file in files:
                if file.endswith(".bpm"):
                    file_name = file.replace(".bpm", "")
                    # open the file and get the first line without the newline symbol
                    with open(os.path.join(giansteps_annotations_path, file), "r") as f:
                        tempo = int(eval(f.readline().replace("\n", "")))
                        annotations[file_name] = tempo

        annotations = pd.DataFrame(annotations.items(), columns=["file_path", "tempo"])
        annotations["split"] = "train"
        annotations["labels"] = None

        dummies, tempi = self.tempo_to_dummies(annotations["tempo"])
        annotations["labels"] = dummies
        annotations["tempo"] = tempi
        annotations["file_path"] = (
            giansteps_audio_path + "/" + annotations["file_path"] + ".mp3"
        )
        annotations = annotations[annotations["tempo"].notna()]
        self.n_classes = 300
        annotations["task"] = "giantsteps"
        idx2class = {i: c for i, c in enumerate(range(300))}

        return annotations, idx2class
