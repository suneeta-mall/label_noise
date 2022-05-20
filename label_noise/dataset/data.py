import logging
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from pytorch_lightning import LightningDataModule
from rarfile import RarFile
from torchvision.datasets.vision import VisionDataset

from .transforms import EvalTransformer, TrainTransformer

__all__ = ["MLRNetDataset", "MLRNetModule"]

_image_col_name = "image"
_seed = 42
_categoris = [
    "airplane",
    "airport",
    "bare soil",
    "baseball diamond",
    "basketball court",
    "beach",
    "bridge",
    "buildings",
    "cars",
    "chaparral",
    "cloud",
    "containers",
    "crosswalk",
    "dense residential area",
    "desert",
    "dock",
    "factory",
    "field",
    "football field",
    "forest",
    "freeway",
    "golf course",
    "grass",
    "greenhouse",
    "gully",
    "habor",
    "intersection",
    "island",
    "lake",
    "mobile home",
    "mountain",
    "overpass",
    "park",
    "parking lot",
    "parkway",
    "pavement",
    "railway",
    "railway station",
    "river",
    "road",
    "roundabout",
    "runway",
    "sand",
    "sea",
    "ships",
    "snow",
    "snowberg",
    "sparse residential area",
    "stadium",
    "swimming pool",
    "tanks",
    "tennis court",
    "terrace",
    "track",
    "trail",
    "transmission tower",
    "trees",
    "water",
    "wetland",
    "wind turbine",
]

_default_category_list = [
    "airplane",
    "airport",
    "buildings",
    "cars",
    "runway",
    "trees",
]


class MLRNetDataset(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        group_pattern: str = "air*",
        category_list: List[str] = _default_category_list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.images_dir = self.root / "Images"
        self.targets_dir = self.root / "Labels"

        if not (self.images_dir.exists() or self.targets_dir.exists()):
            if root / "7j9bv9vwsx-3.zip".exists():
                with zipfile.ZipFile(root / "7j9bv9vwsx-3.zip", "r") as zip_ref:
                    zip_ref.extractall(root)
            else:
                logging.error(
                    f"Download dataset from https://data.mendeley.com/datasets/7j9bv9vwsx/3 and put it in {root}"
                )

        def unpack_fns(fn, dst_location):
            with RarFile(fn) as rf:
                rf.extractall(dst_location)
            fn.unlink()

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            file_str = "**/*.rar"
            if group_pattern:
                file_str = f"**/{group_pattern}.rar"
            _ = {
                executor.submit(unpack_fns, _rar_fn, self.images_dir): _rar_fn
                for _rar_fn in self.images_dir.glob(file_str)
            }

        file_str = "*.csv"
        if group_pattern:
            file_str = f"{group_pattern}.csv"

        ls_fn = list(map(lambda x: pd.read_csv(x, index_col=None, header=0), self.targets_dir.glob(file_str)))
        self._df = pd.concat(ls_fn, axis=0, ignore_index=True)
        # drop columns not very interesting
        self._df = self._df.loc[:, (self._df != 0).any(axis=0)]

        self.categories = category_list
        if not category_list:
            self.categories = list(set(_categoris).intersection(self._df.columns))
            if len(self.categories) > 6:
                random.seed(42)
                self.categories = list(random.sample(self.categories, 6))
        logging.warning("=======================================")
        logging.warning(f"Found {self.categories} labels")
        logging.warning("=======================================")

        self.split = split
        # # drop some samples for speed
        # self._df = self._df.sample(frac=0.3, random_state=_seed)
        train_val = self._df.sample(frac=0.9, random_state=_seed)
        if split == "test":
            self._df = pd.concat([self._df, train_val, train_val]).drop_duplicates(keep=False)
        else:
            train = train_val.sample(frac=0.9, random_state=_seed)
            if split == "val":
                self._df = pd.concat([train_val, train, train]).drop_duplicates(keep=False)
            else:
                self._df = train

        self.images = []
        self.targets = []

        for fn in self._df[_image_col_name]:
            sub_dir = fn.rsplit("_", 1)[0]
            img = self.images_dir / sub_dir / fn
            target = self._df[self._df.image == fn][self.categories].to_numpy()[0]
            self.images.append(img)
            self.targets.append(target)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")
        target = self.targets[index]
        target = torch.from_numpy(target)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    def classes(self):
        return self.categories

    def num_classes(self):
        return len(self.categories)


class MLRNetModule(LightningDataModule):
    def __init__(
        self,
        data_directory: Path,
        batch_size: int = 256,
        image_size: int = 256,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = MLRNetDataset(
                self.data_directory,
                split="train",
                transform=TrainTransformer(),
                group_pattern="air*",
                category_list=_default_category_list,
            )

            self.eval_dataset = MLRNetDataset(
                self.data_directory,
                split="val",
                transform=EvalTransformer(),
                group_pattern="air*",
                category_list=_default_category_list,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = MLRNetDataset(
                self.data_directory,
                split="test",
                transform=EvalTransformer(),
                group_pattern="air*",
                category_list=_default_category_list,
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):

        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
        return data_loader

    def val_dataloader(self):
        eval_sampler = torch.utils.data.SequentialSampler(self.eval_dataset)
        eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            sampler=eval_sampler,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return eval_loader

    def test_dataloader(self):

        test_sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage):
        logging.info("teardown called")

    def categories(self) -> List[str]:
        return _default_category_list
