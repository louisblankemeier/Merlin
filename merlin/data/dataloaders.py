import torch
import monai
from copy import deepcopy
import shutil
import tempfile
from pathlib import Path
from typing import List
from monai.utils import look_up_option
from monai.data.utils import SUPPORTED_PICKLE_MOD

import merlin


class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_path = item_transformed.get("image")
        image_data = {
            "image": item_transformed.get("image")
        }  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode(
                "utf-8"
            )  # Hash based on image data
            # data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            cached_image = torch.load(hashfile)
            _item_transformed["image"] = cached_image
            return _item_transformed

        _image_transformed = self._pre_transform(image_data)["image"]
        _item_transformed["image"] = _image_transformed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        cache_dir: str,
        batchsize: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.datalist = datalist
        self.cache_dir = cache_dir
        self.batchsize = batchsize
        self.dataset = CTPersistentDataset(
            data=datalist,
            transform=merlin.data.ImageTransforms,
            cache_dir=cache_dir,
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )
