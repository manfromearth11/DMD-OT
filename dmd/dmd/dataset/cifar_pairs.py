import h5py
import numpy as np
from torch.utils.data import Dataset


class CIFARPairs(Dataset):
    _shape = (3, 32, 32)  # CHW

    def __init__(self, h5_dataset_path, cache_in_memory: bool = False):
        self.h5_dataset_path = h5_dataset_path
        self.dataset = None
        self.cached = None
        self.array_format = False
        with h5py.File(self.h5_dataset_path, "r") as file:
            self.array_format = "images" in file and "latents" in file
            self.num_samples = len(file["images"]) if self.array_format else len(file["data"])
            if cache_in_memory:
                if not self.array_format:
                    raise ValueError("cache_in_memory=True requires array-format HDF5 with images/latents datasets.")
                self.cached = {
                    "images": file["images"][:],
                    "latents": file["latents"][:],
                    "labels": file["labels"][:] if "labels" in file else np.zeros(self.num_samples, dtype=np.int64),
                    "seeds": file["seeds"][:] if "seeds" in file else np.arange(self.num_samples, dtype=np.int64),
                }

    def __len__(self):
        return self.num_samples

    def _open(self):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_dataset_path, "r")

    def _make_sample(self, index, image, latent, class_idx, seed):
        return {
            "instance_id": index,
            "image": image,
            "latent": latent,
            "class_id": class_idx,
            "seed": seed,
        }

    def __getitem__(self, index):
        # Dataset is created here to avoid errors if (number of workers > 1) in dataloader.
        if self.cached is not None:
            return self._make_sample(
                index,
                self.cached["images"][index],
                self.cached["latents"][index],
                int(self.cached["labels"][index]),
                int(self.cached["seeds"][index]),
            )

        self._open()
        if self.array_format:
            image = self.dataset["images"][index]
            latent = self.dataset["latents"][index]
            class_idx = int(self.dataset["labels"][index]) if "labels" in self.dataset else 0
            seed = int(self.dataset["seeds"][index]) if "seeds" in self.dataset else index
        else:
            sample = self.dataset["data"][str(index)]
            pairs = sample[()]
            attributes = sample.attrs
            image, latent = pairs
            class_idx = attributes["class_idx"]
            seed = attributes["seed"]
        return self._make_sample(index, image, latent, class_idx, seed)

    def __getitems__(self, indices):
        if self.cached is not None:
            indices_np = np.asarray(indices)
            return [
                self._make_sample(int(index), image, latent, int(label), int(seed))
                for index, image, latent, label, seed in zip(
                    indices_np,
                    self.cached["images"][indices_np],
                    self.cached["latents"][indices_np],
                    self.cached["labels"][indices_np],
                    self.cached["seeds"][indices_np],
                )
            ]

        self._open()
        if not self.array_format:
            return [self[index] for index in indices]

        indices_np = np.asarray(indices)
        order = np.argsort(indices_np)
        sorted_indices = indices_np[order]
        restore = np.argsort(order)

        images = self.dataset["images"][sorted_indices][restore]
        latents = self.dataset["latents"][sorted_indices][restore]
        if "labels" in self.dataset:
            labels = self.dataset["labels"][sorted_indices][restore]
        else:
            labels = np.zeros(len(indices), dtype=np.int64)
        if "seeds" in self.dataset:
            seeds = self.dataset["seeds"][sorted_indices][restore]
        else:
            seeds = indices_np

        return [
            self._make_sample(int(index), image, latent, int(label), int(seed))
            for index, image, latent, label, seed in zip(indices_np, images, latents, labels, seeds)
        ]

    @property
    def image_shape(self):
        return list(self._shape)

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
