"""Convert cached EDM CIFAR teacher samples to an efficient DMD HDF5 format."""

import argparse
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm


def convert(cache_dir: Path, output: Path, max_samples: Optional[int] = None) -> None:
    shard_paths = sorted(cache_dir.glob("shard-*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard-*.pt files found in {cache_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    written = 0
    class_counts = {}
    with h5py.File(tmp_output, "w") as hf:
        hf.attrs["source_cache_dir"] = cache_dir.as_posix()
        hf.attrs["format"] = "dmd-cifar-pairs-array"
        image_shape = (3, 32, 32)
        image_chunks = (1024, *image_shape)
        images_out = hf.create_dataset(
            "images",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            chunks=image_chunks,
            dtype=np.float16,
        )
        latents_out = hf.create_dataset(
            "latents",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            chunks=image_chunks,
            dtype=np.float16,
        )
        labels_out = hf.create_dataset("labels", shape=(0,), maxshape=(None,), chunks=(8192,), dtype=np.int64)
        seeds_out = hf.create_dataset("seeds", shape=(0,), maxshape=(None,), chunks=(8192,), dtype=np.int64)

        for shard_path in tqdm(shard_paths, desc="Converting EDM cache shards"):
            shard = torch.load(shard_path, map_location="cpu")
            images = shard["images"].cpu().numpy().astype(np.float16, copy=False)
            latents = shard["latents"].cpu().numpy().astype(np.float16, copy=False)
            shard_count = int(shard.get("num_samples", images.shape[0]))
            if "labels" in shard:
                labels = shard["labels"].cpu().numpy()
            else:
                labels = np.zeros(shard_count, dtype=np.int64)
            seeds = shard["seeds"].cpu().numpy()

            if max_samples is not None:
                shard_count = min(shard_count, max_samples - written)
            if shard_count <= 0:
                break

            start = written
            end = written + shard_count
            for dataset in (images_out, latents_out):
                dataset.resize((end, *image_shape))
            labels_out.resize((end,))
            seeds_out.resize((end,))

            images_out[start:end] = images[:shard_count]
            latents_out[start:end] = latents[:shard_count]
            labels_out[start:end] = labels[:shard_count].astype(np.int64, copy=False)
            seeds_out[start:end] = seeds[:shard_count].astype(np.int64, copy=False)
            for label, count in zip(*np.unique(labels[:shard_count], return_counts=True)):
                label = int(label)
                class_counts[label] = class_counts.get(label, 0) + int(count)
            written = end

            if max_samples is not None and written >= max_samples:
                break

        hf.attrs["num_samples"] = written
        for class_idx, count in sorted(class_counts.items()):
            hf.attrs[f"class_{class_idx}_count"] = count

    os.replace(tmp_output, output)
    print(f"Wrote {written} samples to {output}")
    print("Class counts:", dict(sorted(class_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    convert(args.cache_dir, args.output, args.max_samples)


if __name__ == "__main__":
    main()
