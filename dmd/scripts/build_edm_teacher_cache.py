"""Build an EDM teacher sample cache for DMD-OT CIFAR experiments."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

from dmd.modeling_utils import StackedRandomGenerator, encode_labels, load_edm
from dmd.sampler import edm_sampler


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
    num, height, width, channels = images_np.shape
    grid_w = int(np.ceil(np.sqrt(num)))
    grid_h = int(np.ceil(num / grid_w))
    grid = np.zeros([grid_h * height, grid_w * width, channels], dtype=np.uint8)
    for idx, image in enumerate(images_np):
        y = idx // grid_w
        x = idx % grid_w
        grid[y * height : (y + 1) * height, x * width : (x + 1) * width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        PIL.Image.fromarray(grid[:, :, 0], "L").save(path)
    else:
        PIL.Image.fromarray(grid, "RGB").save(path)


def write_metadata(outdir: Path, metadata: dict) -> None:
    path = outdir / "metadata.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(tmp, path)


def read_progress(outdir: Path) -> Tuple[int, int, Optional[dict]]:
    meta_path = outdir / "metadata.json"
    if not meta_path.is_file():
        return 0, 0, None
    metadata = json.loads(meta_path.read_text())
    return int(metadata.get("num_generated", 0)), int(metadata.get("next_shard_id", 0)), metadata


def save_shard(
    outdir: Path,
    shard_id: int,
    start_index: int,
    latents: torch.Tensor,
    images: torch.Tensor,
    seeds: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Path:
    path = outdir / f"shard-{shard_id:06d}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = {
        "latents": latents,
        "images": images,
        "seeds": seeds,
        "start_index": int(start_index),
        "num_samples": int(images.shape[0]),
    }
    if labels is not None:
        data["labels"] = labels
    torch.save(data, tmp)
    os.replace(tmp, path)
    return path


def make_class_indices(
    start: int,
    batch_size: int,
    label_dim: int,
    class_mode: str,
    rnd: StackedRandomGenerator,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if label_dim == 0:
        return None
    if class_mode == "balanced":
        return torch.arange(start, start + batch_size, device=device, dtype=torch.long) % label_dim
    if class_mode == "random":
        return rnd.randint(label_dim, size=[batch_size], device=device).to(torch.long)
    raise ValueError(f"Unsupported class mode: {class_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="/dataset/$USER/pretrained/edm-cifar10-32x32-uncond-vp.pkl")
    parser.add_argument("--outdir", default="/dataset/$USER/edm_teacher_cache/cifar10-uncond-edm18-500k")
    parser.add_argument("--num", type=int, default=500000)
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--class-mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--preview", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "://" not in args.teacher:
        args.teacher = os.path.abspath(os.path.expandvars(os.path.expanduser(args.teacher)))
    outdir = Path(os.path.expandvars(os.path.expanduser(args.outdir))).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required for practical teacher cache generation.")

    outdir.mkdir(parents=True, exist_ok=True)
    existing_files = [path.name for path in outdir.iterdir() if path.name not in {"metadata.json", "metadata.json.tmp"}]
    if existing_files and not args.resume:
        raise RuntimeError(f"Output directory is not empty: {outdir}. Use --resume to continue.")

    num_generated, shard_id, old_metadata = read_progress(outdir) if args.resume else (0, 0, None)
    if num_generated > args.num:
        raise RuntimeError(f"Cache already contains {num_generated} samples, more than requested {args.num}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f'Loading teacher from "{args.teacher}"...')
    teacher = load_edm(args.teacher, device).eval().requires_grad_(False)
    label_dim = int(teacher.label_dim)
    if label_dim == 0 and args.per_class:
        raise RuntimeError("--per-class requires a conditional teacher.")
    if label_dim != 0 and args.per_class:
        args.num = int(args.per_class) * label_dim

    dtype = dtype_from_name(args.dtype)
    class_counts = [0 for _ in range(label_dim)]
    if old_metadata is not None and old_metadata.get("class_counts") is not None:
        class_counts = [int(x) for x in old_metadata["class_counts"]]

    metadata = {
        "format": "edm-teacher-cache-v1",
        "teacher": args.teacher,
        "num_target": int(args.num),
        "per_class": int(args.per_class),
        "class_mode": args.class_mode if label_dim else "none",
        "num_generated": int(num_generated),
        "next_shard_id": int(shard_id),
        "seed": int(args.seed),
        "sampler": {
            "name": "edm",
            "steps": int(args.steps),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "rho": float(args.rho),
        },
        "storage_dtype": args.dtype,
        "image_shape": [int(teacher.img_channels), int(teacher.img_resolution), int(teacher.img_resolution)],
        "label_dim": label_dim,
        "class_counts": class_counts,
    }
    if old_metadata is not None and "created_time" in old_metadata:
        metadata["created_time"] = old_metadata["created_time"]
    write_metadata(outdir, metadata)

    print(f'Caching {args.num} samples to "{outdir}"...')
    shard_latents = []
    shard_images = []
    shard_seeds = []
    shard_labels = []
    shard_start = num_generated
    preview_images = []
    progress = tqdm(total=args.num, initial=num_generated, unit="img")

    while num_generated < args.num:
        shard_count = sum(x.shape[0] for x in shard_images)
        shard_remaining = args.shard_size - shard_count
        batch_size = min(args.batch, args.num - num_generated, shard_remaining)
        seeds = list(range(args.seed + num_generated, args.seed + num_generated + batch_size))

        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn(
            [batch_size, teacher.img_channels, teacher.img_resolution, teacher.img_resolution],
            device=device,
        )
        class_indices = make_class_indices(num_generated, batch_size, label_dim, args.class_mode, rnd, device)
        class_labels = encode_labels(class_indices, label_dim)

        with torch.no_grad():
            images = edm_sampler(
                teacher,
                latents,
                steps=args.steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                class_labels=class_labels,
                randn_like=rnd.randn_like,
            ).to(torch.float32)

        if args.preview and len(preview_images) < args.preview:
            want = args.preview - len(preview_images)
            preview_images.extend(images[:want].detach().cpu())

        shard_latents.append(latents.detach().to(dtype).cpu())
        shard_images.append(images.detach().to(dtype).cpu())
        shard_seeds.extend(seeds)
        if class_indices is not None:
            class_indices_cpu = class_indices.detach().cpu().to(torch.int64)
            shard_labels.append(class_indices_cpu)
            bincount = torch.bincount(class_indices_cpu, minlength=label_dim).tolist()
            class_counts = [int(a) + int(b) for a, b in zip(class_counts, bincount)]

        num_generated += batch_size
        progress.update(batch_size)

        if sum(x.shape[0] for x in shard_images) == args.shard_size or num_generated == args.num:
            labels_cpu = torch.cat(shard_labels, dim=0) if shard_labels else None
            shard_path = save_shard(
                outdir=outdir,
                shard_id=shard_id,
                start_index=shard_start,
                latents=torch.cat(shard_latents, dim=0),
                images=torch.cat(shard_images, dim=0),
                seeds=torch.as_tensor(shard_seeds, dtype=torch.int64),
                labels=labels_cpu,
            )
            shard_id += 1
            shard_start = num_generated
            shard_latents = []
            shard_images = []
            shard_seeds = []
            shard_labels = []

            metadata["num_generated"] = int(num_generated)
            metadata["next_shard_id"] = int(shard_id)
            metadata["class_counts"] = class_counts
            write_metadata(outdir, metadata)
            print(f"\nSaved shard: {shard_path}", flush=True)

    progress.close()

    if args.preview and preview_images:
        preview_path = outdir / "preview.png"
        save_image_grid(torch.stack(preview_images).to(torch.float32), preview_path)
        print(f"Saved preview: {preview_path}")

    print("Done.")


if __name__ == "__main__":
    main()
