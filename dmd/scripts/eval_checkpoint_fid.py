import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dmd.fid import FID
from dmd.modeling_utils import load_dmd_model


class BalancedLabelDataset(Dataset):
    def __init__(self, size: int = 50000, num_classes: int = 10):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.zeros(3, 32, 32)
        class_idx = index % self.num_classes
        return image, class_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"eval_checkpoint={checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    print(f"checkpoint_global_step={ckpt.get('global_step')} epoch={ckpt.get('epoch')}")

    generator = load_dmd_model(str(checkpoint), device=device, for_training=False)
    generator.eval().requires_grad_(False)
    loader = DataLoader(
        BalancedLabelDataset(size=args.num),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    fid = FID(loader, device=device, ref_path=args.ref)
    value = float(fid(generator))
    print(f"CORRECTED_FID_{args.num}={value}")

    out = Path(args.out) if args.out else checkpoint.parent / f"corrected_fid_{args.num}.json"
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "global_step": ckpt.get("global_step"),
                "epoch": ckpt.get("epoch"),
                "seed": args.seed,
                f"fid_{args.num}": value,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
