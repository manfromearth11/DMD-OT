"""
Main training loop.
Part of this file taken and adopted from devrimcavusoglu/std repository. See
the original file below
https://github.com/devrimcavusoglu/std/blob/main/std/engine.py
"""
import math
import sys
from contextlib import suppress
from pathlib import Path
from typing import List, Optional

import torch
from neptune import Run
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dmd.modeling_utils import encode_labels, forward_diffusion, get_fixed_generator_sigma
from dmd.utils.array import torch_to_pillow
from dmd.utils.common import image_grid
from dmd.utils.logging import MetricLogger


def _save_intermediate_images(
    output_dir: str,
    all_images: List[torch.Tensor],
    prefix: str,
):
    pims = []
    for images in all_images:
        pims.extend(torch_to_pillow(images))
    grid = image_grid(pims, rows=len(all_images), cols=all_images[0].shape[0])
    images_output_dir = Path(output_dir)
    images_output_dir.mkdir(exist_ok=True, parents=True)
    image_path = images_output_dir / f"{prefix}.png"
    grid.save(image_path)
    return grid


def update_parameters(model, loss, optimizer, max_norm):
    optimizer.zero_grad()

    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    loss.backward(create_graph=is_second_order)
    if max_norm is not None:
        clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()


def train_one_epoch(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader_train: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    output_dir: str = None,
    max_norm: float = 10,
    amp_autocast=None,
    neptune_run: Optional[Run] = None,
    wandb_run=None,
    start_step: int = 0,
    max_steps: Optional[int] = None,
    print_freq: int = 10,
    im_save_freq: int = 300,
    train_diffuser: bool = True,
):
    amp_autocast = amp_autocast or suppress
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Set G and mu_fake to train mode, mu_real should be frozen
    generator.requires_grad_(True).train()
    if mu_fake is not None:
        mu_fake.requires_grad_(train_diffuser).train(train_diffuser)
    if mu_real is not None:
        mu_real.requires_grad_(False).eval()

    metric_logger = MetricLogger(delimiter="  ", neptune_run=neptune_run)
    # metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    i = 0
    steps_done = 0
    for pairs in metric_logger.log_every(data_loader_train, print_freq, header):
        if max_steps is not None and start_step + steps_done >= max_steps:
            break
        global_step = start_step + steps_done + 1
        save_images = im_save_freq > 0 and global_step % im_save_freq == 0
        need_free_sample = train_diffuser or getattr(loss_g, "lambda_k", 1.0) != 0 or save_images
        y_ref = pairs["image"].to(device, non_blocking=True).to(torch.float32).clip(-1, 1)
        z_ref = pairs["latent"].to(device, non_blocking=True).to(torch.float32)
        generator_sigma = get_fixed_generator_sigma(y_ref.shape[0], device=device)
        # Scale Z ~ N(0,1) (z and z_ref) w/ sigma(T-1) to match the sigma at T-1
        z = torch.randn_like(y_ref, device=device) * generator_sigma[0, 0] if need_free_sample else None
        z_ref = z_ref * generator_sigma[0, 0]
        class_idx = pairs["class_id"].to(device, non_blocking=True)
        class_ids = encode_labels(class_idx, generator.label_dim)

        with amp_autocast():
            # Update generator
            # tanh after small experiment between (no-postprocess, tanh, clipping)
            x = generator(z, generator_sigma, class_labels=class_ids) if need_free_sample else None
            x_ref = generator(z_ref, generator_sigma, class_labels=class_ids)
            l_g = loss_g(mu_real, mu_fake, x, x_ref, y_ref, class_ids)
            if not math.isfinite(l_g.item()):
                print(f"Generator Loss is {l_g.item()}, stopping training")
                sys.exit(1)

        update_parameters(generator, l_g, optimizer_g, max_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.log_neptune("loss_g", l_g.item())

        if train_diffuser:
            with amp_autocast():
                # Update mu_fake
                t = torch.randint(1, 1000, [x.shape[0]])  # t ~ DU(1,1000) as t=0 leads 1/0^2 -> inf
                l_d = loss_d(mu_fake, x, t, class_ids)
                if not math.isfinite(l_d.item()):
                    print(f"Diffusion Loss is {l_d.item()}, stopping training")
                    sys.exit(1)

            update_parameters(mu_fake, l_d, optimizer_d, max_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            metric_logger.log_neptune("loss_d", l_d.item())
            loss_d_value = l_d.item()
        else:
            loss_d_value = 0.0
        if wandb_run is not None:
            wandb_metrics = {"train/loss_g": l_g.item(), "train/loss_d": loss_d_value}
            for key, value in getattr(loss_g, "last_metrics", {}).items():
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                    value = value.item() if value.numel() == 1 else value.float().mean().item()
                wandb_metrics[f"train/{key}"] = value
            wandb_run.log(wandb_metrics, step=global_step)

        if save_images:
            images_epoch_dir = images_dir / f"epoch_{epoch}"
            images_epoch_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                if mu_real is not None and mu_fake is not None:
                    t_preview = torch.randint(1, 1000, [x.shape[0]])
                    x_t, sigma_t = forward_diffusion(x, t_preview)
                    real_pred = mu_real(x_t, sigma_t, class_labels=class_ids)
                    fake_pred = mu_fake(x_t, sigma_t, class_labels=class_ids)
                    image_rows = [x, real_pred, fake_pred, x_ref, y_ref]
                else:
                    image_rows = [x, x_ref, y_ref]
            grid = _save_intermediate_images(images_epoch_dir, image_rows, f"iter_{i}")
            metric_logger.log_neptune(f"images", grid)

        # if model_ema is not None:
        #     model_ema.update(model)

        metric_logger.update(loss_g=l_g.item(), loss_d=loss_d_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        i += 1
        steps_done += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, steps_done
