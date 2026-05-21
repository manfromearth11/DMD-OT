"""
Main training entry point for DMD training.
Part of this file is taken and adopted from devrimcavusoglu/std. See
the original file below
https://github.com/devrimcavusoglu/std/blob/main/std/main.py
"""

import datetime
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import h5py
import torch
import torchvision.transforms as transforms
from neptune import Run
from torch.backends import cudnn
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import CIFAR10

from dmd import NEPTUNE_CONFIG_PATH, PROJECT_ROOT
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.fid import FID
from dmd.loss import DenoisingLoss, GeneratorLoss
from dmd.modeling_utils import load_edm, load_dmd_model
from dmd.training.training_loop import train_one_epoch
from dmd.utils.common import create_experiment, seed_everything
from dmd.utils.logging import CheckpointHandler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from fvcore.nn import FlopCountAnalysis, flop_count, flop_count_table, parameter_count
    from utils import sfc_flop_jit

    has_fvcore = True
except ImportError:
    has_fvcore = False


class ClassBatchSampler(Sampler):
    """Yield batches whose samples all share one class label."""

    def __init__(self, h5_dataset_path: Path, batch_size: int, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        with h5py.File(h5_dataset_path, "r") as file:
            if "labels" in file:
                labels = file["labels"][:].astype(int).tolist()
            else:
                data = file["data"]
                labels = [int(data[str(idx)].attrs["class_idx"]) for idx in range(len(data))]

        self.class_indices = []
        for class_idx in sorted(set(labels)):
            indices = torch.tensor([idx for idx, label in enumerate(labels) if label == class_idx], dtype=torch.long)
            if indices.numel() < batch_size:
                raise ValueError(f"Class {class_idx} has only {indices.numel()} samples, less than batch_size={batch_size}.")
            self.class_indices.append(indices)
        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = [indices[torch.randperm(indices.numel(), generator=generator)] for indices in self.class_indices]
        positions = [0 for _ in pools]

        for batch_idx in range(self.num_batches):
            class_idx = int(torch.randint(len(pools), (1,), generator=generator).item())
            pool = pools[class_idx]
            pos = positions[class_idx]
            if pos + self.batch_size > pool.numel():
                pool = self.class_indices[class_idx]
                pool = pool[torch.randperm(pool.numel(), generator=generator)]
                pools[class_idx] = pool
                pos = 0
            batch = pool[pos : pos + self.batch_size]
            positions[class_idx] = pos + self.batch_size
            yield batch.tolist()

        self.epoch += 1

    def __len__(self):
        return self.num_batches


class BalancedLabelDataset(Dataset):
    """Dummy CIFAR-shaped eval dataset with balanced class labels."""

    def __init__(self, size: int = 10000, num_classes: int = 10):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.zeros(3, 32, 32)
        class_idx = index % self.num_classes
        return image, class_idx


def train(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    max_norm: float = 10,
    amp_autocast=None,
    neptune_run: Optional[Run] = None,
    wandb_run=None,
    cudnn_benchmark: bool = True,
    is_distributed: bool = False,
    print_freq: int = 10,
    im_save_freq: int = 300,
    max_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    skip_fid: bool = False,
    fid_ref_path: Optional[str] = None,
    train_diffuser: bool = True,
    checkpoint_handler: Optional[CheckpointHandler] = None,
):
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    if cudnn_benchmark:
        cudnn.benchmark = True

    fid = None if skip_fid else FID(data_loader_test, device=device, ref_path=fid_ref_path)

    global_step = 0
    next_eval_step = eval_every_steps
    for epoch in range(epochs):
        if max_steps is not None and global_step >= max_steps:
            break
        if is_distributed:
            data_loader_train.sampler.set_epoch(epoch)

        stop_step = max_steps
        if next_eval_step is not None and global_step < next_eval_step:
            stop_step = next_eval_step if stop_step is None else min(stop_step, next_eval_step)

        train_stats, steps_done = train_one_epoch(
            generator,
            mu_fake,
            mu_real,
            data_loader_train,
            loss_g,
            loss_d,
            optimizer_g,
            optimizer_d,
            device,
            epoch,
            max_norm=max_norm,
            amp_autocast=amp_autocast,
            neptune_run=neptune_run,
            wandb_run=wandb_run,
            start_step=global_step,
            max_steps=stop_step,
            output_dir=checkpoint_handler.checkpoint_dir,
            print_freq=print_freq,
            im_save_freq=im_save_freq,
            train_diffuser=train_diffuser,
        )
        global_step += steps_done
        if steps_done == 0:
            break

        # lr_scheduler.step(epoch)
        model_dict = {
            "model_g": generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            # "model_ema": get_state_dict(model_ema),
            # "args": args,
        }
        if mu_fake is not None and optimizer_d is not None:
            model_dict["model_d"] = mu_fake.state_dict()
            model_dict["optimizer_d"] = optimizer_d.state_dict()
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            # **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "global_step": global_step,
        }
        do_eval = max_steps is None
        if max_steps is not None:
            do_eval = global_step >= max_steps
            if next_eval_step is not None and global_step >= next_eval_step:
                do_eval = True

        if do_eval and fid is not None:
            test_fid = fid(generator)
            if neptune_run is not None:
                neptune_run["test/fid"].append(test_fid)
            if wandb_run is not None:
                wandb_run.log({"test/fid": test_fid}, step=global_step)
            print(f"Test FID: {test_fid}")
            log_stats["test_fid"] = test_fid
            checkpoint_handler.save(model_dict, log_stats, test_fid, epoch)
            while next_eval_step is not None and global_step >= next_eval_step:
                next_eval_step += eval_every_steps
        elif do_eval:
            checkpoint_handler.save(model_dict, log_stats, float("inf"), epoch)
            while next_eval_step is not None and global_step >= next_eval_step:
                next_eval_step += eval_every_steps

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def run(
    model_path: str,
    data_path: str,
    epochs: int,
    teacher_model_path: str = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
    output_dir: str = None,
    batch_size: int = 56,
    eval_batch_size: int = 128,
    num_workers: int = 10,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    dmd_loss_timesteps: int = 1000,
    dmd_loss_lambda: float = 0.25,
    lambda_k: float = 1.0,
    use_ot_reg: bool = False,
    ot_eps: float = 0.05,
    ot_iters: int = 30,
    class_batch: bool = False,
    ot_weight: Optional[str] = None,
    device: str = None,
    log_neptune: bool = False,
    neptune_run_id: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    cudnn_benchmark: bool = True,
    amp_autocast: Optional = None,
    max_norm: float = 10.0,
    print_steps: int = 10,
    im_save_steps: int = 300,
    model_save_steps: int = 1600,
    max_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    skip_fid: bool = False,
    fid_ref_path: Optional[str] = None,
    fid_num_samples: int = 10000,
    train_diffuser: Optional[bool] = None,
    cache_data: bool = False,
    log_wandb: bool = False,
    wandb_project: str = "cifar10-dmd-sanity-test",
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Starts the training phase.

    Args:
        model_path (str): Path to the model.
        data_path (str): Path of the h5 dataset file.
        epochs (int): Number of epochs to train.
        teacher_model_path (str): Path to the frozen EDM teacher.
        output_dir (str): Path to the output directory to save the model.
        batch_size (int): Batch size used in training process. [default: 56]
        eval_batch_size (int): Batch size used in evaluation process. [default: 128]
        num_workers (int): Number of workers for data loader. [default: 10]
        lr (float): Learning rate. [default: 5e-5]
        weight_decay (float): Weight decay for optimizer. [default: 0.01]
        betas (tuple(float, float)): Beta parameters for the optimizer. [default: (0.9, 0.999)]
        dmd_loss_timesteps (int): Number of timesteps to use for DMD loss. [default: 1000]
        dmd_loss_lambda (float): Lambda for the DMD loss. [default: 0.25]
        lambda_k (float): Lambda for DMD distribution matching KL loss. [default: 1.0]
        use_ot_reg (bool): Whether to replace paired LPIPS regression with OT-rematched LPIPS regression.
        ot_eps (float): Entropic Sinkhorn epsilon when `use_ot_reg` is enabled.
        ot_iters (int): Number of Sinkhorn iterations when `use_ot_reg` is enabled.
        class_batch (bool): Whether every training batch should contain samples from one class only.
        ot_weight (Optional[str]): Deprecated compatibility argument. OT always uses N * P[i, j*].
        device (Optional(str)): Device to run the models on. [default: None]
        log_neptune (bool): Whether to log metrics to neptune. [default: False]
        neptune_run_id (Optional(str)): Neptune run id. [default: None]
        resume_from_checkpoint (bool): Whether to resume from a checkpoint. [default: False]
        cudnn_benchmark (bool): Whether to use CUDNN benchmark. [default: True]
        amp_autocast (Optional): Whether to use AMP autocast. [default: None]
        max_norm (Optional[float]): Maximum norm of the gradients. [default: 10.0]
        print_steps (int): Print frequency for metric report. [default: 10]
        im_save_steps (int): Frequency to save image grids. [default: 300]
        model_save_steps (int): Frequency to save the model checkpoint. [default: 1600]
        max_steps (Optional[int]): Stop training after this many optimizer steps. [default: None]
        eval_every_steps (Optional[int]): FID/checkpoint frequency in optimizer steps when max_steps is set.
        skip_fid (bool): Skip FID evaluation and save checkpoints without a validation metric.
        fid_ref_path (Optional[str]): Optional precomputed FID reference stats with `mu` and `sigma`.
        fid_num_samples (int): Number of generated samples for FID when `fid_ref_path` is set.
        train_diffuser (Optional[bool]): Whether to train the fake denoiser used by the KL term.
        cache_data (bool): Load array-format HDF5 training data into CPU RAM at startup.
        log_wandb (bool): Whether to log metrics to Weights & Biases. [default: False]
        wandb_project (str): W&B project name.
        wandb_name (Optional[str]): W&B run name.
        wandb_entity (Optional[str]): W&B entity.
        seed (Optional[int]): Random seed to seed all. [default: 42]
    """
    # sanity check
    if not resume_from_checkpoint and output_dir is None:
        raise ValueError("`output_dir` must be given when `resume_from_checkpoint` is `False`.")
    if resume_from_checkpoint and output_dir is None:
        warnings.warn("`output_dir` is set to `model_path` when `resume_from_checkpoint` is `True`.")
        output_dir = Path(model_path).parent
    output_dir = Path(output_dir)
    seed_everything(seed)
    # Prepare dataloader
    data_path = Path(data_path).resolve()
    training_dataset = CIFARPairs(data_path, cache_in_memory=cache_data)
    if class_batch:
        train_sampler = ClassBatchSampler(data_path, batch_size=batch_size, seed=seed)
        train_loader = DataLoader(training_dataset, batch_sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = None
    if not skip_fid:
        if fid_ref_path is not None:
            test_dataset = BalancedLabelDataset(size=fid_num_samples)
        else:
            test_dataset = CIFAR10(
                root=(PROJECT_ROOT / "data").as_posix(), train=False, download=True, transform=transforms.ToTensor()
            )
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if train_diffuser is None:
        train_diffuser = lambda_k != 0
    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": betas}
    needs_kl_models = bool(lambda_k) or bool(train_diffuser)
    mu_real = load_edm(model_path=teacher_model_path, device=device) if needs_kl_models else None
    if resume_from_checkpoint:
        generator, generator_optimizer, mu_fake, diffuser_optimizer = load_dmd_model(model_path=model_path, device=device, for_training=True, optimizer_kwargs=optimizer_kwargs)
    else:
        generator = load_edm(model_path=model_path, device=device)
        mu_fake = load_edm(model_path=model_path, device=device) if needs_kl_models else None

        # Create optimizers
        generator_optimizer = AdamW(params=generator.parameters(), **optimizer_kwargs)
        diffuser_optimizer = AdamW(params=mu_fake.parameters(), **optimizer_kwargs) if mu_fake is not None else None

    # Create losses
    generator_loss = GeneratorLoss(
        timesteps=dmd_loss_timesteps,
        lambda_reg=dmd_loss_lambda,
        lambda_k=lambda_k,
        use_ot_reg=use_ot_reg,
        ot_eps=ot_eps,
        ot_iters=ot_iters,
    ).to(device)
    diffusion_loss = DenoisingLoss().to(device)

    checkpoint_handler = CheckpointHandler(
        checkpoint_dir=output_dir, lower_is_better=True
    )  # hardcoded lower_is_better for experimentation

    neptune_run = None
    if log_neptune:
        # create neptune run
        neptune_run = create_experiment(NEPTUNE_CONFIG_PATH, run_id=neptune_run_id)
        neptune_run["training_args"] = {
            "model_path": model_path,
            "teacher_model_path": teacher_model_path,
            "data_path": data_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "dmd_loss_timesteps": dmd_loss_timesteps,
            "dmd_loss_lambda": float(dmd_loss_lambda),
            "lambda_k": float(lambda_k),
            "use_ot_reg": bool(use_ot_reg),
            "ot_eps": float(ot_eps),
            "ot_iters": int(ot_iters),
            "class_batch": bool(class_batch),
            "device": str(device),
            "cudnn_benchmark": cudnn_benchmark,
            "amp_autocast": amp_autocast,
            "max_norm": max_norm,
            "print_steps": print_steps,
            "im_save_steps": im_save_steps,
            "model_save_steps": model_save_steps,
            "max_steps": max_steps,
            "eval_every_steps": eval_every_steps,
            "skip_fid": bool(skip_fid),
            "fid_ref_path": fid_ref_path,
            "fid_num_samples": int(fid_num_samples),
            "train_diffuser": bool(train_diffuser),
            "cache_data": bool(cache_data),
        }

    wandb_run = None
    if log_wandb:
        import wandb

        wandb_config = {
            "model_path": model_path,
            "teacher_model_path": teacher_model_path,
            "data_path": data_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "dmd_loss_timesteps": dmd_loss_timesteps,
            "dmd_loss_lambda": float(dmd_loss_lambda),
            "lambda_k": float(lambda_k),
            "use_ot_reg": bool(use_ot_reg),
            "ot_eps": float(ot_eps),
            "ot_iters": int(ot_iters),
            "class_batch": bool(class_batch),
            "device": str(device),
            "max_norm": max_norm,
            "print_steps": print_steps,
            "im_save_steps": im_save_steps,
            "model_save_steps": model_save_steps,
            "max_steps": max_steps,
            "eval_every_steps": eval_every_steps,
            "skip_fid": bool(skip_fid),
            "fid_ref_path": fid_ref_path,
            "fid_num_samples": int(fid_num_samples),
            "train_diffuser": bool(train_diffuser),
            "cache_data": bool(cache_data),
            "seed": seed,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=wandb_config,
        )

    # start training
    train(
        generator=generator,
        mu_real=mu_real,
        mu_fake=mu_fake,
        data_loader_train=train_loader,
        data_loader_test=test_loader,
        device=device,
        loss_g=generator_loss,
        loss_d=diffusion_loss,
        optimizer_g=generator_optimizer,
        optimizer_d=diffuser_optimizer,
        epochs=epochs,
        neptune_run=neptune_run,
        wandb_run=wandb_run,
        cudnn_benchmark=cudnn_benchmark,
        amp_autocast=amp_autocast,
        max_norm=max_norm,
        print_freq=print_steps,
        im_save_freq=im_save_steps,
        max_steps=max_steps,
        eval_every_steps=eval_every_steps,
        skip_fid=skip_fid,
        fid_ref_path=fid_ref_path,
        train_diffuser=train_diffuser,
        checkpoint_handler=checkpoint_handler,
    )

    if neptune_run:
        neptune_run.stop()
    if wandb_run:
        wandb_run.finish()
