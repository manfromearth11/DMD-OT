from typing import Optional

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm
from tqdm import tqdm

from dmd.modeling_utils import encode_labels, get_fixed_generator_sigma


def _import_edm_dnnlib():
    try:
        import dnnlib  # type: ignore

        return dnnlib
    except ImportError:
        edm_dir = Path.home() / "edm"
        if edm_dir.exists():
            sys.path.insert(0, str(edm_dir))
        import dnnlib  # type: ignore

        return dnnlib


class FID:
    def __init__(self, data_loader, device="cuda", ref_path: Optional[str] = None) -> None:
        self.dataloader = data_loader
        self.device = device
        self.ref_path = ref_path
        self.ref_mu = None
        self.ref_sigma = None
        if ref_path is not None:
            with np.load(ref_path) as ref:
                self.ref_mu = ref["mu"]
                self.ref_sigma = ref["sigma"]
        dnnlib = _import_edm_dnnlib()
        detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
        with dnnlib.util.open_url(detector_url, verbose=False) as f:
            self.inception_model = pickle.load(f).to(self.device)
        self.inception_model.eval()
        self.detector_kwargs = dict(return_features=True)

    def get_inception_features(self, image_batch):
        if image_batch.dtype != torch.uint8:
            image_batch = (image_batch.clamp(0, 1) * 255).round().to(torch.uint8)
        inception_output = self.inception_model(image_batch.to(self.device), **self.detector_kwargs)
        return inception_output.data.cpu().numpy()

    @torch.no_grad()
    def __call__(self, generator):
        was_training = generator.training
        generator.eval()
        inception_feature_batches_fake = []
        try:
            for image, class_idx in tqdm(
                self.dataloader, desc=f"FID - Fake Data Feature Extraction", total=len(self.dataloader)
            ):
                z = torch.randn_like(image, device=self.device)
                # Scale Z ~ N(0,1) (z and z_ref) w/ 80.0 to match the sigma_t at T_n
                g_sigma = get_fixed_generator_sigma(z.shape[0], device=self.device)
                z = z * g_sigma[0, 0]  # scalar product
                class_idx = class_idx.to(self.device, non_blocking=True)
                class_ids = encode_labels(class_idx, generator.label_dim)
                fake_image_batch = generator(z, g_sigma, class_labels=class_ids)
                fake_image_batch = (fake_image_batch + 1) / 2.0  # Normalizing the pixel values from [-1,1] to [0,1]
                inception_feature_batch = self.get_inception_features(fake_image_batch)
                inception_feature_batches_fake.append(inception_feature_batch)
            inception_features_fake = np.concatenate(inception_feature_batches_fake)

            mu_fake, sigma_fake = inception_features_fake.mean(axis=0), cov(inception_features_fake, rowvar=False)
            if self.ref_mu is not None and self.ref_sigma is not None:
                mu_real, sigma_real = self.ref_mu, self.ref_sigma
            else:
                inception_feature_batches_real = []
                for image_batch, _ in tqdm(
                    self.dataloader, desc=f"FID - Real Data Feature Extraction", total=len(self.dataloader)
                ):
                    image_batch = image_batch.to(self.device, non_blocking=True).to(torch.float32)
                    inception_feature_batch = self.get_inception_features(image_batch)
                    inception_feature_batches_real.append(inception_feature_batch)
                inception_features_real = np.concatenate(inception_feature_batches_real)
                mu_real, sigma_real = inception_features_real.mean(axis=0), cov(inception_features_real, rowvar=False)
        finally:
            if was_training:
                generator.train()
        ssdiff = np.sum((mu_fake - mu_real) ** 2.0)
        cov_mean = sqrtm(sigma_fake.dot(sigma_real))
        if iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        frechet_inception_distance = ssdiff + trace(sigma_fake + sigma_real - 2.0 * cov_mean)
        return frechet_inception_distance
