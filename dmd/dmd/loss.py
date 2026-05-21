import math

import torch
import torch.nn.functional as F
from piq import LPIPS
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torchvision.transforms import Resize

from dmd.modeling_utils import forward_diffusion


class DistributionMatchingLoss(_Loss):
    """
    Loss function for DMD (Algorithm 2) proposed in
    "One-step Diffusion with Distribution Matching Distillation".
    """

    def __init__(self, timesteps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timesteps = timesteps

    def forward(
        self, mu_real: Module, mu_fake: Module, x: torch.Tensor, class_ids: torch.Tensor = None
    ) -> torch.Tensor:
        b, c, w, h = x.shape

        # In practice T_min, T_max choices follows DreamFusion as follows
        T_min, T_max = int(0.02 * self.timesteps), int(0.98 * self.timesteps)
        timestep = torch.randint(T_min, T_max, [b])
        noisy_x, sigma_t = forward_diffusion(x, timestep)

        with torch.no_grad():
            pred_fake_image = mu_fake(noisy_x, sigma_t, class_labels=class_ids)
            pred_real_image = mu_real(noisy_x, sigma_t, class_labels=class_ids)

        weighting_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3], keepdim=True)  # Eqn. 8
        grad = (pred_fake_image - pred_real_image) / weighting_factor
        diff = (x - grad).detach()  # stop-gradient
        return 0.5 * F.mse_loss(x, diff, reduction=self.reduction)


class GeneratorLoss(_Loss):
    """
    Combined loss for the generator model. See § 3.4 (Final Objective).
    D_KL + lambda_reg * L_reg
    """

    def __init__(
        self,
        timesteps: int = 1000,
        lambda_reg: float = 0.25,
        lambda_k: float = 1.0,
        use_ot_reg: bool = False,
        ot_eps: float = 0.05,
        ot_iters: int = 30,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dmd_loss = DistributionMatchingLoss(timesteps)
        self.lpips_vector = LPIPS(reduction="none")
        self.resize = Resize(224)
        self.lambda_reg = lambda_reg
        self.lambda_k = lambda_k
        self.use_ot_reg = use_ot_reg
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters
        self.last_metrics = {}

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        """Balanced entropic OT with uniform marginals."""
        n, m = cost.shape
        log_mu = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
        log_nu = cost.new_full((m,), -torch.log(cost.new_tensor(float(m))))
        log_k = -cost / self.ot_eps
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(self.ot_iters):
            u = log_mu - torch.logsumexp(log_k + v[None, :], dim=1)
            v = log_nu - torch.logsumexp(log_k + u[:, None], dim=0)

        return torch.exp(log_k + u[:, None] + v[None, :])

    def _hard_anchor(self, P: torch.Tensor, x: torch.Tensor):
        """Row-wise argmax teacher-anchor selection."""
        anchor_idx = P.argmax(dim=1)
        anchor_mass = P.gather(1, anchor_idx[:, None]).squeeze(1)
        return x[anchor_idx], anchor_mass, anchor_idx

    def _class_mask(self, x_class_ids: torch.Tensor, y_class_ids: torch.Tensor = None) -> torch.Tensor:
        if x_class_ids is None:
            return None
        if y_class_ids is None:
            y_class_ids = x_class_ids
        if x_class_ids.ndim > 1:
            x_class_ids = x_class_ids.argmax(dim=1)
        if y_class_ids.ndim > 1:
            y_class_ids = y_class_ids.argmax(dim=1)
        return x_class_ids[:, None] == y_class_ids[None, :]

    def _pairwise_lpips_cost(self, x_ref: torch.Tensor, y_ref: torch.Tensor, class_ids: torch.Tensor):
        n = x_ref.shape[0]
        m = y_ref.shape[0]
        if class_ids is not None and class_ids.shape[0] != n:
            raise ValueError("class_ids must have one entry per student sample.")
        if class_ids is not None and m != n:
            raise ValueError("Class-aware OT with different student/teacher batch sizes needs teacher class ids.")
        class_mask = self._class_mask(class_ids)

        with torch.no_grad():
            self.lpips_vector.model.to(x_ref)
            x_features = self.lpips_vector.get_features(x_ref)
            y_features = self.lpips_vector.get_features(y_ref)
            cost = x_ref.new_zeros((n, m), dtype=torch.float32)
            for x_feature, y_feature, weight in zip(x_features, y_features, self.lpips_vector.weights):
                n_x, channels, height, width = x_feature.shape
                n_y = y_feature.shape[0]
                spatial_size = height * width
                x_flat = x_feature.float().flatten(2)
                y_flat = y_feature.float().flatten(2)
                scale = weight.to(device=x_flat.device, dtype=x_flat.dtype).flatten().sqrt()
                scale = scale.view(1, channels, 1) / math.sqrt(spatial_size)
                x_weighted = (x_flat * scale).reshape(n_x, -1)
                y_weighted = (y_flat * scale).reshape(n_y, -1)
                cost = cost + x_weighted.square().sum(dim=1)[:, None]
                cost = cost + y_weighted.square().sum(dim=1)[None, :]
                cost = cost - 2.0 * (x_weighted @ y_weighted.t())
            cost = cost.clamp_min(0)
            if class_mask is not None:
                cost = cost.masked_fill(~class_mask, float("inf"))

            valid_cost = cost[torch.isfinite(cost)]
            valid_mean = valid_cost.mean()
            info = {
                "ot/cost_pair_mean": valid_mean,
                "ot/cost_pair_std": valid_cost.std(unbiased=False),
                "ot/cost_pair_min": valid_cost.min(),
                "ot/cost_pair_max": valid_cost.max(),
            }
            if class_mask is not None:
                info["ot/valid_pair_frac"] = class_mask.float().mean()
            cost = cost / (valid_mean + 1e-8)
            cost = cost.masked_fill(~torch.isfinite(cost), 1e6)

        return cost, info

    def _lpips_per_sample(self, x_ref: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        return self.lpips_vector(x_ref, y_ref).reshape(x_ref.shape[0], -1).mean(dim=1)

    def forward(
        self,
        mu_real: Module,
        mu_fake: Module,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        y_ref: torch.Tensor,
        class_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Optional OT re-matching followed by DMD's LPIPS regression loss."""
        self.last_metrics = {}
        x_ref = (x_ref + 1) / 2.0
        y_ref = (y_ref + 1) / 2.0
        if self.use_ot_reg:
            x_loss = self.resize(x_ref)
            y_loss_all = self.resize(y_ref)
            cost, ot_metrics = self._pairwise_lpips_cost(x_loss, y_loss_all, class_ids)
            P = self._sinkhorn(cost)
            _, anchor_mass, anchor_idx = self._hard_anchor(P, y_ref)
            y_loss = y_loss_all[anchor_idx]
            loss_reg_per_sample = self._lpips_per_sample(x_loss, y_loss)
            ot_weight = anchor_mass * x_ref.shape[0]
            loss_reg = (loss_reg_per_sample * ot_weight).mean()

            with torch.no_grad():
                n, m = P.shape
                row_target = 1.0 / n
                col_target = 1.0 / m
                row_error = (P.sum(dim=1) - row_target).abs()
                col_error = (P.sum(dim=0) - col_target).abs()
                selected_cost = cost[torch.arange(n, device=cost.device), anchor_idx]
                selected_y = y_ref[anchor_idx]
                pixel_selected_mse = (x_ref - selected_y).square().flatten(1).mean(dim=1)
                ot_metrics.update(
                    {
                        "ot/transport_sum": P.sum(),
                        "ot/row_error_mean": row_error.mean(),
                        "ot/row_error_max": row_error.max(),
                        "ot/col_error_mean": col_error.mean(),
                        "ot/col_error_max": col_error.max(),
                        "ot/selected_cost_mean": selected_cost.mean(),
                        "ot/selected_cost_min": selected_cost.min(),
                        "ot/selected_cost_max": selected_cost.max(),
                        "ot/selected_mass_mean": anchor_mass.mean(),
                        "ot/selected_mass_min": anchor_mass.min(),
                        "ot/selected_mass_max": anchor_mass.max(),
                        "ot/weight_mean": ot_weight.mean(),
                        "ot/weight_min": ot_weight.min(),
                        "ot/weight_max": ot_weight.max(),
                        "ot/unique_teacher_anchors": anchor_idx.unique().numel(),
                        "ot/diag_match_frac": (anchor_idx == torch.arange(n, device=anchor_idx.device)).float().mean(),
                        "ot/pixel_selected_mse": pixel_selected_mse.mean(),
                        "lpips/selected_mean": loss_reg_per_sample.detach().mean(),
                        "lpips/selected_min": loss_reg_per_sample.detach().min(),
                        "lpips/selected_max": loss_reg_per_sample.detach().max(),
                    }
                )
                if class_ids is not None:
                    class_idx = class_ids.argmax(dim=1) if class_ids.ndim > 1 else class_ids
                    ot_metrics["ot/class_match_frac"] = (class_idx == class_idx[anchor_idx]).float().mean()
                self.last_metrics.update(ot_metrics)
        else:
            pixel_paired_mse = (x_ref - y_ref).square().flatten(1).mean()
            x_ref = self.resize(x_ref)
            y_ref = self.resize(y_ref)
            loss_reg_per_sample = self._lpips_per_sample(x_ref, y_ref)
            loss_reg = loss_reg_per_sample.mean()
            self.last_metrics.update(
                {
                    "lpips/paired_mean": loss_reg_per_sample.detach().mean(),
                    "lpips/paired_min": loss_reg_per_sample.detach().min(),
                    "lpips/paired_max": loss_reg_per_sample.detach().max(),
                    "pixel/paired_mse": pixel_paired_mse.detach(),
                }
            )
        if self.lambda_k:
            loss_kl = self.dmd_loss(mu_real, mu_fake, x, class_ids)
        else:
            loss_kl = x_ref.new_zeros(())
        loss = self.lambda_k * loss_kl + self.lambda_reg * loss_reg
        self.last_metrics.update(
            {
                "loss/reg": loss_reg.detach(),
                "loss/reg_weighted": (self.lambda_reg * loss_reg).detach(),
                "loss/kl": loss_kl.detach(),
                "loss/kl_weighted": (self.lambda_k * loss_kl).detach(),
                "loss/total": loss.detach(),
            }
        )
        return loss


class DenoisingLoss(_Loss):
    """
    Loss function for DMD (Equation 6 / Algorithm 3) proposed in
    "One-step Diffusion with Distribution Matching Distillation".
    """

    def forward(
        self, mu_fake: Module, x: torch.Tensor, t: torch.Tensor, class_ids: torch.Tensor = None
    ) -> torch.Tensor:
        x_t, sigma_t = forward_diffusion(x.detach(), t)  # stop grad
        pred_fake_image = mu_fake(x_t, sigma_t, class_labels=class_ids)
        # Algorithm SNR + 1 / sigma_data^2 for EDM (sigma_data = 0.5)
        weight = 1 / sigma_t**2 + 1 / mu_fake.sigma_data**2
        return torch.mean(weight[:, None, None, None] * (pred_fake_image - x.detach()) ** 2)
