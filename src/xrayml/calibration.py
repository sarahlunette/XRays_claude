"""Temperature scaling (revision plan 1.2). Focal loss and oversampling distort probabilities;
recalibrate on the validation split *before* choosing operating points if the product shows
a probability. Ranking (AUC) is unchanged, so a threshold chosen on raw scores maps 1:1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(labels, dtype=torch.float64).reshape(-1)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits) / temperature))
