"""Exponential Moving Average (EMA) for Mean Teacher framework.

The Teacher model's weights are updated as:
    phi_t = alpha * phi_{t-1} + (1 - alpha) * theta_t

where theta is the Student's parameters and alpha is the EMA decay rate
(typically 0.99 or 0.999). This makes the Teacher an ensemble of historical
Student snapshots, producing more stable and higher-quality pseudo-labels.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMATeacher:
    """Manages an EMA copy of a Student model (the Teacher).

    Args:
        student_model: The Student model to track.
        decay: EMA decay rate. Higher = slower update = more stable Teacher.
            0.99 → Teacher forgets Student in ~100 steps.
            0.999 → Teacher forgets Student in ~1000 steps.
    """

    def __init__(self, student_model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep copy Student as initial Teacher weights
        self.teacher = deepcopy(student_model)
        # Teacher never requires gradients
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def update(self, student_model: nn.Module):
        """Update Teacher weights via EMA after each optimizer step.

        Call this AFTER optimizer.step() in every training iteration.
        """
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), student_model.parameters()
            ):
                t_param.data.mul_(self.decay).add_(
                    s_param.data, alpha=1.0 - self.decay
                )

    def __call__(self, x):
        """Forward pass through Teacher (always in eval mode, no grad)."""
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(x)

    def state_dict(self):
        """Return Teacher model state dict for checkpointing."""
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        """Load Teacher model state dict."""
        self.teacher.load_state_dict(state_dict)

    def to(self, device):
        """Move Teacher to device."""
        self.teacher.to(device)
        return self
