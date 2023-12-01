import math

import lightning.pytorch as pl
import torch.optim as optim
from torch.optim import Optimizer

from .nano_gpt import GPT, GPTConfig


class CustomScheduler(optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler that increases linearly for n_linear_steps,
    then decays cosine annealing for n_decay_steps,
    then stays at lr_final for the remaining steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_linear_steps (int): Number of steps for linear increase.
        n_decay_steps (int): Number of steps for cosine decay.
        lr_init (float, optional): Initial learning rate. Default is 0.
        lr_max (float, optional): Maximum learning rate. Default is 1e-5.
        lr_final (float, optional): Final learning rate. Default is 1e-6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_linear_steps: int,
        n_decay_steps: int,
        lr_init: float = 0,
        lr_max: float = 1e-5,
        lr_final: float = 1e-6,
    ):
        self.n_linear_steps = n_linear_steps
        self.n_decay_steps = n_decay_steps

        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final

        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step <= self.n_linear_steps:
            lr = self.lr_init + (self.lr_max - self.lr_init) * current_step / (
                self.n_linear_steps
            )
        elif current_step <= self.n_linear_steps + self.n_decay_steps:
            lr = (
                0.5
                * math.cos(
                    (current_step - self.n_linear_steps)
                    / (self.n_decay_steps)
                    * math.pi
                )
                + 0.5
            ) * (self.lr_max - self.lr_final) + self.lr_final
        else:
            lr = self.lr_final
        return [lr for _ in self.base_lrs]


class LitGPT(pl.LightningModule):
    def __init__(self, gpt_config: GPTConfig):
        super().__init__()

        self.model = GPT(gpt_config)
        self.learning_rate = 5e-4  # max learning rate

    def forward(self, idx, targets=None):
        return self.model(idx, targets=targets)

    def training_step(self, batch, batch_idx):
        tokens = batch
        src, tgt = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, loss = self(src, tgt)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch
        src, tgt = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, loss = self(src, tgt)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            weight_decay=1e-1,
            learning_rate=self.learning_rate,
            betas=(0.9, 0.999),
            device_type="cuda",
        )

        sched_config = {
            "scheduler": CustomScheduler(
                optimizer,
                n_linear_steps=1000,
                n_decay_steps=10000,
                lr_init=0.0,
                lr_max=self.learning_rate,
                lr_final=self.learning_rate / 10,
            ),
            "frequency": 1,
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": sched_config}

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k
        )

    def generate_top_p(self, idx, max_new_tokens, top_p=0.8):
        return self.model.generate_top_p(idx, max_new_tokens, top_p=top_p)
