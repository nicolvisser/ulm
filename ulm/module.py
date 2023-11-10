import select
from typing import Self
import lightning.pytorch as pl

from .nano_gpt import GPT, GPTConfig


class LitGPT(pl.LightningModule):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.model = GPT(config)

    def forward(self, idx, targets=None):
        return self.model(idx, targets=targets)

    def training_step(self, batch, batch_idx):
        tokens = batch
        src, tgt = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, loss = self(src, tgt)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch
        src, tgt = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, loss = self(src, tgt)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers(
            weight_decay=0.01,
            learning_rate=1e-3,
            betas=(0.9, 0.999),
            device_type="cuda",
        )

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k
        )
