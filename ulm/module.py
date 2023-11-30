import lightning.pytorch as pl

from .nano_gpt import GPT, GPTConfig


class LitGPT(pl.LightningModule):
    def __init__(self, config: GPTConfig, learning_rate: float):
        super().__init__()

        self.learning_rate = learning_rate
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
            weight_decay=1e-1,
            learning_rate=self.learning_rate,
            betas=(0.9, 0.999),
            device_type="cuda",
        )

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k
        )

    def generate_top_p(self, idx, max_new_tokens, top_p=0.9):
        return self.model.generate_top_p(idx, max_new_tokens, top_p=top_p)
