import json

import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

from dataset import TokenizedDataset
from ulm.module import GPTConfig, LitGPT


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .json file with model config",
    prompt=True,
)
@click.option(
    "--train_ids_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .pt file with ids of training tokens",
    prompt=True,
)
@click.option(
    "--val_ids_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .pt file with ids of validation tokens",
    prompt=True,
)
@click.option(
    "--version_name",
    type=str,
    help="A checkpoint version name",
    prompt=True,
)
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size",
)
@click.option(
    "--learning_rate",
    type=float,
    default=5e-4,
    help="Learning rate",
)
@click.option(
    "--num_workers",
    type=int,
    default=8,
    help="Number of workers",
)
@click.option(
    "--max_steps",
    type=int,
    default=100000,
    help="Number of epochs",
)
@click.option(
    "--train_num_samples",
    type=int,
    default=10000,
    help="Number of samples in training dataset per epoch",
)
@click.option(
    "--val_num_samples",
    type=int,
    default=1000,
    help="Number of samples in validation dataset",
)
def train(
    config_path,
    train_ids_path,
    val_ids_path,
    version_name,
    batch_size,
    learning_rate,
    num_workers,
    max_steps,
    train_num_samples,
    val_num_samples,
):
    with open(config_path, "r") as f:
        config = GPTConfig(**json.load(f))

    train_dataset = TokenizedDataset(
        path_to_ids=train_ids_path,
        block_size=config.block_size,
        num_samples=train_num_samples,
    )

    val_dataset = TokenizedDataset(
        path_to_ids=val_ids_path,
        block_size=config.block_size,
        num_samples=val_num_samples,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    torch.set_float32_matmul_precision("medium")

    model = LitGPT(config, learning_rate=learning_rate)

    logger = WandbLogger(
        project="ulm-nano-gpt",
        name=version_name,
        save_dir="./",
    )

    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        verbose=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_weights_only=False,
    )

    logger.log_hyperparams(
        {
            **config.__dict__,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        logger=logger,
        max_steps=max_steps,
        log_every_n_steps=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    train()
