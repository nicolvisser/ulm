import json

import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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
    help="Batch size",
)
@click.option(
    "--learning_rate",
    type=float,
    help="Max learning rate. Final learning rate is 10x lower.",
)
@click.option(
    "--num_workers",
    type=int,
    help="Number of workers",
)
@click.option(
    "--max_steps",
    type=int,
    help="Max number of steps before training stops",
)
@click.option(
    "--train_num_samples",
    type=int,
    help="Number of samples in training dataset per epoch",
)
@click.option(
    "--val_num_samples",
    type=int,
    default=1000,
    help="Number of samples in validation dataset",
)
@click.option(
    "--val_check_interval",
    type=float,
    default=1.0,
    help="Interval (in epochs) to check validation dataset",
)
@click.option(
    "--log_every_n_steps",
    type=int,
    default=1,
    help="Interval (in steps) to log training metrics",
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
    val_check_interval,
    log_every_n_steps,
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
    n_steps_per_epoch = len(train_dataset) // batch_size

    model = LitGPT(
        config,
        learning_rate=learning_rate,
        n_steps_per_epoch=n_steps_per_epoch,
    )

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

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

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
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback,
            lr_monitor_callback,
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    train()
