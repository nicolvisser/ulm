import json
from pathlib import Path
from attr import asdict

import click
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

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

    model = LitGPT(config)

    tensorboard = pl_loggers.TensorBoardLogger(save_dir="", version=version_name)

    log_dir = Path(tensorboard.log_dir)

    if log_dir.exists():
        raise ValueError(
            f"A log with version name, {version_name}, already exists in {tensorboard.log_dir}. Please choose another version name or delete the existing version."
        )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, save_last=True, monitor="val_loss"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tensorboard,
        max_steps=max_steps,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    train()
