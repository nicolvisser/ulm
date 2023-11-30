import json

import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch

from dataset import TokenizedDataset
from ulm.module import GPTConfig, LitGPT

from callbacks import LogSampleContinuationCallback


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
    "--n_units",
    type=int,
    help="Ther number of k-means units with which the units are encoded",
)
@click.option(
    "--dp_lambda",
    type=float,
    help="The lambda paramters for DPDP with which the units are encoded",
)
@click.option(
    "--tokenizer_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .model file with sentencepiece tokenizer",
)
@click.option(
    "--num_workers",
    type=int,
    help="Number of workers",
    prompt=True,
)
@click.option(
    "--val_check_interval",
    type=float,
    default=1.0,
    help="Interval (in epochs) to check validation dataset",
    prompt=True,
)
@click.option(
    "--log_every_n_steps",
    type=int,
    default=1,
    help="Interval (in steps) to log training metrics",
    prompt=True,
)
def train(
    config_path,
    train_ids_path,
    val_ids_path,
    version_name,
    n_units,
    dp_lambda,
    tokenizer_path,
    num_workers,
    val_check_interval,
    log_every_n_steps,
):
    with open(config_path, "r") as f:
        config = GPTConfig(**json.load(f))

    train_loader = DataLoader(
        dataset=TokenizedDataset(
            path_to_ids=train_ids_path,
            block_size=config.block_size,
            num_samples=50000,
        ),
        batch_size=50,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=TokenizedDataset(
            path_to_ids=val_ids_path,
            block_size=config.block_size,
            num_samples=100,
        ),
        batch_size=50,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    torch.set_float32_matmul_precision("medium")

    model = LitGPT(config)

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

    sample_continuation_callback = LogSampleContinuationCallback(
        prompt_audio_path="prompt.wav",
        n_units=n_units,
        dp_lambda=dp_lambda,
        tokenizer_path=tokenizer_path,
    )

    logger.log_hyperparams(config.__dict__)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        logger=logger,
        max_steps=100000,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback,
            lr_monitor_callback,
            sample_continuation_callback,
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    train()
