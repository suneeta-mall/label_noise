from pathlib import Path
from typing import List

import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from label_noise.dataset import MLRNetModule
from label_noise.model import ResNeXT

app = typer.Typer()


@app.command("train")
def train(
    data_directory: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    output_dir: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    epochs: int = typer.Option(7, help="Number of epochs to run for."),
    batch_size: int = typer.Option(256, help="Number of batches."),
    pretrained: bool = typer.Option(False, help="Use pretrained model weights"),
    num_devices: List[int] = typer.Option(None, help="gpu device id"),
    num_workers: int = typer.Option(12, help="Number of workers for data loading"),
    lr: float = typer.Option(0.01, help="initial learning rate"),
    momentum: float = typer.Option(0.9, help="momentum"),
    weight_decay: float = typer.Option(1e-4, help="weight decay (default: 1e-4)"),
    amp: bool = typer.Option(False, help="Use torch.cuda.amp for mixed precision training"),
):

    torch.cuda.empty_cache()
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    data_module = MLRNetModule(
        data_directory,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    tb_logger = TensorBoardLogger(save_dir=output_dir / "logs")

    trainer = Trainer(
        logger=tb_logger,
        gpus=num_devices,
        auto_select_gpus=True,
        max_epochs=epochs,
        callbacks=[
            LearningRateMonitor(log_momentum=True, logging_interval="epoch"),
            EarlyStopping("val_loss", patience=10, check_finite=True),
            ModelCheckpoint(dirpath=output_dir, filename="model-{epoch:02d}", every_n_epochs=1),
            RichProgressBar(refresh_rate=1),
        ],
        precision=16 if amp else 32,
    )

    model = ResNeXT(
        pretrained,
        num_classes=6,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    trainer.fit(model, data_module)
    trainer.save_checkpoint(output_dir / "model.ckpt")

    script = model.to_torchscript()
    torch.jit.save(script, output_dir / "model.pt")

    trainer.test(model, data_module)


@app.command("test")
def test(
    data_directory: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    output_dir: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    epochs: int = typer.Option(7, help="Number of epochs to run for."),
    batch_size: int = typer.Option(256, help="Number of batches."),
    num_devices: List[int] = typer.Option(None, help="gpu device id"),
    num_workers: int = typer.Option(12, help="Number of workers for data loading"),
    amp: bool = typer.Option(False, help="Use torch.cuda.amp for mixed precision training"),
    accumulate_grad: int = typer.Option(1, help="Use accumulate_grad"),
):

    torch.cuda.empty_cache()
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    data_module = MLRNetModule(
        data_directory,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = ResNeXT.load_from_checkpoint(checkpoint_path="example.ckpt")
    model.eval()

    tb_logger = TensorBoardLogger(save_dir=output_dir / "logs")

    trainer = Trainer(
        logger=tb_logger,
        gpus=num_devices,
        auto_select_gpus=True,
        max_epochs=epochs,
        callbacks=[
            LearningRateMonitor(log_momentum=True, logging_interval="epoch"),
            EarlyStopping("val_loss", patience=10, check_finite=True),
            ModelCheckpoint(dirpath=output_dir, filename="model-{epoch:02d}", every_n_epochs=1),
            RichProgressBar(refresh_rate=1),
        ],
        precision=16 if amp else 32,
        accumulate_grad_batches=accumulate_grad,
    )

    trainer.test(model, data_module)


if __name__ == "__main__":
    app()
