from pathlib import Path
from typing import List

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from cleanlab import filter
from cleanlab.filter import find_label_issues
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from label_noise.dataset import MLRNetModule
from label_noise.dataset.transforms import UnNormalise
from label_noise.model import ResNeXT

app = typer.Typer()


@app.command("vis")
def viz(
    data_directory: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    batch_size: int = typer.Option(256, help="Number of batches."),
    num_workers: int = typer.Option(12, help="Number of workers for data loading"),
):

    data_module = MLRNetModule(
        data_directory,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_module.setup("test")

    plt.ion()
    plt.show()
    for x, y in data_module.test_dataloader():
        plt.imshow(np.transpose(x[0].cpu().numpy(), [1, 2, 0]))
        title = f"Possible error: given label {[data_module.categories()[a] for a in np.where(y.cpu().numpy()[0] == 1)[0].tolist()]}"
        plt.title(title)
        plt.draw()
        plt.pause(0.1)        


@app.command("check-noise")
def check_noise(
    data_directory: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    model_fn: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    output_dir: Path = typer.Option(None, help="Path where datafiles are located if they need verification"),
    batch_size: int = typer.Option(256, help="Number of batches."),
    num_devices: List[int] = typer.Option(None, help="gpu device id"),
    num_workers: int = typer.Option(12, help="Number of workers for data loading"),
    with_vis: bool = typer.Option(False, help="Weather to show visualization"),
):

    torch.cuda.empty_cache()
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    data_module = MLRNetModule(
        data_directory,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup("test")

    model = ResNeXT.load_from_checkpoint(checkpoint_path=model_fn)
    model.eval()

    tb_logger = TensorBoardLogger(save_dir=output_dir / "logs")

    trainer = Trainer(
        logger=tb_logger,
        gpus=num_devices,
        auto_select_gpus=True,
        max_epochs=256,
        callbacks=[
            RichProgressBar(refresh_rate=1),
        ],
    )

    results = trainer.predict(model, data_module)
    predictions = torch.cat(tuple(results[i][0] for i in range(len(results))))
    images = torch.cat(tuple(results[i][1][0] for i in range(len(results))))
    truth = torch.cat(tuple(results[i][1][1] for i in range(len(results))))

    reverse_ohe_truth = [torch.where(x_i == 1)[0].cpu().numpy().tolist() for x_i in torch.unbind(truth, dim=0)]

    ordered_label_errors = find_label_issues(
        reverse_ohe_truth,
        predictions.cpu().numpy(),
        return_indices_ranked_by="normalized_margin",  # Orders label errors
        multi_label=True,
    )

    if with_vis:
        plt.ion()
        plt.show()
        for idx in ordered_label_errors:
            img = UnNormalise()(images[idx])
            plt.imshow(np.transpose(img.cpu().numpy(), [1, 2, 0]))
            expected_str = [f"{b}:{np.round(float(a), 2)}" for a, b in zip(predictions[idx], data_module.categories())]
            title = f"Possible error: \nTruth: {[data_module.categories()[a] for a in reverse_ohe_truth[idx]]}\nExpected: {expected_str}"
            plt.title(title)
            plt.draw()
            plt.pause(0.1)

    # weird OOD examples
    self_confidence_label_quality_scores = cleanlab.rank.get_self_confidence_for_each_label(
        reverse_ohe_truth,
        predictions.cpu().numpy(),
    )
    if with_vis:
        plt.ion()
        plt.show()
        plt.hist(self_confidence_label_quality_scores)
        plt.pause(1)
        plt.boxplot(self_confidence_label_quality_scores)
        plt.pause(1)

    entropy_label_quality_scores = cleanlab.rank.get_confidence_weighted_entropy_for_each_label(
        reverse_ohe_truth,
        predictions.cpu().numpy(),
    )

    if with_vis:
        plt.ion()
        plt.show()
        plt.hist(entropy_label_quality_scores)
        plt.pause(1)
        plt.boxplot(entropy_label_quality_scores)
        plt.pause(1)    

    confident_joint_matrix = cleanlab.count.compute_confident_joint(
        reverse_ohe_truth,
        predictions.cpu().numpy(),
        multi_label=True,
    )


if __name__ == "__main__":
    app()
