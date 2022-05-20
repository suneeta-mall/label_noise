import torch
import torch.utils.data
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import accuracy, f1_score


class ResNeXT(LightningModule):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model = torchvision.models.resnet18(
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        preds = self.model(x)
        return preds

    def _calculate_loss_metrics(self, logits, y, prefix):
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float())

        s_logits = torch.sigmoid(logits)

        accuracy_val = accuracy(s_logits > 0.5, y, average=None, num_classes=self.num_classes)
        f1 = f1_score(s_logits, y, num_classes=self.num_classes, threshold=0.5, average="none")

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for i in range(self.num_classes):
            self.log(
                f"{prefix}_accuracy_{i}",
                accuracy_val[i],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"{prefix}_f1_{i}",
                f1[i],
                on_step=True,
                on_epoch=True,
            )

        return loss, accuracy_val, f1

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss, accuracy_val, f1 = self._calculate_loss_metrics(logits, y, prefix="train")

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss, accuracy_val, f1 = self._calculate_loss_metrics(logits, y, prefix="val")

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)

        loss, accuracy_val, f1 = self._calculate_loss_metrics(logits, y, prefix="test")
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        logits = self.forward(x)
        s_logits = torch.sigmoid(logits)
        return s_logits, (x, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [scheduler]
