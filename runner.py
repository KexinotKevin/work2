import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from model_r import LGUNet_rela


class BrainRunner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = LGUNet_rela(args)
        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, g_data, lb_data):
        return self.model(g_data, lb_data, g_data.batch)

    def training_step(self, batch, batch_idx):
        g_data, lb_data = batch
        lb_pred = self(g_data, lb_data)
        loss = torch.abs(lb_pred - lb_data).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        g_data, lb_data = batch
        lb_pred = self(g_data, lb_data)
        loss = torch.abs(lb_pred - lb_data).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_loss_history.append(float(loss.detach().cpu()))

    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_loss_history.append(float(loss.detach().cpu()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.l2_penalty,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3, 5, 10, 20, 30], gamma=0.6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
