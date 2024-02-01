import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights, resnet18, ResNet18_Weights


class AE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Encoder, Input shape: (N, 1, 256, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )

        self.loss_fn = nn.MSELoss()
        self.config = config

    def forward(self, x: Tensor):
        print('x shape: ', x.shape)
        x = self.encoder(x)
        print('after encoder shape: ', x.shape)
        x = self.decoder(x)
        print('after decoder shape: ', x.shape)
        return x

    def detect_anomaly(self, x: Tensor):
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def training_step(self, batch: Tensor, batch_idx):
        x = batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        x = batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])


# if __name__ == '__main__':
#     input_datya = torch.randn(1, 1, 128, 128)
#     model = AE({'lr': 0.001})
#     output = model(input_datya)


