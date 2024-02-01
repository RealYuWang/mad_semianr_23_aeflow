import torch
import torch.nn as nn
import torch.distributions as dist
import pytorch_lightning as pl
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


# For Flow model
class ResNetCouplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetCouplingBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.last_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def custom_relu(self, x):
        return torch.max(torch.tensor(0.0).to(x.device), x)

    def forward(self, x):
        x = self.custom_relu(self.block(x))
        x = self.last_conv(x)
        return x


class AEFLOW(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Encoder, Inputshape: (N, 1, 256, 256) Output shape: (N, 1024, 16, 16)
        self.encoder = nn.Sequential(*list(wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).children())[:7])
        self.encoder[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Normalizing flow with 8 steps
        self.flow = SequenceINN(1024, 16, 16)
        for _ in range(8):
            self.flow.append(AllInOneBlock, subnet_constructor=ResNetCouplingBlock, permute_soft=False)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=1, output_padding=0),
        )
        self.config = config

        # 冻结预训练模型的权重
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.flow(x)[0]
        x = self.decoder(x)
        return x

    def on_train_start(self) -> None:
        torch.autograd.set_detect_anomaly(True)

    def detect_anomaly(self, x):
        # reconstruction score
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        # ssim_loss = cv2.compare_ssim
        # anomaly_score_rec = -torch.sum(anomaly_map, dim=(1, 2, 3))

        # flow score
        normal_dist = dist.Normal(0, 1)
        z_ = self.flow(self.encoder(x))[0]
        log_prob = normal_dist.log_prob(z_)
        anomaly_score_flow = -log_prob.mean().item()
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score_flow
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimizer

    def training_step(self, batch):
        x = batch
        x_recon = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(x_recon, x)
        # Flow loss (negative log prob density of p(z))
        normal_dist = dist.Normal(0, 1)
        z = self.encoder(x)

        log_prob = normal_dist.log_prob(z)
        flow_loss = -log_prob.mean().item()
        total_loss = 0.5 * recon_loss + 0.5 * flow_loss
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)

        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(x_recon, x)
        # Flow loss (negative log likelihood of p(z))
        normal_dist = dist.Normal(0, 1)
        z = self.encoder(x)
        log_prob = normal_dist.log_prob(z)
        flow_loss = -log_prob.mean().item()
        loss = 0.5 * recon_loss + 0.5 * flow_loss

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


# if __name__ == '__main__':
#     input_data = torch.randn(1, 1, 256, 256)
#     model = AEFLOW('{lr: 2e-4}')
#     output = model(input_data)
#     print(output.shape)