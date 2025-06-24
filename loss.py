import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    """
    Функция потерь для генератора SRGAN.

    Состоит из:
    - Content (Perception) Loss: среднеквадратическая ошибка между признаками VGG16 предсказанных и эталонных изображений (HR);
    - Image Loss: среднеквадратическая ошибка между предсказанным и эталонным изображением (HR);
    - Adversarial Loss: потери от дискриминатора;
    - TV Loss: Total Variation Loss для сглаживания артефактов.

    Формула:
    L = Image loss + 0.001 * Adversarial loss + 0.006 * Content loss + 2e-8 * TVLoss
    """
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # Используем предобученную VGG16
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

        self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        """
        Вычисляет итоговую функцию потерь генератора SRGAN.

        На входе:
        - out_labels (Tensor): предсказания дискриминатора;
        - out_images (Tensor): изображение, сгенерированное генератором;
        - target_images (Tensor): эталонное (HR) изображение высокого разрешения.

        На выходе:
        - Итоговая скалярная функция потерь генератора.
        """       
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Content Loss
        content_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * content_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    """
    Total Variation Loss — для устранения артефактов в изображениях.

    Аргументы:
    - tv_loss_weight (float): вес TV-loss в общей функции потерь.
    """
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """
        Вычисляет Total Variation Loss.

        На входе:
        - x (Tensor): изображение (B, C, H, W).

        На выходе:
        - Скалярная TV-loss.
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


"""
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, out_labels, out_images, target_images):

        L_SR = L_X_SR + 10^-3 * L_Gen_SR 

        # Adversarial Loss (BCE)
        adversarial_loss = self.bce_loss(out_labels, torch.ones_like(out_labels))

        # Content Loss (VGG MSE)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Итоговая функция потерь
        return perception_loss + 0.001 * adversarial_loss
"""


if __name__ == "__main__":
    g_loss = GeneratorLoss()
