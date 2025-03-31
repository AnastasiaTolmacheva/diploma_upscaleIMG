import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
from torchvision import transforms
import PIL.Image as pil_image


def convert_rgb_to_y(img):
    """
    Конвертирует изображение RGB в Y-канал (яркость) цветового пространства YCbCr.
    """
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def convert_rgb_to_ycbcr(img):
    """
    Конвертирует изображение RGB в цветовое пространство YCbCr.
    """
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def convert_ycbcr_to_rgb(img):
    """
    Конвертирует изображение из YCbCr обратно в RGB.
    """
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def psnr(img1, img2):
    """
    Вычисляет пиковое отношение сигнала к шуму (PSNR) между двумя изображениями
    """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    """
    Вспомогательный класс для расчета среднего значения и хранения статистики.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_image_file(filename):
    """
    Проверяет, является ли файл изображением
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


def calculate_valid_crop_size(crop_size, upscale_factor):
    """
    Вычисляет корректный размер обрезки изображения
    """
    return crop_size - (crop_size % upscale_factor)


def image_to_tensor(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразует изображение в тензор и масштабирует значения в диапазон [0, 1]
    ])
    return transform(image).unsqueeze(0)  # Добавляем ось батча (изменение формы с (C, H, W) на (1, C, H, W))


def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy().squeeze(0)  # Убираем размерность батча (из (1, C, H, W) в (C, H, W))
    image = image.transpose(1, 2, 0)  # Переводим из (C, H, W) в (H, W, C)
    return pil_image.fromarray((image * 255).astype('uint8'))  # Масштабируем и преобразуем в uint8


def train_hr_transform(crop_size):
    """
    Трансформация для обрезки и перевода изображения в тензор
    """
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    """
    Трансформация для уменьшения размера изображения и перевода в тензор
    """
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    """
    Трансформация для отображения изображений
    """
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])