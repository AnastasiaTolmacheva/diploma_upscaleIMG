import torch
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil_image


def convert_rgb_to_y(img):
    """
    Преобразование RGB-изображение в Y-канал (яркость) цветового пространства YCbCr.

    На входе:
    - img (np.ndarray или torch.Tensor): RGB-изображение в виде массива NumPy или тензора PyTorch;
        * np.ndarray имеет форму (H, W, 3), где H — высота, W — ширина;
        * torch.Tensor может иметь форму (3, H, W) или (1, 3, H, W) (если батч содержит одно изображение).

    На выходе:
    - Y-канал изображения (np.ndarray или torch.Tensor) значениями яркости по формуле: 
    Y = 16 + (64.738 * R + 129.057 * G + 25.064 * B) / 256

    Исключения:
    - если тип изображения не поддерживается (ни np.ndarray, ни torch.Tensor).
    """
    # Обработка изображения NumPy: извлекаем каналы R, G, B и рассчитываем Y по формуле
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    
    # Обработка изображения PyTorch
    # Если изображение в батче (1, 3, H, W), убираем размерность батча
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def convert_rgb_to_ycbcr(img):
    """
    Преобразование изображение из RGB в цветовое пространство YCbCr.

    На входе:
    - img (np.ndarray или torch.Tensor): RGB-изображение с тремя каналами;
        * Если np.ndarray, ожидается форма (H, W, 3) с каналами R, G, B;
        * Если torch.Tensor, форма (3, H, W) или (1, 3, H, W). 

    На выходе:
    - np.ndarray или torch.Tensor: изображение в формате YCbCr, форма (H, W, 3)

    Формулы перевода (по стандарту JPEG/ITU-R BT.601):
      Y = 16 + (65.738 * R + 129.057 * G + 25.064 * B) / 256
      Cb = 128 + (-37.945 * R -  74.494 * G + 112.439 * B) / 256
      Cr = 128 + (112.439 * R - 94.154 * G - 18.285 * B) / 256

    Исключения:
    - если тип изображения не поддерживается.
    """
    # Обработка NumPy-изображения (H, W, 3)
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    
    # Обработка PyTorch-тензора (3, H, W) или (1, 3, H, W)
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:   # (1, 3, H, W) -> (3, H, W)
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)   # (3, H, W) -> (H, W, 3)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def convert_ycbcr_to_rgb(img):
    """
    Преобразование изображение из цветового пространства YCbCr обратно в RGB.

    На входе:
    - img (np.ndarray или torch.Tensor): изображение в формате YCbCr;
        * Для np.ndarray ожидается форма (H, W, 3);
        * Для torch.Tensor — (3, H, W) или (1, 3, H, W), где 0-й канал — Y, 1-й — Cb, 2-й — Cr.

    На выходе:
    - np.ndarray или torch.Tensor: изображение в формате RGB, форма (H, W, 3).

    Используемые формулы (обратные к ITU-R BT.601):
      R = 298.082 * Y / 256 + 408.583 * Cr / 256 - 222.921
      G = 298.082 * Y / 256 - 100.291 * Cb / 256 - 208.120 * Cr / 256 + 135.576
      B = 298.082 * Y / 256 + 516.412 * Cb / 256 - 276.836
    """
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)    # (1, 3, H, W) -> (3, H, W)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)   # (3, H, W) -> (H, W, 3)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def psnr(img1, img2):
    """
    Вычисление PSNR между двумя изображениями.

    На входе:
    - img1, img2 (torch.Tensor): два изображения одинакового размера, нормализованные в диапазоне [0, 1].

    На выходе:
    - torch.Tensor: значение PSNR dB.
    """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    """
    Класс для хранения и обновления среднего арифметического.
    Используется для отслеживания среднего значения метрик (потерь и PSNR) по эпохам.

    Атрибуты:
    - val: текущее значение
    - avg: текущее среднее значение
    - sum: сумма всех значений
    - count: общее количество обновлений

    Методы:
    - reset(): сбрасывает все значения
    - update(val, n): добавляет новое значение val с весом n (по умолчанию 1)
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


def is_image_file(filename: str) -> bool:
    """
    Проверка является ли файл изображением.
    Поддерживаемые форматы: .png, .jpg, .jpeg, .tiff, .tif (без учёта регистра)

    На входе:
    - filename (str): имя файла или путь.

    На выходе:
    - bool: True, если файл — изображение, иначе False.
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))


def normalize(val: float, min_val: float, max_val: float) -> float:
    """
    Нормализация значения в диапазон [0, 1].

    На входе:
    - val (float): значение, которое нужно нормализовать;
    - min_val (float): минимальное значение диапазона;
    - max_val (float): максимальное значение диапазона.

    На выходе:
    - float: нормализованное значение от 0 до 1.
             Если max_val <= min_val, возвращает 1.0.
    """
    return (val - min_val) / (max_val - min_val) if max_val > min_val else 1.0


def calculate_valid_crop_size(crop_size: int, scale_factor: int) -> int:
    """
    Вычисление ближайшего размера обрезки, кратного коэффициенту увеличения.

    Параметры:
    - crop_size (int): исходный размер обрезки.
    - scale_factor (int): коэффициент увеличения (2, 3 или 4).

    Возвращает:
    - int: допустимый размер обрезки, кратный scale_factor.
    """
    return crop_size - (crop_size % scale_factor)


def image_to_tensor(image):
    """
    Преобразование изображения PIL в тензор PyTorch.

    На выходе:
    - image (PIL.Image): изображение PIL.

    На выходе:
    - torch.Tensor: тензор изображения формы (1, C, H, W) с диапазоном значений [0, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразует изображение в тензор и масштабирует значения в диапазон [0, 1]
    ])
    return transform(image).unsqueeze(0)  # Добавляем ось батча (C, H, W) -> (1, C, H, W)


def tensor_to_image(tensor):
    """
    Преобразование тензора PyTorch в изображение PIL.

    На выходе:
    - tensor (torch.Tensor): тензор изображения формы (1, C, H, W).

    На выходе:
    - PIL.Image: изображение PIL (H, W, C) с диапазоном [0, 255].
    """
    image = tensor.cpu().detach().numpy().squeeze(0)  # (1, C, H, W) -> (C, H, W)
    image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return pil_image.fromarray((image * 255).astype('uint8'))


"""
def train_hr_transform(crop_size):

    Трансформирует: обрезка изображения и преобразование в тензор.

    На выходе:
    - crop_size (int): размер обрезки изображения высокого разрешения.

    На выходе:
    - torchvision.transforms.Compose: трансформации изображения.

    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):

    Трансформирует изображение, создавая LR из HR с помощью бикубической интерполяции.

    На выходе:
    - crop_size (int): размер изображения высокого разрешения;
    - upscale_factor (int): коэффициент уменьшения.

    На выходе:
    - torchvision.transforms.Compose: трансформации изображения: преобразование в PIL, уменьшение и преобразование в тензор.

    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():

    Трансформация для отображения изображений

    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
"""