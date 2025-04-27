import h5py
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, CenterCrop, Resize, ToPILImage
from torchvision.transforms import ToTensor, CenterCrop, Resize
from utils import (
    is_image_file,
    calculate_valid_crop_size
)


class TrainDatasetSRCNN(Dataset):
    """
    Набор данных для обучения SRCNN.

    На выходе:
    - h5_file (str): путь к .h5 файлу, содержащему LR и HR изображения.

    На выходе:
    - кортеж (lr, hr): оба изображения нормализованы в диапазон [0, 1], размер (1, H, W).
    """
    def __init__(self, h5_file):
        super(TrainDatasetSRCNN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDatasetSRCNN(Dataset):
    """
    Набор данных для оценки SRCNN.

    На входе:
    - h5_file (str): путь к .h5 файлу, содержащему LR и HR изображения.

    На выходе:
    - кортеж (lr_upscaled, hr): оба изображения нормализованы в диапазон [0, 1], размер (1, H, W).
    """
    def __init__(self, h5_file):
        super(EvalDatasetSRCNN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            idx_str = str(idx)
            lr = f['lr'][idx_str][:] / 255.0
            hr = f['hr'][idx_str][:] / 255.0

        return np.expand_dims(lr, 0), np.expand_dims(hr, 0)


    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDatasetESPCN(Dataset):
    """
    Набор данных для обучения ESPCN.

    На выходе:
    - h5_file (str): путь к .h5 файлу, содержащему LR и HR изображения.

    На выходе:
    - кортеж (lr, hr): оба изображения нормализованы в диапазон [0, 1], размер (1, H, W).
    """
    def __init__(self, h5_file):
        super(TrainDatasetESPCN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDatasetESPCN(Dataset):
    """
    Набор данных для оценки ESPCN.

    На входе:
    - h5_file (str): путь к .h5 файлу, который содержит датасет.

    На выходе:
    - кортеж (lr, hr): оба изображения нормализованы в диапазон [0, 1], размер (1, H, W).
    """
    def __init__(self, h5_file):
        super(EvalDatasetESPCN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDatasetSRGAN(Dataset):
    """
    Набор данных для обучения SRGAN из директории изображений.

    На входе:
    - dataset_dir (str): путь к директории с изображениями;
    - crop_size (int): желаемый размер фрагмента для обучения;
    - upscale_factor (int): коэффициент увеличения разрешения.

    На выходе:
    - кортеж (lr, hr): обрезанные и масштабированные изображения в виде тензоров, размер (3, H, W).
    """
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetSRGAN, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        self.hr_transform = Compose([
            RandomCrop(crop_size),  # Вырезаем случайный фрагмент
            ToTensor(),
        ])

        self.lr_transform = Compose([
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
        ])

    def __getitem__(self, index):
        hr = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr = self.lr_transform(hr)
        return lr, hr
    
    def __len__(self):
        return len(self.image_filenames)


class EvalDatasetSRGAN(Dataset):
    """
    Набор данных для оценки SRGAN, включает создание LR из HR и восстановление изображения.
    
    На входе:
    - dataset_dir (str): путь к директории с изображениями;
    - upscale_factor (int): коэффициент увеличения разрешения.

    Возвращает:
    - кортеж (lr_image, hr_image): изображения в виде тензоров, размер (3, H, W).
    """
    def __init__(self, dataset_dir, upscale_factor):
        super(EvalDatasetSRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        hr_image = CenterCrop(crop_size)(hr_image)

        # Создаем изображение LR (низкого разрешения) с использованием бикубической интерполяции
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        lr_image = lr_scale(hr_image)

        return ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)
