import h5py
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, CenterCrop, Resize, Normalize, ToPILImage
import cv2
from torchvision.transforms import ToTensor, CenterCrop, Resize
from utils import (
    is_image_file,
    calculate_valid_crop_size
)


class TrainDatasetSRCNN(Dataset):
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
    def __init__(self, h5_file):
        super(EvalDatasetSRCNN, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            idx_str = str(idx)
            lr = f['lr'][idx_str][:] / 255.0
            hr = f['hr'][idx_str][:] / 255.0

        h, w = hr.shape
        lr_upscaled = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        return np.expand_dims(lr_upscaled, 0), np.expand_dims(hr, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TrainDatasetESPCN(Dataset):
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
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetSRGAN, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        # Обновляем трансформации
        self.hr_transform = Compose([
            RandomCrop(crop_size),  # Вырезаем случайный фрагмент
            ToTensor(),
            # Normalize(mean=[0.5], std=[0.5])  # Нормализация
        ])

        self.lr_transform = Compose([
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
            # Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.image_filenames)


class EvalDatasetSRGAN(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(EvalDatasetSRGAN, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)
