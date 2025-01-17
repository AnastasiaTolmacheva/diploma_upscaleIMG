import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from skimage.metrics import structural_similarity as ssim
import pyiqa
import statistics


def upscale_image(weights_file, input_folder, output_folder, output_log, scale=2):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Загрузка модели
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # Инициализация модели для вычисления NIQE
    niqe_model = pyiqa.create_metric('niqe')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Списки для хранения значений PSNR, SSIM и NIQE
    psnr_list = []
    ssim_list = []
    niqe_list = []

    # Открываем файл для записи логов
    with open(output_log, 'w') as log_file:

        # Обработка всех изображений в папке
        for image_name in os.listdir(input_folder):
            if 'LR' not in image_name:  # Фильтрация по наличию 'LR' в названии файла
                continue

            image_file = os.path.join(input_folder, image_name)

            # Открытие изображения
            image = pil_image.open(image_file).convert('RGB')

            # Изменение размера изображения с помощью бикубической интерполяции
            image_bicubic = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

            # Подготовка изображения для модели
            image_np = np.array(image_bicubic).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image_np)

            y = ycbcr[..., 0]
            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            # Прогон изображения через модель
            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)

            # Вычисление PSNR
            psnr_value = calc_psnr(y, preds)
            psnr_list.append(psnr_value)
            log_file.write(f'PSNR for {image_name}: {psnr_value:.2f}\n')

            # Вычисление SSIM
            hr_image_file = image_file.replace('LR', 'HR')
            if os.path.exists(hr_image_file):
                hr_image = pil_image.open(hr_image_file).convert('RGB')
                hr_ycbcr = convert_rgb_to_ycbcr(np.array(hr_image).astype(np.float32))
                hr_y = hr_ycbcr[..., 0] / 255.
                hr_y = torch.from_numpy(hr_y).to(device).unsqueeze(0).unsqueeze(0)

                preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                hr_np = hr_y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                ssim_value = ssim(hr_np, preds_np, data_range=255.0)
                ssim_list.append(ssim_value)
                log_file.write(f'SSIM for {image_name}: {ssim_value:.4f}\n')
            else:
                log_file.write(f'HR image for {image_name} not found. Skipping SSIM calculation.\n')

            # Вычисление NIQE с помощью pyiqa
            preds_image = pil_image.fromarray(preds_np.astype(np.uint8))  # Преобразование в формат PIL
            niqe_value = niqe_model(preds_image)  # Вычисление NIQE
            niqe_list.append(niqe_value.item())
            log_file.write(f'NIQE for {image_name}: {niqe_value.item():.4f}\n')
            log_file.write('--------------\n')

            # Конвертация предсказаний обратно в изображение
            preds_np = preds_np.astype(np.float32)
            output_image = np.array([preds_np, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output_image = np.clip(convert_ycbcr_to_rgb(output_image), 0.0, 255.0).astype(np.uint8)
            output_image = pil_image.fromarray(output_image)

            # Сохранение результата
            output_image.save(os.path.join(output_folder, image_name.replace('.', '_srcnn_x{}.'.format(scale))))

        # Вычисление и запись медианы для PSNR, SSIM и NIQE
        median_psnr = statistics.median(psnr_list) if psnr_list else None
        median_ssim = statistics.median(ssim_list) if ssim_list else None
        median_niqe = statistics.median(niqe_list) if niqe_list else None

        log_file.write(f'\nMedian PSNR: {median_psnr:.2f}\n')
        log_file.write(f'Median SSIM: {median_ssim:.4f}\n')
        log_file.write(f'Median NIQE: {median_niqe:.4f}\n')

    print(f'Processing completed. Results saved in {output_folder} and {output_log}')


weights_file = 'D:\\SRCNN\\output_01\\x2\\best.pth'
input_folder = 'D:\\SRCNN\\Set14'
output_folder = 'D:\\SRCNN\\output_test'
output_log = 'D:\\SRCNN\\output_test\\results_log.txt'

upscale_image(weights_file, input_folder, output_folder, output_log, scale=2)
