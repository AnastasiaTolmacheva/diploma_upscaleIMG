import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import platform
import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pyiqa
import statistics
from model import Generator

# Настройки
UPSCALE_FACTOR = 2
MODEL_PATH = 'D:\\SRGAN0\\epochs\\netG_epoch_2_80.pth'
INPUT_FOLDER = 'D:/SRGAN0/Set14/SRF_2/'
OUTPUT_FOLDER = 'D:\\SRGAN0\\output_test\\'
OUTPUT_LOG = 'D:\\SRGAN0\\output_test\\results_log.txt'

# Загружаем модель генератора
netG = Generator(UPSCALE_FACTOR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netG = netG.to(device)
netG.load_state_dict(torch.load(MODEL_PATH))
netG.eval()

# Инициализация модели для вычисления NIQE
niqe_model = pyiqa.create_metric('niqe')

# Функция для преобразования изображения в тензор
def image_to_tensor(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

# Функция для преобразования тензора обратно в изображение
def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy().squeeze(0)
    image = image.transpose(1, 2, 0)
    return Image.fromarray((image * 255).astype('uint8'))

# Вычисление метрики PSNR
def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Основная функция для обработки изображений
def process_images(input_folder, output_folder, output_log):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Открываем файл для записи логов
    with open(output_log, 'w') as log_file:
        # Системная информация
        log_file.write(f"System Information:\n")
        log_file.write(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Operating System: {platform.system()} {platform.release()}\n")
        log_file.write(f"Python Version: {platform.python_version()}\n")
        log_file.write(f"Development Environment: {platform.python_implementation()} {torch.__version__}\n")
        log_file.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}\n")

        log_file.write(f"\nModel Name: SRGAN\n")
        log_file.write(f"Input Data Source: {input_folder}\n")
        log_file.write(f"Scaling Factor: x{UPSCALE_FACTOR}\n")
        log_file.write(f"\nResults:\n")
        log_file.write('--------------\n')

        psnr_list = []
        ssim_list = []
        niqe_list = []

        # Обработка всех изображений в папке
        for image_name in os.listdir(input_folder):
            if 'LR' not in image_name:
                continue

            image_file = os.path.join(input_folder, image_name)

            # Преобразуем входное изображение
            lr_image = image_to_tensor(image_file).to(device)

            # Прогоняем через генератор
            with torch.no_grad():
                sr_image = netG(lr_image)

            # Преобразуем результат в изображение
            sr_image_pil = tensor_to_image(sr_image)

            # Сохранение изображения
            sr_output_path = os.path.join(output_folder, f"SR_{image_name}")
            sr_image_pil.save(sr_output_path)

            # Вычисляем PSNR
            hr_image_file = image_file.replace('LR', 'HR')
            if os.path.exists(hr_image_file):
                hr_image = image_to_tensor(hr_image_file).to(device)
                
                psnr_value = calc_psnr(sr_image, hr_image)
                psnr_list.append(psnr_value)
                log_file.write(f'PSNR for {image_name}: {psnr_value:.2f}\n')

                # Вычисление SSIM
                sr_image_np = np.array(sr_image_pil).astype(np.float32)
                hr_image_pil = Image.open(hr_image_file).convert('RGB')
                hr_image_np = np.array(hr_image_pil).astype(np.float32)
                ssim_value = ssim(hr_image_np, sr_image_np, data_range=255.0, multichannel=True, win_size=7, channel_axis=2)
                ssim_list.append(ssim_value)
                log_file.write(f'SSIM for {image_name}: {ssim_value:.4f}\n')
            else:
                log_file.write(f'HR image for {image_name} not found. Skipping PSNR/SSIM calculation.\n')

            # Вычисляем NIQE
            niqe_value = niqe_model(sr_image_pil)
            niqe_list.append(niqe_value.item())
            log_file.write(f'NIQE for {image_name}: {niqe_value.item():.4f}\n')
            log_file.write('--------------\n')

        # Вычисление медианы метрик
        median_psnr = statistics.median(psnr_list) if psnr_list else None
        median_ssim = statistics.median(ssim_list) if ssim_list else None
        median_niqe = statistics.median(niqe_list) if niqe_list else None

        log_file.write(f'\nMedian PSNR: {median_psnr:.2f}\n')
        log_file.write(f'Median SSIM: {median_ssim:.4f}\n')
        log_file.write(f'Median NIQE: {median_niqe:.4f}\n')

    print(f'Processing completed. Results saved in {output_folder} and {output_log}')

# Запуск обработки изображений
process_images(INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_LOG)
