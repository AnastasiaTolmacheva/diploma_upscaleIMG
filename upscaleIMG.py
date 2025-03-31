import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import Image
import platform
import datetime
import statistics
from skimage.metrics import structural_similarity as ssim
import pyiqa
from models import SRCNN, ESPCN, Generator
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, psnr, image_to_tensor, tensor_to_image


def log_message(log_filename, message):
    """
    Функция для записи логов в файл и вывода сообщений в консоль.

    Входные параметры:
    - log_filename (str): Путь к файлу для записи логов.
    - message (str): Сообщение, которое будет записано в лог.
    """
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(formatted_message + "\n")
    print(formatted_message)


def process_srgann(input_folder, output_folder, model, scale, device, niqe_model, mode, log_filename):
    psnr_list = []
    ssim_list = []
    niqe_list = []

    for image_name in os.listdir(input_folder):
        try:
            if mode == "test" and 'LR' not in image_name:
                continue

            image_file = os.path.join(input_folder, image_name)
            lr_image = Image.open(image_file).convert('RGB')  # Открываем изображение в формате RGB
            lr_image_tensor = image_to_tensor(lr_image).to(device)  # Преобразуем его в тензор

            with torch.no_grad():
                sr_image = model(lr_image_tensor)

            # Преобразуем SR изображение в PIL формата RGB
            sr_image_np = sr_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            sr_image_np = np.clip(sr_image_np * 255.0, 0, 255).astype(np.uint8)  # Масштабируем значения в диапазон [0, 255]

            sr_image_pil = Image.fromarray(sr_image_np)  # Преобразуем в PIL изображение

            # Сохранение результата
            model_tag = "srgan"
            output_image_path = os.path.join(output_folder, f"{image_name.replace('.', f'_{model_tag}_x{scale}.')}")
            sr_image_pil.save(output_image_path)

            # Вычисление PSNR и SSIM только в режиме "test"
            if mode == "test":
                hr_image_file = image_file.replace('LR', 'HR')
                if os.path.exists(hr_image_file):
                    hr_image = Image.open(hr_image_file).convert('RGB')  # Открываем HR изображение
                    hr_image_tensor = image_to_tensor(hr_image).to(device)  # Преобразуем в тензор
                    psnr_value = psnr(sr_image, hr_image_tensor)
                    psnr_list.append(psnr_value)
                    log_message(log_filename, f'PSNR for {image_name.replace(".", f"_{model_tag}_x{scale}.")} : {psnr_value:.2f}')

                    # SSIM
                    hr_image_np = np.array(hr_image).astype(np.float32)
                    ssim_value = ssim(hr_image_np, sr_image_np, data_range=255.0, multichannel=True, win_size=7, channel_axis=2)
                    ssim_list.append(ssim_value)
                    log_message(log_filename, f'SSIM for {image_name.replace(".", f"_{model_tag}_x{scale}.")} : {ssim_value:.4f}')
                else:
                    log_message(log_filename, f'HR image for {image_name} not found. Skipping PSNR/SSIM calculation.')

            # NIQE (всегда вычисляется)
            niqe_value = niqe_model(sr_image_pil)
            niqe_list.append(niqe_value.item())
            log_message(log_filename, f'NIQE for {image_name.replace(".", f"_{model_tag}_x{scale}.")} : {niqe_value.item():.4f}')

        except Exception as e:
            log_message(log_filename, f"Error processing {image_name}: {str(e)}")

    # Вычисление медианы метрик
    try:
        median_psnr = statistics.median(psnr_list) if psnr_list else None
        median_ssim = statistics.median(ssim_list) if ssim_list else None
        median_niqe = statistics.median(niqe_list) if niqe_list else None

        if mode == "test":
            log_message(log_filename, f'Median PSNR: {median_psnr:.2f}')
            log_message(log_filename, f'Median SSIM: {median_ssim:.4f}')
        log_message(log_filename, f'Median NIQE: {median_niqe:.4f}')
    
    except Exception as e:
        log_message(log_filename, f"Error calculating median metrics: {str(e)}")


def upscale_image(model_name, weights_file, input_folder, output_folder, scale, mode):
    """
    """
    log_filename = f"logs/upscale_{args.model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_x{args.scale}.txt"

    try:
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Выбор модели
        if model_name == "SRCNN":
            model = SRCNN().to(device)
        elif model_name == "ESPCN":
            model = ESPCN(scale_factor=scale).to(device)
        elif model_name == "SRGAN":
            model = Generator(scale).to(device)
            model.load_state_dict(torch.load(weights_file))
            model.eval()

        # Загрузка весов для SRCNN и ESPCN
        if model_name in ["SRCNN", "ESPCN"]:
            state_dict = model.state_dict()
            for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)

        model.eval()

        # Инициализация модели для NIQE
        niqe_model = pyiqa.create_metric('niqe')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        python_version = platform.python_version()
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Weights: {weights_file}")
        log_message(log_filename, f"Data for upscaling: {input_folder}")
        log_message(log_filename, f"Results: {output_folder}")
        log_message(log_filename, "=== Starting upscaling images ===")
        log_message(log_filename, f"Parameters: \n- Model: ESPCN\n- Scaling factor: {scale}\n- Mode: {mode}")

        psnr_list = []
        ssim_list = []
        niqe_list = []

        if model_name == "SRGAN":
            # Для SRGAN вызываем отдельную функцию
            process_srgann(input_folder, output_folder, model, scale, device, niqe_model, mode, log_filename)
            return  # Завершаем выполнение функции, так как обработка завершена внутри process_srgann
        
    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}")

    for image_name in os.listdir(input_folder):
        try:
            # В тестовом режиме пропускаем файлы без 'LR' в имени
            if mode == "test" and "LR" not in image_name:
                continue

            image_file = os.path.join(input_folder, image_name)
            image = pil_image.open(image_file).convert('RGB')

            if model_name == "SRCNN":
                image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

            image_np = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image_np)

            y = ycbcr[..., 0] / 255.
            y = torch.from_numpy(y).to(device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)

            # PSNR только в тестовом режиме
            if mode == "test":
                hr_image_file = image_file.replace('LR', 'HR')
                if os.path.exists(hr_image_file):
                    hr_image = pil_image.open(hr_image_file).convert('RGB')
                    hr_ycbcr = convert_rgb_to_ycbcr(np.array(hr_image).astype(np.float32))
                    hr_y = hr_ycbcr[..., 0] / 255.
                    hr_y = torch.from_numpy(hr_y).to(device).unsqueeze(0).unsqueeze(0)

                    preds_resized = torch.nn.functional.interpolate(preds, size=(hr_y.size(2), hr_y.size(3)), mode='bicubic', align_corners=False)
                    psnr_value = psnr(hr_y, preds_resized)
                    psnr_list.append(psnr_value)
                    log_message(log_filename, f'PSNR for {image_name}: {psnr_value:.2f}')
                else:
                    log_message(log_filename, f'HR image for {image_name} not found. Skipping PSNR calculation.')

            # SSIM (только в тестовом режиме)
            if mode == "test":
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
                    log_message(log_filename, f'SSIM for {image_name}: {ssim_value:.4f}')
                else:
                    log_message(log_filename, f'HR image for {image_name} not found. Skipping SSIM calculation.')

            # NIQE
            preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            preds_image = pil_image.fromarray(preds_np.astype(np.uint8))
            niqe_value = niqe_model(preds_image)
            niqe_list.append(niqe_value.item())
            log_message(log_filename, f'NIQE for {image_name}: {niqe_value.item():.4f}')

            # Конвертация YCbCr в RGB
            cb_resized = torch.nn.functional.interpolate(
                torch.from_numpy(ycbcr[..., 1]).unsqueeze(0).unsqueeze(0),
                size=preds_np.shape, mode='bicubic', align_corners=False).squeeze().numpy()
            cr_resized = torch.nn.functional.interpolate(
                torch.from_numpy(ycbcr[..., 2]).unsqueeze(0).unsqueeze(0),
                size=preds_np.shape, mode='bicubic', align_corners=False).squeeze().numpy()

            output_image = np.array([preds_np, cb_resized, cr_resized]).transpose([1, 2, 0])
            output_image = np.clip(convert_ycbcr_to_rgb(output_image), 0.0, 255.0).astype(np.uint8)
            output_image = pil_image.fromarray(output_image)

            # Сохранение результата
            output_image.save(os.path.join(output_folder, image_name.replace('.', f'_{model_name}_x{scale}.')))
      
        except Exception as e:
            log_message(log_filename, f"Error calculating median metrics: {str(e)}")

    try:
        # Медианные значения
        if mode == "test":
            median_psnr = statistics.median(psnr_list) if psnr_list else None
            median_ssim = statistics.median(ssim_list) if ssim_list else None
            log_message(log_filename, f'Median PSNR: {median_psnr:.2f}')
            log_message(log_filename, f'Median SSIM: {median_ssim:.4f}')

        median_niqe = statistics.median(niqe_list) if niqe_list else None
        log_message(log_filename, f'Median NIQE: {median_niqe:.4f}')
            
    except Exception as e:
        log_message(log_filename, f"Error calculating median metrics: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale imges.")
    parser.add_argument("--model", default="SRCNN", type=str, choices=["SRCNN", "ESPCN", "SRGAN"], help="Model name (SRCNN, ESPCN, SRGAN)")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    parser.add_argument("--scale",  default=2, type=int, choices=[2, 3, 4], help="Upscaling factor (2, 3, 4)")
    parser.add_argument("--mode", type=str, required=True, choices=["test", "real"], help="Mode of operation (test - you have LR and HR images, real - ypu have only LR images)")
    args = parser.parse_args()

    upscale_image(args.model, args.weights, args.input, args.output, args.scale, args.mode)
