import argparse
import os
import traceback
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
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, psnr, image_to_tensor, normalize
from collections import defaultdict


def log_message(log_filename: str, message: str) -> None:
    """
    Запись логов в файл и вывод сообщений в консоль с меткой даты.

    На входе:
    - log_filename (str): путь к файлу для записи логов;
    - message (str): сообщение, которое будет записано в лог.
    
    На выходе:
    - записывает лог в файл и выводит его в консоль.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(formatted_message + "\n")
    print(formatted_message)


def init_log_file(model_name: str, scale: int, input_folder: str, output_folder: str, mode: str, weights: dict[str, str]) -> str:
    """
    Инициализирует лог-файл и записывает в него основную информацию о запуске.

    На входе:
    - model_name (str): имя модели ("SRCNN", "ESPCN", "SRGAN" или "all" для всех моделей);
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - input_folder (str): путь к папке с изображениями низкого разрешения;
    - output_folder (str): путь к папке, куда будут сохранены результаты;
    - mode (str): режим работы (test - тестирование, вычисляются метрики PSNR, SSIM, NIQE; real - реальный, вычисляется NIQE);
    - weights (dict[str, str]): словарь, содержащий пути к весам для каждой модели.

    Возвращает:
    - str: путь к созданному лог-файлу.
    """
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/upscale_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_x{scale}.txt"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()
    log_message(log_filename, f"Device: {device_name}")
    python_version = platform.python_version()
    log_message(log_filename, f"Programming language: Python {python_version}")
    if model_name == "all":
        for k, v in weights.items():
            log_message(log_filename, f"Weights ({k}): {v}")
    else:
        log_message(log_filename, f"Weights ({model_name}): {weights.get(model_name)}")
    log_message(log_filename, f"Data for upscaling: {input_folder}")
    log_message(log_filename, f"Results: {output_folder}")
    log_message(log_filename, "=== Starting upscaling images ===")
    log_message(log_filename, f"Parameters: \n- Model: {model_name}\n- Scaling factor: {scale}\n- Mode: {mode}")
    return log_filename


def process_srgann(model_name: str, input_folder: str, output_folder: str, model, scale: int, device, niqe_model, mode: str, log_filename: str, metrics_dict: dict) -> None:
    """
    Учеличение разрешения изображений с помощью модели SRGAN.

    На входе:
    - model_name (str): имя модели - SRGAN;
    - input_folder (str): директория с изображениями для увеличения;
    - output_folder (str): директория для сохранения результата увеличения разрешения;
    - model (models.Generator): веса модели;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - device (torch.device): платформа, на которой происходит инференс;
    - niqe_model (pyiqa.models): модель NIQE для вычисления метрики;
    - mode (str): режим работы (test - тестирование, вычисляются метрики PSNR, SSIM, NIQE; real - реальный, вычисляется NIQE);
    - log_filename (str):  путь к файлу для записи логов;
    - metrics_dict (dict): словарь для сохранения метрик по каждому изображению.

    На выходе:
    - изображения с увеличенным разрешением в папке output_folder.
    """
    psnr_list = []  # Список значений PSNR
    ssim_list = []  # Список значений SSIM
    niqe_list = []  # Список значений NIQE

    if os.path.isfile(input_folder):
        # Обрабатываем одиночное изображение
        image_paths = [input_folder]
    else:
        # Обрабатываем все изображения в папке
        image_paths = [
            os.path.join(input_folder, f) 
            for f in os.listdir(input_folder) 
            if os.path.isfile(os.path.join(input_folder, f))
        ]

    for image_file in image_paths:
        try:
            image_name = os.path.basename(image_file)
            # В режиме test обрабатываем только изображения с меткой LR
            if mode == "test" and "LR" not in image_name:
                continue

            if os.path.isfile(input_folder):
                image_file = input_folder
            else:
                image_file = os.path.join(input_folder, image_file)
            lr_image = Image.open(image_file).convert('RGB')  # Открываем изображение в формате RGB
            lr_image_tensor = image_to_tensor(lr_image).to(device)  # Преобразуем его в тензор

            # Получаем SR изображение с помощью модели (без градиентов)
            with torch.no_grad():
                sr_image = model(lr_image_tensor)

            # Преобразуем SR изображение в PIL формата RGB
            sr_image_np = sr_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            sr_image_np = np.clip(sr_image_np * 255.0, 0, 255).astype(np.uint8)  # Масштабируем значения в диапазон [0, 255]

            output_image = Image.fromarray(sr_image_np)  # Преобразуем в PIL изображение

            # Сохранение результата
            name, ext = os.path.splitext(image_name)
            output_image_path = os.path.join(output_folder, f"{name}_{model_name}_x{scale}{ext}")
            output_image.save(output_image_path)

            # Ключ для словаря метрик (зависит от режима)
            image_key = image_name if mode == "real" else image_name.replace("LR", "HR")
            entry = {"model": model_name}

            # Вычисление PSNR и SSIM только в режиме test
            if mode == "test":
                hr_image_file = image_file.replace('LR', 'HR')
                if os.path.exists(hr_image_file):
                    hr_image = Image.open(hr_image_file).convert('RGB')
                    hr_image_np = np.array(hr_image).astype(np.float32)
                    sr_image_np_float = sr_image_np.astype(np.float32)

                    # Переводим RGB -> YCbCr и берём только Y-канал
                    hr_ycbcr = convert_rgb_to_ycbcr(hr_image_np)
                    sr_ycbcr = convert_rgb_to_ycbcr(sr_image_np_float)

                    hr_y = hr_ycbcr[..., 0]
                    sr_y = sr_ycbcr[..., 0]

                    # Нормализуем и переводим в тензоры
                    hr_y_tensor = torch.from_numpy(hr_y / 255.).unsqueeze(0).unsqueeze(0).to(device)
                    sr_y_tensor = torch.from_numpy(sr_y / 255.).unsqueeze(0).unsqueeze(0).to(device)

                    # PSNR и SSIM по Y-каналу
                    psnr_value = psnr(sr_y_tensor, hr_y_tensor)
                    psnr_list.append(psnr_value)
                    log_message(log_filename, f'PSNR for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {psnr_value:.2f} dB')

                    ssim_value = ssim(hr_y, sr_y, data_range=255.0)
                    ssim_list.append(ssim_value)
                    log_message(log_filename, f'SSIM for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {ssim_value:.4f}')

                    entry["psnr"] = psnr_value
                    entry["ssim"] = ssim_value
                else:
                    log_message(log_filename, f'HR image for {image_name} not found. Skipping PSNR/SSIM calculation.')  # Логируем ошибку

            # NIQE (всегда вычисляется)
            niqe_value = niqe_model(output_image)
            niqe_list.append(niqe_value.item())
            log_message(log_filename, f'NIQE for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {niqe_value.item():.4f}')  # Логируем NIQE

            entry["niqe"] = niqe_value.item()
            metrics_dict[image_key].append(entry)

        except Exception as e:
            log_message(log_filename, f"Error processing {image_name}: {str(e)}\n{traceback.format_exc()}")  # Логируем ошибку

    # Вычисление медианных метрик
    try:
        median_psnr = statistics.median(psnr_list) if psnr_list else None
        median_ssim = statistics.median(ssim_list) if ssim_list else None
        median_niqe = statistics.median(niqe_list) if niqe_list else None

        # Логируем медианные метрики
        if mode == "test":
            log_message(log_filename, f'Median PSNR: {median_psnr:.2f} dB')
            log_message(log_filename, f'Median SSIM: {median_ssim:.4f}')
        log_message(log_filename, f'Median NIQE: {median_niqe:.4f}')
    
    except Exception as e:
        log_message(log_filename, f"Error calculating median metrics: {str(e)}\n{traceback.format_exc()}")  # Логируем ошибку 


def upscale_image(model_name: str, weights_file: str, input_folder: str, output_folder: str, scale: int, mode: str, log_filename: str, metrics_dict: dict):
    """
    Основная функция для увеличения разрешения изображений.

    На входе:
    - model_name (str): имя модели (SRCNN, ESPCN, SRGAN);
    - weights_file (str): путь до файла с весами модели;
    - input_folder (str): директория с изображениями для увеличения;
    - output_folder (str): директория для сохранения результата увеличения разрешения;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - mode (str): мод инференса (test - тестирование, вычисляются метрики PSNR, SSIM, NIQE; real - реальный, вычисляется NIQE);
    - log_filename (str):  путь к файлу для записи логов;
    - metrics_dict (dict): словарь для сохранения метрик по каждому изображению.

    На выходе:
    - изображения с увеличенным разрешением в папке output_folder.
    """
    try:
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Выбор и инициализация модели
        if model_name == "SRCNN":
            model = SRCNN().to(device)
        elif model_name == "ESPCN":
            model = ESPCN(scale_factor=scale).to(device)
        elif model_name == "SRGAN":
            model = Generator(scale).to(device)
            model.load_state_dict(torch.load(weights_file))
            model.eval()  # Переводим модель в режим инференса

        # Загрузка весов для SRCNN и ESPCN
        if model_name in ["SRCNN", "ESPCN"]:
            state_dict = model.state_dict()
            for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
        model.eval()  # Переводим модель в режим инференса

        # Инициализация модели для NIQE
        niqe_model = pyiqa.create_metric('niqe')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        psnr_list = []  # Список значений PSNR
        ssim_list = []  # Список значений SSIM
        niqe_list = []  # Список значений NIQE

        if model_name == "SRGAN":
            process_srgann(model_name, input_folder, output_folder, model, scale, device, niqe_model, mode, log_filename, metrics_dict)
            return
        
    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}\n{traceback.format_exc()}") # Логируем ошибку
        return

    if os.path.isfile(input_folder):
        # Обрабатываем одиночное изображение
        image_paths = [input_folder]
    else:
        # Обрабатываем все изображения в папке
        image_paths = [
            os.path.join(input_folder, f) 
            for f in os.listdir(input_folder) 
            if os.path.isfile(os.path.join(input_folder, f))
        ]

    for image_file in image_paths:
        try:
            image_name = os.path.basename(image_file)
            # В режиме test обрабатываем только изображения с меткой LR
            if mode == "test" and "LR" not in image_name:
                continue

            if os.path.isfile(input_folder):
                image_file = input_folder
            else:
                image_file = os.path.join(input_folder, image_file)
            image = pil_image.open(image_file).convert('RGB')  # Преобразуем в RGB

            # Для SRCNN заранее увеличиваем изображение с помощью бикубической интерполяции
            if model_name == "SRCNN":
                image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

            # Перевод изображения в формат YCbCr и выделение Y-канала
            image_np = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image_np)
            y = ycbcr[..., 0] / 255.
            y = torch.from_numpy(y).to(device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)    # Предсказание модели, ограничение диапазона [0, 1]

            entry = {"model": model_name}

            # PSNR и SSIM (только в тестовом режиме)
            if mode == "test":
                hr_image_file = image_file.replace('LR', 'HR')
                if os.path.exists(hr_image_file):
                    hr_image = pil_image.open(hr_image_file).convert('RGB')
                    hr_ycbcr = convert_rgb_to_ycbcr(np.array(hr_image).astype(np.float32))
                    hr_y = hr_ycbcr[..., 0] / 255.
                    hr_y = torch.from_numpy(hr_y).to(device).unsqueeze(0).unsqueeze(0)

                    # preds_resized = torch.nn.functional.interpolate(preds, size=(hr_y.size(2), hr_y.size(3)), mode='bicubic', align_corners=False)

                    # PSNR
                    # psnr_value = psnr(hr_y, preds_resized)
                    psnr_value = psnr(hr_y, preds)
                    psnr_list.append(psnr_value)
                    log_message(log_filename, f'PSNR for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {psnr_value:.2f} dB') # Логируем PSNR

                    # SSIM
                    # preds_np = preds_resized.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                    preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                    hr_np = hr_y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

                    ssim_value = ssim(hr_np, preds_np, data_range=255.0)
                    ssim_list.append(ssim_value)
                    log_message(log_filename, f'SSIM for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {ssim_value:.4f}') # Логируем SSIM

                    entry["psnr"] = psnr_value
                    entry["ssim"] = ssim_value
                else:
                    log_message(log_filename, f'HR image for {image_name} not found. Skipping PSNR and SSIM calculation.') # Логируем ошибку

            # NIQE
            preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            preds_image = pil_image.fromarray(preds_np.astype(np.uint8))
            niqe_value = niqe_model(preds_image)
            niqe_list.append(niqe_value.item())
            log_message(log_filename, f'NIQE for {image_name.replace(".", f"_{model_name}_x{scale}.")}: {niqe_value.item():.4f}') # Логируем NIQE

            entry["niqe"] = niqe_value.item()
            image_key = image_name if mode == "real" else image_name.replace("LR", "HR")
            metrics_dict[image_key].append(entry)

            # Конвертация YCbCr в RGB (восстановление цветного изображения)
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
            name, ext = os.path.splitext(image_name)
            output_image_path = os.path.join(output_folder, f"{name}_{model_name}_x{scale}{ext}")
            output_image.save(output_image_path)
      
        except Exception as e:
            log_message(log_filename, f"Error processing {image_name}: {str(e)}\n{traceback.format_exc()}") # Логируем ошибки

    # Вычисление медианных метрик
    try:
        median_psnr = statistics.median(psnr_list) if psnr_list else None
        median_ssim = statistics.median(ssim_list) if ssim_list else None
        median_niqe = statistics.median(niqe_list) if niqe_list else None

        # Логируем медианные метрики
        if mode == "test":
            log_message(log_filename, f'Median PSNR: {median_psnr:.2f} dB')
            log_message(log_filename, f'Median SSIM: {median_ssim:.4f}')
        log_message(log_filename, f'Median NIQE: {median_niqe:.4f}')
            
    except Exception as e:
        log_message(log_filename, f"Error calculating median metrics: {str(e)}\n{traceback.format_exc()}")  # Логируем ошибку


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale images.")
    parser.add_argument("--model", default="SRCNN", type=str, choices=["SRCNN", "ESPCN", "SRGAN", "a"], help="Model name (SRCNN, ESPCN, SRGAN) or 'a' for all")

    parser.add_argument("--weights_srcnn", type=str, help="Path to SRCNN model weights")
    parser.add_argument("--weights_espcn", type=str, help="Path to ESPCN model weights")
    parser.add_argument("--weights_srgan", type=str, help="Path to SRGAN model weights")

    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    parser.add_argument("--scale", default=2, type=int, choices=[2, 3, 4], help="Upscaling factor")
    parser.add_argument("--mode", type=str, required=True, choices=["test", "real"],
                        help="Mode: 'test' with HR images or 'real' for LR only")
    args = parser.parse_args()

    all_models = ["SRCNN", "ESPCN", "SRGAN"]
    weights_dict = {
        "SRCNN": args.weights_srcnn,
        "ESPCN": args.weights_espcn,
        "SRGAN": args.weights_srgan
    }
    metrics_dict = defaultdict(list)

    # Если указано использовать все модели
    if args.model == "a":
        try:
            # Проверка наличия весов для всех моделей
            for m in all_models:
                if not weights_dict[m]:
                    print(f"Missing weights for model: {m}")
                    exit(1)

            # Инициализация файла логов
            log_filename = init_log_file("all", args.scale, args.input, args.output, args.mode, weights_dict)

            # Применение каждой модели по очереди
            for model_name in all_models:
                log_message(log_filename, f"\n=== Upscaling with {model_name} ===") # Логируем имя модели
                model_output = os.path.join(args.output, model_name)
                os.makedirs(model_output, exist_ok=True)
                try:
                    upscale_image(model_name, weights_dict[model_name], args.input, model_output, args.scale, args.mode, log_filename, metrics_dict)
                except Exception as e:
                    log_message(log_filename, f"Error during {model_name}: {str(e)}\n{traceback.format_exc()}") # Логируем ошибку

            log_message(log_filename, f"\nSelecting best images per LR file:")  # Логируем прцесс выбора лучшего изображения

            for image_name, metric_list in metrics_dict.items():
                if not metric_list:
                    continue

                best_model = None
                best_score = float("-inf")

                if args.mode == "test":
                    # Расчет гармонического среднего по метрикам PSNR, SSIM и NIQE (обратный)
                    psnrs = [m.get("psnr", 0) for m in metric_list]
                    ssims = [m.get("ssim", 0) for m in metric_list]
                    inv_niqes = [1.0 / m.get("niqe", 1e-6) for m in metric_list]

                    for m in metric_list:
                        # Нормализация метрик
                        norm_psnr = normalize(m.get("psnr", 0), min(psnrs), max(psnrs))
                        norm_ssim = normalize(m.get("ssim", 0), min(ssims), max(ssims))
                        norm_inv_niqe = normalize(1.0 / m.get("niqe", 1e-6), min(inv_niqes), max(inv_niqes))

                        # Гармоническое среднее нормализованных значений
                        if norm_psnr > 0 and norm_ssim > 0 and norm_inv_niqe > 0:
                            hmean = 3.0 / (1.0 / norm_psnr + 1.0 / norm_ssim + 1.0 / norm_inv_niqe)
                        else:
                            hmean = 0.0

                        # Обновление лучшей модели
                        if hmean > best_score:
                            best_score = hmean
                            best_model = m["model"]

                    log_message(log_filename, f"{image_name} → {image_name.replace('HR', 'SR')}_{best_model}_x{args.scale} (Harmonic = {best_score:.4f})") # Логируем лучший результат

                else:   # режим real - выбор модели с наименьшим NIQE
                    best_entry = min(metric_list, key=lambda x: x.get("niqe", float("inf")))
                    best_model = best_entry["model"]
                    niqe_val = best_entry.get("niqe", -1)
                    log_message(log_filename, f"{image_name} → {image_name}_{best_model}_x{args.scale} (NIQE = {niqe_val:.4f})")

        except Exception as e:
            log_message(log_filename, f"An error occurred: {str(e)}\n{traceback.format_exc()}") # Логируем ошибку

    else:   # Обработка случая, когда указана только одна модель
        model_weights = weights_dict[args.model]
        if not model_weights:
            print(f"Weights not provided for model {args.model}")
            exit(1)

        # Инициализация лог-файла
        log_filename = init_log_file(args.model, args.scale, args.input, args.output, args.mode, weights_dict)
        log_message(log_filename, f"\n=== Upscaling with {args.model} ===")

        # Создание директории для результатов
        model_output = os.path.join(args.output, args.model)
        os.makedirs(model_output, exist_ok=True)

        upscale_image(args.model, model_weights, args.input, model_output, args.scale, args.mode, log_filename, metrics_dict)
