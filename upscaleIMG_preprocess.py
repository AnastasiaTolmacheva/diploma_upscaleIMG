import argparse
import glob
import h5py
import numpy as np
import os
import datetime
import PIL.Image as pil_image
from utils import convert_rgb_to_y


def log_message(log_filename, message):
    """
    Функция для записи логов в файл и вывода сообщений в консоль.

    Входные параметры:
    - log_filename (str): Путь к файлу для записи логов.
    - message (str): Сообщение, которое будет записано в лог.

    Выход:
    Лог записывается в файл и выводится в консоль в зависимости от уровня логирования.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"

    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(formatted_message + "\n")

    print(formatted_message)


def process_images(args, is_train):
    """
    Функция для предобработки изображений моделей SRCNN или ESPCN.

    Входные параметры:
    - args (Namespace): Аргументы командной строки (определяются через argparse).
    - is_train (bool): Флаг, указывающий на режим обучения (True - обучение, False - оценка/валидация).

    Выход:
    - None. Обрабатывает изображения и сохраняет их в HDF5 файл.
    """

    os.makedirs("logs", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    # Формируем имя H5-файла: {dataset}_{model}_x{scale}.h5
    dataset_name = os.path.basename(os.path.normpath(args.images_dir))  # Получаем название папки
    h5_filename = f"{dataset_name}_{args.model}_x{args.scale}.h5"
    output_path = os.path.join("datasets", h5_filename)

    # Формирование имени лог-файла
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"logs/preprocess_{args.model}_{start_time}.txt"

    h5_file = h5py.File(output_path, 'w')

    # Логирование параметров
    log_message(log_filename, f"Parameters: \n- Model: {args.model}\n- Images directory: {args.images_dir}\n"
                          f"- Prepared file name: {output_path}\n- Scaling factor: {args.scale}\n"
                          f"- Patch size: {args.patch_size}\n- Stride: {args.stride}\n- Evaluation: {args.eval}")
    log_message(log_filename, "=== Starting image preprocessing ===")
    
    try:
        image_paths = sorted(glob.glob('{}/*'.format(args.images_dir)))
        log_message(log_filename, f"{len(image_paths)} images found for processing.")

        # Создаем группы для хранения данных
        if not is_train:
            lr_group = h5_file.create_group('lr')
            hr_group = h5_file.create_group('hr')

        if is_train:
            lr_patches = []
            hr_patches = []

        for i, image_path in enumerate(image_paths):
            log_message(log_filename, f"Processing image {i + 1}/{len(image_paths)}: {image_path}")

            # 1. Открываем изображение в RGB
            hr = pil_image.open(image_path).convert('RGB')

            # 2. Приводим размеры к ближайшим, кратным `scale`
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

            # 3. Создаем изображение низкого разрешения (LR)
            lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)

            # 4. В SRCNN после уменьшения мы снова увеличиваем LR-изображение до исходного размера
            if args.model =="SRCNN":
                lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

            # 5. Преобразуем изображения в массивы numpy (тип float32)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)

            # 6. Извлекаем Y-канал из YCbCr
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            if is_train:
                # 7. Разбиваем изображение на патчи
                patch_count = 0
                for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
                    for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                        lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])

                        if args.model == "SRCNN":
                            hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
                        elif args.model == "ESPCN":
                            hr_patches.append(
                                hr[i * args.scale:i * args.scale + args.patch_size * args.scale,
                                j * args.scale:j * args.scale + args.patch_size * args.scale])
                        patch_count += 1

                log_message(log_filename, f"{patch_count} patches created from {image_path}")

            else:
                # 8. Если режим eval (валидация), сохраняем полные изображения
                lr_group.create_dataset(str(i), data=lr)
                hr_group.create_dataset(str(i), data=hr)

        if is_train:
            # 9. Сохраняем патчи в HDF5
            h5_file.create_dataset('lr', data=lr_patches)
            h5_file.create_dataset('hr', data=hr_patches)

        h5_file.close()
        log_message(log_filename, f"Data successfully saved to {output_path}")
        log_message(log_filename, "Preprocessing completed")

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess images for SRCNN or ESPCN.")
    parser.add_argument("--model", type=str, choices=["SRCNN", "ESPCN"], required=True, help="Model name (SRCNN, ESPCN)")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--scale", default=2, type=int, choices=[2, 3, 4], help="Scaling factor for super-resolution")
    parser.add_argument("--patch_size", type=int, default=33, help="Size of image patches")
    parser.add_argument("--stride", type=int, default=14, help="Stride for patch extraction")
    parser.add_argument("--eval", action='store_true', help="Enable evaluation mode (save full images instead of patches)")

    args = parser.parse_args()

    process_images(args, not args.eval)
