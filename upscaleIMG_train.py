import argparse
import os
import traceback
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import SRCNN, ESPCN, Generator, Discriminator
from datasets import TrainDatasetSRCNN, EvalDatasetSRCNN, TrainDatasetSRGAN, EvalDatasetSRGAN, TrainDatasetESPCN, EvalDatasetESPCN
from utils import AverageMeter, psnr, convert_rgb_to_y
from loss import GeneratorLoss
import platform
import matplotlib.pyplot as plt
import datetime
import time


def log_message(log_filename: str, message: str) -> None:
    """
    Запись логов в файл и вывод сообщений в консоль с меткой даты.

    На входе:
    - log_filename (str): путь к файлу для записи логов;
    - message (str): сообщение, которое будет записано в лог.

    На выходе:
    - записывает лог в файл и выводит его в консоль.
    """
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(formatted_message + "\n")
    print(formatted_message)


def plot(model: str, train_losses: list[float], val_losses: list[float], train_psnrs: list[float], val_psnrs: list[float], 
         scale: int, epochs: int, batch_size: int, log_filename: str) -> None:
    """
    Функция для построения графиков с тренировочными и валидационными значениями функции потерь и PSNR по эпохам.

    На входе:
    - model (str): название модели (SRCNN, EPSCN, SRGAN);
    - train_losses (list): значения функции потерь на тренировочной выборке по эпохам;
    - val_losses (list): значения функции потерь на валидационной выборке по эпохам;
    - train_psnrs (list): значения PSNR на тренировочной выборке по эпохам;
    - val_psnrs (list): значения PSNR на валидационной выборке по эпохам;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - epochs (int): количество эпох;
    - batch_size (int): размер батча;
    - log_filename (str): путь к файлу для записи логов.

    На выходе:
    - график в директории logs в формате PNG.
    - логи с подробной информацией о работе в директории logs.
    """
    # Формируем имя файла с графиком
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"logs/graph_{model}_{timestamp}.png"

    plt.figure(figsize=(12, 6))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", color="red", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # График PSNR
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_psnrs) + 1), train_psnrs, label="Train PSNR", color="green")
    plt.plot(range(1, len(val_psnrs) + 1), val_psnrs, label="Val PSNR", color="orange", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR per Epoch")
    plt.legend()

    # Заголовок и сохранение
    plt.suptitle(f"{model} Training\n{timestamp}\nScale={scale}, Epochs={epochs}, Batch={batch_size}")
    plt.subplots_adjust(top=0.85)
    plt.savefig(plot_filename)
    plt.close()

    log_message(log_filename, f"Graph saved: {plot_filename}")  # Логируем результат


def train_srcnn(train_data: str, val_data: str, output: str, scale: int, epochs: int, batch_size: int) -> None:
    """
    Обучение модели SRCNN на заданных тренировочных и валидационных данных.

    На входе:
    - train_data (str): путь к файлу .h5 с тренировочными изображениями (предобработанными);
    - val_data (str): путь к файлу .h5 с валидационными изображениями (предобработанными);
    - output (str): директория для сохранения весов модели;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - epochs (int): количество эпох;
    - batch_size (int): размер батча.

    На выходе:
    - веса модели, сохраненные в указанную папку output после каждой эпохи;
    - логи с подробной информацией о работе в директории logs;
    - графики обучения и валидации с потерями и PSNR в формате PNG.
    """
    if not os.path.exists(output):
        os.makedirs(output)
    
    try:
        # Настройки устройства (GPU или CPU)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Логирование системной информации
        python_version = platform.python_version()
        log_filename = f"logs/train_SRCNN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        log_message(log_filename, f"Training platform: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Train data: {train_data}")
        log_message(log_filename, f"Validation data: {val_data}")
        log_message(log_filename, f"Output folder: {output}")

        # Логирование начала процесса обучения
        log_message(log_filename, "=== Starting model training ===")
        log_message(log_filename, f"Parameters: \n- Model: SRCNN\n- Scaling factor: {scale}\n- Epochs: {epochs}\n- Batch size: {batch_size}")

        # Инициализация модели, функции потерь и оптимизатора
        model = SRCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': 1e-5}
        ], lr=1e-4)

        # Загрузка данных для обучения и валидации
        train_dataset = TrainDatasetSRCNN(train_data)
        eval_dataset = EvalDatasetSRCNN(val_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=1)

        train_losses = []   # Список со значениями функции потерь на тренировочных данных
        val_losses = []    # Список со значениями функции потерь на валидационных данных
        train_psnrs = []    # Список со значениями PSNR на тренировочных данных
        val_psnrs = []  # Список со значениями PSNR на валидационных данных
        epoch_times = []    # Список с длительностями эпох

        # Цикл обучения и валидации
        for epoch in range(1, epochs + 1):
            model.train()  # Переводим модель в режим тренировки
            start_time = time.time() # Начинаем замер времени выполнения эпохи
            epoch_losses_meter = AverageMeter()  # Средний показатель потерь для эпохи
            epoch_psnr_meter = AverageMeter()  # Средний показатель PSNR для эпохи
            
            # Цикл по данным обучения (прогресс вычисляется по числу батчей)
            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch: {epoch}/{epochs}')
                for data in train_loader:
                    inputs, labels = data   # Входные изображения LR и HR
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)   # Прогон данных через модель, получение результата
                    loss = criterion(preds, labels)
                    epoch_losses_meter.update(loss.item(), len(inputs)) # Обновление средней потери
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    # Обновляем веса модели

                    # Подсчет PSNR на тренировочных данных
                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    epoch_psnr_meter.update(psnr_value, len(inputs))

                    # Обновляем прогресс-бар
                    train_bar.set_postfix(loss=f'{epoch_losses_meter.avg:.6f}', psnr=f'{psnr_value:.2f} dB')
                    train_bar.update(1)
            
            # Сохранение значений функциии потерь и PSNR после завершения эпохи
            end_time = time.time()
            train_losses.append(epoch_losses_meter.avg)
            train_psnrs.append(epoch_psnr_meter.avg.detach().cpu().numpy())
            epoch_times.append(end_time - start_time)

            # Сохранение модели после каждой эпохи
            torch.save(model.state_dict(), os.path.join(output, f'srcnn_epoch_{epoch}_scale_x{scale}.pth'))

            model.eval()  # Переводим модель в режим валидации
            val_loss_meter = AverageMeter()  # Средний показатель потерь для валидации
            val_psnr_meter = AverageMeter()  # Средний показатель PSNR для валидации

            # Цикл по данным валидации
            with tqdm(total=len(eval_loader)) as val_bar:
                val_bar.set_description(f'Validation {epoch}/{epochs}')
                for data in eval_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    val_loss = criterion(preds, labels)
                    val_loss_meter.update(val_loss.item(), len(inputs))

                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    val_psnr_meter.update(psnr_value, len(inputs))

                    val_bar.set_postfix(psnr=f'{psnr_value:.2f} dB', val_loss=f'{val_loss_meter.avg:.6f}')
                    val_bar.update(1)

            # Сохранение значений функциии потерь и PSNR
            val_losses.append(val_loss_meter.avg)
            val_psnrs.append(val_psnr_meter.avg.detach().cpu().numpy())

            # Логируем информацию о каждой эпохе
            log_message(log_filename, f"Epoch {epoch}: Train Loss = {epoch_losses_meter.avg:.6f}, Train PSNR = {epoch_psnr_meter.avg:.2f} dB, Val Loss = {val_loss_meter.avg:.6f}, Val PSNR = {val_psnr_meter.avg:.2f} dB")

        # Создаем график
        plot("SRCNN", train_losses, val_losses, train_psnrs, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}\n{traceback.format_exc()}") # Логируем, если произошла ошибка


def train_espcn(train_data: str, val_data: str, output: str, scale: int, epochs: int, batch_size: int) -> None:
    """
    Обучение модели ESPCN на заданных тренировочных и валидационных данных.

    На входе:
    - train_data (str): путь к файлу .h5 с тренировочными изображениями (предобработанными);
    - val_data (str): путь к файлу .h5 с валидационными изображениями (предобработанными);
    - output (str): директория для сохранения весов модели;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - epochs (int): количество эпох;
    - batch_size (int): размер батча.

    На выходе:
    - веса модели, сохраненные в указанную папку output после каждой эпохи;
    - логи с подробной информацией о работе в директории logs;
    - графики обучения и валидации с потерями и PSNR в формате PNG.
    """
    if not os.path.exists(output):
        os.makedirs(output)
    
    try:
        # Настройки устройства (GPU или CPU)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        python_version = platform.python_version()

        # Логирование системной информации
        log_filename = f"logs/train_ESPCN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Train data: {train_data}")
        log_message(log_filename, f"Validation data: {val_data}")
        log_message(log_filename, f"Output folder: {output}")

        # Логирование начала процесса обучения
        log_message(log_filename, "=== Starting model training ===")
        log_message(log_filename, f"Parameters: \n- Model: ESPCN\n- Scaling factor: {scale}\n- Epochs: {epochs}\n- Batch size: {batch_size}")
        
        # Инициализация модели, функции потерь и оптимизатора
        model = ESPCN(scale_factor=scale).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Загрузка данных для обучения и валидации
        train_dataset = TrainDatasetESPCN(train_data)
        eval_dataset = EvalDatasetESPCN(val_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1)

        train_losses = []   # Список со значениями функции потерь на тренировочных данных
        val_losses = []    # Список со значениями функции потерь на валидационных данных
        train_psnrs = []    # Список со значениями PSNR на тренировочных данных
        val_psnrs = []  # Список со значениями PSNR на валидационных данных
        epoch_times = []    # Список с длительностями эпох

        # Цикл обучения и валидации
        for epoch in range(1, epochs + 1):
            model.train()  # Переводим модель в режим тренировки
            start_time = time.time() # Начинаем замер времени выполнения эпохи
            epoch_losses_meter = AverageMeter()  # Средний показатель потерь для эпохи
            epoch_psnr_meter = AverageMeter()  # Средний показатель PSNR для эпохи

            # Цикл по данным обучения (прогресс вычисляется по числу батчей)
            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch: {epoch}/{epochs}')
                for data in train_loader:
                    inputs, labels = data   # Входные изображения LR и HR
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)   # Прогон данных через модель, получение результата
                    loss = criterion(preds, labels)
                    epoch_losses_meter.update(loss.item(), len(inputs))  # Обновление средней потери
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    # Обновляем веса модели

                    # Подсчет PSNR на тренировочных данных
                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    epoch_psnr_meter.update(psnr_value, len(inputs))

                    # Обновляем прогресс-бар
                    train_bar.set_postfix(loss=f'{epoch_losses_meter.avg:.6f}', psnr=f'{epoch_psnr_meter.avg:.2f} dB')
                    train_bar.update(1)

            # Сохранение значений функциии потерь и PSNR после завершения эпохи
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            train_losses.append(epoch_losses_meter.avg)
            train_psnrs.append(epoch_psnr_meter.avg.detach().cpu().numpy())

            # Сохранение модели после каждой эпохи
            torch.save(model.state_dict(), os.path.join(output, f'espcn_epoch_{epoch}_scale_x{scale}.pth'))

            # Оценка модели на валидационных данных
            model.eval()  # Переводим модель в режим валидации
            val_loss_meter = AverageMeter()  # Средний показатель потерь для валидации
            val_psnr_meter = AverageMeter()  # Средний показатель PSNR для валидации

            # Цикл по данным валидации
            with tqdm(total=len(eval_loader)) as val_bar:
                val_bar.set_description(f'Validation {epoch}/{epochs}')
                for data in eval_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    # Подсчет функции потерь на валидации
                    val_loss = criterion(preds, labels)
                    val_loss_meter.update(val_loss.item(), len(inputs))

                    # Подсчет PSNR на валидации
                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    val_psnr_meter.update(psnr_value, len(inputs))

                    val_bar.set_postfix(psnr=f'{psnr_value:.2f} dB', val_loss=f'{val_loss_meter.avg:.6f}')
                    val_bar.update(1)

            # Сохранение значений функциии потерь и PSNR
            val_losses.append(val_loss_meter.avg)
            val_psnrs.append(val_psnr_meter.avg.detach().cpu().numpy())

            # Логируем информацию о каждой эпохе
            log_message(log_filename, f"Epoch {epoch}: Train Loss = {epoch_losses_meter.avg:.6f}, Train PSNR = {epoch_psnr_meter.avg:.2f} dB, Val Loss = {val_loss_meter.avg:.6f}, Val PSNR = {val_psnr_meter.avg:.2f} dB")

        # Создаем график
        plot("ESPCN", train_losses, val_losses, train_psnrs, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}\n{traceback.format_exc()}") # Логируем, если произошла ошибка


def train_srgan(train_data: str, val_data: str, output: str, scale: int, epochs: int, batch_size: int, crop_size: int) -> None:
    """
    Обучение модели SRGAN на заданных тренировочных и валидационных данных.

    На входе:
    - train_data (str): директория с изображениями для обучения;
    - val_data (str): директория с изображениями для валидации;
    - output (str): директория для сохранения весов модели;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4);
    - epochs (int): количество эпох;
    - batch_size (int): размер батча;
    - crop_size (int): размер обрезанного изображения.

    На выходе:
    - веса модели, сохраненные в указанную папку output после каждой эпохи;
    - логи с подробной информацией о работе в директории logs;
    - графики обучения и валидации с потерями и PSNR в формате PNG.
    """
    if not os.path.exists(output):
        os.makedirs(output)
    
    try:
        # Настройка устройства (GPU или CPU)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Логирование системной информации
        python_version = platform.python_version()
        log_filename = f"logs/train_SRGAN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Train data: {train_data}")
        log_message(log_filename, f"Validation data: {val_data}")
        log_message(log_filename, f"Output folder: {output}")

        # Логирование начала процесса обучения
        log_message(log_filename, "=== Starting model training ===")
        log_message(log_filename, f"Parameters: \n- Model: SRGAN\n- Scaling factor: {scale}\n- Epochs: {epochs}\n- Batch size: {batch_size}\n- Crop size: {crop_size}")

        # Инициализация генератора и дискриминатора, функции потерь и оптимизатора
        netG, netD = Generator(scale).to(device), Discriminator().to(device)
        generator_criterion = GeneratorLoss().to(device)
        optimizerG, optimizerD = optim.Adam(netG.parameters()), optim.Adam(netD.parameters())

        # Загрузка данных для обучения и валидации
        train_set = TrainDatasetSRGAN(train_data, crop_size, upscale_factor=scale)
        val_set = EvalDatasetSRGAN(val_data, upscale_factor=scale)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

        train_losses = []   # Список со значениями функции потерь на тренировочных данных
        val_losses = []    # Список со значениями функции потерь на валидационных данных
        train_psnrs = []    # Список со значениями PSNR на тренировочных данных
        val_psnrs = []  # Список со значениями PSNR на валидационных данных
        epoch_times = []    # Список с длительностями эпох
        
        # Цикл обучения и валидации
        for epoch in range(1, epochs + 1):
            netG.train()    # Переводим генератор в режим обучения
            netD.train()    # Переводим дискриминатор в режим обучения
            start_time = time.time() # Начинаем замер времени выполнения эпохи
            epoch_losses_meter = AverageMeter()  # Средний показатель потерь для эпохи
            epoch_psnr_meter = AverageMeter()  # Средний показатель PSNR для эпохи

            # Цикл по данным обучения (прогресс вычисляется по числу батчей)
            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch {epoch}/{epochs}')
                for data, target in train_loader:
                    curr_batch_size = data.size(0)
                    real_img, z = target.to(device), data.to(device)    # real_img - эталон (HR), z - вход (LR)
                    fake_img = netG(z)  # Генерация изображения высокого разрешения

                    # Обучение генератора
                    optimizerG.zero_grad()
                    g_loss = generator_criterion(netD(fake_img).mean(), fake_img, real_img)
                    g_loss.backward()
                    optimizerG.step()

                    # Обучение дискриминатора
                    real_out, fake_out = netD(real_img).mean(), netD(fake_img.detach()).mean()
                    d_loss = 1 - real_out + fake_out
                    optimizerD.zero_grad()
                    d_loss.backward()
                    optimizerD.step()

                    epoch_losses_meter.update(g_loss.item(), curr_batch_size)

                    # Преобразуем изображения в Y-канал перед вычислением PSNR
                    fake_img_y = convert_rgb_to_y(fake_img)
                    real_img_y = convert_rgb_to_y(real_img)
                    # Вычисление PSNR на Y-канале
                    train_psnr_value = psnr(fake_img_y, real_img_y)
                    epoch_psnr_meter.update(train_psnr_value.cpu().item())

                    train_bar.set_postfix(loss=f'{epoch_losses_meter.avg:.6f}', psnr=f'{epoch_psnr_meter.avg:.2f} dB')
                    train_bar.update(1)

            # Сохранение значений функциии потерь и PSNR после завершения эпохи
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            train_losses.append(epoch_losses_meter.avg)
            train_psnrs.append(epoch_psnr_meter.avg)

            # Цикл по данным валидации
            netG.eval()    # Переводим генератор в режим валидации
            val_loss_meter = AverageMeter()  # Средний показатель потерь для валидации
            val_psnr_meter = AverageMeter()  # Средний показатель PSNR для валидации

            # Цикл по данным валидации
            with torch.no_grad():
                with tqdm(total=len(val_loader)) as val_bar:
                    val_bar.set_description(f'Validation {epoch}/{epochs}')
                    for val_lr, val_hr in val_loader:
                        val_lr, val_hr = val_lr.to(device), val_hr.to(device)
                        sr_img = netG(val_lr)

                        # Подсчет функции потерь
                        val_g_loss = generator_criterion(netD(sr_img).mean(), sr_img, val_hr)
                        val_loss_meter.update(val_g_loss.item())

                        # Преобразуем изображения в Y-канал перед вычислением PSNR на валидации
                        sr_img_y = convert_rgb_to_y(sr_img)
                        val_hr_y = convert_rgb_to_y(val_hr)

                        # Подсчет PSNR на Y-канале
                        val_psnr_value = psnr(sr_img_y, val_hr_y)
                        val_psnr_meter.update(val_psnr_value.cpu().item())

                        val_bar.set_postfix(psnr=f'{val_psnr_meter.avg:.2f} dB', val_loss=f'{val_loss_meter.avg:.6f}')
                        val_bar.update(1)

            # Сохранение значений функциии потерь и PSNR
            val_losses.append(val_loss_meter.avg)
            val_psnrs.append(val_psnr_meter.avg)

            # Логируем информацию о каждой эпохе
            log_message(log_filename, f"Epoch {epoch}: Train Loss = {epoch_losses_meter.avg:.6f}, Train PSNR = {epoch_psnr_meter.avg:.2f} dB, Val Loss = {val_loss_meter.avg:.6f}, Val PSNR = {val_psnr_meter.avg:.2f} dB")

            # Сохранение весов генератора и дискриминатора после каждой эпохи
            torch.save(netG.state_dict(), os.path.join(output, f'srgan_G_epoch_{epoch}_scale_x{scale}.pth'))
            torch.save(netD.state_dict(), os.path.join(output, f'srgan_D_epoch_{epoch}_scale_x{scale}.pth'))

        # Создаем график
        plot("SRGAN", train_losses, val_losses, train_psnrs, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}\n{traceback.format_exc()}")  # Логируем, если произошла ошибка


def check_scale_in_filename(data_path: str, scale: int) -> bool:
    """
    Проверяет, что в пути данных присутствует ожидаемый скейл-фактор в имени файла (x2, x3, x4).

    На входе:
    - data_path (str): путь до файла для обучения или валидации;
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4).

    На выходе:
    - bool: булевое значение.
    """
    scale_str = f"x{scale}"
    file_name = os.path.basename(data_path)  # Извлекаем имя файла
    if scale_str not in file_name:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument("--model", default="SRCNN", type=str, choices=["SRCNN", "ESPCN", "SRGAN"], help="Model name (SRCNN, ESPCN, SRGAN)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--scale", default=2, type=int, choices=[2, 3, 4], help="Upscaling factor (2, 3, 4)")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--crop_size", default=88, type=int, help="Crop size for training")
    
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.crop_size <= 0:
        print("Invalid value: epochs, batch_size and crop_size must be more than 0.")
    else:
        # Проверка, что в данных указан правильный скейл только для SRCNN и ESPCN
        if args.model in ["SRCNN", "ESPCN"]:
            if not check_scale_in_filename(args.train_data, args.scale) or not check_scale_in_filename(args.val_data, args.scale):
                print(f"Error: The scale factor '{args.scale}' is not present in the filenames of train daat or val data.")
            else:
                if args.model == "SRCNN":
                    train_srcnn(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size)
                elif args.model == "ESPCN":
                    train_espcn(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size)
        elif args.model == "SRGAN":
            train_srgan(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size, args.crop_size)
