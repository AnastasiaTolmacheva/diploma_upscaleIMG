import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import SRCNN, ESPCN, Generator, Discriminator
from datasets import TrainDatasetSRCNN, EvalDatasetSRCNN, TrainDatasetSRGAN, EvalDatasetSRGAN, TrainDatasetESPCN, EvalDatasetESPCN
from utils import AverageMeter, psnr
from loss import GeneratorLoss
import platform
import matplotlib.pyplot as plt
import datetime
import time


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


def plot(model, train_losses, val_psnrs, scale, epochs, batch_size, log_filename):
    """Строит и сохраняет график обучения"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"logs/graph_{model}_{timestamp}.png"

    # Сохранение графика
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_psnrs) + 1), val_psnrs, label="Val PSNR", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR per Epoch")
    plt.legend()

    plt.suptitle(f"{model} Training\n{timestamp}\nScale={scale}, Epochs={epochs}, Batch={batch_size}")
    plt.subplots_adjust(top=0.75)

    plt.savefig(plot_filename)
    plt.close()

    log_message(log_filename, f"Graph saved: {plot_filename}")


def train_srcnn(train_data, val_data, output, scale, epochs, batch_size):
    """
    Функция для обучения модели SRCNN

    На входе:
    - train_data (str): директория с изображениями для обучения
    - val_data (str): директория с изображениями для валидации
    - output (str): директория для сохранения весов модели
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4)
    - epochs (int): количество эпох
    - batch_size (int): размер батча
    """
    if not os.path.exists(output):
        os.makedirs(output)
    
    try:
        # Настройки устройства (GPU или CPU)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        python_version = platform.python_version()

        # Логирование системной информации
        log_filename = f"logs/train_SRCNN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        log_message(log_filename, f"Training platform: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Train data: {train_data}")
        log_message(log_filename, f"Validation data: {val_data}")

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

        train_losses = []   # Словарь со значениями функции потерь на тренировочных данных
        val_psnrs = []  # Словарь со значениями PSNR на валидационных данных
        epoch_times = []    # Словарь с длительностями эпох

        # Основной цикл обучения
        for epoch in range(1, epochs + 1):
            model.train()  # Переводим модель в режим тренировки
            start_time = time.time() # Замеряем время каждой эпохи
            epoch_losses = AverageMeter()  # Средний показатель потерь для эпохи
            
            # Цикл по данным обучения
            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch: {epoch}/{epochs}')
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)
                    loss = criterion(preds, labels)
                    epoch_losses.update(loss.item(), len(inputs))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_bar.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                    train_bar.update(1)
            
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            train_losses.append(epoch_losses.avg)

            # Сохранение модели после каждой эпохи
            torch.save(model.state_dict(), os.path.join(output, f'srcnn_epoch_{epoch}_scale_x{scale}.pth'))
            
            # Оценка модели на валидационных данных
            model.eval()
            epoch_psnr = AverageMeter()
            with tqdm(total=len(eval_loader)) as val_bar:
                val_bar.set_description(f'Validation {epoch}/{epochs}')
                for data in eval_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    # Подсчет PSNR
                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    epoch_psnr.update(psnr_value, len(inputs))

                    # Обновление прогресс-бара для валидации
                    val_bar.set_postfix(psnr=f'{psnr_value:.2f} dB')
                    val_bar.update(1)

            val_psnrs.append(epoch_psnr.avg.cpu().numpy())  # Переводим результат на CPU и преобразуем в numpy для работы с графиками
            log_message(log_filename, f"Epoch {epoch}: Loss = {epoch_losses.avg:.6f}, PSNR = {epoch_psnr.avg:.2f} dB")

        # Сохраняем лог и график
        plot("SRCNN", train_losses, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}")


def train_espcn(train_data, val_data, output, scale, epochs, batch_size):
    """
    Функция для обучения модели ESPCN

    На входе:
    - train_data (str): директория с изображениями для обучения
    - val_data (str): директория с изображениями для валидации
    - output (str): директория для сохранения весов модели
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4)
    - epochs (int): количество эпох
    - batch_size (int): размер батча
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

        train_losses = []   # Словарь со значениями функции потерь на тренировочных данных
        val_psnrs = []  # Словарь со значениями PSNR на валидационных данных
        epoch_times = []    # Словарь с длительностями эпох

        for epoch in range(1, epochs + 1):
            model.train()   
            start_time = time.time()
            epoch_losses = AverageMeter()

            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch: {epoch}/{epochs}')
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)
                    loss = criterion(preds, labels)
                    epoch_losses.update(loss.item(), len(inputs))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_bar.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                    train_bar.update(1)

            end_time = time.time()
            epoch_times.append(end_time - start_time)
            train_losses.append(epoch_losses.avg)

            # Сохранение модели после каждой эпохи
            torch.save(model.state_dict(), os.path.join(output, f'espcn_epoch_{epoch}_scale_x{scale}.pth'))

            # Оценка модели на валидационных данных
            model.eval()
            epoch_psnr = AverageMeter()
            with tqdm(total=len(eval_loader)) as val_bar:
                val_bar.set_description(f'Validation {epoch}/{epochs}')
                for data in eval_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    # Подсчет PSNR
                    psnr_value = psnr(preds.cpu(), labels.cpu())
                    epoch_psnr.update(psnr_value, len(inputs))

                    # Обновление прогресс-бара для валидации
                    val_bar.set_postfix(psnr=f'{psnr_value:.2f} dB')
                    val_bar.update(1)

            val_psnrs.append(epoch_psnr.avg.cpu().numpy())  # Переводим результат на CPU и преобразуем в numpy для работы с графиками
            log_message(log_filename, f"Epoch {epoch}: Loss = {epoch_losses.avg:.6f}, PSNR = {epoch_psnr.avg:.2f} dB")

        # Сохраняем лог и график
        plot("ESPCN", train_losses, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}")


def train_srgan(train_data, val_data, output, scale, epochs, batch_size, crop_size):
    """
    Функция для обучения модели SRGAN

    На входе:
    - train_data (str): директория с изображениями для обучения
    - val_data (str): директория с изображениями для валидации
    - output (str): директория для сохранения весов модели
    - scale (int): коэффициент увеличения разрешения (2, 3 или 4)
    - epochs (int): количество эпох
    - batch_size (int): размер батча
    - crop_size (int): размер обрезанного изображения
    """
    if not os.path.exists(output):
        os.makedirs(output)
    
    try:
        # Настройка устройства (GPU или CPU)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        python_version = platform.python_version()
        
        # Логирование системной информации
        log_filename = f"logs/train_SRGAN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        log_message(log_filename, f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}")
        log_message(log_filename, f"Programming language: Python {python_version}")
        log_message(log_filename, f"Train data: {train_data}")
        log_message(log_filename, f"Validation data: {val_data}")

        # Логирование начала процесса обучения
        log_message(log_filename, "=== Starting model training ===")
        log_message(log_filename, f"Parameters: \n- Model: SRGAN\n- Scaling factor: {scale}\n- Epochs: {epochs}\n- Batch size: {batch_size}\n- Crop size: {crop_size}")

        # Инициализация сетей
        netG, netD = Generator(scale).to(device), Discriminator().to(device)
        generator_criterion = GeneratorLoss().to(device)
        optimizerG, optimizerD = optim.Adam(netG.parameters()), optim.Adam(netD.parameters())

        # Загрузка данных
        train_set = TrainDatasetSRGAN(train_data, crop_size, upscale_factor=scale)
        val_set = EvalDatasetSRGAN(val_data, upscale_factor=scale)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

        # Отслеживание метрик
        train_losses = []   # Словарь со значениями функции потерь на тренировочных данных
        val_psnrs = []  # Словарь со значениями PSNR на валидационных данных
        epoch_times = []    # Словарь с длительностями эпох
        val_losses = []

        for epoch in range(1, epochs + 1):
            netG.train()
            netD.train()
            start_time = time.time()
            epoch_losses = AverageMeter()
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

            with tqdm(total=len(train_loader)) as train_bar:
                train_bar.set_description(f'Epoch {epoch}/{epochs}')
                for data, target in train_loader:
                    batch_size = data.size(0)
                    running_results['batch_sizes'] += batch_size

                    real_img, z = target.to(device), data.to(device)
                    fake_img = netG(z)

                    optimizerG.zero_grad()
                    g_loss = generator_criterion(netD(fake_img).mean(), fake_img, real_img)
                    g_loss.backward()
                    optimizerG.step()

                    real_out, fake_out = netD(real_img).mean(), netD(fake_img.detach()).mean()
                    d_loss = 1 - real_out + fake_out

                    optimizerD.zero_grad()
                    d_loss.backward()
                    optimizerD.step()

                    running_results['g_loss'] += g_loss.item() * batch_size
                    running_results['d_loss'] += d_loss.item() * batch_size
                    running_results['d_score'] += real_out.item() * batch_size
                    running_results['g_score'] += fake_out.item() * batch_size

                    epoch_losses.update(g_loss.item(), batch_size)  # Обновляем потери
                    train_bar.set_postfix(d_loss=f'{d_loss.item():.4f}', g_loss=f'{g_loss.item():.4f}')
                    train_bar.update(1)

            end_time = time.time()
            epoch_times.append(end_time - start_time)
            train_losses.append(running_results['g_loss'] / running_results['batch_sizes'])

            # Валидация с прогресс-баром
            netG.eval()
            epoch_psnr = AverageMeter()
            val_g_loss_sum = 0
            val_psnr_sum = 0
            with torch.no_grad():
                with tqdm(total=len(val_loader)) as val_bar:
                    val_bar.set_description(f'Validation {epoch}/{epochs}')
                    for val_lr, val_hr, _ in val_loader:
                        val_lr, val_hr = val_lr.to(device), val_hr.to(device)
                        sr_img = netG(val_lr)

                        # Подсчет PSNR
                        val_psnr_value = psnr(sr_img, val_hr)
                        val_psnr_sum += val_psnr_value.cpu().item()
                        epoch_psnr.update(val_psnr_value.cpu().item())

                        # Подсчет потерь генератора на валидации
                        val_g_loss = generator_criterion(netD(sr_img).mean(), sr_img, val_hr)
                        val_g_loss_sum += val_g_loss.item()

                        val_bar.set_postfix(psnr=f'{val_psnr_value.cpu().item():.2f} dB')
                        val_bar.update(1)

            val_losses.append(val_g_loss_sum / len(val_loader))
            val_psnrs.append(val_psnr_sum / len(val_loader))

            # Логирование метрик после каждой эпохи
            log_message(log_filename, f"Epoch {epoch}: Train Loss = {epoch_losses.avg:.6f}, Val Loss = {val_losses[-1]:.6f}, PSNR = {epoch_psnr.avg:.2f} dB")

            torch.save(netG.state_dict(), os.path.join(output, f'srgan_G_epoch_{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(output, f'srgan_D_epoch_{epoch}.pth'))

        # Сохраняем лог и график
        plot("SRGAN", train_losses, val_psnrs, scale, epochs, batch_size, log_filename)

    except Exception as e:
        log_message(log_filename, f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument("--model", default="SRCNN", type=str, choices=["SRCNN", "ESPCN", "SRGAN"], help="Model name (SRCNN, ESPCN, SRGAN)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--scale", default=2, type=int, choices=[2, 3, 4], help="Upscaling factor (2, 3, 4)")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--crop_size", default=88, type=int, help="Crop size for training")
    
    args = parser.parse_args()
    
    if args.model == "SRCNN":
        train_srcnn(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size)
    elif args.model == "ESPCN":
        train_espcn(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size)
    elif args.model == "SRGAN":
        train_srgan(args.train_data, args.val_data, args.output, args.scale, args.epochs, args.batch_size, args.crop_size)