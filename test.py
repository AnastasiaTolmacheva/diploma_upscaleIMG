import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


def upscale_image(weights_file, image_file, scale=2):
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

    # Открытие изображения
    image = pil_image.open(image_file).convert('RGB')

    # Изменение размера изображения с помощью бикубической интерполяции
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

    # Сохранение изображения после bicubic апскейлинга
    image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    # Подготовка изображения для модели
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    # Прогон изображения через модель
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    # Вычисление PSNR
    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    # Конвертация предсказаний обратно в изображение
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)

    # Сохранение результата
    output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))


upscale_image('D:\\SRCNN\\output_01\\x2\\best.pth', 'D:\\SRCNN\\Set14\\image_SRF_2\\img_001_SRF_2_LR.png', scale=2)
