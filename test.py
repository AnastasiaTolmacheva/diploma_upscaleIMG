import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


def upscale_image(weights_file, image_file, scale=3):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Инициализация модели
    model = ESPCN(scale_factor=scale).to(device)

    # Загрузка весов
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # Открытие изображения
    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale

    # Изменение размеров изображений
    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    # Предварительная обработка изображений
    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    # Подсчет PSNR
    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    # Постобработка и сохранение результата
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_file.replace('.', '_espcn_x{}.'.format(scale)))


weights_file = 'D:\\ESPCN\\model_1\\x2\\best.pth'
image_file = 'D:\\ESPCN\\Set14\\image_SRF_2\\img_001_SRF_2_LR.png'
scale = 2 # Масштаб


upscale_image(weights_file, image_file, scale)
