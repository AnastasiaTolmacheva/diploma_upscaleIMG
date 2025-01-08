import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y


def train(args):
    h5_file = h5py.File(args['output_path'], 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args['images_dir']))):
        print(f"Processing image: {image_path}")
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args['scale']) * args['scale']
        hr_height = (hr.height // args['scale']) * args['scale']
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args['scale'], hr_height // args['scale']), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args['scale'], lr.height * args['scale']), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args['patch_size'] + 1, args['stride']):
            for j in range(0, lr.shape[1] - args['patch_size'] + 1, args['stride']):
                lr_patches.append(lr[i:i + args['patch_size'], j:j + args['patch_size']])
                hr_patches.append(hr[i:i + args['patch_size'], j:j + args['patch_size']])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    print("Starting evaluation...")
    h5_file = h5py.File(args['output_path'], 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args['images_dir'])))):
        print(f"Processing image: {image_path}")  # Добавляем вывод для отладки
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args['scale']) * args['scale']
        hr_height = (hr.height // args['scale']) * args['scale']
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args['scale'], hr_height // args['scale']), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args['scale'], lr.height * args['scale']), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()
    print(f"Data saved to {args['output_path']}")


if __name__ == '__main__':
    # Прямо здесь задать параметры
    args = {
        'images_dir': 'D:\SRCNN\images',  # Путь к папке с изображениями
        'output_path': 'd:/SRCNN/output.h5',  # Путь для сохранения файла h5
        'patch_size': 33,  # Размер патча
        'stride': 14,  # Шаг
        'scale': 2,  # Масштаб
        'eval': False  # Если True, будет выполнена оценка, иначе - обучение
    }

    if not args['eval']:
        train(args)
    else:
        eval(args)
