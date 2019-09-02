import os
import pilgram
from PIL import Image
from shutil import copyfile

splits = ['train', 'valid', 'test']

for split in splits:
    split_root = '/home/adnan/Datasets/New-InnerEye-dataset/{0}'.format(split)
    original_images_dir_path = os.path.join(split_root, 'original')
    original_image_files = os.listdir(original_images_dir_path)

    filters = ['_1977', 'aden', 'brannan', 'brooklyn', 'clarendon', 'earlybird', 'gingham', 'hudson', 'inkwell', 'kelvin', 'lark', 'lofi', 'maven', 'mayfair', 'moon', 'nashville', 'perpetua', 'reyes', 'rise', 'slumber', 'stinson', 'toaster', 'valencia', 'walden', 'willow', 'xpro2']

    for filter in filters:
        for original_image_file in original_image_files:
            if filter == '_1977':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram._1977(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'aden':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.aden(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'brannan':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.brannan(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'brooklyn':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.brooklyn(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'clarendon':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.clarendon(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'earlybird':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.earlybird(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'gingham':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.gingham(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'hudson':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.hudson(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'inkwell':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.inkwell(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'kelvin':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.kelvin(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'lark':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.lark(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'lofi':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.lofi(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'maven':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.maven(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'mayfair':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.mayfair(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'moon':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.moon(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'nashville':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.nashville(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'perpetua':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.perpetua(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'reyes':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.reyes(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'rise':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.rise(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'slumber':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.slumber(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'stinson':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.stinson(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'toaster':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.toaster(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'valencia':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.valencia(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'walden':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.walden(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'willow':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.willow(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            elif filter == 'xpro2':
                _path = os.path.join(split_root, filter)
                os.makedirs(_path, exist_ok=True)
                pilgram.xpro2(Image.open(os.path.join(original_images_dir_path, original_image_file))).save(os.path.join(_path, original_image_file))
            else:
                continue
        print('Done with:', filter)

for split in splits:
    data_root = '/home/adnan/Datasets/InnerEye-dataset/{0}'.format(split)
    split_root = '/home/adnan/Datasets/New-InnerEye-dataset/{0}'.format(split)
    original_images_dir_path = os.path.join(split_root, 'original')
    original_image_files = os.listdir(original_images_dir_path)

    filters = ['gotham', 'lomo', 'sepia']

    for filter in filters:
        os.makedirs(os.path.join(split_root, filter), exist_ok=True)
        for i, original_image_file in enumerate(original_image_files):
            src = os.path.join(data_root, filter, original_image_file)
            dst = os.path.join(split_root, filter, original_image_file)
            copyfile(src, dst)
            print(split + ' ' + filter + ' ' + str(i))
