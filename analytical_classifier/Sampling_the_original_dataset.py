import os
from shutil import copyfile

splits = ['train', 'valid', 'test']

data_root = '/home/adnan/Datasets/InnerEye-dataset'
data_dest = '/home/adnan/Datasets/New-InnerEye-dataset'

for split in splits:
    image_files = os.listdir(os.path.join(data_root, split, 'original'))
    for i, image_file in enumerate(image_files):
        if split == 'train' and i > 7000:
            break
        if split == 'valid' and i > 250:
            break
        if split == 'test' and i > 500:
            break
        src = os.path.join(data_root, split, 'original', image_file)
        dst = os.path.join(data_dest, split, 'original', image_file)
        copyfile(src, dst)
        print(split + ' ' + str(i))
