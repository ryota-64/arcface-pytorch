import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128), random_erase=True):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                PadingWithLongerSize(),
                T.Resize(self.input_shape[1:]),
                # T.Grayscale(),
                T.ToTensor(),
                normalize,
                T.RandomErasing(),
            ])

        else:
            self.transforms = T.Compose([
                PadingWithLongerSize(),
                T.Resize(self.input_shape[1:]),
                # T.Grayscale(),
                T.ToTensor(),
                normalize,

            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        img_data = Image.open(img_path)
        img_data = img_data.convert('L')
        img_data = self.transforms(img_data)
        label = np.int32(splits[1])
        return img_data.float(), label

    def __len__(self):
        return len(self.imgs)


class PadingWithLongerSize(object):
    """
    長い方の辺を基準に正方形になるようにpaddingする
    左上が基準
    """
    def __init__(self):
        pass

    def __call__(self, image):
        """
        Args:
            image (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        size = image.size[:2]
        longer_flip = np.argmax(size)
        padding_flip = np.argmin(size)
        padding_num = size[longer_flip] - size[padding_flip]
        right_pad = padding_num if padding_flip == 0 else 0
        bottom_pad = padding_num if padding_flip == 1 else 0
        self.transform = [
            T.Pad((0, 0, right_pad, bottom_pad))
        ]
        for t in self.transform:
            image = t(image)
        return image
