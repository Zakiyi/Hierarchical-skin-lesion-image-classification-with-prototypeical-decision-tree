import cv2
import PIL
import torch
import numpy as np
import PIL.Image as Image
import albumentations as A
import torchvision.transforms as transforms
from _collections import OrderedDict


class RandomMaskOut(torch.nn.Module):
    def __init__(self, max_holes=10, max_height=10, max_width=10, min_holes=None, min_height=None,
                 min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.CoarseDropout(max_holes, max_height, max_width, min_holes, min_height, min_width,
                                         fill_value, mask_fill_value, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, blur_limit=(3, 7), sigma_limit=0.5, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.GaussianBlur(blur_limit, sigma_limit, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, var_limit=(10.0, 30.0), mean=0, per_channel=True, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.GaussNoise(var_limit, mean, per_channel, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class Augmentations:
    def __init__(self, size, is_train=False):

        transform_list = []

        if is_train:
            # random resize and crop
            transform_list.append(transforms.RandomResizedCrop(size=size,
                                                               scale=(0.7, 1.3),
                                                               ratio=(0.7, 1.3)
                                                               ))
            # random flip
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())

            transform_list.append(RandomGaussianNoise(p=0.3))

            # random color trans
            transform_list.append(transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                                         saturation=(0.8, 1.2),
                                                         hue=(-0.05, 0.05))
                                  )

            transform_list.append(RandomMaskOut(p=0.3))
            transform_list.append(RandomGaussianBlur(p=0.3))
        else:

            transform_list.append(transforms.Resize(size + 32))
            transform_list.append(transforms.CenterCrop(size))

        # normalize to tensor
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        self.transform = transforms.Compose(transform_list)  # output: T, H, W, C

    def __call__(self, img):
        img = self.transform(img)
        return img
    #     self.ta_transform = self._get_crop_transform(tta)
    #
    # def _get_crop_transform(self, method='ten'):
    #
    #     if method == 'ten':
    #         crop_tf = transforms.Compose([
    #             transforms.Resize((self.size + 32, self.size + 32)),
    #             transforms.TenCrop((self.size, self.size))
    #         ])
    #
    #     if method == 'inception':
    #         crop_tf = InceptionCrop(
    #             self.size,
    #             resizes=tuple(range(self.size + 32, self.size + 129, 32))
    #         )
    #
    #     after_crop = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(self.mean, self.std),
    #     ])
    #     return transforms.Compose([
    #         crop_tf,
    #         transforms.Lambda(
    #             lambda crops: torch.stack(
    #                 [after_crop(crop) for crop in crops]))
    #     ])


if __name__ == "__main__":
    img = Image.open('xxx.png')

    augmentor = Augmentations()
    imt = augmentor(img)
    imt = transforms.ToPILImage()(imt)
    imt.show()
