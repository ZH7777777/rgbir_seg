# coding:utf-8
import numpy as np
from PIL import Image
from PIL import Image
from PIL import Image, ImageOps
import random
import numpy as np
# from ipdb import set_trace as st
import random
import numbers
import math
import collections
import cv2
class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class ConvertFromInts(object):
    def __call__(self, image, labels=None):
        return image.astype(np.float32), labels
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image,  labels=None):
        if random.randint(0,1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image,  labels
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels=None):
        if random.randint(0,1):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, labels
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image,  labels=None):
        if random.randint(0,1):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image,labels
# class RandomBrightness():
#     def __init__(self, bright_range=0.15, prob=0.9):
#         #super(RandomBrightness, self).__init__()
#         self.bright_range = bright_range
#         self.prob = prob
#
#     def __call__(self, image, label):
#         if np.random.rand() < self.prob:
#             bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
#             image = (image * bright_factor).astype(image.dtype)
#
#         return image, label

class RandomFlip_multilabel():
    def __init__(self, prob=0.5):
        super(RandomFlip_multilabel, self).__init__()
        self.prob = prob

    def __call__(self, image, label,salient_label,boundary_label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
            salient_label=salient_label[:,::-1]
            boundary_label=boundary_label[:,::-1]
        return image, label,salient_label,boundary_label


class RandomCrop_multilabel():
    def __init__(self, crop_rate=0.1, prob=1.0):
        super(RandomCrop_multilabel, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label,salient_label,boundary_label):
        if np.random.rand() < self.prob:
            h,w, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            # image = image[w1:w2, h1:h2]
            # label = label[w1:w2, h1:h2]
            image = image[h1:h2, w1:w2]
            label = label[h1:h2, w1:w2]
            salient_label=salient_label[h1:h2, w1:w2]
            boundary_label=boundary_label[h1:h2, w1:w2]

        return image, label,salient_label,boundary_label

class RandomFlip():
    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label


class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            h,w, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            # image = image[w1:w2, h1:h2]
            # label = label[w1:w2, h1:h2]
            image = image[h1:h2, w1:w2]
            label = label[h1:h2, w1:w2]

        return image, label

class RandomCrop2(object):
    def __init__(self, crop_rate=0.1, prob=1.0):
        super(RandomCrop2, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # print(image.shape)
            # print(label.shape)
            h,w,c = image.shape

            h1 = np.random.randint(0, 240) #480*0.1=48  0-48
            w1 = np.random.randint(0, 320) #640*0.1=64  有问题 没有裁全 整个图只裁了左上角 其他浪费了

            h2 = h1+240
            w2 = w1+320
            # h2 = np.random.randint(h - h * self.crop_rate, h + 1)
            # w2 = np.random.randint(w - w * self.crop_rate, w + 1)

            # image = image[w1:w2, h1:h2]
            # label = label[w1:w2, h1:h2]
            image = image[h1:h2, w1:w2]
            label = label[h1:h2, w1:w2]

        return image, label
class RandomCrop3:
    """Crop images to given size.
    Parameters
    ----------
      crop_size: a tuple specifying crop size,
                 which can be larger than original size.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, label):
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        a=image.size[0]
        b=self.crop_size[1]
        if image.size[0] < self.crop_size[1]:
            image = ImageOps.expand(image, (self.crop_size[1] - image.size[0], 0), fill=0)
            label = ImageOps.expand(label, (self.crop_size[1] - label.size[0], 0), fill=255)
        if image.size[1] < self.crop_size[0]:
            image = ImageOps.expand(image, (0, self.crop_size[0] - image.size[1]), fill=0)
            label = ImageOps.expand(label, (0, self.crop_size[0] - label.size[1]), fill=255)

        i, j, h, w = self.get_params(image, self.crop_size)
        image = image.crop((j, i, j + w, i + h))
        label = label.crop((j, i, j + w, i + h))

        return image, label
class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            image[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return image, label


# class RandomBrightness():
#     def __init__(self, bright_range=0.15, prob=0.9):
#         super(RandomBrightness, self).__init__()
#         self.bright_range = bright_range
#         self.prob = prob
#
#     def __call__(self, image, label):
#         if np.random.rand() < self.prob:
#             bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
#             image = (image * bright_factor).astype(image.dtype)
#
#         return image, label


class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )

            image = (image + noise).clip(0,255).astype(image.dtype)

        return image, label
        


