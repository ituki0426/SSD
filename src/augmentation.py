import cv2
import numpy as np
from numpy import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgn, boxes=None, labels=None):
        for t in self.transforms:
            imgn, boxes, labels = t(imgn, boxes, labels)
        return imgn, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "Contrast upper must be >= lower."
        assert self.lower >= 0, "Contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV') -> None:
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300) -> None:
        self.size = size

    def __call__(self, image, boxes=None, labels=None) -> None:
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class SubtractMeans(object):
    def __init__(self, mean) -> None:
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None) -> None:
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
    inter = np.clip((max_xy-min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0]*inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0])*(box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0])*(box_b[:, 3]-box_b[:, 1]))
    union = area_a+area_b-inter
    return inter/union


class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )
        self.sample_options = np.array(self.sample_options, dtype=object)

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            # トレースの最大値
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3*width, width)
                h = random.uniform(0.3*height, height)
                if h/w < 0.5 or h/w > 2:
                    continue
                left = random.uniform(width-w)
                top = random.uniform(height-h)

                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                oberlap = jaccard_numpy(boxes, rect)

                if oberlap.min() < min_iou and oberlap.max() > max_iou:
                    continue

                current_image = current_image[rect[1]                                              :rect[3], rect[0]:rect[2], :]

                center = (boxes[:, :2]+boxes[:, 2:])/2.0

                m1 = (rect[0] < center[:, 0])*(rect[1] < center[:, 1])

                m2 = (rect[2] > center[:, 0])*(rect[3] > center[:, 1])

                mask = m1*m2

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()

                current_labels = labels[mask]

                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2])

                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(
                    current_boxes[:, 2:], rect[2:])

                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels
