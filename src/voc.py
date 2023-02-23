from augmentation import Compose, ConvertFromInts, ToAbsoluteCoords

class DataTransform(object):
    def __init__(self, input_size, color_mean) -> None:
        self.transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
            ])
        }