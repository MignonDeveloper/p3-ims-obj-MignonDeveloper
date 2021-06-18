from albumentations import (Compose, Normalize, OneOf)
from albumentations import (RandomBrightnessContrast, RandomContrast)
from albumentations import (HorizontalFlip, Resize, ShiftScaleRotate, Rotate, RandomResizedCrop, CenterCrop, VerticalFlip)
from albumentations import (GridDistortion, CLAHE, ElasticTransform, GaussNoise, Cutout)
from albumentations.pytorch import ToTensorV2

class BaseTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class AugLastTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            HorizontalFlip(p=0.5),
            CLAHE(clip_limit=(1, 8), tile_grid_size=(10, 10), p=0.3),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=(-0.46, 0.40)),
                ElasticTransform(alpha=1.68, sigma=48.32, alpha_affine=44.97),
            ], p=0.3),
            RandomResizedCrop(p=0.3, height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            ShiftScaleRotate(p=0.3, shift_limit=(-0.06, 0.06), scale_limit=(-0.10, 0.10), rotate_limit=(-20, 20)),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class FinalTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=(-0.46, 0.40), value=(0, 0, 0)),
                ElasticTransform(alpha=1.68, sigma=48.32, alpha_affine=44.97,value=(0, 0, 0)),
                RandomResizedCrop(height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.33))
            ], p=.3),
            ShiftScaleRotate(shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-90, 90),p=0.3),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class BaseTestAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transformer(image=image)