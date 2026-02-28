import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                max_holes=4, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8, p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
