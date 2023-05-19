import albumentations as A
import cv2

pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            always_apply=True, contrast_limit=0.2, brightness_limit=0.2
        ),
        A.OneOf(
            [
                A.MotionBlur(always_apply=True),
                A.GaussNoise(always_apply=True),
                A.GaussianBlur(always_apply=True),
            ],
            p=0.5,
        ),
        A.Rotate(always_apply=True, limit=20, border_mode=cv2.BORDER_REPLICATE),
    ]
)

A.save(pipeline, "custom_pipeline_example.yml", data_format="yaml")
