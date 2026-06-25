from torchvision.transforms import transforms


def transform_preprocessed_image():
   return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ]
    )