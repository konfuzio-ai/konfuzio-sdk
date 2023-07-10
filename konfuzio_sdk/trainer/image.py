"""Preprocessing and data augmentation for images."""
import PIL.ImageOps

from konfuzio_sdk.extras import torchvision


class InvertImage:
    """Invert (negate) images."""

    def __call__(self, sample):
        """Apply image invertion."""
        image = PIL.ImageOps.invert(sample)
        return image


class ImagePreProcessing:
    """Define the images pre processing transformations."""

    def __init__(self, transforms: dict = {'target_size': (1000, 1000)}):
        """Collect the transformations to be applied to the images."""
        self.pre_processing_operations = []
        transforms_keys = transforms.keys()

        if 'invert' in transforms_keys and transforms['invert']:
            self.pre_processing_operations.append(InvertImage())

        if 'target_size' in transforms_keys:
            self.pre_processing_operations.append(torchvision.transforms.Resize(transforms['target_size']))

        if 'grayscale' in transforms_keys and transforms['grayscale']:
            # num_output_channels = 3 because pre-trained models use 3 channels
            self.pre_processing_operations.append(torchvision.transforms.Grayscale(num_output_channels=3))

        self.pre_processing_operations.append(torchvision.transforms.ToTensor())

    def get_transforms(self):
        """Get the transformations to be applied to the images."""
        transforms = torchvision.transforms.Compose(self.pre_processing_operations)
        return transforms


class ImageDataAugmentation:
    """Defines the images data augmentation transformations."""

    def __init__(self, transforms: dict = {'rotate': 5}, pre_processing_operations=None):
        """
        Collect the transformations to be applied to the images.

        Order of operations is pre defined here.
        """
        self.pre_processing_operations = pre_processing_operations
        if pre_processing_operations is None:
            self.pre_processing_operations = []

        # Removing to_tensor transformation coming from pre processing. It must be done at the end.
        self.pre_processing_operations = [
            p for p in self.pre_processing_operations if not isinstance(p, torchvision.transforms.transforms.ToTensor)
        ]

        if 'rotate' in transforms.keys():
            self.pre_processing_operations.append(torchvision.transforms.RandomRotation(transforms['rotate']))

        self.pre_processing_operations.append(torchvision.transforms.ToTensor())

    def get_transforms(self):
        """Get the transformations to be applied to the images."""
        transforms = torchvision.transforms.Compose(self.pre_processing_operations)
        return transforms


def create_transformations_dict(possible_transforms, args=None):
    """Create a dictionary with the image transformations accordingly with input args."""
    input_dict = {}
    if args is None:
        args = {'invert': False, 'target_size': (1000, 1000), 'grayscale': True, 'rotate': 5}

    if isinstance(args, dict):
        for transform in possible_transforms:
            if args[transform] is not None:
                input_dict[transform] = args[transform]
    else:
        for transform in possible_transforms:
            if args.__dict__[transform] is not None:
                input_dict[transform] = args.__dict__[transform]

    if len(input_dict.keys()) == 0:
        return None

    return input_dict
