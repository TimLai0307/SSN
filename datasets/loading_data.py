import torchvision.transforms as standard_transforms
from .SHHA import loading_data

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def building_data(data_root):
    # the pre-proccessing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), #轉為張量
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = loading_data(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = loading_data(data_root, train=False, transform=transform)

    return train_set, val_set
