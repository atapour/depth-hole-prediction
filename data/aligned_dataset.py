import os.path
import random
from torch.utils.data import Dataset
from PIL import Image
from data.image_folder import make_dataset

class AlignedDataset(Dataset):
    def __init__(self, root, rgb_transforms=[], rgb_torchvision_transforms=[], gt_transforms=[]):

        self.root = root
        self.dir_rgb = os.path.join(root, 'RGB')
        self.dir_gt = os.path.join(root, 'GT')

        self.rgb_paths = make_dataset(self.dir_rgb)
        self.gt_paths = make_dataset(self.dir_gt)

        self.rgb_paths = sorted(self.rgb_paths)
        self.gt_paths = sorted(self.gt_paths)

        self.rgb_transforms = rgb_transforms
        self.gt_transforms = gt_transforms
        self.rgb_torchvision_transforms = rgb_torchvision_transforms


    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        gt_path = self.gt_paths[index]

        rgb_img = Image.open(rgb_path).convert('RGB')
        gt_img = Image.open(gt_path)

        rgb_size = rgb_img.size
        gt_size = gt_img.size

        for transform in self.rgb_torchvision_transforms:
            if random.random() < transform.probability:
                rgb_img = transform(rgb_img)

        rgb_img = self.rgb_transforms(rgb_img)
        gt_img = self.gt_transforms(gt_img)

        return {'rgb': rgb_img, 'gt': gt_img,
                'rgb_paths': rgb_path, 'gt_paths': gt_path,
                'rgb_sizes': rgb_size, 'gt_sizes': gt_size}

    def __len__(self):
        return len(self.rgb_paths)
