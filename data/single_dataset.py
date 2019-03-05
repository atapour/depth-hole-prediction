from torch.utils.data import Dataset
from PIL import Image
from data.image_folder import make_dataset

class SingleDataset(Dataset):
    def __init__(self, root, rgb_transforms=[]):

        self.root = root

        self.rgb_paths = make_dataset(self.root)
        self.rgb_paths = sorted(self.rgb_paths)
        self.rgb_transforms = rgb_transforms

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]

        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_size = rgb_img.size
        rgb_img = self.rgb_transforms(rgb_img)

        return {'rgb': rgb_img, 'rgb_paths': rgb_path, 'rgb_sizes': rgb_size}

    def __len__(self):
        return len(self.rgb_paths)
