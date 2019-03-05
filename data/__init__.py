from PIL import Image
import torch.utils.data
from torchvision.transforms import ToTensor, Compose, Grayscale, ColorJitter
from data.aligned_dataset import AlignedDataset
from data.single_dataset import SingleDataset

# this function scales a PIL image (Segmentation GT) and scales it to 256x1024
def scale_gt_to_256_factor(img):

    return img.resize((1024, 256), Image.NEAREST)
#-----------------------------------------

# this function scales a PIL image (RGB input) and scales it to 256x1024
def scale_rgb_to_256_factor(img):

    return img.resize((1024, 256), Image.ANTIALIAS)
#-----------------------------------------

# this function takes a PIL image and center crops it to 256x1024
def center_crop_to_256_factor(img):

    w, h = img.size
    return img.crop((w//2-512, h//2-128, w//2+512, h//2+128))
#-----------------------------------------

# this function takes a  PIL image and devides it by 255. this is for the label images in which the values are 0 or 255.
def to_range(img):

    return img/255

# this function takes a tensor and turns it into a long tensor if it is an int tensor. This one is used for the segmentation ground truth, so no divisions.
def to_longTensor_gt(img):
    # convert a IntTensor to a LongTensor
    if isinstance(img, torch.IntTensor):
        return img.type(torch.LongTensor)
    elif isinstance(img, torch.LongTensor):
        return img
    else:
        return img
#-----------------------------------------

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
#-----------------------------------------

# this class takes a torch tensor and adds some noise. This can be very helpful in preventing instability when thr input image is completely uniform, which of course cannot happen here. 
class AddNoise(object):

    def __init__(self,size):
        self.noise = torch.randn(*size) * 0.0000001

    def __call__(self,x):
        return x + self.noise
#-----------------------------------------

# this function returns the transforms applied to the images
def get_transform(args):

    if args.transform == 'resize':

        rgb_transforms = Compose([scale_rgb_to_256_factor, ToTensor(), AddNoise((3, 256, 1024))])
        gt_transforms = Compose([scale_gt_to_256_factor, ToTensor(), to_longTensor_gt])

    elif args.transform == 'crop':

        rgb_transforms = Compose([center_crop_to_256_factor, ToTensor(), AddNoise((3, 256, 1024))])
        gt_transforms = Compose([center_crop_to_256_factor, ToTensor(), to_longTensor_gt])

    else:
        raise ValueError('the value (%s) for --transform is not valid.' % args.transform)

    torchvision_transforms = []

    if args.phase == 'train':
        grayscale = Grayscale(num_output_channels=3)
        grayscale.probability = 0.075
        torchvision_transforms.append(grayscale)
        colorJitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        colorJitter.probability = 0.1
        torchvision_transforms.append(colorJitter)

        return rgb_transforms, torchvision_transforms, gt_transforms

    elif args.phase == 'test':

        return rgb_transforms

# this function only creates the dataset from the aligned dataset class
def create_dataset(args):

    if args.phase == 'train':
        rgb_transforms, torchvision_transforms, gt_transforms = get_transform(args)
        return AlignedDataset(args.data_root, rgb_transforms, torchvision_transforms, gt_transforms)
    elif args.phase == 'test':
        rgb_transforms = get_transform(args)
        return SingleDataset(args.data_root, rgb_transforms)


# this function creates the dataloader
def create_loader(args):

    data_loader = DataLoader()
    data_loader.initialize(args)
    print("The training data has been loaded.")
    return data_loader

# The DataLoader class based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class DataLoader():

    def initialize(self, args):

        self.dataset = create_dataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=args.phase == 'train', num_workers=int(args.num_workers), drop_last=args.phase == 'train')

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data
