import time
import os
import ntpath
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

# this class is used to print and display results using Visdom based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class Display():
    def __init__(self, args):
        self.display_id = args.display_id
        self.win_size = args.display_winsize
        self.name = args.name
        self.args = args
        if self.display_id > 0:
            import visdom
            self.ncols = args.display_ncols
            self.vis = visdom.Visdom(server=args.display_server, port=args.display_port, env=args.display_env, raise_exceptions=True)

        dir = os.path.join(args.checkpoints_dir, args.name)
        mkdir(args.checkpoints_dir)
        mkdir(dir)
        self.log_name = os.path.join(dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))

                label_html = ''
                label_html_row = ''
                images = []
                labels = []
                idx = 0

                for label, image in visuals.items():

                    image_numpy = return_numpy_array(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    labels.append(label + ' ')
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row

                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=''.join(labels)))

                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = return_numpy_array(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label), win=self.display_id + idx)
                    idx += 1

    # losses: dictionary of error labels and values
    def plot_current_loss(self, epoch, counter_ratio, loss):

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': ['Loss']}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([np.float(loss)])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)

        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_loss(self, epoch, i, loss, t, t_data):
        message = '(epoch: %d, batches: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)

        message += '%s: %.3f ' % ('loss', loss)

        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


# this function return a numpy array when given an image. For the network output images (one channel images) colorization to black and white is performed here.
def return_numpy_array(img, imtype=np.uint8):

    if isinstance(img, torch.Tensor):
        image_tensor = img.data
    else:
        return img

    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)

# this function returns a binary cross entropy loss with jaccard weight. Taken from: https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py
class LossBinary:

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss

# this function is used to make the necessary directory structures
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            print(path)
            mkdir(path)
    else:
        mkdir(paths)

# this function is used to make a directory if it does not already exist
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# this function is used during the testing phase to write the output to disk
def save_images(paths, images, size=None):

    image_dir = paths[1]
    image_path = paths[0]

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im in images.items():

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)

        im = return_numpy_array(im)

        if size is not None:
            im = cv2.resize(im, size)

        save_image(im, save_path)

def save_image(image, image_path):
    image = image.astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(image_path)
