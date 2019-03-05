import argparse
import colorama
import os

class Arguments():

    def __init__(self):

        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--name', required=True, type=str, help='experiment name')
        parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'], help='determining whether the model is being trained or used for inference. Since this is the train_arguments file, this needs to set to train!!')
        parser.add_argument('--data_root', required=True, type=str, help='path to data directory.')
        parser.add_argument('--batch_size', default=25, type=int, help='It is the size of your batch.')
        parser.add_argument('--num_epochs', default=500, type=int, help='number of training epochs to run')
        parser.add_argument('--input_nc', default=3, type=int, help='number of channels in the input image')
        parser.add_argument('--output_nc', default=64, type=int, help='number of channels in the output image.')
        parser.add_argument('--num_downs', default=8, type=int, help='number of downscaling done within the architecture')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers used in the dataloader.')
        parser.add_argument('--ngf', default=32, type=int, help='number of filters in first convolutional layer in the network.')
        parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
        parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='which is training being done on.')
        parser.add_argument('--transform', default='resize', type=str, choices=['resize', 'crop'], help='inputs to the network must be 1024x256. You can choose to resize or crop them to those dimenstion.')
        parser.add_argument('--jaccard-weight', default=0.5, type=float, help='This: https://en.wikipedia.org/wiki/Jaccard_index.')
        parser.add_argument('--resume', action='store_true', help='resume training from most recent checkpoint.')
        parser.add_argument('--which_checkpoint', type=str, default='latest', help='the checkpoint to be loaded to resume training. Checkpoints are identified and saved by the number of steps passed during training.')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='the path to where the model is saved.')
        parser.add_argument('--print_freq', default=2, type=int, help='how many steps before printing the loss values to the standard output for inspection purposes only.')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for visdom.')
        parser.add_argument('--display_freq', type=int, default=2, help='frequency of showing training results on screen using visdom.')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display.')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display.')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main").')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display.')
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int, help='how many steps before saving one sequence of images to disk for inspection purposes only.')

        self.initialized = True

        return parser

    def get_args(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_args(self, args):

        # setting up the colors:
        reset = colorama.Style.RESET_ALL
        magenta = colorama.Fore.MAGENTA
        blue = colorama.Fore.BLUE

        txt = '\n'

        txt += '{}The default argumnets are displayed in blue!{}\n'.format(blue, reset)
        txt += '{}The specified argumnets are displayed in magenta!{}\n'.format(magenta, reset)


        txt += '\n'
        txt += '-------------------- Arguments --------------------\n'

        for k, v in sorted(vars(args).items()):

            comment = ''
            default = self.parser.get_default(k)

            color = blue if v == default else magenta

            if v != default:
                comment = '\t[default: %s]' % str(default)

            txt += '{}{:>25}: {:<30}{}{}\n'.format(color, str(k), str(v), comment, reset)

        txt += '----------------------- End -----------------------'
        txt += '\n'

        print(txt)

    def parse(self):

        args = self.get_args()
        self.print_args(args)
        self.args = args

        return self.args
