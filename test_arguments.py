import argparse
import os
import colorama

class Arguments():

    def __init__(self):

        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--phase', default='test', type=str, choices=['train', 'test'], help='determining whether the model is being trained or used for inference. Since this is the test_arguments file, this needs to be test!!')
        parser.add_argument('--test_checkpoint_path', type=str, default='./pretrained_weights/hole_predictor.pth', help='during inference, the path to checkpoint is needed.')
        parser.add_argument('--results_path', type=str, default='./results', help='during inference, the path to save the results. The directory is not created and must already be there.')
        parser.add_argument('--data_root', default='./example', type=str, help='path to data directory, where the input RGB image are.')
        parser.add_argument('--batch_size', default=25, type=int, help='It is the size of your batch.')
        parser.add_argument('--num_epochs', default=500, type=int, help='number of training epochs to run')
        parser.add_argument('--input_nc', default=3, type=int, help='number of channels in the input image')
        parser.add_argument('--output_nc', default=64, type=int, help='number of channels in the output image.')
        parser.add_argument('--num_downs', default=8, type=int, help='number of downscaling done within the architecture')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers used in the dataloader.')
        parser.add_argument('--ngf', default=32, type=int, help='number of filters in first convolutional layer in the network.')
        parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='which is training being done on.')
        parser.add_argument('--transform', default='resize', type=str, choices=['resize', 'crop'], help='inputs to the network must be 1024x256. You can choose to resize or crop them to those dimenstion.')

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

        txt += '{}The default arguments are displayed in blue!{}\n'.format(blue, reset)
        txt += '{}The specified arguments are displayed in magenta!{}\n'.format(magenta, reset)


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
