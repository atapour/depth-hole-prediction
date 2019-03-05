import os
import glob
from collections import OrderedDict
import colorama
import torch
from utils import LossBinary
from model.network import HolePredictor_Network


class TheModel():

    # this initializes all requirements for the model
    def initialize(self, args):

        self.args = args
        self.phase = args.phase
        self.device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')

        self.net = HolePredictor_Network(input_nc=args.input_nc, output_nc=args.output_nc, num_downs=args.num_downs, num_classes=1, ngf=args.ngf)
        self.net.to(self.device)

        if self.phase == 'train':
            self.checkpoint_save_dir = os.path.join(args.checkpoints_dir, args.name)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
            self.criterion = LossBinary(jaccard_weight=args.jaccard_weight)

        elif self.phase == 'test':
            self.results_path = args.results_path

    # this function sets up the model by loading and printing the model if necessary
    def set_up(self, args):

        if self.phase == 'test':
            if args.test_checkpoint_path is not None:

                print('loading the checkpoint from %s' % args.test_checkpoint_path)

                state_dict = torch.load(args.test_checkpoint_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                self.net.load_state_dict(state_dict)

            else:
                raise Exception('For inference, a checkpoint path must be passed as an argument.')

        else:
            if args.resume:
                if not os.listdir(self.checkpoint_save_dir):
                    raise Exception('The specified checkpoints directory is empty. Resuming is not possible.')
                if args.which_checkpoint == 'latest':
                    checkpoints = glob.glob(os.path.join(self.checkpoint_save_dir, '*.pth'))
                    checkpoints.sort()
                    latest = checkpoints[-1]
                    step = latest.split('_')[1]
                elif args.which_checkpoint != 'latest' and args.which_checkpoint.isdigit():
                    step = args.which_checkpoint
                else:
                    raise Exception('The specified checkpoint to load in invalid.')
                self.load_networks(step)
        self.print_networks()

    # data inputs are assigned
    def assign_inputs(self, input):

        self.rgb = input['rgb'].to(self.device)

        if self.phase == 'train':
            self.gt = input['gt'].to(self.device)
            self.gt_paths = input['gt_paths']

        elif self.phase == 'test':
            self.rgb_paths = input['rgb_paths']
            self.rgb_sizes = input['rgb_sizes']
            self.rgb_sizes = (int(self.rgb_sizes[0]), int(self.rgb_sizes[1]))

    # forward pass
    def forward(self):

        self.out = self.net(self.rgb)

    # backward pass with the loss
    def backward(self):

        self.loss = self.criterion(self.out, self.gt)
        self.loss.backward()

    # optimize the model parameters
    def optimize(self):

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    # this function is only used during inference
    def test(self):

        self.forward()

    # this function saves model checkpoints to disk
    def save_networks(self, step):

        save_filename = 'checkpoint_%s_steps.pth' % (step)
        save_path = os.path.join(self.checkpoint_save_dir, save_filename)

        print('saving the checkpoint to %s' % save_path)

        torch.save(self.net.state_dict(), save_path)

    # this function loads model checkpoints from disk
    def load_networks(self, step):

        load_filename = 'checkpoint_%s_steps.pth' % (step)
        load_path = os.path.join(self.checkpoint_save_dir, load_filename)

        print('loading the checkpoint from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.net.load_state_dict(state_dict)

    # this function prints the network information
    def print_networks(self):

        # setting up the pretty colors:
        reset = colorama.Style.RESET_ALL
        blue = colorama.Fore.BLUE
        red = colorama.Fore.RED

        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()

        print('{}{}{}{}{}{}{}{}'.format(blue, 'There are a total number of ', red, num_params, ' parameters', blue, ' in the model.', reset))
        print('')

    # this function returns the loss value
    def get_loss(self):

        return self.loss

    # this function returns the images involved in the training for saving and displaying
    def get_images(self):

        im_ret = OrderedDict()

        im_ret['rgb'] = self.rgb
        im_ret['gt'] = self.gt

        output = self.out.detach().clone()
        output[output < 0.5] = 0
        output[output >= 0.5] = 1

        im_ret['out'] = output

        return im_ret

    # this function returns the output image and the RGB image during testing
    def get_test_outputs(self):

        im_ret = OrderedDict()

        im_ret['rgb'] = self.rgb

        output = self.out.detach().clone()

        # The following normalization and thresholding is purely for visualization purposes and is not based on any analysis.
        # Proper analysis and thresholding need to take to place depending on the application of the hole predictor.
        output /= output.max()/1.0
        output[output < 0.1] = 0
        output[output >= 0.1] = 1

        im_ret['holes'] = output

        return im_ret

    # this function returns RGB image path to save the image during testing
    def get_test_paths(self):
        return self.rgb_paths, self.results_path

    # this function returns the size of the image so it can be resized properly before saving
    def get_image_size(self):
        return self.rgb_sizes
 