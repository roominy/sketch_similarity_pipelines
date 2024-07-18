import numpy as np
import torch
import os
import functools
import torch.nn as nn

from PIL import Image
from .networks import ResnetGenerator, init_weights, print_network


class Pix2PixModel:

    def __init__(self, input_nc=3, output_nc=1, ngf=64, init_type='normal', use_cuda=True,
                 which_epoch='latest', pretrain_path='./src/pretrained_model/photo_sketching/', which_direction='AtoB',
                 suffix='contour'):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.init_type = init_type
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.which_epoch = which_epoch
        self.which_direction = which_direction
        self.suffix = suffix
        self.pretrain_path = pretrain_path
        self.initialize()

    def name(self):
        return 'Pix2PixModel'

    def initialize(self):

        # load/define network
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        self.netG = ResnetGenerator(self.input_nc, self.output_nc, self.ngf, norm_layer=norm_layer, n_blocks=9)

        init_weights(self.netG, init_type=self.init_type)

        self.load_network(self.netG, 'G', self.which_epoch)

        self.netG = self.netG.to(self.device)

        print('---------- Networks initialized -------------')
        print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.image_paths = input['paths']
        if 'w' in input:
            self.input_w = input['w']
        if 'h' in input:
            self.input_h = input['h']

    def forward(self, input):
        self.input = input['input'].to(self.device)
        self.image_paths = input['paths']
        if 'w' in input:
            self.input_w = input['w']
        if 'h' in input:
            self.input_h = input['h']
        return self.netG(self.input)

    # no backprop gradients
    def test(self, input):
        with torch.no_grad():
            return self.forward(input)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def write_image(self, input, output, out_dir):
        image_numpy = output.detach()[0][0].cpu().float().numpy()
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        image_pil = image_pil.resize((input['w'][0], input['h'][0]), Image.BICUBIC)
        name, _ = os.path.splitext(os.path.basename(input['paths'][0]))
        out_path = os.path.join(out_dir, self.suffix + '-' + name.split('-')[-1] + '.jpeg')
        image_pil.save(out_path)

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.pretrain_path, save_filename)
        network.load_state_dict(torch.load(save_path))
