"""
Loader for Mimic Dataset
"""
import numpy as np
import logging

import imageio
import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset


class MimicDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("MimicDataLoader")
        self.data_path = '../data/s3/mimic_iv__multiclass_multioutput__json_files'
        self.output = [
            "atelectasis",
            "cardiomegaly",
            "edema",
            "lung_opacity",
            "pleural_effusion",
            "pneumonia",
        ]
        
        if config.data_mode == "imgs":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "tabular_numpy_train":
            self.logger.info("Loading DATA...")
            self.data_path = "../data/s3/tensors"
            self.feature_names = np.loadtxt(self.data_path + '/tabular_feature_names.txt', delimiter=',', dtype='str')
            self.label_names = np.loadtxt(self.data_path + '/tabular_label_names.txt', delimiter=',', dtype='str')
            
            self.train_loader = DataLoader(
                torch.load(self.data_path + '/tabular_train.pt'),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            self.valid_loader = DataLoader(
                torch.load(self.data_path + '/tabular_valid.pt'),
                batch_size=self.config.batch_size,
            )
            
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
            self.logger.info("Loaded tabular_numpy_train")

        elif config.data_mode == "tabular_numpy_test":
            self.logger.info("Loading DATA...")
            self.data_path = "../data/s3/tensors"
            self.feature_names = np.loadtxt(self.data_path + '/tabular_feature_names.txt', delimiter=',', dtype='str')
            self.label_names = np.loadtxt(self.data_path + '/tabular_label_names.txt', delimiter=',', dtype='str')
            test = TensorDataset(self.data_path + '/tabular_test.pt')

            self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass

