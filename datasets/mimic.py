"""
Loader for Mimic Dataset
"""
import numpy as np
import pandas as pd
import logging

import imageio
import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset


class MimicDatasetTabular(Dataset):
    def __init__(self, path):
        self.feature_names = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp','dbp']
        self.label_names = ['atelectasis', 'cardiomegaly', 'edema', 'lung_opacity', 'pleural_effusion', 'pneumonia']
        self.df = pd.read_csv(path, usecols=[*self.feature_names, *self.label_names])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        features_df = self.df[self.feature_names].iloc[index]
        labels_df = self.df[self.label_names].iloc[index]

        features = torch.tensor(features_df.values, dtype=torch.float32)
        atelectasis = torch.tensor([labels_df['atelectasis']], dtype=torch.float32)
        cardiomegaly = torch.tensor([labels_df['cardiomegaly']], dtype=torch.float32)
        edema = torch.tensor([labels_df['edema']], dtype=torch.float32)
        lung_opacity = torch.tensor([labels_df['lung_opacity']], dtype=torch.float32)
        pleural_effusion = torch.tensor([labels_df['pleural_effusion']], dtype=torch.float32)
        pneumonia = torch.tensor([labels_df['pneumonia']], dtype=torch.float32)
        
        return {
            'features': features,
            'atelectasis': atelectasis,
            'cardiomegaly': cardiomegaly,
            'edema': edema,
            'lung_opacity': lung_opacity,
            'pleural_effusion': pleural_effusion,
            'pneumonia': pneumonia
        }

class MimicDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("MimicDataLoader")
        self.data_path = './data/s3/mimic_iv__multiclass_multioutput__json_files'
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
            self.data_path = "./data/s3"
            self.feature_names = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp','dbp']
            self.label_names = ['atelectasis', 'cardiomegaly', 'edema', 'lung_opacity', 'pleural_effusion', 'pneumonia']
            
            train_set = MimicDatasetTabular(self.data_path + '/tabular_train.csv')
            valid_set = MimicDatasetTabular(self.data_path + '/tabular_valid.csv')
            
            self.train_loader = DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config.batch_size,
            )
            
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
            self.logger.info("Loaded tabular_numpy_train")

        elif config.data_mode == "tabular_numpy_test":
            self.logger.info("Loading DATA...")
            self.data_path = "./data/s3"
            self.feature_names = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp','dbp']
            self.label_names = ['atelectasis', 'cardiomegaly', 'edema', 'lung_opacity', 'pleural_effusion', 'pneumonia']

            test_set = MimicDatasetTabular(self.data_path + '/tabular_test.csv')

            self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
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

