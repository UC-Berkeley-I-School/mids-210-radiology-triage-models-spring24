import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent
from graphs.models.tabularAttempt import TabularModel
from graphs.losses.tabular_loss import MultiHeadBCE
from datasets.mimic import MimicDataLoader

# import your classes here

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class TabularAttemptAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = TabularModel(self.config)

        # define data_loader
        self.data_loader = MimicDataLoader(self.config)

        # define loss
        self.loss = MultiHeadBCE()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.betas[0], self.config.betas[1]))

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_loss = 1

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.device = torch.device("cuda")
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='Tabular')

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        file_name = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # save state
        torch.save(state, self.config.checkpoint_dir + file_name)
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch += 1
            self.train_one_epoch()
            valid_loss = self.validate()
            is_best = valid_loss < self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # set model to training mode
        self.model.train()
        counter = 0
        epoch_loss = AverageMeter()
        
        for i, data in tqdm(enumerate(self.data_loader.train_loader), total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch)):
            self.current_iteration += 1

            # extract features and labels
            features = data['features'].to(self.device)
            target1 = data['atelectasis'].to(self.device)
            target2 = data['cardiomegaly'].to(self.device)
            target3 = data['edema'].to(self.device)
            target4 = data['lung_opacity'].to(self.device)
            target5 = data['pleural_effusion'].to(self.device)
            target6 = data['pneumonia'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            targets = (target1, target2, target3, target4, target5, target6)
            loss = self.loss(outputs, targets)
            epoch_loss.update(loss.item())

            loss.backward()
            self.optimizer.step()
            # if i % self.config.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         self.current_epoch, i * len(data), len(self.data_loader.train_loader.dataset),
            #            100. * i / len(self.data_loader.train_loader), loss.item()))
        self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
        print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        epoch_loss = AverageMeter()
        for i, data in tqdm(enumerate(self.data_loader.valid_loader), total=self.data_loader.valid_iterations,
                          desc="Validating Epoch-{}-".format(self.current_epoch)):
            # extract features and labels
            features = data['features'].to(self.device)
            target1 = data['atelectasis'].to(self.device)
            target2 = data['cardiomegaly'].to(self.device)
            target3 = data['edema'].to(self.device)
            target4 = data['lung_opacity'].to(self.device)
            target5 = data['pleural_effusion'].to(self.device)
            target6 = data['pneumonia'].to(self.device)

            pred = self.model(features)
            targets = (target1, target2, target3, target4, target5, target6)
            cur_loss = self.loss(pred, targets)
            epoch_loss.update(cur_loss.item())

        self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val))
        return epoch_loss.val

    def test(self):
        """
        test function TODO
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("finalizing operation")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
