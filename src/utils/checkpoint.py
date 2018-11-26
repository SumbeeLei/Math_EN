from __future__ import print_function
import os
import time
import shutil

import torch

class Checkpoint():

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, optimizer, epoch, step, train_acc_list, test_acc_list, loss_list, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.train_acc_list = train_acc_list
        self.test_acc_list = test_acc_list 
        self.loss_list = loss_list 
        self._path = path
        self.flag = 0

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save_according_name(self, experiment_dir, filename, args=None):
        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, filename)
        path = self._path
        '''
        if self.flag == 0:
            self.flag = 1
            with open("./record.log", 'a') as f:
                f.write(path+'\n')
                f.write('parameters: {}'.format(vars(args)))
                f.write('\n\n')
        '''
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer,
                    'train_acc_list': self.train_acc_list,
                    'test_acc_list': self.test_acc_list,
                    'loss_list': self.loss_list
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))
        return path

    def save_according_time(self, experiment_dir, args):
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path
        if self.flag == 0:
            self.flag = 1
            with open("./record.log", 'a') as f:
                f.write(path+'\n')
                f.write('parameters: {}'.format(vars(args)))
                f.write('\n\n')

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer,
                    'train_acc_list': self.train_acc_list,
                    'test_acc_list': self.test_acc_list,
                    'loss_list': self.loss_list
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        return path

    @classmethod
    def load(cls, path):
        print("Loading checkpoints from {}".format(path))
        resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
        model = torch.load(os.path.join(path, cls.MODEL_NAME))
        model.flatten_parameters()
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          train_acc_list = [],
                          test_acc_list = [],
                          loss_list = [],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])

    @classmethod
    def get_certain_checkpoint(cls, experiment_path, filename):
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        return os.path.join(checkpoints_path, filename)
