import sys
import random
import numpy as np
sys.path.append('/home/demolwang/demolwang/math_word_problem/critical-based/auto_mwp/seq2seq_v2/src')
from model import EncoderRNN, DecoderRNN_1, Seq2seq
from utils import NLLLoss, Optimizer, Checkpoint, Evaluator

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import pdb

class SupervisedTrainer(object):
    def __init__(self, vocab_dict, vocab_list, decode_classes_dict, decode_classes_list, cuda_use, \
                  loss, print_every, teacher_schedule, checkpoint_dir_name):
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.class_dict = decode_classes_dict
        self.class_list = decode_classes_list

        random_seed = 10
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.cuda_use = cuda_use
        self.loss = loss 
        if self.cuda_use == True:
            self.loss.cuda()

        self.print_every = print_every

        self.teacher_schedule = teacher_schedule

        Checkpoint.CHECKPOINT_DIR_NAME = checkpoint_dir_name

        
    def _convert_f_e_2_d_sybmbol(self, target_variable):
        new_variable = []
        batch,colums = target_variable.size()
        for i in range(batch):
            tmp = []
            for j in range(colums):
                #idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].data[0]]]
                idx = self.decode_classes_dict[self.vocab_list[target_variable[i][j].item()]]
                tmp.append(idx)
            new_variable.append(tmp)
        return Variable(torch.LongTensor(np.array(new_variable)))


    def _train_batch(self, input_variables, input_lengths,target_variables, target_lengths, model,\
                         template_flag, teacher_forcing_ratio, mode, batch_size, post_flag):
        decoder_outputs, decoder_hidden, symbols_list = \
                                      model(input_variable = input_variables, 
                                      input_lengths = input_lengths, 
                                      target_variable = target_variables, 
                                      template_flag = template_flag,
                                      teacher_forcing_ratio = teacher_forcing_ratio, 
                                      mode = mode,
                                      use_rule = False,
                                      use_cuda = self.cuda_use,
                                      vocab_dict = self.vocab_dict,
                                      vocab_list = self.vocab_list,
                                      class_dict = self.class_dict,
                                      class_list = self.class_list)
        # cuda
        target_variables = self._convert_f_e_2_d_sybmbol(target_variables)
        if self.cuda_use:
            target_variables = target_variables.cuda()

        pad_in_classes_idx = self.decode_classes_dict['PAD_token']
        batch_size = len(input_lengths)

        match = 0
        total = 0

        seq = symbols_list
        seq_var = torch.cat(seq, 1)

        self.loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            # cuda step_output = step_output.cuda()
            if self.cuda_use:
                step_output = step_output.cuda()
            target = target_variables[:, step]
            self.loss.eval_batch(step_output.contiguous().view(batch_size, -1), target)
            non_padding = target.ne(pad_in_classes_idx)
            correct = seq[step].view(-1).eq(target).masked_select(non_padding).sum().item()#data[0]
            match += correct
            total += non_padding.sum().item()#.data[0]

        right = 0
        for i in range(batch_size):
            for j in range(target_variables.size(1)):
                #if target_variables[i][j].data[0] != pad_in_classes_idx  and \
                              #target_variables[i][j].data[0] == seq_var[i][j].data[0]:
                if target_variables[i][j].item() != pad_in_classes_idx  and \
                              target_variables[i][j].item() == seq_var[i][j].item():
                    continue
                #elif target_variables[i][j].data[0] == 1:
                elif target_variables[i][j].item() == 1:
                    right += 1
                    break
                else:
                    break

        model.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.get_loss(), [right, match, total]


    def _train_epoches(self, data_loader, model, batch_size, start_epoch, start_step, n_epoch, \
                            mode, template_flag, teacher_forcing_ratio, post_flag):
        print_loss_total = 0

        train_list = data_loader.math23k_train_list
        test_list = data_loader.math23k_test_list
        valid_list = data_loader.math23k_valid_list
        train_list = train_list + valid_list# = data_loader.math23k_valid_list
        steps_per_epoch = len(train_list)/batch_size
        total_steps = steps_per_epoch * n_epoch

        step = start_step
        step_elapsed = 0

        threshold = [0]+[1]*9

        max_ans_acc = 0 

        for epoch in range(start_epoch, n_epoch + 1):
            epoch_loss_total = 0

            #marker if self.teacher_schedule:

            batch_generator = data_loader.get_batch(train_list, batch_size, True)
            
            right_count = 0
            match = 0
            total_m = 0
            total_r = 0

            model.train(True)
            for batch_data_dict in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables = batch_data_dict['batch_encode_pad_idx']
                input_lengths = batch_data_dict['batch_encode_len']
                target_variables = batch_data_dict['batch_decode_pad_idx']
                target_lengths = batch_data_dict['batch_decode_len']

                #cuda
                input_variables = Variable(torch.LongTensor(input_variables))
                target_variables = Variable(torch.LongTensor(target_variables))

                if self.cuda_use:
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                loss, com_list = self._train_batch(input_variables = input_variables, 
                                                   input_lengths = input_lengths, 
                                                   target_variables = target_variables, 
                                                   target_lengths = target_lengths, 
                                                   model = model, 
                                                   template_flag = template_flag,
                                                   teacher_forcing_ratio = teacher_forcing_ratio,
                                                   mode = mode, 
                                                   batch_size = batch_size,
                                                   post_flag = post_flag)


                right_count += com_list[0]
                total_r += batch_size

                match += com_list[1]
                total_m += com_list[2]

                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    print ('step: %d, Progress: %d%%, Train %s: %.4f, Teacher_r: %.2f' % (
                           step,
                           step*1.0 / total_steps * 100,
                           self.loss.name,
                           print_loss_avg,
                           teacher_forcing_ratio))


            model.eval()
            train_temp_acc, train_ans_acc =\
                                        self.evaluator.evaluate(model = model,
                                                                data_loader = data_loader,
                                                                data_list = train_list,
                                                                template_flag = True,
                                                                batch_size = batch_size,
                                                                evaluate_type = 0,
                                                                use_rule = False,
                                                                mode = mode,
                                                                post_flag=post_flag)
            #valid_temp_acc, valid_ans_acc =\
            #                            self.evaluator.evaluate(model = model,
            #                                                    data_loader = data_loader,
            #                                                    data_list = valid_list,
            #                                                    template_flag = True,
            #                                                    batch_size = batch_size,
            #                                                    evaluate_type = 0,
            #                                                    use_rule = False,
            #                                                    mode = mode,
            #                                                    post_flag=post_flag)
            test_temp_acc, test_ans_acc =\
                                        self.evaluator.evaluate(model = model,
                                                                data_loader = data_loader,
                                                                data_list = test_list,
                                                                template_flag = True,
                                                                batch_size = batch_size,
                                                                evaluate_type = 0,
                                                                use_rule = False,
                                                                mode = mode,
                                                                post_flag=post_flag)
            self.train_acc_list.append((epoch, step, train_ans_acc))
            self.test_acc_list.append((epoch, step, test_ans_acc))
            self.loss_list.append((epoch, epoch_loss_total/steps_per_epoch))

            #if valid_ans_acc > max_ans_acc:
            #    max_ans_acc = valid_ans_acc
            #    th_checkpoint = Checkpoint(model=model,
            #                               optimizer=self.optimizer,
            #                               epoch=epoch,
            #                               step=step,
            #                               train_acc_list = self.train_acc_list,
            #                               test_acc_list = self.test_acc_list,
            #                               loss_list = self.loss_list).\
            #                                 save_according_name("./experiment", 'best')

            #print ("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, validate_acc: %.2f, %.2f, test_acc: %.2f, %.2f"\
            #      % (epoch, step, train_temp_acc, train_ans_acc, valid_temp_acc, valid_ans_acc, test_temp_acc, test_ans_acc))
            print ("Epoch: %d, Step: %d, train_acc: %.2f, %.2f, test_acc: %.2f, %.2f"\
                  % (epoch, step, train_temp_acc, train_ans_acc, test_temp_acc, test_ans_acc))


    def train(self, model, data_loader, batch_size, n_epoch, template_flag, \
                        resume=False, optimizer=None, mode=0, teacher_forcing_ratio=0, post_flag=False):
        self.evaluator = Evaluator(vocab_dict = self.vocab_dict,
                                   vocab_list = self.vocab_list,
                                   decode_classes_dict = self.decode_classes_dict,
                                   decode_classes_list = self.decode_classes_list,
                                   loss = NLLLoss(),
                                   cuda_use = self.cuda_use)
        if resume:
            checkpoint_path = Checkpoint.get_certain_checkpoint("./experiment", "best")
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            start_step = resume_checkpoint.step
            self.train_acc_list = resume_checkpoint.train_acc_list
            self.test_acc_list = resume_checkpoint.test_acc_list
            self.loss_list = resume_checkpoint.loss_list
        else:
            start_epoch = 1
            start_step = 0
            self.train_acc_list = []
            self.test_acc_list = []
            self.loss_list = []

            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=0)
            self.optimizer = optimizer

        self._train_epoches(data_loader = data_loader, 
                            model = model, 
                            batch_size = batch_size,
                            start_epoch = start_epoch, 
                            start_step = start_step, 
                            n_epoch = n_epoch,
                            mode = mode,
                            template_flag = template_flag,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            post_flag = post_flag)


