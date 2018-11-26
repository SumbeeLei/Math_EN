from __future__ import print_function, division

import torch
from torch.autograd import Variable
import numpy as np
import json

from loss import NLLLoss
from equ_tools import *
import pdb

class Evaluator(object):
    def __init__(self, loss=NLLLoss(), cuda_use=False, batch_size=64):
        self.loss = loss
        self.cuda_use  = cuda_use
        if cuda_use:
            self.loss.cuda()
        self.batch_size = batch_size

    def _convert_f_e_2_d_sybmbol(self, data_loader, target_variable):
        new_variable = []
        batch,colums = target_variable.size()
        for i in range(batch):
            tmp = []
            for j in range(colums):
                idx = data_loader.decode_classes_dict[ \
                      data_loader.vocab_list[target_variable[i][j].data[0]]]
                tmp.append(idx)
            new_variable.append(tmp)
        return Variable(torch.LongTensor(np.array(new_variable)))

    def print_analysis(self, target_var, seq_var, batch_data_dict, batch_info, data_loader):
        ret_data = {}
        ret_data['iIndex'] = batch_info[0]
        ret_data['text'] = batch_info[1]
        ret_data['equ'] = batch_info[2]
        ret_data['num_list'] = batch_info[3]
        print ('index', batch_info[0])
        print ('text:', batch_info[1].encode('utf-8'))
        print ('equ:', batch_info[2])
        print (target_var.data.cpu().numpy())
        string = ''
        for num in target_var.data.cpu().numpy():
            if num == 0:
                break
            string += data_loader.decode_classes_list[num]+' '
        print (string)
        print (seq_var.data.cpu().numpy())
        ret_data['target_temp'] = string
        string = ''
        for num in seq_var.data.cpu().numpy():
            if num == 0:
                break
            string += data_loader.decode_classes_list[num]+' '
        print (string)
        ret_data['gen_temp'] = string
        return ret_data


    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem: 
                index = alphabet.index(elem[-1])
                new_equ_list.append(num_list[index])
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def idx_to_equation(self, seq_var, num_list, classes_list):
        #print (seq_var.data.cpu().numpy())
        #print (num_list)
        equ_list = []
        for num in seq_var.data.cpu().numpy():
            if num == 0:
                break
            equ_list.append(classes_list[num])
        equ_list = equ_list[2:]
        #print (equ_list)
        #print (num_list)
        try:
            equ_list = self.inverse_temp_to_num(equ_list, num_list)
        except:
            return 'inverse error'
        equ_string = ''.join(equ_list)
        equ_list = split_equation(equ_string)
        try:
            ans = solve_equation(equ_list)
            float(ans)
            return ans
        except:
            return 'compute error'

        #print (solve_equation(equ_list))

    def gt_compute(self, solution):
        o_ans = solution[0]
        if '%' in o_ans:
            o_ans = float(o_ans[:-1])/100
        else:
            if '(' in o_ans :
                o_ans = split_equation(o_ans)
                o_ans = solve_equation(o_ans)
            o_ans = float(o_ans)
        return o_ans


    def evaluate(self, model, data_loader, train=False, evaluate_type=0, print_flag=False, extr_flag=False):
        if train == False:
            batch_generator = data_loader.get_batch('test', self.batch_size) 
            total_num = len(data_loader.test_list)
        else:
            train_list = data_loader.math23k_train_list
            batch_generator = data_loader.get_batch('train', self.batch_size) 
            total_num = len(data_loader.train_list)
        if evaluate_type == 0:
            teacher_forcing_ratio = 0.0
        else:
            teacher_forcing_ratio = 1.0

        count = 0
        acc_right = 0

        data_rectify_dict = {'true': {}, 'false':{}}
        extract_dict = {}

        for batch_data_dict in batch_generator:
            input_variables = batch_data_dict['batch_encode_pad_idx']
            input_lengths = batch_data_dict['batch_encode_len']
            target_variables = batch_data_dict['batch_decode_pad_idx']
            target_lengths = batch_data_dict['batch_decode_len']

            batch_index = batch_data_dict['batch_index']
            batch_text = batch_data_dict['batch_text']
            batch_equ = batch_data_dict['batch_equ']
            batch_num_list = batch_data_dict['batch_num_list']
            batch_solution = batch_data_dict['batch_solution']

            input_variables = Variable(torch.LongTensor(input_variables))
            target_variables = Variable(torch.LongTensor(target_variables))
            if self.cuda_use:
                input_variables = input_variables.cuda()
            if self.cuda_use:
                target_variables = target_variables.cuda()

            (decoder_outputs, decoder_hidden, other), encoder_hidden = \
                                   model(input_variables, input_lengths, \
                                     target_variables, teacher_forcing_ratio=teacher_forcing_ratio)

            #print (encoder_hidden)#.size()
            #print (encoder_hidden[0].squeeze(0).cpu().data.numpy())
            hidden_np = encoder_hidden[0].squeeze(0).cpu().data.numpy()
            #pdb.set_trace()

            seqlist = other['sequence']
            seq_var = torch.cat(seqlist, 1)
            target_variables = self._convert_f_e_2_d_sybmbol(data_loader, target_variables)
            if self.cuda_use:
               target_variables = target_variables.cuda()

            for i in range(self.batch_size):
                flag = 0
                if extr_flag == True:
                    print (batch_index[i])
                    extract_dict[batch_index[i]] = hidden_np[i].tolist()
                    
                for j in range(target_variables.size(1)):
                    if seq_var[i][j].data[0] != 0 and \
                                target_variables[i][j].data[0] == seq_var[i][j].data[0]:
                        pass
                    elif seq_var[i][j].data[0] == 0:
                        flag = 1
                        break
                    else:
                        break
                if print_flag == True:
                    ret_data = self.print_analysis(target_variables[i], seq_var[i], batch_data_dict, [batch_index[i], batch_text[i], batch_equ[i], batch_num_list[i]], data_loader)
                if 1:
                    ans = self.idx_to_equation(seq_var[i], batch_num_list[i], data_loader.decode_classes_list)
                    o_ans = self.gt_compute(batch_solution[i])
                    if 'error' in ans: 
                        acc_right += 0
                    else:
                        #print (ans, o_ans,)
                        if abs(float(ans) - float(o_ans)) <1e-5:
                            if print_flag == True:
                                print ('True')
                                ret_data['target_ans'] = o_ans
                                ret_data['gen_ans'] = ans
                                data_rectify_dict['true'][batch_index[i]] = ret_data
                            acc_right+=1
                        else:
                            if print_flag == True:
                                ret_data['target_ans'] = o_ans
                                ret_data['gen_ans'] = ans
                                data_rectify_dict['false'][batch_index[i]] = ret_data
                                print ('False')
                            pass
                        #print ()
                        #print ()
                    
                count += flag

            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step]
            #pdb.set_trace() 
        #print (count, total_num, count*1.0/total_num)
        if print_flag == True:
            if train == False:
                rl_filename  = 'rl_data_test.json'
            else:
                rl_filename  = 'rl_data_train.json'
            with open(rl_filename, 'w') as f:
                json.dump(data_rectify_dict, f)
        if extr_flag == True:
            if train == False:
                rl_filename  = 'extract_hidden_test.json'
            else:
                rl_filename  = 'extract_hidden_train.json'
            with open(rl_filename, 'w') as f:
                json.dump(extract_dict, f)
        return count*1.0/total_num, acc_right*1.0/total_num


