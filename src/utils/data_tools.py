import numpy as np
import json
import re
import random
import pdb

def read_data_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_data_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def is_number(word):
    if word[0] == '(' and word[-1] == ')':
        return True
    #if word[0] == '(' and word[1].isdigit() and not word[-1].isdigit():
    #    return True
    if '(' in word and ')' in word and '/' in word and not word[-1].isdigit():
         return True
    if word[-1] == '%' and len(word)>1:
        return True
    if word[0].isdigit():
        return True
    try:
        float(word)
        return True
    except:
        return False
'''
mask number
1
'''
def split_num_and_unit(word):
    num = ''
    unit = ''
    for idx in range(len(word)):
        char = word[idx]
        if char.isdigit() or char in ['.', '/', '(', ')']:
            num += char
        else:
            unit += char
    return num, unit#.encode('utf-8')

def mask_num(seg_text_list, equ_str):
    alphas = 'abcdefghijklmnopqrstuvwxyz'
    num_list  = []
    mask_seg_text = []
    count = 0 
    for word in seg_text_list:
        if word == '':
            continue
        if is_number(word):
            mask_seg_text.append("temp_"+alphas[count])
            if '%' in word:
                mask_seg_text.append('%')
            num_list.append(word)
            count += 1
        else:
            mask_seg_text.append(word)
    mask_equ_list = []
    s_n = sorted([(w,i) for i,w in enumerate(num_list)], key=lambda x: len(str(x[0])), reverse=True)
    if '3.14%' not in equ_str and '3.1416' not in equ_str:
        equ_str = equ_str.replace('3.14', '&PI&', 15)
    for num, idx in s_n:
        num = num_list[idx]
        equ_str = equ_str.replace(num, '&temp_'+alphas[idx]+'&', 15)
    equ_list = []
    num_set = ['0','1','2','3','4','5','6','7','8','9','%', '.']
    for elem in equ_str.split('&'):
        if 'temp' in elem or 'PI' in elem:
            equ_list.append(elem)
        else:
            start = ''
            for char in elem:
                if char not in num_set:
                    if start != '':
                        equ_list.append(start)
                    equ_list.append(char)
                    start = ''
                else:
                    start += char
            if start != '':
                equ_list.append(start)
    reverse_equ_list = equ_list[::-1]
    #reverse_equ_list.append('END_token')
    #equ_list.append('END_token')
    return mask_seg_text, num_list, equ_list, reverse_equ_list

def extract_number_and_align_per(data_per):
    wp_id = data_per['iIndex']
    seg_text = data_per['sQuestion']
    equation = data_per['iEquation']
    equation = re.sub('\[', '(', equation)
    equation = re.sub('\]', ')', equation)
    ans = data_per['solution']

    num_list = []
    word_seg_list = []
    for word in seg_text.split(' '):
        if word == '' or word == ' ':
            continue 
        if is_number(word):
            if '(' not in word and '%' not in word and not word[-1].isdigit():
                num, unit = split_num_and_unit(word)
                word_seg_list.append(num)
                word_seg_list.append(unit)
            elif '(' in word and word[-1] != ')' and not word[-1].isdigit():
                num, unit = split_num_and_unit(word)
                word_seg_list.append(num)
                word_seg_list.append(unit)
            else:
                word_seg_list.append(word) 
        else:
            word_seg_list.append(word)
                 
    mask_seg_text_list, num_list, mask_equ_list, mask_inv_equ_list = mask_num(word_seg_list, equation)
    return [mask_seg_text_list, mask_equ_list, num_list, mask_inv_equ_list]


def extract_number_and_align_all(data_all):
    data_dict = {}
    for elem in data_all:
        key = elem['iIndex']
        if str(key) in ['8882', '10430']:
            continue
        tuple_per = extract_number_and_align_per(elem)
        data_dict[key] = {}
        data_dict[key]['tuple'] = tuple_per
        data_dict[key]['sQuestion'] = elem['sQuestion'][:]
        data_dict[key]['iEquation'] = elem['iEquation'][:]
        data_dict[key]['solution'] = elem['solution'][:]
        data_dict[key]['iIndex'] = elem['iIndex']
    return data_dict

'''
2
'''
def pad_sen(sen_idx_list, max_len=115, pad_idx=1):
    return sen_idx_list + [pad_idx]*(max_len-len(sen_idx_list))
    
def pad_sen_inv(sen_idx_list, max_len=115, pad_idx=1):
    return [pad_idx]*(max_len-len(sen_idx_list)) + sen_idx_list

def string_2_idx_sen(sen,  vocab_dict):
    new = []
    for word in sen:
        if word not in vocab_dict:
             new.append(vocab_dict['UNK_token'])
        else:
             new.append(vocab_dict[word])
    return new

def process_dict_for_one_emb(vocab1_list, vocab2_list):
    new_vocab_list = vocab1_list[:]
    for word in vocab2_list:
        if word not in new_vocab_list:
             new_vocab_list.append(word)
    new_vocab_dict = dict([(word, idx) for idx, word in enumerate(new_vocab_list)])
    return new_vocab_list, new_vocab_dict

def split_train_test(data_dict, random_seed):
    '''
    modified later using cross-validation
    '''
    random.seed(random_seed)
    total_num = len(data_dict.keys())
    data_list = data_dict.items()
    random.shuffle([(key, info_dict) for key, info_dict in data_list])
    train_list = data_list[:total_num/10*9]
    test_list = data_list[total_num/10*9:]
    del data_list
    return train_list, test_list

def split_by_feilong_23k(data_dict):
    t_path = "./data/id_ans_test" 
    v_path = "./data/valid_ids.json"
    valid_ids = read_data_json(v_path)
    test_ids = []
    with open(t_path, 'r') as f:
        for line in f:
            test_id = line.strip().split('\t')[0]
            test_ids.append(test_id)
    train_list = []
    test_list = []
    valid_list = []
    for key, value in data_dict.items():
        if key in test_ids:
            test_list.append((key, value))
        elif key in valid_ids:
            valid_list.append((key, value))
        else:
            train_list.append((key, value))
    return train_list, valid_list, test_list
    


def split_train_valid_test(data_dict, random_seed):
    '''
    modified later using cross-validation
    '''
    random.seed(random_seed)
    total_num = len(data_dict.keys())
    data_list = data_dict.items()
    random.shuffle([(key, info_dict) for key, info_dict in data_list])
    train_list = data_list[:total_num/10*8]
    validate_list = data_list[total_num/10*8: total_num/10*9]
    test_list = data_list[total_num/10*9:]
    del data_list
    return train_list, validate_list, test_list
