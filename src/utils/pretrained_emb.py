# encoding: utf-8
import gensim
import numpy as np
import json
import re
import jieba
import pdb
from data_tools import *

def read_data_math23k():
    with open("./data/math_json_wy.json", 'r') as f:
        return json.load(f)

def read_data_5w7():
     with open('./data/ex_data/external_data.json', 'r') as f:
         external_data = json.load(f)
     return external_data

def preprocess_data_23k(data_all):
    data_dict = extract_number_and_align_all(data_all)
    #print len(data_dict)
    #pdb.set_trace()
    return data_dict

def find_fenshu(s):
    start = s.find('\\frac')
    l = []
    loc = []
    while start != -1:
        mid = s.find('}', start+1)
        end = s.find('}', mid+1)
        l.append(s[start:end+1])
        loc.append((start, end+1))
        start = s.find('\\frac', end+1)
    return l, loc

def find_all_nums(ss):
    num = ''
    flag = 0
    l = []
    loc = []
    for i in range(len(ss)):
        elem = ss[i]
        try:
            float(elem)
            flag = 1
            num += elem
        except:
            if (elem == '.' or elem=='%') and len(num)>0:
                flag = 1
                num += elem
            else:
                flag = 0
        if flag == 0 and len(num) > 0:
            l.append(num)
            loc.append(i-1)
            num = ''
    if len(num) > 0:
        l.append(num)
        loc.append(i-1)
    return l, loc

def is_fenshu_loc(idx, fenshu_loc):
    for loc in fenshu_loc:
        if idx >= loc[0]  and idx <= loc[1]:
            return True
    return False

def obtain_num(fenshu_l, fenshu_loc, nums_l, nums_loc):
    l = []
    fenshu_l = []
    for i in range(len(nums_l)):
        if is_fenshu_loc(nums_loc[i], fenshu_loc):
            fenshu_l.append(nums_l[i])
            if len(fenshu_l) >= 2:
                l.append(fenshu_l[:])
                fenshu_l = []
        else:
            l.append(nums_l[i])
    #print l
    new_l = []
    fen_l = []
    for elem in l:
        if type(elem) == type([]):
            s = '('+elem[0]+'/'+elem[1]+')'
            new_l.append(s)
            fen_l.append(s)
        else:
            new_l.append(elem)
    return new_l, fen_l

def add_space_temp_in_text(ss):
    idx = 0
    while 1:
        idx = ss.find('temp', idx)
        if idx == -1:
            break
        n_idx = idx + 5
        ss = ss[:n_idx]+' '+ss[n_idx:]
        idx = n_idx
    return ss

def replace_and_cut(num_list, ss):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    num_list = sorted(enumerate(num_list), key = lambda k: len(k[1]), reverse=True)
    #print '----', num_list
    #print '----', ss.encode('utf-8')
    for i in range(len(num_list)):
        num = num_list[i][1]
        idx = num_list[i][0]
        ss = re.sub(num, 'temp'+ alphabets[idx], ss)
    ss = add_space_temp_in_text(ss)
    split_ss = list(jieba.cut(ss, cut_all=False))
    for i in range(len(split_ss)):
        if 'temp' in split_ss[i]:
            split_ss[i] = split_ss[i][0:4]+'_'+split_ss[i][-1]
    split_ss = filter(lambda a: a!=' ', split_ss)
    #print ' '.join(split_ss).encode('utf-8')
    #print 
    return split_ss

def preprocess_data_5w7(data_dict ):
    new_data = {}
    for k, v in data_dict.items():
        if len(v['text']) >2:
            s = v['text'].strip()
        else:
            s = v['problemtext'].strip()
        ans = v['ans'].strip()
        s = re.sub(u'choice\{0\}', u'多少', s)
        #ss = s.encode('utf-8')
        ss = s
        #print s.encode('utf-8')
        fenshu_l, fenshu_loc = find_fenshu(ss)
        nums_l, nums_loc = find_all_nums(ss)
        num_list, fen_l = obtain_num(fenshu_l, fenshu_loc, nums_l, nums_loc)
        if len(num_list) > 26:
            continue
        #print nums_l, nums_loc
        #print num_list, fen_l, fenshu_loc

        ss_new = ''
        emmm = 0 
        for i in range(len(fenshu_l)):
            start = fenshu_loc[i][0]
            end = fenshu_loc[i][1]
            ss[:start]
            try:
                fen_l[i]
            except:
                #print fen_l, i, fenshu_loc, fenshu_l
                break

            ss[end:]
            ss_new += ss[emmm:start]+fen_l[i]#+ss[end:]
            emmm = end
        ss_new += ss[emmm:]
        ss_list = replace_and_cut(num_list, ss_new)
        new_data[k] = {}
        new_data[k]['text'] = ss_new
        new_data[k]['cut'] = ss_list
        #new_data[k]['
        #print ss_new.encode('utf-8')
        #print 
    print len(new_data)
    return new_data

def word2vec():
    '''
    dict: pad_token, end_token, unk_token, and equ symbols
    save emb as numpy
    '''
    data_23k_dict = read_data_math23k()
    data_5w7_dict = read_data_5w7()

    data_23k_dict = preprocess_data_23k(data_23k_dict)
    data_5w7_dict = preprocess_data_5w7(data_5w7_dict)

    new_data ={}
    sentences = []
    for k, v in data_23k_dict.items():
        cut = v['tuple'][0]
        sentences.append(cut)
        for elem in cut:
            new_data[elem] = new_data.get(elem, 0) + 1

    for k, v in data_5w7_dict.items():
        cut = v['cut'] # list 
        sentences.append(cut)
        for elem in cut:
            new_data[elem] = new_data.get(elem, 0) + 1

    from gensim.models import word2vec
    model = word2vec.Word2Vec(sentences, size=128, min_count=1)
    token_list = ['PAD_token', 'UNK_token', 'END_token']
    #np.zeros((1 
    emb_vectors = []
    emb_vectors.append(np.zeros((128)))
    emb_vectors.append(np.random.rand((128))/1000.0)
    emb_vectors.append(np.random.rand((128))/1000.0)
    for k, v in new_data.items():
        #print k.encode('utf-8')
        #print model.wv[k]
        #print 
        token_list.append(k)
        emb_vectors.append(np.array(model.wv[k]))
    emb_vectors = np.array(emb_vectors)
    np.save("./data/emb.npy", emb_vectors)
    with open("./data/token_list.json", 'w') as f:
        json.dump(token_list, f)
    load_emb = np.load('./data/emb.npy')
    pdb.set_trace()
    print 'ok'
    #for k, v in new_data.items():
    #    print k.encode('utf-8'), v


word2vec()
