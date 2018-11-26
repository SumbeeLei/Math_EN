#encoding:utf-8
from config import *
from .data_tools import *
from .equ_tools import *
from .chinese import convertChineseDigitsToArabic
import numpy as np
import json
import pdb
import jieba

#args = get_args()

class DataLoader():
    def __init__(self, args, emb_dim=128):
        self.args = args
        self.math23k_train_list = read_data_json("./data/train23k_processed.json")
        self.math23k_valid_list = read_data_json("./data/valid23k_processed.json")
        self.math23k_test_list = read_data_json("./data/test23k_processed.json")

        self.emb_vectors, self.vocab_list, self.decode_classes_list = self.preprocess_and_word2vec(emb_dim)
        self.vocab_dict = dict([(elem, idx) for idx, elem in enumerate(self.vocab_list)])
        self.vocab_len = len(self.vocab_list) 
        self.decode_classes_dict = dict([(elem, idx) for idx, elem in enumerate(self.decode_classes_list)])
        self.classes_len = len(self.decode_classes_list)   
        print ("data processed done!")

    def preprocess_and_word2vec(self, emb_dim):
        new_data ={}
        sentences = []
        equ_dict = {}
        for elem in self.math23k_train_list:
            sentence = elem['text'].strip().split(' ')
            if self.args.post_flag == False:
                equation = elem['target_template'][2:]#.strip().split(' ')
            else:
                equation = elem['target_norm_post_template'][2:]#.strip().split(' ')
            for equ_e in equation:
                if equ_e not in equ_dict:
                    equ_dict[equ_e] = 1
            sentences.append(sentence)
            for elem in sentence:
                new_data[elem] = new_data.get(elem, 0) + 1

        from gensim.models import word2vec
        model = word2vec.Word2Vec(sentences, size=emb_dim, min_count=1)

        token_list = ['PAD_token', 'UNK_token', 'END_token']
        ext_list = ['PAD_token', 'END_token']
        emb_vectors = []
        emb_vectors.append(np.zeros((emb_dim)))
        emb_vectors.append(np.random.rand((emb_dim))/1000.0)
        emb_vectors.append(np.random.rand((emb_dim))/1000.0)

        for k, v in new_data.items(): 
            token_list.append(k)
            emb_vectors.append(np.array(model.wv[k]))

        for equ_k in equ_dict.keys():
            ext_list.append(equ_k)
        print ("encode_len:", len(token_list), "decode_len:", len(ext_list))
        print ("de:",ext_list)
        for elem in ext_list:
            if elem not in token_list:
                token_list.append(elem)
                emb_vectors.append(np.random.rand((emb_dim))/1000.0)
        emb_vectors = np.array(emb_vectors)
         
        return emb_vectors, token_list, ext_list 

    def inverse_temp_to_num(self, equ_list, num_list):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        new_equ_list = []
        for elem in equ_list:
            if 'temp' in elem:
                index = alphabet.index(elem[-1])
                new_equ_list.append(str(num_list[index]))
            elif 'PI' == elem:
                new_equ_list.append('3.14')
            else:
                new_equ_list.append(elem)
        return new_equ_list

    def check_(self, equ, num_list, t_ans):
        equ_list = self.inverse_temp_to_num(equ, num_list)
        ans = post_solver(equ_list)
        print (t_ans, '--', ans, abs(float(t_ans) - float(ans)) < 1e-5  )
        
        
    def _data_batch_preprocess(self, data_batch, template_flag):
        batch_encode_idx = []
        batch_decode_idx = []
        batch_encode_len = []
        batch_decode_len = []

        batch_idxs = []
        batch_text = []
        batch_num_list = []
        batch_solution = []

        for elem in data_batch:
            idx = elem["id"]
            encode_sen = elem['text']
            encode_sen_idx = string_2_idx_sen(encode_sen.strip().split(' '), self.vocab_dict)
            batch_encode_idx.append(encode_sen_idx)
            batch_encode_len.append(len(encode_sen_idx))

            if template_flag == True:
                if self.args.post_flag == False:
                    decode_sen = elem['target_template'][2:]
                else:
                    decode_sen = elem['target_norm_post_template'][2:]
                #pdb.set_trace()
                #print decode_sen
                #try:
                #    self.check_(decode_sen, elem['num_list'], elem['answer'])
                #except:
                #    pass
                decode_sen.append('END_token')
                decode_sen_idx = string_2_idx_sen(decode_sen, self.vocab_dict)
                batch_decode_idx.append(decode_sen_idx)
                batch_decode_len.append(len(decode_sen_idx))

            batch_idxs.append(idx)
            batch_text.append(encode_sen)
            batch_num_list.append(elem['num_list'])
            batch_solution.append(elem['answer'])

        max_encode_len =  max(batch_encode_len)
        batch_encode_pad_idx = []

        if template_flag == True:
            max_decode_len =  max(batch_decode_len)
            batch_decode_pad_idx = []

        for i in range(len(data_batch)):
            encode_sen_idx = batch_encode_idx[i]
            encode_sen_pad_idx = pad_sen(\
                               encode_sen_idx, max_encode_len, self.vocab_dict['PAD_token'])
            batch_encode_pad_idx.append(encode_sen_pad_idx)

            if template_flag:
                decode_sen_idx = batch_decode_idx[i]
                decode_sen_pad_idx = pad_sen(\
                              decode_sen_idx, max_decode_len, self.vocab_dict['PAD_token'])
                              #decode_sen_idx, max_decode_len, self.decode_classes_dict['PAD_token'])
                batch_decode_pad_idx.append(decode_sen_pad_idx)

        batch_data_dict = dict()
        batch_data_dict['batch_encode_idx'] = batch_encode_idx
        batch_data_dict['batch_encode_len'] = batch_encode_len
        batch_data_dict['batch_encode_pad_idx'] = batch_encode_pad_idx

        batch_data_dict['batch_index'] = batch_idxs
        batch_data_dict['batch_text'] = batch_text
        batch_data_dict['batch_num_list'] = batch_num_list
        batch_data_dict['batch_solution'] = batch_solution

        if template_flag:
            batch_data_dict['batch_decode_idx'] = batch_decode_idx
            batch_data_dict['batch_decode_len'] = batch_decode_len
            batch_data_dict['batch_decode_pad_idx'] = batch_decode_pad_idx

        if len(data_batch) != 1:
            new_batch_data_dict = self._sorted_batch(batch_data_dict)
        else:
            new_batch_data_dict = batch_data_dict
        return new_batch_data_dict

    def _sorted_batch(self, batch_data_dict):
        new_batch_data_dict = dict()
        batch_encode_len = np.array(batch_data_dict['batch_encode_len'])
        sort_idx = np.argsort(-batch_encode_len)
        for key, value in batch_data_dict.items():
            new_batch_data_dict[key] = np.array(value)[sort_idx]
        return new_batch_data_dict


    def get_batch(self, data_list, batch_size, template_flag = False, verbose=0):
        batch_num = int(len(data_list)/batch_size)+1
        for idx in range(batch_num):
            batch_start = idx*batch_size
            batch_end = min((idx+1)*batch_size, len(data_list))
            #print batch_start, batch_end, len(data_list)
            batch_data_dict = self._data_batch_preprocess(data_list[batch_start: batch_end],\
                                                               template_flag)
            yield batch_data_dict
        
def test():
    args.post_flag = True
    data_loader = DataLoader(args)
    gen_data = data_loader.get_batch(data_loader.math23k_test_list, 32, True)
    for elem in gen_data:
        pass
    print ("test done")
        #print len(elem)
    #for elem in gen_data:
    #    pass
        #print elem.keys()
    #data_23k = DataMath23k(False) 
    #data_57k = DataUnlabel57k(False)
    #word2vec = Word2vec(data_23k, data_57k, False)
    #print word2vec.emb_vectors.shape

def test_57k():
    data_57k = DataUnlabel57k(False)

def test_w2v():
    data_23k = DataMath23k(False) 
    data_57k = DataUnlabel57k(False)
    word2vec = Word2vec(data_23k, data_57k, False)
#test()
#test_57k()
#test_w2v()
