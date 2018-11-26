#encoding:utf-8
from data_tools import *
import numpy as np
import json
import pdb
import jieba


class DataMath23k():
    def __init__(self):
        filename = "./data/math_json_wy.json"
        self._data_list = read_data_json(filename)
        self._data_dict = extract_number_and_align_all(self._data_list)
        self.convert_style()

    def convert_style(self):
        '''
        idx: temp_text_str, temp_list, num_list, ans
        '''
        data_dict = {}
        for key, elem in self._data_dict.items():
            elem_tuple = elem['tuple']
            data_dict[key] = {'text':'','target_template':[],'gen_template':[],'num_list':[],'ans':''}
            data_dict[key]['text'] = ' '.join(elem_tuple[0])
            data_dict[key]['target_template'] = elem_tuple[2]
            data_dict[key]['num_list'] = elem_tuple[1]
            data_dict[key]['ans'] = elem['solution']
        self.data_dict = data_dict
            
        

class DataUnlabel57k():
    def __init__(self):
        filename = "./data/ex_data/external_data.json"
        self._data_dict = read_data_json(filename)
        self.attain_style()

    def split_num_and_unit(self, num_unit):
        num = ''
        unit = ''
        for idx in range(len(num_unit)):
            char = num_unit[idx]
            if char.isdigit() or char in ['.', '/', '(', ')', '%']:
                num += char
            else:  
                unit += char
        return num, unit

    def process_frac(self, num_elem):
        start = num_elem.find('\\frac')
        elem = ''
        if start > 0 :
            num_pre = num_elem[:start]
        else:
            num_pre = ''
        if start != -1:
            #print num_elem.encode('utf-8'),
            s_1 = num_elem.find('{', start+1)
            e_1 = num_elem.find('}', s_1+1)
            s_2 = num_elem.find('{', e_1)
            e_2 = num_elem.find('}', s_2)
            #print '('+num_elem[s_1+1:e_1]+'/'+ num_elem[s_2+1:e_2]+')'
            elem = '('+num_elem[s_1+1:e_1]+'/'+ num_elem[s_2+1:e_2]+')' + num_elem[e_2+1:]
            elem = num_pre+elem
            return elem
        else:
            return num_elem

    def chinese_num(self, num_elem):
        chinese_nums = [u'一',u'二',u'三',u'四',u'五',u'六',u'七',u'八',u'九',u'十']
        chinese_dict = dict([(num, idx+1) for idx, num in enumerate(chinese_nums)])
        flag = False
        for chinese_num in chinese_nums:
            #print [chinese_num], [num_elem], '--', chinese_num.encode('utf-8'), num_elem.encode('utf-8')
            if chinese_num in num_elem:
                flag = True
                break
        elem = ''
        if flag == True:
            for per in num_elem:
                if per in chinese_dict:
                    elem += str(chinese_dict[per])
                else:
                    elem += per
            return elem
        else:
            return num_elem

    def process_num_list(self, num_line):
        num_list = num_line[6:].split(', ')
        pure_num_list = []
        for elem in num_list:
            elem = elem[elem.find('[')+1:elem.find(']')]
            elem = self.process_frac(elem)
            elem = self.chinese_num(elem)
            num, unit = self.split_num_and_unit(elem)
            ## marker num是数字 unit是单位， unit暂时不用，所以暂时不管，说不定还有用
            pure_num_list.append(num)    
            #print elem.encode('utf-8'),'---',num, unit.encode('utf-8'), '\t', 
        return pure_num_list 

    def process_cut_line(self, line):
        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        line_list = list(jieba.cut(line))
        filter_line_list = filter(lambda x: x!='[' and x!=']', line_list)
        new_line_list = []
        for elem in filter_line_list:
            if 'tmp' in elem:
                new_line_list.append('temp_'+alphabets[int(elem[3:])]) 
            else:
                new_line_list.append(elem)
        return ' '.join(new_line_list)

    def process_ans(self, ans):
        chinese_nums = [u'一',u'二',u'三',u'四',u'五',u'六',u'七',u'八',u'九',u'十']
        ans = ans.strip()
        try: 
            float(ans)
            return ans
        except:
            #ans = ans.encode('utf-8')
            pass
            flag = 0
            for elem in ans:
                if u'\u4e00' <= elem <= u'\u9fff':
                    flag = 1
            if flag == 1:
                #print ans.encode('utf-8')
                self.count += 1
                
            
            '''
            new_ans = ''
            for elem in ans:
                if elem in chinese_nums:
                    new_ans += str(chinese_nums.index(elem)+1)
                else:
                    new_ans += elem
            '''
                
        

    def attain_style(self):
        data_dict = {}
        count = 0
        self.count = 0
        for key, elem in self._data_dict.items():
            line = elem['normltext']
            num_line = elem['formal-procedure'].strip()
            ans = elem['ans']
            if 'tmp' not in num_line:
                continue
            num_list = self.process_num_list(num_line)
            if len(num_list) > 10:
                continue
            line_str = self.process_cut_line(line) 
            data_dict[count] = {'text':'','target_template':[],'gen_template':[],'num_list':[],'ans':''}
            data_dict[count]['text'] = line_str
            data_dict[count]['num_list'] = num_list
            data_dict[count]['ans'] = self.process_ans(ans)
            count += 1
        self.data_dict = data_dict
        print self.count
        
    def print_analysis(self):
        max_num_list = -1 
        num_len_dict = {}
        count = 0
        for key, elem in self._data_dict.items():
            print 'iIndex', key
            print elem['normltext'].encode('utf-8')
            line = elem['normltext']
            #print elem['problemtext'].encode('utf-8')
            num_line = elem['formal-procedure'].strip()
            if 'tmp' not in num_line:
                continue
            num_list = self.process_num_list(num_line)
            if len(num_list) > 10:
                continue
            line_list = self.process_cut_line(line) 
            print '--', ' '.join(line_list).encode('utf-8')
            #num_len_dict[len(num_list)] = num_len_dict.get(len(num_list), 0) +1
            #if len(num_list) > max_num_list:
            #    max_num_list = len(num_list)
            print elem['formal-procedure'].encode('utf-8')
            print num_list
            print elem['ans'].encode('utf-8')
            print 
        #print max_num_list
        #pdb.set_trace()
        pass

class Data_1():
    def __init__(self):
        pass

class Data_2():
    def __init__(self):
        pass

class DataLoader():
    def __init__(self):
        pass

def test():
    data_math23k = DataMath23k()
    #pdb.set_trace()
    data_ublabel57k = DataUnlabel57k()
    #data_ublabel57k.print_analysis()
    #for key, elem in data_ublabel57k.data_dict.items():
    #    print elem['ans'].strip().encode('utf-8')


test()
