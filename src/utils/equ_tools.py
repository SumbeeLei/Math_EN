import re
import numpy as np

def split_equation(equation):
    num_set = ['0','1','2','3','4','5','6','7','8','9','%', '.', 'P', 'I']
    start = ''
    equ_list = []
    for char in equation:
        if char not in num_set:
            if start != '':
                equ_list.append(start)
            equ_list.append(char)
            start = ''
        else:
            start += char
    if start != '':
        equ_list.append(start)
    return equ_list

def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
    for elem in equ_list:
        elem = str(elem)
        if elem == '(':
            stack.append('(')
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if stack == []:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            #if elem == 'PI':
            #    post_equ.append('3.14')
            #else:
            #    post_equ.append(elem)
            post_equ.append(elem)
    while stack != []:
        post_equ.append(stack.pop())
    return post_equ


def post_solver(post_equ):
    stack = [] 
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        elem = str(elem)
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))
            elif elem == '/':
                stack.append(str(op_v_2/op_v_1))
            else:
                stack.append(str(op_v_2**op_v_1))
    return stack.pop()

def solve_equation(equ_list):
    post_equ = postfix_equation(equ_list)
    ans = post_solver(post_equ)
    return ans

def equ_api_1(equ, ans):
    '''
    template: list, x = xxxxx 
    return boolean variable
    '''
    equ = ''.join(equ)
    equ = equ[2:]
    equ = split_equation(equ)
    p_ans = solve_equation(equ)
    #print p_ans
    if abs(float(p_ans) - float(ans))<1e-5:
        return True
    return False


