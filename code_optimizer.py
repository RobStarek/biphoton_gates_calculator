import re
import json
from sympy.printing.numpy import NumPyPrinter

def print_numpy_expression(expession):
    printer = NumPyPrinter()
    code = printer.doprint(expession)
    return code
    
def replace_bracketed_goniometry(code_str, fun, i_offset = 0):    
    """
    Replaces top level of 'fun(...)' exppressions with
    substitutions like wxxxx in the code.
    Returns list of substitutions.
    """
    inits = {}
    new_code = code_str[:]
    code_length = len(new_code)
    token_length = len(fun)
    idx = 0
    go = True
    counter = 0
    safe_counter = 0

    while go:
        left_idx = new_code.find(fun, idx)
        if left_idx < 0:
            go = False
            break
        go2 = True
        succ = False
        #find next )
        right_idx = new_code.find(')', left_idx+token_length)
        while go2:            
            if right_idx == -1:
                break
            sub_text = new_code[left_idx:right_idx+1]
            left_bracket_count = sub_text.count('(')
            right_bracket_count = sub_text.count(')')
            safe_counter += 1
            if safe_counter>10_000:
                raise Exception('too many iteration')
            if left_bracket_count == right_bracket_count:
                go2 = False
                succ = True
                break            
            else:
                right_idx = new_code.find(')', right_idx+1)
        if succ:
            key = f'w{counter+i_offset:04d}'
            expression = new_code[left_idx:right_idx+1]
            inits[key] = expression
            counter = counter + 1
            succ = False
            while expression in new_code:
                new_code = new_code.replace(expression, key)
            idx = left_idx+1
        else:
            raise Exception('parenthesis not closed?')
        if right_idx == -1:
            break
    return inits, new_code

def generate_enum_and_init(expression, func_name = None):
    """
    generate enum class and replacement dictionary from sympy expression
    """
    if func_name is None:
        func_name = 'foo'
    symbols = [str(element) for element in expression.free_symbols]
    arguments = sorted(symbols)
    enum_body = '\n'.join([f'    {key} = {i}' for i, key in enumerate(arguments)])
    enum_code = f"class {func_name}_symbols (IntEnum):\n"+enum_body    
    replacement_dict = {key : f'args[{i}]' for i, key in enumerate(arguments)}
    return enum_code, replacement_dict

def substitutions(code_str, replacement_dict):
    new_code = code_str[:]
    for old, new in replacement_dict.items():
        regex = r'(?![^\w])' + old + r'(?=[^\w])'        
        new_code = re.sub(regex, new, new_code)
    return new_code

SUBS_TOKENS = ['numpy.sin(', 'numpy.cos(', 'numpy.exp(']
def generate_code(expr, func_name=None):
    if func_name is None:
        func_name = 'foo'    
    code = print_numpy_expression(expr)
    offset = 0
    all_inits = dict()
    #replace omnipresent pi constant
    if 'numpy.pi' in code:
        code = code.replace('np.pi', 'pi')
        all_inits['pi'] = 'numpy.pi'    
    #replace top level goniometric expressions
    for token in SUBS_TOKENS:
        inits, code = replace_bracketed_goniometry(code, token, offset)
        all_inits.update(inits)
        offset = len(all_inits)
    enum_code, replacement_dictionary = generate_enum_and_init(expr, func_name)  
    code = code.replace('\n','')    
    #tabbed_code = '\n'.join([f'\t{line}' for line in code.split('\n')])
    foo_init_code = ""
    for key, sub in all_inits.items():
        foo_init_code = foo_init_code + f'\n    {key} = {sub.replace("numpy.","np.")}'
    foo_init_code = substitutions(foo_init_code, replacement_dictionary)
    code = substitutions(code, replacement_dictionary)
    function_code = f"def {func_name}(args):\n" + foo_init_code + '\n    return ' + code.replace('numpy.','np.')
    big_code = f'import numpy as np\nfrom enum import IntEnum\n\n{enum_code}\n\n{function_code}'
    return big_code
