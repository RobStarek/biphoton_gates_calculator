# -*- coding: utf-8 -*-
"""
Open linear optical component sequence (so called netlist) saved as json
and generate sympy expression or numpy matrix describing the input-output mapping of a single
photon.

Netlist element structure:
    "<component name>" : {
    "type" : "<component class name>",
    "input_modes" : <mapping list of global input modes to local input modes>,
    "output_modes" : <mapping list of global output modes to local output modes>,
    "layer" : <layer number>,
    "arguments" : <list of arguments passed to constructor>,
    "kw_args" : <dictionarry of keyword arguments passed to constructor>
    }

Rules for netlist:
- "circuits" are linear, no recursion allowed
- a mode can be assigned only once in a single layer
  i.e., two components with same layer number cant be connected to the same mode in
- components must have uniquie names
- mandatory entries: type, input/output_modes, layer
- it is recommended to write the netlist in order, although
  it is probably not neccessary
- chain the elements as the order dictionary in the json file
- layers should go from 0 with integer step, withou skipping, e.g. 0, 1, 2, 3, ...

How to write mapping lists:
- the length should be equal to the corresponding inner number of modes
- entry j at index i connects inner mode i to the global mode j
- order of the list is therefore important
- entry can be also None, in that case a vacuum is connect to that inner mode

How to add more component types:
- see components.py, subclass Component abstract base class
"""

from functools import reduce
import numpy as np
import sympy as sp
from components import COMPONENTS

#from sympy.printing.numpy import NumPyPrinter
# # move codeprinting elsewhere
# def print_numpy_expression(expession):
#     printer = NumPyPrinter()
#     code = printer.doprint(expession)
#     return code


def get_layers(netlist_json: dict) -> set:
    """
    Get a set of layers present in a netlist.
    Args:
        netlist_json : dictionary fulfilling rules for netlists

    Returns:
        sorted set containing netlist layer numbers
    """
    return set(
        sorted(
            [component['layer'] for name, component in netlist_json.items()]
        )
    )


def get_elements_names_in_layer(netlist_json: dict, layer: int = 0) -> list:
    """
    Get all component/element names in given netlist layer
    Args:
        netlist_json : dictionary fulfilling rules for netlists
        layer: integer index of the layer
    Returns:
        list of element names
    """
    element_list = [name for name, component in netlist_json.items(
    ) if component['layer'] == layer]
    return element_list


def estimate_layer_mode_numbers(netlist_json: dict, layer: int = 0) -> tuple:
    """
    Count how many modes are needed in a layer of netlist.
    Args:
        netlist_json : dictionary fulfilling rules for netlists
        layer: integer index of the layer
    Returns:
        input_modes, output_modes ... integers with number of needed modes on in/out side
        of the layer
    """
    element_keys = get_elements_names_in_layer(netlist_json, layer)
    max_mode_index_in = 0
    max_mode_index_out = 0
    for key in element_keys:
        input_map = netlist_json[key]["input_modes"]
        output_map = netlist_json[key]["output_modes"]
        max_mode_index_in = max(
            *[i for i in input_map if i is not None], max_mode_index_in)
        max_mode_index_out = max(
            *[i for i in output_map if i is not None], max_mode_index_out)
    input_modes = max_mode_index_in + 1
    output_modes = max_mode_index_out + 1
    return input_modes, output_modes


def check_layer_for_mode_conflicts(netlist_json, layer=0):
    """
    Just check whether there are not multiple components connected to a single mode
    withing a single layer of netlist.
    Args:
        netlist_json : dictionary fulfilling rules for netlists
        layer: integer index of the layer
    Returns:
        None
    Raises:
        ValueError if there is a conflict.
    """
    element_keys = get_elements_names_in_layer(netlist_json, layer)
    used_modes_in = []
    used_modes_out = []
    for key in element_keys:
        used_modes_in.extend(
            [i for i in netlist_json[key]["input_modes"] if i is not None])
        used_modes_out.extend(
            [i for i in netlist_json[key]["output_modes"] if i is not None])
    if len(used_modes_in) > len(set(used_modes_in)):
        raise ValueError(
            f"Netlist is invalid, there is some input mode conflict within layer {layer}.")
    if len(used_modes_out) > len(set(used_modes_out)):
        raise ValueError(
            f"Netlist is invalid, there is some output mode conflict within layer {layer}.")
    return None


def extract_sympy_symbols(netlist_json: dict) -> set:
    """
    Get list of all sympy symbol names in the netlist.
    Args:
        netlist_json : dictionary fulfilling rules for netlists
    Returns:
        symbols : list of strings with symbol names
    """
    symbols = []
    for i_layer in get_layers(netlist_json):
        element_names = get_elements_names_in_layer(netlist_json, i_layer)
        for element in element_names:
            kw_args = netlist_json[element].get('kw_args', {})
            if kw_args.get('sympy', False):
                layer_symbols = [
                    arg for arg in netlist_json[element]["arguments"] if isinstance(arg, str)]
                symbols.extend(layer_symbols)
    return set(symbols)

# -> instance of Component
def instantiate_component(netlist_json: dict, element: str, modes_in: int, modes_out: int):
    """
    Take netlist dictionary element construct its instance.
    Args:
        netlist_json : dictionary fulfilling rules for netlists
        element : string with name of the element
        modes_in : number of input modes in element's layer
        modes_in : number of output modes in element's layer
    Returns:
        instance ... constructed from the netlist entry, a subclass of Component base class
    """
    map_in = netlist_json[element]["input_modes"]
    map_out = netlist_json[element]["output_modes"]
    component_type = netlist_json[element]['type']
    arguments = netlist_json[element].get('arguments', [])
    keyword_arguments = netlist_json[element].get('kw_args', {})
    # print(netlist_json[element])
    # print(map_in)
    # print(map_out)
    # print(component_type)
    # print(arguments)
    instance = COMPONENTS[component_type](
        map_in, map_out, modes_in, modes_out, *arguments, **keyword_arguments)
    # print(np.round(instance.get_numpy_matrix(),3))
    # print('----')
    return instance


def instantiate_netlist_components(netlist_json: dict) -> list:
    """
    Instantiate all components in the netlist and arrange them
    into list of dictionaries.
    Args:
        netlist_json : dictionary fulfilling rules for netlists
    Returns:
        layers ... list of dictinaries, the dicts have keys with component names
            and values with component instances
    """
    layers = []
    for i_layer in get_layers(netlist_json):
        check_layer_for_mode_conflicts(netlist_json, i_layer)
        modes_in, modes_out = estimate_layer_mode_numbers(
            netlist_json, i_layer)
        #print("@", i_layer, modes_in, modes_out)
        element_names = get_elements_names_in_layer(netlist_json, i_layer)
        layers.append({})
        for element in element_names:
            instance = instantiate_component(
                netlist_json, element, modes_in, modes_out)
            layers[i_layer][element] = instance
    return layers


def np_calculate_effective_matrix(layers: list) -> np.ndarray:
    """
    Calculate effective mapping matrix from the list of netlist element instances.
    Args:
        layers : list of dicts containing the Component subclass instances,
           to be used with instantiate_netlist_components() function.
    Returns:
        effective_matrix ... numpy matrix
    """
    effective_matrix = None
    for i, layer in enumerate(layers):
        matrices = [instance.get_numpy_matrix()
                    for name, instance in layer.items()]
        layer_matrix = sum(matrices)
        if i == 0:
            effective_matrix = layer_matrix
        else:
            effective_matrix = layer_matrix @ effective_matrix
    return effective_matrix


def sp_calculate_effective_matrix(layers: list, simplify: bool = False)\
     -> sp.matrices.dense.MutableDenseMatrix:
    """
    Calculate effective mapping matrix from the list of netlist element instances.
    Args:
        layers : list of dicts containing the Component subclass instances,
           to be used with instantiate_netlist_components() function.
        simplify : set to True to attempt some sympy-simplification
    Returns:
        effective_matrix ... sympy matrix
    """
    effective_matrix = None
    for i, layer in enumerate(layers):
        matrices = [instance.get_sympy_matrix()
                    for name, instance in layer.items()]
        layer_matrix = reduce(lambda x, y: x+y, matrices)
        if i == 0:
            effective_matrix = layer_matrix
        else:
            effective_matrix = layer_matrix * effective_matrix
    if not simplify:
        return effective_matrix
    return sp.simplify(effective_matrix)

# #Example
# import json
# with open('netlist_swap_sp2.json') as mf:
#     netlist = json.load(mf)
# objs = instantiate_netlist_components(netlist)
# exp1 = sp_calculate_effective_matrix(objs, simplify=True)
# print(exp1)
# symbs = extract_sympy_symbols(netlist)
# print(symbs)
