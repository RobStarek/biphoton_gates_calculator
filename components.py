# -*- coding: utf-8 -*-
"""
Definition of component types to be used in netlists.
"""

from abc import ABC, abstractmethod
#import enum
import numpy as np
import sympy as sp
import WaveplateBox as wp

# Single-qubit logic basis states
SP_LO = sp.matrices.Matrix([[1], [0]])
SP_HI = sp.matrices.Matrix([[0], [1]])


def np_map_modes(M: np.ndarray,
                 mapping_in: list,
                 mapping_out: list,
                 dim_in: int,
                 dim_out: int
                 ) -> np.ndarray:
    """
    Pad matrix M to represent mapping from dim_in to dim_out
    and the route outer input modes to local input modes
    and local output modes to outer output modes.
    args:
    ----
        M : input local matrix
        mapping_in : list of integers
        mapping_out : list of integers
            the list is ordered as the local modes and its values
            sign the mapping to outer modes
        dim_in : number of modes of outer input space
        dim_out : number of modes of outer output space
    returns:
    ----
        matrix : matrix M representation in higher Hilber space
            with applied mode mapping
    """
    matrix1 = np.zeros((dim_out, dim_in), dtype=complex)
    for i, k in enumerate(mapping_out):
        for j, l in enumerate(mapping_in):
            if k is not None and l is not None:
                matrix1[k, l] = M[i, j]
    return matrix1


def sp_map_modes(
        M: np.ndarray,
        mapping_in: list,
        mapping_out: list,
        dim_in: int,
        dim_out: int
) -> sp.matrices.dense.MutableDenseMatrix:
    """
    like np_map_modes, but for sympy matrices
    """
    #n, m = M.shape
    matrix1 = sp.zeros(dim_out, dim_in)
    for i, k in enumerate(mapping_out):
        for j, l in enumerate(mapping_in):
            if k is not None and l is not None:
                matrix1[k, l] = M[i, j]
    return matrix1


def sp_linpol(x):
    """
    linear polarization vector
    """
    return sp.functions.cos(x)*SP_LO + sp.functions.sin(x)*SP_HI


def sp_waveplate(rot, retardance):
    """
    sympy expression for general rotated retarder/waveplate
    Args:
        rot - either str or float, rotation of waveplate in radians
        retardance - phase delay between two local polarizations,
            it could be str or a fixed float value
    """
    ket1 = sp.functions.cos(rot)*SP_LO + sp.functions.sin(rot)*SP_HI
    ket2 = sp.functions.sin(rot)*SP_LO - sp.functions.cos(rot)*SP_HI
    pt1 = ket1*(sp.transpose(ket1))
    pt2 = ket2*(sp.transpose(ket2))
    if isinstance(retardance, str):
        sp_arg = sp.symbols(retardance)*sp.I
        return pt1 + pt2*sp.functions.exp(sp_arg)
    else:
        if retardance == np.pi:
            return pt1 - pt2
        return pt1 + pt2*sp.functions.exp(1j*retardance)


class Component(ABC):
    """
    Class template for any component.
    Be sure to implement the mandatory methods.
    __init__ passes all its arguments to the load_argument so
    do the construction there.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_arguments(*args, **kwargs)

    @abstractmethod
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        """
        Here most of calculation should happen.
        """

    @abstractmethod
    def get_sympy_matrix(self):
        """getter for symbolic python"""

    @abstractmethod
    def get_numpy_matrix(self):
        """getter for numpy"""


class PathPhaseShifter(Component):
    """
    Phase shift paths, keep phase difference between polarizations within path.
    Note: phase shifter has as many paths as many arguments (*args) we enter.
    The arguments are phases.
    """

    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []
        self.shifts = args
        n_paths = len(args)

        if not kwargs.get('sympy', False):
            shifts_arr = np.array(args)
            np_local_matrix = np.kron(
                np.diag(np.exp(1j*shifts_arr)), np.eye(2))
            np_global_matrix = np_map_modes(
                np_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = np_global_matrix
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix)
        else:
            diagonal = []
            for arg in args:
                if isinstance(arg, str):
                    sp_arg = sp.functions.exp(sp.I*sp.Symbol(arg))
                    diagonal.append(sp_arg)  # two appends does the kronecker
                    diagonal.append(sp_arg)
                else:
                    # this retyping is needed for some reason
                    np_arg = complex(np.exp(1j*arg))
                    diagonal.append(np_arg)
                    diagonal.append(np_arg)

            self.npy_matrix = np.eye((n_paths*2))
            sp_local_matrix = sp.diag(*diagonal)
            sp_global_matrix = sp_map_modes(
                sp_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.sp_matrix = sp_global_matrix

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


class HalfWaveplate(Component):
    """
    Ideal half-wave plate.
    """
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []
        self.angle = args[0]
        self.dretardance = args[1]

        if not kwargs.get('sympy', False):
            np_local_matrix = wp.WP(self.angle, np.pi+self.dretardance)            
            np_global_matrix = np_map_modes(
                np_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = np_global_matrix
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix)
        else:
            self.npy_matrix = np.eye(2)
            name = self.angle            
            if self.dretardance == 0:
                sp_local_matrix = sp_waveplate(name, np.pi)
            else:
                sp_local_matrix = sp_waveplate(name, sp.pi + sp.Symbol(self.dretardance))
            sp_global_matrix = sp_map_modes(
                sp_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.sp_matrix = sp_global_matrix

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


class QuarterWaveplate(Component):
    """ideal quarter-wave plate"""
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []
        self.angle = args[0]
        self.dretardance = args[1]

        if not kwargs.get('sympy', False):
            np_local_matrix = wp.WP(self.angle, np.pi/2+self.dretardance)
            np_global_matrix = np_map_modes(
                np_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = np_global_matrix
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix)
        else:
            self.npy_matrix = np.eye(2)
            name = self.angle            
            if self.dretardance == 0:
                sp_local_matrix = sp_waveplate(name, sp.pi/2)
            else:
                sp_local_matrix = sp_waveplate(name, sp.pi/2 + sp.Symbol(self.dretardance))
            sp_global_matrix = sp_map_modes(
                sp_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.sp_matrix = sp_global_matrix

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


class GeneralWaveplate(Component):
    """a rotated retarded (general waveplate)"""
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []
        self.angle = args[0]
        self.retardance = args[1]

        if not kwargs.get('sympy', False):
            np_local_matrix = wp.WP(self.angle, self.retardance)
            np_global_matrix = np_map_modes(
                np_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = np_global_matrix
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix)
        else:
            self.npy_matrix = np.eye(2)
            name = self.angle            
            sp_local_matrix = sp_waveplate(name, self.retardance)
            sp_global_matrix = sp_map_modes(
                sp_local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.sp_matrix = sp_global_matrix

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


class BeamSplitterTwoPorts(Component):
    """Beam splitter with two ports, each port supports 2 polarizations."""
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []
        
        if kwargs.get('sympy', False):
            Th, Tv = sp.Symbol(args[0]), sp.Symbol(args[1])
            th = sp.functions.sqrt(Th)
            tv = sp.functions.sqrt(Tv)
            rh = sp.functions.sqrt(1-Th)
            rv = sp.functions.sqrt(1-Tv)
            local_matrix = sp.Matrix([
                [th, 0, rh, 0],
                [0, tv, 0, rv],
                [-rh, 0, th, 0],
                [0, -rv, 0, tv]
            ])
            global_matrix = sp_map_modes(
                local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = np.eye(global_matrix.shape[0])
            self.sp_matrix = global_matrix
        else:
            Th, Tv = args[0], args[1]
            th = Th**0.5
            tv = Tv**0.5
            rh = (1-Th)**0.5
            rv = (1-Tv)**0.5
            local_matrix = np.array([
                [th, 0, rh, 0],
                [0, tv, 0, rv],
                [-rh, 0, th, 0],
                [0, -rv, 0, tv]
            ])
            global_matrix = np_map_modes(
                local_matrix, mapping_in, mapping_out, dim_in, dim_out)
            self.npy_matrix = global_matrix
            self.sp_matrix = sp.Matrix(global_matrix)

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


class Attenuation(Component):
    """Atternuator component. It could be also used to construct beam displacers or mode-swapper."""
    def load_arguments(self, mapping_in, mapping_out, dim_in, dim_out, *args, **kwargs):
        if mapping_in is None:
            mapping_in = []
        if mapping_out is None:
            mapping_out = []

        if dim_in is None:
            dim_in = len(args)
        if dim_out is None:
            dim_out = len(args)

        local_matrix = np.diag(args)
        global_matrix = np_map_modes(
            local_matrix, mapping_in, mapping_out, dim_in, dim_out)
        self.npy_matrix = global_matrix
        if kwargs.get('integer'):
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix.real.astype(int))
        else:
            self.sp_matrix = sp.matrices.Matrix(self.npy_matrix)

    def get_numpy_matrix(self):
        return self.npy_matrix

    def get_sympy_matrix(self):
        return self.sp_matrix


COMPONENTS = {
    'hwp': HalfWaveplate,
    'qwp': QuarterWaveplate,
    'wp': GeneralWaveplate,
    'BS2pol': BeamSplitterTwoPorts,
    'shift': PathPhaseShifter,
    'atten': Attenuation
}
