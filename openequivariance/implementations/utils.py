import functools
import warnings
import math

import numpy as np 

from openequivariance.implementations.TensorProductBase import TensorProductBase
from openequivariance.implementations.e3nn_lite import Irrep, _MulIr, Irreps, Instruction, TPProblem

def sparse_outer_product_work(cg : np.ndarray) -> int: 
    return np.sum(np.max(cg != 0, axis=2))

def convenience_namer(L1 : Irreps, L2 : Irreps, L3 : Irreps):
    return f"({L1}x{L2}->{L3})"

# Non Zeros 
@functools.lru_cache(typed=True)
def count_cg_non_zero(l1, l2, l3) -> int:
    return np.count_nonzero(TensorProductBase.load_cg_tensor(l1, l2, l3))

def calculate_total_nnz(tpp : TPProblem) -> int:
        """
        To make sure you don't over count repeat CGs which get used multiple times 
        """
        nnz_by_l_combo = {}
        for ins in tpp.instructions: # type : Instruction
            l1 = tpp.irreps_in1[ins.i_in1].ir.l 
            l2 = tpp.irreps_in2[ins.i_in2].ir.l
            l3 = tpp.irreps_out[ins.i_out].ir.l
            assert isinstance(l1, int)
            assert isinstance(l2, int)
            assert isinstance(l3, int)
            nnz_by_l_combo[(l1,l2,l3)] =  count_cg_non_zero(l1,l2,l3)
        return sum(nnz_by_l_combo.values())

def calc_weight_offsets(tpp : TPProblem) -> list[int]:
    """
    Returns a list of weight offsets for every instruction. 
    """
    assert isinstance(tpp, TPProblem)
    offset = 0
    offsets = []
    for ins in tpp.instructions:
        assert isinstance(ins, Instruction) 
        offsets.append(offset)
        if ins.has_weight:
            flatsize = math.prod(ins.path_shape)
            offset += flatsize
    return offsets     

 
def filter_and_analyze_problem(problem):
    '''
    Centralized function that stops unhandled problem configurations,
    returns a dictionary of useful information about the problem. 
    '''
    for inst in problem.instructions:
        assert inst.connection_mode == problem.instructions[0].connection_mode, \
            f"All instructions must have the same connection mode, got {inst.connection_mode} and {problem.instructions[0].connection_mode}"

    assert problem.instructions[0].connection_mode in ["uvu", "uvw"], \
            f"Connection mode must be 'uvu' or 'uvw', got {problem.instructions[0].connection_mode}"

    assert problem.irrep_dtype == problem.weight_dtype, \
            f"irrep_dtype and weight_dtype must be the same, got {problem.irrep_dtype} and {problem.weight_dtype}"

    assert len(problem.instructions) > 0, \
            "Tensor product has no valid instructions!"

    result = {
        "is_uvw": problem.instructions[0].connection_mode == "uvw", 
    }
    return result