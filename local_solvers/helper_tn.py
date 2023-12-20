# from setup.configs import *
# import numpy as np
import quimb

from setup.defaults import *
import time
import quimb.tensor as qtn


def sum_eff_TNs(eff_tns: Sequence['qtn.TensorNetwork'], transpose_bonds=None):

    transpose_bonds = eff_tns[0].outer_inds() if transpose_bonds is None else transpose_bonds

    A_eff_ = None
    for A_eff_tn in eff_tns:
        A_eff_tens = qtn.tensor_contract(*A_eff_tn.tensors, preserve_tensor=True)
        A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
        if len(transpose_bonds) > 0:
            A_eff_tens.transpose(*transpose_bonds, inplace=True)

        if A_eff_ is None:
            A_eff_ = A_eff_tens.copy()
        else:
            A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
    return A_eff_

