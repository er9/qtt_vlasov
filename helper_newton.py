import numpy as np

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup_.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

MPO_type = Union['qtn.MatrixProductOperator', MatrixProductOperatorTN, IndexedMPO]
MPS_type = Union['qtn.MatrixProductState', MatrixProductStateTN, IndexedMPS]


import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix


def newton_krylov(F, J, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        # Compute residual
        r = eval_cost(F, x)

        # Check convergence
        if np.linalg.norm(r) < tol:
            break

        # Solve Jacobian system using Krylov method
        J_x = eval_jac(J, x)
        dx, info = gmres(J_x, -r)

        # Update solution
        if info == 0:
            x += dx
        else:
            print("Krylov solver failed to converge")

    return x

def block_newton_krylov():
    raise NotImplementedError

def eval_cost(func: 'qtn.TensorNetwork', x: 'qtn.Tensor'):
    raise NotImplementedError

def eval_jac(jac: 'qtn.TensorNetwork', x: 'qtn.Tensor'):
    raise NotImplementedError

def krylov_solve(A: 'qtn.TensorNetwork', b: 'qtn.Tensor'):
    """ solve Ax = b --> x = A^{-1} b
        CGD?
    """
    raise NotImplementedError

def block_krylov_solve(A_dict: dict[tuple[int,int]: qtn.TensorNetwork], b: dict[int, qtn.Tensor]):
    """ solve Ax = b --> x = A^{-1} b in block form
    """
    raise NotImplementedError