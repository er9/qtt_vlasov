""" v2 works.
    this is a cleaner version + elemental multiplication of f(x) * g(y)
    where f, g are element-wise operations
"""

from setup_.configs import *
import time
# from scipy import linalg
import scipy.linalg as linalg
import quimb.tensor as qtn
import helper_quimb as helper
from local_solvers_old.local_solver import *
from local_solvers_old.environments import EnvironmentSide, CrossEnvironment
# from quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']

from enum import Enum, IntEnum



def check_left_orthog(mpx: Union['qtn.MatrixProductState', Sequence[qtn.Tensor]],
                      select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L - 1):
        tens = mpx[i].copy()

        bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L-1 else []
        bond_p = [s.format(i) for s in site_ind_ids]

        tens = tens.fuse({'eff': bond_p + bond_l})
        tens.transpose('eff', *bond_r, inplace=True)
        try:
            env = tens.data[select_inds[i], :]
        except KeyError:
            print(f'select ind not defined for {i}')
            return i

        err = np.linalg.norm(env - np.eye(env.shape[0]))
        if err > 1.0e-10:
            print('not left orthog', err) #, env)
            break

    return i


def check_right_orthog(mpx: Union['qtn.MatrixProductState', Sequence['qtn.Tensor']],
                       select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L - 1, 0, -1):
        tens = mpx[i].copy()

        bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L - 1 else []
        bond_p = [mpx.site_ind_id.format(i)]

        tens = tens.fuse({'eff': bond_p + bond_r})
        tens.transpose('eff', *bond_l, inplace=True)
        try:
            env = tens.data[select_inds[i], :]
        except KeyError:
            print(f'select ind not defined for {i}')
            return i

        err = np.linalg.norm(env - np.eye(env.shape[0]))
        if err > 1.0e-10:
            print('not right orthog', err) #, env)
            break

    return i


def check_orthog(mpx: 'qtn.MatrixProductState', select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str]
                 ) -> tuple[int, int]:
    print('check orthog')
    left = check_left_orthog(mpx, select_inds, site_ind_ids)
    right = check_right_orthog(mpx, select_inds, site_ind_ids)
    return left, right

########


def maxvol_inds(A: 'np.ndarray', max_iters=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, do_qr=True, ):
    """
    modified so that even if n < r, environment will be identity
    :param A:  n x r matrix. want to find r x r submatrix of A
    :param max_iters:
    :param conv_tol:
    :param do_qr:
    :return: inds to obtain submatrix of A
    """
    n, r = A.shape  ## A is 2D matrix: alpha_{i-1} * d_{i} x alpha_{i}
    # print('A shape', A.shape)
    if n <= r:
        r = n

    if do_qr:
        Q, R_dmp = np.linalg.qr(A)
    else:
        Q = A.copy()

    p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
    p = np.argsort(p)       # so that P A = L U, as in matlab
    inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)

    if n <= r:
        # return slice(0,n)
        return inds

    submat = Q[inds, :]  # permute rows of A
    B = Q @ np.linalg.inv(submat)       # Q Q^-1 = B = [I, Z]

    ## start iterations
    it, err = 0, 1.0
    while it < max_iters:  # and err < conv_tol:
        # print('B', B)
        max_ind = np.argmax(np.abs(B))
        i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])
        # print('i0,j0', i0, j0)
        max_val = B[i0, j0]
        if np.abs(max_val) <= 1 + 1.0e-8:
            break

        ## rank 1 update of B from switching rows i0 and inds[j0]
        old_row = inds[j0]
        B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
        inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)

        it += 1

    # print('final B', B[p,:], np.max(B))
    # print('final B', B[inds, :], np.max(B))
    # submat = Q[inds, :]
    # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])

    return np.sort(inds)


# def maxvol_inds(A: 'np.ndarray', max_iters=DEFAULT_MAX_TOT_ITER, conv_tol=CONV_TOL, do_qr=True, ):
#     """
#     :param A:  n x r matrix. want to find r x r submatrix of A
#     :param max_iters:
#     :param conv_tol:
#     :param do_qr:
#     :return: inds to obtain submatrix of A
#     """
#
#     n, r = A.shape  ## A is 2D matrix: alpha_{i-1} * d_{i} x alpha_{i}
#     # print('A shape', A.shape)
#     if n <= r:
#         return slice(0,n)
#
#     if do_qr:
#         Q, R_dmp = np.linalg.qr(A)
#     else:
#         Q = A.copy()
#
#     p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
#     p = np.argsort(p)       # so that P A = L U, as in matlab
#     inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)
#
#     submat = Q[inds, :]  # permute rows of A
#     B = Q @ np.linalg.inv(submat)       # Q Q^-1 = B = [I, Z]
#
#     ## start iterations
#     it, err = 0, 1.0
#     while it < max_iters:  # and err < conv_tol:
#         # print('B', B)
#         max_ind = np.argmax(np.abs(B))
#         i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])
#         # print('i0,j0', i0, j0)
#         max_val = B[i0, j0]
#         if np.abs(max_val) <= 1 + 1.0e-8:
#             break
#
#         ## rank 1 update of B from switching rows i0 and inds[j0]
#         old_row = inds[j0]
#         B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
#         inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)
#
#         it += 1
#
#     # print('final B', B[p,:], np.max(B))
#     # print('final B', B[inds, :], np.max(B))
#     # submat = Q[inds, :]
#     # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])
#
#     return np.sort(inds)


def canonize_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
                       inplace=True):

    L = len(tens_list)
    tens_list = tens_list if inplace else [t.copy() for t in tens_list]
    select_inds_list = []

    for i in range(0, L-1):
        right_inds, unshared_inds = tens_list[i].filter_bonds(tens_list[i+1])
        ## shared and unshared bonds
        phys_inds = [s.format(site_inds[i]) for s in site_ind_ids]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        assert(len(left_inds)<=1), 'not specific enough if have multiple virtual "left inds"'

        ## must follow specific order for fusing unshared bonds
        fuse_inds = phys_inds + left_inds

        ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
        ## canon form: want C (A_)^-1
        ## C is essentially the tensor core
        ## Let C = QT -->
        ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1

        q, r = qtn.tensor_split(tens_list[i], fuse_inds, absorb='right', method='svd',
                                bond_ind=f'__tmp_{i}__')

        q.transpose(*fuse_inds, f'__tmp_{i}__', inplace=True)
        r.transpose(f'__tmp_{i}__', *right_inds, inplace=True)

        ## obtain dominant submatrix
        q_ = q.fuse({f'xx{i}': fuse_inds}, inplace=False)
        q_.transpose(f'xx{i}', f'__tmp_{i}__', inplace=True)
        # nr, nc = q_.shape[0], q_.shape[-1]
        inds = maxvol_inds(q_.data, do_qr=False)  ## n, r
        q_.modify(apply=lambda x: x[inds,:])    ## submat

        q.modify(apply=lambda x: np.tensordot(x, np.linalg.inv(q_.data), axes=(-1,0)))
        r.modify(apply=lambda x: np.dot(q_.data, x))

        ## update sites in MPS
        q.transpose_like(tens_list[i], inplace=True)
        tens_list[i].modify(data=q.data)

        # print('next site', tens_list[i+1].norm())
        next_site = qtn.tensor_contract(r, tens_list[i + 1])
        next_site.transpose_like(tens_list[i + 1], inplace=True)
        tens_list[i + 1].modify(data=next_site.data)
        # print('new next site', tens_list[i + 1].norm())

        select_inds_list += [inds]

    return tens_list, select_inds_list


def cross_split(A: np.ndarray, do_qr=False, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False):
    ### decompose matrix using cross interpolation; obtain low-rank approximation

    n, r = A.shape  ## A is 2D matrix: (alpha_{i-1} * d_{i}) x alpha_{i}

    # print('A shape', A.shape)
    # if n <= r:
    #     print('here')
    #     mat1, mat2 = np.linalg.qr(A)
    #     if return_inds:
    #         return mat1, mat2, slice(0,n)
    #     else:
    #         return mat1, mat2

    if do_qr or n < r:
        Q, R_dmp = np.linalg.qr(A)
    else:
        Q = A.copy()

    n, r = Q.shape

    p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
    p = np.argsort(p)       # so that P A = L U, as in matlab
    inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)

    submat = Q[inds, :]  # permute rows of A

    B = Q @ np.linalg.inv(submat)       # Q Q^-1 = B = [I, Z]

    ## start iterations
    it, err = 0, 1.0
    while it < max_iters:  # and err < conv_tol:
        # print('B', B)
        max_ind = np.argmax(np.abs(B))
        i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])
        # print('i0,j0', i0, j0)
        max_val = B[i0, j0]
        if np.abs(max_val) <= 1 + 1.0e-8:
            break

        ## rank 1 update of B from switching rows i0 and inds[j0]
        old_row = inds[j0]
        B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
        inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)

        it += 1

    ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
    ## canon form: want C (A_)^-1
    ## C is essentially the tensor core
    ## Let C = QT -->
    ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1

    # inds = np.sort(inds)

    T1 = B
    submat = Q[inds,:]
    # print('submat', submat)
    T2 = (submat @ R_dmp) if do_qr else (submat)
    # print('T2 norm', np.linalg.norm(T2))

    # print('final B', B[p,:], np.max(B))
    # print('final B', B[inds, :], np.max(B))
    # print('final B', B[np.sort(inds), :], np.max(B))
    # submat = Q[inds, :]
    # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])

    # print('err B', np.linalg.norm(B - (Q @ np.linalg.inv(submat))))
    # print('err A', np.linalg.norm(A[:, :] - T1 @ T2))

    if return_inds:
        return T1, T2, inds  # np.sort(inds)
    else:
        return T1, T2



def cross_compress(A: np.ndarray, max_bond, do_qr=True, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False,
                   conv_tol=DEFAULT_CONV_TOL):
    ### decompose matrix using cross interpolation; obtain low-rank approximation

    n, r = A.shape  ## A is 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}

    if max_bond is None or n < max_bond:
        return cross_split(A, do_qr=do_qr, max_iters=max_iters, return_inds=return_inds)

    r = int( min(r,max_bond) )

    ## randomly select r elements in A; make A n x r
    inds_r = slice(0, r, 1)

    prev_A = np.zeros(A.shape, dtype=A.dtype)
    it = 0

    while it < max_iters:

        RA = A[inds_r, :].T
        inds_c = maxvol_inds(RA, do_qr=do_qr)

        CA = A[:, inds_c]
        T1, submat, inds_r = cross_split(CA, do_qr=do_qr, max_iters=max_iters, return_inds=True)

        it_A = T1 @ A[inds_r, :]        ## A = A[:, inds_c] @ submat^-1 @ A[inds_r, :]
        err = np.linalg.norm(it_A - prev_A) / np.linalg.norm(it_A)
        # print('err', err)
        prev_A = it_A
        it += 1

        if it == max_iters:
            T2 = A[inds_r, :]
            print('compress err', err)

        if err < conv_tol:
            T2 = A[inds_r, :]
            break


    if return_inds:
        return T1, T2, inds_r  # np.sort(inds)
    else:
        return T1, T2


# def cross_split_H(A: np.ndarray, do_qr=False, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False):
#     ### decompose Hermitian matrix using cross interpolation; obtain low-rank approximation
#
#     n, r = A.shape  ## A is 2D Hermitian matrix: (alpha_{i-1} * d_{i}) x alpha_{i}
#
#     if do_qr or n < r:
#         Q, R_dmp = np.linalg.qr(A)
#     else:
#         Q = A.copy()
#
#     n, r = Q.shape
#
#     p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
#     p = np.argsort(p)       # so that P A = L U, as in matlab
#     inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)
#
#     submat = Q[inds, :]  # permute rows of A
#
#     B = Q @ np.linalg.inv(submat)       # Q Q^-1 = B = [I, Z]
#
#     ## start iterations
#     it, err = 0, 1.0
#     while it < max_iters:  # and err < conv_tol:
#         # print('B', B)
#         max_ind = np.argmax(np.abs(B))
#         i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])
#         # print('i0,j0', i0, j0)
#         max_val = B[i0, j0]
#         if np.abs(max_val) <= 1 + 1.0e-8:
#             break
#
#         ## rank 1 update of B from switching rows i0 and inds[j0]
#         old_row = inds[j0]
#         B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
#         inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)
#
#         it += 1
#
#     ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
#     ## canon form: want C (A_)^-1
#     ## C is essentially the tensor core
#     ## Let C = QT -->
#     ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1
#
#     T1 = B
#     submat = Q[inds,:]
#     # print('submat', submat)
#     T2 = (submat @ R_dmp) if do_qr else (submat)
#     # print('T2 norm', np.linalg.norm(T2))
#
#     if return_inds:
#         return T1, T2, inds  # np.sort(inds)
#     else:
#         return T1, T2
#
# def cross_compress_H(A: np.ndarray, max_bond, do_qr=True, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False,
#                      conv_tol=DEFAULT_CONV_TOL):
#     ### decompose matrix using cross interpolation; obtain low-rank approximation
#
#     n, r = A.shape  ## A is Hermitian 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}
#
#     # if max_bond is None or n < max_bond:
#     #     return cross_split(A, do_qr=do_qr, max_iters=max_iters, return_inds=return_inds)
#
#     r = int( min(r,max_bond) )
#
#     ## randomly select r elements in A; make A n x r
#     inds_r = slice(0, r, 1)
#
#     prev_A = np.zeros(A.shape, dtype=A.dtype)
#     it = 0
#
#     while it < max_iters:
#
#         RA = A[inds_r, :].T
#         inds_c = maxvol_inds(RA, do_qr=do_qr)
#
#         CA = A[:, inds_c]
#         T1, submat, inds_r = cross_split(CA, do_qr=do_qr, max_iters=max_iters, return_inds=True)
#
#         it_A = T1 @ A[inds_r, :]        ## A = A[:, inds_c] @ submat^-1 @ A[inds_r, :]
#         err = np.linalg.norm(it_A - prev_A) / np.linalg.norm(it_A)
#         # print('err', err)
#         prev_A = it_A
#         it += 1
#
#         if it == max_iters:
#             T2 = A[inds_r, :]
#             print('compress err', err)
#
#         if err < conv_tol:
#             T2 = A[inds_r, :]
#             break
#
#
#     if return_inds:
#         return T1, T2, inds_r  # np.sort(inds)
#     else:
#         return T1, T2




def canonize(mps: 'qtn.MatrixProductState', i, cur_orthog=None, select_inds=None):
    """ inplace canonicalization via cross
    """

    tens_left, tens_right = [], []
    max_ind = mps.L - 1
    # i = mps.get_mps_ind(i)
    mps_inds = list(range(mps.L))
    updated_inds_l, updated_inds_r = [], []
    for lx in mps_inds:
        if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
            updated_inds_l += [lx]
            tens_left += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
        if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
            if lx not in updated_inds_l:
                updated_inds_r += [lx]
            tens_right += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]

    _, sel_inds_list_l = canonize_tens_list(tens_left, [mps.site_ind_id], updated_inds_l, inplace=True)
    _, sel_inds_list_r = canonize_tens_list(tens_right[::-1], [mps.site_ind_id], updated_inds_r[::-1],
                                            inplace=True)

    ###
    select_inds = {} if select_inds is None else select_inds
    for i in range(len(sel_inds_list_l)):
        select_inds[updated_inds_l[i]] = sel_inds_list_l[i]
    for i in range(len(sel_inds_list_r)):
        select_inds[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    ## updated select inds; no update for site i
    # for i in range(len(sel_inds_list_l)):
    #     mps.ket_select_inds_l[updated_inds_l[i]] = sel_inds_list_l[i]
    # for i in range(len(sel_inds_list_r)):
    #     mps.ket_select_inds_r[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    # print('updated inds', updated_inds)
    # if mps.in_ind == mps.out_ind:
    #     mps.set_bra_from_ket(sites=updated_inds_l + updated_inds_r)
        ## bra select inds should have already been updated--same object as self.ket_select_inds_x

    return mps, select_inds



####
#######################
###  Local Solvers  ###
#######################

class CrossSolver(LocalSolver, ABC):
    """ primary functions:
        - f(x)
        - f(x) + g(y)
        - TODO: the following
        - solve:
            - with least squares, conjugate gradient descent (CGD)
        - apply, add + compress
        - eigsolve:  (DMRG)
            minimizes <psi|H|psi> / <psi|psi>, equivalently minimizes <psi|H|psi> - lambda <psi|psi>
            sweep through sites i:
                H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
                N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
                --> minimize <psi|H|psi> - lambda <psi|psi>
                --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
                --> H_eff * T[i] - lambda * N_eff * T[i] = 0
                lambda will be the energy of the state
        operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        mps_ind_range:  in case only working on a portion of MPS (indices aren't 0 to L)
    """

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence[Optional['MPO_type']]] = None,
                 operators_H: Optional[Sequence[Optional['MPO_type']]] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 ket_select_inds_l: dict[int, tuple[int, ...]] = None,
                 ket_select_inds_r: dict[int, tuple[int, ...]] = None,
                 bra_select_inds_l: dict[int, tuple[int, ...]] = None,
                 bra_select_inds_r: dict[int, tuple[int, ...]] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Optional[Sequence[int]] = None,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['OperatorEnv']] = None,
                 A_envsRs: Optional[Sequence['OperatorEnv']] = None,
                 b_envsLs: Optional[Sequence['TargetEnv']] = None,
                 b_envsRs: Optional[Sequence['TargetEnv']] = None,
                 # combine_operators_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 combine_targets_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 # left_envs: Optional[Sequence['Environment']] = None,
                 # right_envs: Optional[Sequence['Environment']] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, is_H=False,
                 enrich_rank=0):

        self.ket_select_inds_l = {}
        self.ket_select_inds_r = {}
        self.bra_select_inds_l = {}
        self.bra_select_inds_r = {}
        self.k_envL, self.k_envR = None, None

        super().__init__(ket_state, targets=targets, operators=operators, operators_H=operators_H, bra_state=bra_state,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds, virtual=virtual,
                         A_envsLs=A_envsLs, b_envsLs=b_envsLs, A_envsRs=A_envsRs, b_envsRs=b_envsRs,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         max_bond=max_bond, is_H=is_H,
                         combine_targets_func=combine_targets_func)

        if A_envsRs is None and A_envsLs is None and b_envsRs is None and b_envsLs is None:
            self.initialize_envs(ket_select_inds_l, ket_select_inds_r, bra_select_inds_l, bra_select_inds_r,
                                 op_env0_Ls=op_env0_Ls, op_env0_Rs=op_env0_Rs,
                                 target_env0_Ls=target_env0_Ls, target_env0_Rs=target_env0_Rs)

        bra_horizontal_bonds = {i: self.bra.bond(i, i+1) for i in range(self.L-1)}
        self.k_envL = TargetEnv_Cross(self.L, EnvironmentSide.LEFT, self.ket,
                                      bra=self.ket_select_inds_l,
                                      bra_horizontal_bonds=bra_horizontal_bonds,
                                      # init_env=ket_env0_L,
                                      mps_inds=self.mps_inds)
        self.k_envR = TargetEnv_Cross(self.L, EnvironmentSide.RIGHT, self.ket,
                                      bra=self.ket_select_inds_r,
                                      bra_horizontal_bonds=bra_horizontal_bonds,
                                      # init_env=ket_env0_R,
                                      mps_inds=self.mps_inds)

        # ## enrichment
        # self.Z_enrich = None
        # if enrich_rank > 0:
        #     self.Z_enrich = self.create_like(max_bond=enrich_rank, enrich_rank=0, copy=False, virtual=True)
        #     # print('Z target', self.Z_enrich.targets[0])
        #     # print('Z enrich', type(self.Z_enrich))

    @property
    def left_envs(self) -> list[Environment]:
        return self.A_envsLs + self.b_envsLs + [self.k_envL]

    @property
    def right_envs(self) -> list[Environment]:
        return self.A_envsRs + self.b_envsRs + [self.k_envR]

    @property
    def bra_site_ind_id(self):
        """ desired bra site index """
        return self.operators[-1].upper_ind_id if len(self.operators) > 0 else self.ket.site_ind_id

    @property
    def target_site_ind(self):
        """ desired site ind id for targets
        """
        return self.bra_site_ind_id


    def initialize_envs(self, ket_select_inds_l, ket_select_inds_r, bra_select_inds_l, bra_select_inds_r,
                        op_env0_Ls=None, op_env0_Rs=None, target_env0_Ls=None, target_env0_Rs=None):

        ## operator environments <x_j| A_ji |x_i>
        if self.num_operators > 0:
            if self.A_envsLs is None or len(self.A_envsLs) == 0:
                ket_select_inds_l = {} if ket_select_inds_l is None else ket_select_inds_l
                ket_select_inds_r = {} if ket_select_inds_r is None else ket_select_inds_r
                bra_select_inds_l = ket_select_inds_l if self.in_ind == self.out_ind else bra_select_inds_l
                bra_select_inds_r = ket_select_inds_r if self.in_ind == self.out_ind else bra_select_inds_r

                self.A_envsLs = [OperatorEnv_Cross(self.L, EnvironmentSide.LEFT,
                                                   ket_select_inds_l,
                                                   bra=bra_select_inds_l,
                                                   operator=self.operators[ib],
                                                   init_env=op_env0_Ls[ib] if op_env0_Ls is not None else None,
                                                   end_env=op_env0_Rs[ib] if op_env0_Rs is not None else None, )
                                 for ib in range(self.num_operators)]

                self.A_envsRs = [OperatorEnv_Cross(self.L, EnvironmentSide.RIGHT,
                                                   ket_select_inds_r,
                                                   bra=bra_select_inds_r,
                                                   operator=self.operators[ib],
                                                   init_env=op_env0_Rs[ib] if op_env0_Rs is not None else None,
                                                   end_env=op_env0_Ls[ib] if op_env0_Ls is not None else None, )
                                 for ib in range(self.num_operators)
                                 ]

            else:
                ket_select_inds_l = self.A_envsLs[0].ket
                bra_select_inds_l = self.A_envsLs[0].bra
                ket_select_inds_r = self.A_envsRs[0].ket
                bra_select_inds_r = self.A_envsRs[0].bra

                # self.ket_select_inds_r = {} if ket_select_inds_r is None else ket_select_inds_r
                # self.bra_select_inds_r = self.ket_select_inds_r if in_ind == out_ind else bra_select_inds_r

        ## target environments <x_i|b_i>
        if self.num_targets > 0 and self.in_ind == self.out_ind:
            if self.b_envsLs is None or len(self.b_envsLs) == 0:
                ket_select_inds_l = {} if ket_select_inds_l is None else ket_select_inds_l
                ket_select_inds_r = {} if ket_select_inds_r is None else ket_select_inds_r

                self.b_envsLs = [TargetEnv_Cross(self.L, EnvironmentSide.LEFT,
                                                 self.targets[ib], bra=ket_select_inds_l,
                                                 init_env=target_env0_Ls[
                                                     ib] if target_env0_Ls is not None else None,
                                                 end_env=target_env0_Rs[ib] if target_env0_Rs is not None else None)
                                 for ib in range(self.num_targets)]

                self.b_envsRs = [TargetEnv_Cross(self.L, EnvironmentSide.RIGHT,
                                                 self.targets[ib], bra=ket_select_inds_r,
                                                 init_env=target_env0_Rs[
                                                     ib] if target_env0_Rs is not None else None,
                                                 end_env=target_env0_Ls[
                                                     ib] if target_env0_Ls is not None else None, )
                                 for ib in range(self.num_targets)]

            else:
                ket_select_inds_l = self.b_envsLs[0].bra
                ket_select_inds_r = self.b_envsRs[0].bra

        self.ket_select_inds_l = {} if ket_select_inds_l is None else ket_select_inds_l
        self.ket_select_inds_r = {} if ket_select_inds_r is None else ket_select_inds_r
        self.bra_select_inds_l = {} if bra_select_inds_l is None else bra_select_inds_l
        self.bra_select_inds_r = {} if bra_select_inds_r is None else bra_select_inds_r

    @property
    def Z_ket(self):
        return self.Z_enrich.bra if self.Z_enrich else None

    @property
    def Z_maxbond(self):
        return self.Z_enrich.max_bond if self.Z_enrich else None

    def create_like(self, copy=True, new_ket=None, **kwargs):
        raise NotImplementedError

    # def copy(self, deep=True):
    #     new_solver = super().copy(deep=deep)
    #     # new_solver = self.create_like(copy=deep)
    #     # new_solver.err = self.err
    #     # new_solver.is_conv = self.is_conv
    #     new_solver.ket_select_inds_l = self.ket_select_inds_l
    #     new_solver.ket_select_inds_r = self.ket_select_inds_r
    #     new_solver.bra_select_inds_l = self.bra_select_inds_l
    #     new_solver.bra_select_inds_r = self.bra_select_inds_r
    #     return new_solver

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        super().reinitialize_envs()
        self.ket_select_inds_r.clear()
        self.ket_select_inds_l.clear()
        self.bra_select_inds_r.clear()
        self.bra_select_inds_l.clear()
        # for env in self.left_envs:
        #     if env is not None:  env.reinitialize()
        #
        # for env in self.right_envs:
        #     if env is not None:  env.reinitialize()
        #
        # self.err = np.inf
        # self.is_conv = False

    def reinitialize_envs_i(self, i):
        """ if boundary envs are None, uses old values
        """
        super().reinitialize_envs_i(i)
        # self.ket_select_inds_r.pop(i, None)
        # self.ket_select_inds_l.pop(i, None)
        # self.bra_select_inds_r.pop(i, None)
        # self.bra_select_inds_l.pop(i, None)
        # for env in self.left_envs:
        #     if i == 0:  continue
        #     if env is not None:  env.reinitialize_i(i)
        #
        # for env in self.right_envs:
        #     if i == self.L - 1:  continue
        #     if env is not None:  env.reinitialize_i(i)

    def canonize(self, i, cur_orthog=None):

        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        i = self.get_mps_ind(i)
        updated_inds_l, updated_inds_r = [], []
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds_l += [lx]
                tens_left += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds_l:
                    updated_inds_r += [lx]
                tens_right += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]

        _, sel_inds_list_l = canonize_tens_list(tens_left, [self.ket.site_ind_id], updated_inds_l, inplace=True)
        _, sel_inds_list_r = canonize_tens_list(tens_right[::-1], [self.ket.site_ind_id], updated_inds_r[::-1],
                                                inplace=True)

        ## updated select inds; no update for site i
        for i in range(len(sel_inds_list_l)):
            self.ket_select_inds_l[updated_inds_l[i]] = sel_inds_list_l[i]
        for i in range(len(sel_inds_list_r)):
            self.ket_select_inds_r[updated_inds_r[-1-i]] = sel_inds_list_r[i]

        # print('updated inds', updated_inds)
        if self.in_ind == self.out_ind:
            self.set_bra_from_ket(sites=updated_inds_l + updated_inds_r)
            ## bra select inds should have already been updated--same object as self.ket_select_inds_x


    def left_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i + 1)
        canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id], [ind1, ind2], inplace=True)
        if self.in_ind == self.out_ind:
            self.set_bra_from_ket(sites=[i, i + 1])

    def right_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i - 1)
        canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id], [ind1, ind2], inplace=True)
        if self.in_ind == self.out_ind:
            self.set_bra_from_ket(sites=[i, i + 1])


    #########################

    def sweep_l2r(self, nsites, *args, verbose=False, **kwargs):
        raise NotImplementedError

    def sweep_r2l(self, nsites, *args, verbose=False, **kwargs):
        raise NotImplementedError

    def _site_solve(self, left_site_pos: int, nsites: int, **kwargs) -> tuple[qtn.Tensor, Numeric]:
        site_pos = list(range(left_site_pos, left_site_pos + nsites))
        site_inds = self.mps_inds[left_site_pos: left_site_pos + nsites]
        site_l = np.min(site_inds)
        site_r = np.max(site_inds)
        inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm(), current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        ### this is also the selected submatrix if self is in cross-canonical form
        b_eff = self.get_b_eff(left_site_pos, nsites)

        ## solve for T[i]
        # x_eff = b_eff.reindex(bra_to_ket_inds)   ### ket is not orthogonal
        x_eff = b_eff.copy()
        # if self.func is not None:
        #     x_eff.modify(apply=lambda x: self.func(x))
        x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
        ## b_eff_exponent already in b_eff tensor

        ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
        ## but it should be canonical?
        tensors = [x_eff]
        if site_l > 0:
            inv_env_l = self.k_envL[site_l].copy()
            inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
            tensors += [inv_env_l]
        if site_r < self.L - 1:
            inv_env_r = self.k_envR[site_r].copy()
            inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
            tensors += [inv_env_r]

        x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
        x_eff.transpose(*(inds_l + [self.ket.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)
        # print('x eff data', x_eff.norm(), x_eff.data)

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm(x_eff.data - current_x.data)

        return x_eff, site_err

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', **kwargs):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('update 1 site', site_i.norm(), direction)
        if direction == SweepDirection.RIGHT:
            ## update ket
            ind1 = self.mps_inds[i]
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            ## canonicalize ket to next site
            if i < self.L - 1:
                ind2 = self.mps_inds[i + 1]
                _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                self.ket_select_inds_l[ind1] = select_inds[0]

            #     if self.in_ind == self.out_ind:
            #         self.set_bra_from_ket(sites=[i, i + 1])
            # else:
            #     ## update bra
            #     if self.in_ind == self.out_ind:
            #         self.set_bra_from_ket(sites=[i])

        else:
            ## update k
            ind1 = self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = self.mps_inds[i - 1]
                _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                self.ket_select_inds_r[ind1] = select_inds[0]

            #     ## update bra
            #     if self.in_ind == self.out_ind:
            #         self.set_bra_from_ket(sites=[i, i - 1])
            # else:
            #     ## update bra
            #     if self.in_ind == self.out_ind:
            #         self.set_bra_from_ket(sites=[i])

        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i + 1]
            split_inds, unshared_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            right_inds = [ik for ik in self.ket[ind2].inds if ik not in split_inds]

            phys_inds = [self.ket.site_ind_id.format(ind1)]
            left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
            left_inds = phys_inds + left_inds

            site_i.transpose(*left_inds, *right_inds, inplace=True)
            shape_l = site_i.shape[:len(left_inds)]
            shape_r = site_i.shape[-len(right_inds):]
            site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))
            # mat1, mat2, select_inds = cross_split( site_i_mat, do_qr=True, return_inds=True)
            mat1, mat2, select_inds = cross_compress(site_i_mat, self.max_bond, do_qr=True, return_inds=True)


            mat1 = mat1.reshape(shape_l + (-1,))
            mat2 = mat2.reshape((-1,) + shape_r)

            ## update ket
            self.ket[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
            self.ket[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
            self.ket_select_inds_l[ind1] = select_inds

            # ## update bra
            # if self.in_ind == self.out_ind:
            #     self.set_bra_from_ket(sites=[i, i + 1])

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, unshared_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            right_inds = [ik for ik in self.ket[ind2].inds if ik not in split_inds]

            phys_inds = [self.ket.site_ind_id.format(ind1)]
            left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
            left_inds = phys_inds + left_inds

            site_i.transpose(*left_inds, *right_inds, inplace=True)
            shape_l = site_i.shape[:len(left_inds)]
            shape_r = site_i.shape[-len(right_inds):]
            site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))
            # mat1, mat2, select_inds = cross_split(site_i_mat, do_qr=True, return_inds=True)
            mat1, mat2, select_inds = cross_compress(site_i_mat, self.max_bond, do_qr=True, return_inds=True)

            mat1 = mat1.reshape(shape_l + (-1,))
            mat2 = mat2.reshape((-1,) + shape_r)

            ## update ket
            self.ket[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
            self.ket[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
            self.ket_select_inds_r[ind1] = select_inds

            # ## update bra
            # if self.in_ind == self.out_ind:
            #     self.set_bra_from_ket(sites=[i, i - 1])

        return


    def _update_2site_svd(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i + 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site1.data, inds=site1.inds)
            self.ket[ind2].modify(data=site2.data, inds=site2.inds)

            _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                [ind1, ind2], inplace=True)
            self.ket_select_inds_l[ind1] = select_inds[0]

            # ## update bra
            # if self.in_ind == self.out_ind:
            #     self.set_bra_from_ket(sites=[i, i + 1])

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                [ind1, ind2], inplace=True)
            self.ket_select_inds_r[ind1] = select_inds[0]

            # ## update bra
            # if self.in_ind == self.out_ind:
            #     self.set_bra_from_ket(sites=[i, i - 1])

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def check_err(self, get_approx_err: bool=True, left_site_ind: int=0, nsites: int=1) -> 'float':
        return self.err

    ####
    def initialize_Z(self, direction=SweepDirection.LEFT):

        if self.Z_maxbond == 0:
            return

        if direction == SweepDirection.RIGHT:
            for i in range(self.L-1):
                if i == 0:
                    tens = np.random.rand(self.ket.phys_dim(), self.Z_maxbond)
                else:
                    tens = np.random.rand(self.ket.phys_dim()*self.Z_maxbond, self.Z_maxbond)

                q, r = np.linalg.qr(tens)
                z_inds = maxvol_inds(q)     # n x r (open virtual)
                self.Z_enrich.select_inds_l[i] = z_inds

            # print(self.Z_ket)
            # print(self.targets[0])
            # print(self.Z_enrich.select_inds_l)
            self.Z_enrich.build_all_envs_left(1, canonize=False)

        elif direction == SweepDirection.LEFT:
            for i in range(self.L - 1, 0, -1):
                if i == self.L - 1:
                    tens = np.random.rand(self.ket.phys_dim(), self.Z_maxbond)
                else:
                    tens = np.random.rand(self.ket.phys_dim()*self.Z_maxbond, self.Z_maxbond)

                q, r = np.linalg.qr(tens)
                z_inds = maxvol_inds(q)  # n x r (open virtual)
                self.Z_enrich.select_inds_r[i] = z_inds

            self.Z_enrich.build_all_envs_right(1, canonize=False)

        else:
            raise ValueError

        return

