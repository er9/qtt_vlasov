""" v2 works.
    this is a cleaner version + elemental multiplication of f(x) * g(y)
    where f, g are element-wise operations
"""

from setup.configs import *
import time
# from scipy import linalg
import scipy.linalg as linalg
import quimb.tensor as qtn
# import helper_quimb as helper
from local_solvers.defaults import *
from local_solvers.mps_classes import MPS

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']



def check_left_orthog(mpx: Union['qtn.MatrixProductState', Sequence[qtn.Tensor]],
                      select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L):
        tens = mpx[i].copy()

        bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L-1 else []
        bond_p = [s.format(i) for s in site_ind_ids]

        tens = tens.fuse({'eff': bond_p + bond_l})
        tens.transpose('eff', *bond_r, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            env = tens.data[select_inds[i], :]
            err = np.linalg.norm(env - np.eye(env.shape[0]))
        except (KeyError, ValueError, IndexError):
            print(f'select ind not defined or incompatible for {i}')
            return i

        if err > 1.0e-10:
            print('not left orthog', i, err) #, env)
            break

    return i


def check_right_orthog(mpx: Union['qtn.MatrixProductState', Sequence['qtn.Tensor']],
                       select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L - 1, -1, -1):
        tens = mpx[i].copy()

        bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L - 1 else []
        bond_p = [s.format(i) for s in site_ind_ids]  # [mpx.site_ind_id.format(i)]

        tens = tens.fuse({'eff': bond_p + bond_r})
        tens.transpose('eff', *bond_l, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            env = tens.data[select_inds[i], :]
            err = np.linalg.norm(env - np.eye(env.shape[0]))
        except (KeyError, ValueError, IndexError):
            print(f'select ind not defined or incompatible for {i} ')
            return i

        if err > 1.0e-10:
            print('not right orthog', i, err) #, env)
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

        q, r = qtn.tensor_split(tens_list[i], fuse_inds, absorb='right',
                                method = 'qr',
                                # method='svd', cutoff=0.0,
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


def compress_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
                       inplace=True, max_bond=None):

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

        tens = tens_list[i].transpose(*fuse_inds, *right_inds, inplace=True)
        shape_left = tens.shape[:len(fuse_inds)]
        shape_right = tens.shape[len(fuse_inds):]
        tens = tens.fuse({f'xx{i}': fuse_inds, f'oo{i}': right_inds})
        q, r, inds = cross_compress(tens.data, do_qr=True, max_bond=max_bond, return_inds=True)
        q = q.reshape(*shape_left, -1)
        r = r.reshape(-1, *shape_right)

        tens_list[i].modify(data=q)

        r_tens = qtn.Tensor(r, inds=(f'tmp{i}', *right_inds))
        next_site = qtn.tensor_contract(r_tens, tens_list[i + 1])
        next_site.transpose_like(tens_list[i + 1], inplace=True)
        tens_list[i + 1].modify(data=next_site.data)

        # # q, r = qtn.tensor_split(tens_list[i], fuse_inds, absorb='right',
        # #                         method = 'qr',
        # #                         # method='svd', cutoff=0.0,
        # #                         bond_ind=f'__tmp_{i}__')
        # #
        # # q.transpose(*fuse_inds, f'__tmp_{i}__', inplace=True)
        # # r.transpose(f'__tmp_{i}__', *right_inds, inplace=True)
        # #
        # # ## obtain dominant submatrix
        # # q_ = q.fuse({f'xx{i}': fuse_inds}, inplace=False)
        # # q_.transpose(f'xx{i}', f'__tmp_{i}__', inplace=True)
        # # # nr, nc = q_.shape[0], q_.shape[-1]
        # # inds = maxvol_inds(q_.data, do_qr=False)  ## n, r
        # # q_.modify(apply=lambda x: x[inds,:])    ## submat
        # #
        # # q.modify(apply=lambda x: np.tensordot(x, np.linalg.inv(q_.data), axes=(-1,0)))
        # # r.modify(apply=lambda x: np.dot(q_.data, x))
        #
        # ## update sites in MPS
        # q.transpose_like(tens_list[i], inplace=True)
        # tens_list[i].modify(data=q.data)
        #
        # # print('next site', tens_list[i+1].norm())
        # next_site = qtn.tensor_contract(r, tens_list[i + 1])
        # next_site.transpose_like(tens_list[i + 1], inplace=True)
        # tens_list[i + 1].modify(data=next_site.data)
        # # print('new next site', tens_list[i + 1].norm())

        select_inds_list += [inds]

    return tens_list, select_inds_list


# def split_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
#                     select_inds: list[Sequence[int]], inplace=True):
#
#     L = len(tens_list)
#     tens_list = tens_list if inplace else [t.copy() for t in tens_list]
#
#
#     for i in range(0, L-1):
#         right_inds, unshared_inds = tens_list[i].filter_bonds(tens_list[i+1])
#         ## shared and unshared bonds
#         phys_inds = [s.format(site_inds[i]) for s in site_ind_ids]
#         left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
#         assert(len(left_inds)<=1), 'not specific enough if have multiple virtual "left inds"'
#
#         ## must follow specific order for fusing unshared bonds
#         fuse_inds = phys_inds + left_inds
#
#         ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
#         ## canon form: want C (A_)^-1
#         ## C is essentially the tensor core
#         ## Let C = QT -->
#         ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1
#
#         tens = tens_list[i].transpose(*fuse_inds, *right_inds, inplace=True)
#         shape_left = tens.shape[:len(fuse_inds)]
#         shape_right = tens.shape[len(fuse_inds):]
#         tens = tens.fuse({f'xx{i}': fuse_inds, f'oo{i}': right_inds})
#
#         q, r = split_by_inds(tens.data, select_inds[i], do_qr=True)
#         q = q.reshape(*shape_left, -1)
#         r = r.reshape(-1, *shape_right)
#
#         tens_list[i].modify(data=q)
#
#         r_tens = qtn.Tensor(r, inds=(f'tmp{i}', *right_inds))
#         next_site = qtn.tensor_contract(r_tens, tens_list[i + 1])
#         next_site.transpose_like(tens_list[i + 1], inplace=True)
#         tens_list[i + 1].modify(data=next_site.data)
#
#     return tens_list



# def split_by_inds(A: np.ndarray, sel_inds: Sequence[int], do_qr=False):
#     ### decompose matrix using specified sel_inds
#
#     n, r = A.shape  ## A is 2D matrix: (alpha_{i-1} * d_{i}) x alpha_{i}
#
#     if do_qr or n < r:
#         Q, R_dmp = np.linalg.qr(A)
#     else:
#         Q = A.copy()
#
#     # n, r = Q.shape
#
#     # p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
#     # p = np.argsort(p)       # so that P A = L U, as in matlab
#     # inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)
#
#     inds = np.array(sel_inds)
#     # inds = sel_inds
#     submat = Q[inds, :]  # permute rows of A
#
#     T1 = Q @ np.linalg.inv(submat)       # Q Q^-1 = B = [I, Z]
#
#     ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
#     ## canon form: want C (A_)^-1
#     ## C is essentially the tensor core
#     ## Let C = QT -->
#     ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1
#
#     submat = Q[inds,:]
#     T2 = (submat @ R_dmp) if do_qr else (submat)
#     # print('T2 norm', np.linalg.norm(T2))
#
#     return T1, T2


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




def canonize(mps: 'qtn.MatrixProductState', i:int, cur_orthog:int=None, select_inds: dict[int,Sequence[int]]=None):
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


def update_1site(mps: MPS, left_site_pos: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', max_bond:int = None):
    """ update ket, bra with new_site
        i: int of mps site
    """
    # print('update 1 site', site_i.norm(), direction)
    if direction == SweepDirection.RIGHT:
        ## update ket
        ind1 = left_site_pos  # self.mps_inds[i]
        site_i.transpose_like(mps[ind1], inplace=True)
        mps[ind1].modify(data=site_i.data)

        ## canonicalize ket to next site
        if ind1 < mps.L - 1:
            ind2 = ind1 + 1  # self.mps_inds[i + 1]
            # _, select_inds = compress_tens_list([mps[ind1], mps[ind2]],
            _, select_inds = compress_tens_list([mps[ind1], mps[ind2]],
                                                site_ind_ids=[mps.site_ind_id], site_inds=[ind1, ind2],
                                                inplace=True, max_bond=max_bond)
            # self.ket_select_inds[ind1] = select_inds[0]
            mps.select_inds[ind1] = select_inds[0]
            mps.cur_orthog = ind2

    else:
        ## update k
        ind1 = left_site_pos   # self.mps_inds[i]
        mps[ind1].modify(data=site_i.data, inds=site_i.inds)

        ## canonicalize ket to next site
        if ind1 > 0:
            ind2 = ind1 - 1  # self.mps_inds[i - 1]
            # _, select_inds = compress_tens_list([mps[ind1], mps[ind2]],
            _, select_inds = compress_tens_list([mps[ind1], mps[ind2]],
                                                 site_ind_ids=[mps.site_ind_id],
                                                 site_inds=[ind1, ind2],
                                                 inplace=True, max_bond=max_bond)
            # self.ket_select_inds[ind1] = select_inds[0]
            mps.select_inds[ind1] = select_inds[0]
            mps.cur_orthog = ind2

    return


def update_2site(mps: 'MPS', left_site_pos: int, site_i: 'qtn.Tensor', direction: 'SweepDirection',
                 max_bond: int=None,):
    """ update ket, bra with new_site
        i: mps_site
        only decimate / only keep select inds --- can't guarantee select_inds are consistent?
    """
    # print('update 2 site', site_i.norm(), direction)
    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1   # self.mps_inds[i], self.mps_inds[i + 1]
        split_inds, unshared_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        right_inds = [ik for ik in mps[ind2].inds if ik not in split_inds]

        phys_inds = [mps.site_ind_id.format(ind1)]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        left_inds = phys_inds + left_inds

        site_i.transpose(*left_inds, *right_inds, inplace=True)
        shape_l = site_i.shape[:len(left_inds)]
        shape_r = site_i.shape[-len(right_inds):]
        site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))
        mat1, mat2, select_inds = cross_compress(site_i_mat, max_bond, do_qr=True, return_inds=True)

        mat1 = mat1.reshape(shape_l + (-1,))
        mat2 = mat2.reshape((-1,) + shape_r)

        ## update ket
        mps.select_inds[ind1] = select_inds
        mps[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
        mps[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))

    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos
        split_inds, unshared_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        right_inds = [ik for ik in mps[ind2].inds if ik not in split_inds]

        phys_inds = [mps.site_ind_id.format(ind1)]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        left_inds = phys_inds + left_inds

        site_i.transpose(*left_inds, *right_inds, inplace=True)
        shape_l = site_i.shape[:len(left_inds)]
        shape_r = site_i.shape[-len(right_inds):]
        site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))
        # mat1, mat2, select_inds = cross_split(site_i_mat, do_qr=True, return_inds=True)
        mat1, mat2, select_inds = cross_compress(site_i_mat, max_bond, do_qr=True,
                                                 return_inds=True)

        mat1 = mat1.reshape(shape_l + (-1,))
        mat2 = mat2.reshape((-1,) + shape_r)

        ## update ket
        mps[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
        mps[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
        mps.select_inds[ind1] = select_inds

    mps.cur_orthog = ind2

    return


def _update_2site_svd(mps: 'MPS', left_site_pos: int, site_i: 'qtn.Tensor', direction: 'SweepDirection',
                      max_bond: int = None):
    """ update ket, bra with new_site
        i: mps_site
    """
    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1  # self.mps_inds[i], self.mps_inds[i + 1]
        split_inds, left_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=max_bond)

        ## update ket
        mps[ind1].modify(data=site1.data, inds=site1.inds)
        mps[ind2].modify(data=site2.data, inds=site2.inds)

        _, select_inds = canonize_tens_list([mps[ind1], mps[ind2]],
                                             site_ind_ids=[mps.site_ind_id],
                                             site_inds=[ind1, ind2], inplace=True)
        mps.select_inds[ind1] = select_inds[0]

    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos
        split_inds, left_inds = mps[ind1].filter_bonds(mps[ind2])  # inds only on ket[i]
        site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=max_bond)

        ## update ket
        mps[ind1].modify(data=site2.data, inds=site2.inds)
        mps[ind2].modify(data=site1.data, inds=site1.inds)

        _, select_inds = canonize_tens_list([mps[ind1], mps[ind2]],
                                            site_ind_ids=[mps.site_ind_id],
                                            site_inds=[ind1, ind2], inplace=True)
        mps.select_inds[ind1] = select_inds[0]

    mps.cur_orthog = ind2

    return




