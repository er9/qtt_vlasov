""" v2 works.
    this is a cleaner version + elemental multiplication of f(x) * g(y)
    where f, g are element-wise operations
"""

from setup.configs import *
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


def maxvol_inds(A: 'np.ndarray', max_iters=DEFAULT_MAX_TOT_ITER, conv_tol=CONV_TOL, do_qr=True, ):
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




####
#######################
###  Local Solvers  ###
#######################

class CrossLocalSolver(LocalSolver):
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
    env_class = CrossEnvironment

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence[Optional['MPO_type']]] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Optional[Sequence[int]] = None,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['CrossEnvironment']] = None,
                 A_envsRs: Optional[Sequence['CrossEnvironment']] = None,
                 b_envsLs: Optional[Sequence['CrossEnvironment']] = None,
                 b_envsRs: Optional[Sequence['CrossEnvironment']] = None,
                 # left_envs: Optional[Sequence['CrossEnvironment']] = None,
                 # right_envs: Optional[Sequence['CrossEnvironment']] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, enrich_rank=0):

        super().__init__(ket_state, targets=targets, operators=operators,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
                         op_env0_Rs=op_env0_Rs, op_env0_Ls=op_env0_Ls, A_envsRs=A_envsRs, A_envsLs=A_envsLs,
                         target_env0_Rs=target_env0_Rs, target_env0_Ls=target_env0_Ls,
                         b_envsLs=b_envsLs, b_envsRs=b_envsRs,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         max_bond=max_bond, virtual=virtual)

        ## environments
        self.select_inds_r = {}  ## select inds from maxvol at each tensor site (right to left)
        self.select_inds_l = {}  ## select inds from maxvol at each tensor site (left to right)

        ## enrichment
        self.Z_enrich = None
        if enrich_rank > 0:
            self.Z_enrich = self.create_like(max_bond=enrich_rank, enrich_rank=0, copy=False, virtual=True)
            # print('Z target', self.Z_enrich.targets[0])
            # print('Z enrich', type(self.Z_enrich))


    # @property
    # def L(self):
    #     return len(self.mps_inds)  # self.mps_inds[1] - self.mps_inds[0]
    #
    # @property
    # def num_targets(self):
    #     return len(self._targets)
    #
    # @property
    # def num_operators(self):
    #     return len(self._operators)
    #
    # @property
    # def left_envs(self):
    #     return self._left_envs
    #
    # @property
    # def right_envs(self):
    #     return self._right_envs
    #
    # @property
    # def ket(self):
    #     return self._ket
    #
    # @ket.setter
    # def ket(self, new_ket: qtn.MatrixProductState):
    #     self._ket = new_ket if self.virtual else new_ket.copy()
    #     for target in self.targets:
    #         target.site_ind_id = new_ket.site_ind_id
    #     if self.num_operators > 0:
    #         for op in self.operators:
    #             op.lower_ind_id = new_ket.site_ind_id
    #         # self.operators[0].lower_ind_id = new_ket.site_ind_id
    #     self.reinitialize_envs()
    #
    # @property
    # def bra_site_ind_id(self):
    #     """ desired bra site index """
    #     return self.operators[-1].upper_ind_id if len(self.operators) > 0 else self.ket.site_ind_id
    #
    #
    # def update_ket_tens(self, new_ket: Sequence['qtn.Tensor'], sites=None, reinit_envs=True,
    #                     new_exponent=None):
    #     """ inplace update of bra with specified sites
    #     """
    #
    #     sites = range(self.L) if sites is None else sites
    #     # sites = self.mps_inds if sites is None else sites
    #     for ix, new_ket_tens in zip(sites, new_ket):
    #         i = self.mps_inds[ix]
    #         self._ket[i].modify(data=new_ket_tens.data, inds=tuple(new_ket_tens.inds))
    #
    #         if reinit_envs:
    #             self.reinitialize_envs_i(ix)
    #
    #     if new_exponent is not None:
    #         self._ket.exponent = new_exponent
    #     return
    #
    #
    # @property
    # def target_site_ind(self):
    #     """ desired site ind id for targets
    #     """
    #     return self.bra_site_ind_id
    #
    # @property
    # def targets(self):
    #     return self._targets
    #
    # @targets.setter
    # def targets(self, new_targets: Union['qtn.MatrixProductState', Sequence[qtn.MatrixProductState]]):
    #     if new_targets is None:
    #         self._targets = []
    #     elif isinstance(new_targets, qtn.TensorNetwork):
    #         self._targets = [new_targets if self.virtual else new_targets.copy()]
    #     else:
    #         self._targets = []
    #         for new_target in new_targets:
    #             new_target = new_target if self.virtual or new_target is None else new_target.copy()
    #             if new_target is not None:
    #                 new_target.site_ind_id = self.target_site_ind  # self.bra.site_ind_id
    #                 if not self.virtual:
    #                     new_target.mangle_inner_()
    #             self._targets += [new_target]
    #     self.reinitialize_envs()    ## technically wouldn't have to update norm envs
    #
    #     # target_ovlps = 0.0
    #     # for target in self._targets:
    #     #     if target is not None:
    #     #         target_ovlps += helper.ovlp(target, target.conj())
    #     # for ti, tj in np.ndindex(len(self._targets), len(self._targets)):
    #     #     if ti > tj:
    #     #         if self._targets[ti] is not None and self._targets[tj] is not None:
    #     #             out = helper.ovlp(self._targets[ti], self._targets[tj].conj())
    #     #             target_ovlps += 2 * np.real(out)
    #     # self.targets_norm = np.sqrt(target_ovlps)
    #
    #     # print('targets norm', self.targets_norm)
    #
    # @property
    # def operators(self) -> list['MPO_type']:
    #     return self._operators
    #
    # @operators.setter
    # def operators(self, new_operators: Sequence['MPO_type']):
    #     self._operators = [] if new_operators is None else \
    #         [mpo if self.virtual or mpo is None else mpo.copy() for mpo in new_operators]
    #
    #     upper_ind_id = None  # if self.bra is None else self.bra.site_ind_id
    #     for op in self._operators:
    #         if op is None:  continue
    #         # op.upper_ind_id = self.bra.site_ind_id
    #         if op.upper_ind_id == self.ket.site_ind_id:
    #             op.upper_ind_id = op.upper_ind_id + '_out'
    #         op.lower_ind_id = self.ket.site_ind_id
    #         if upper_ind_id is not None:
    #             op.upper_ind_id = upper_ind_id
    #         # op.mangle_inner_()
    #         upper_ind_id = op.upper_ind_id
    #
    #     self.reinitialize_envs()

    @property
    def Z_ket(self):
        return self.Z_enrich.bra if self.Z_enrich else None

    @property
    def Z_maxbond(self):
        return self.Z_enrich.max_bond if self.Z_enrich else None

    def create_like(self, copy=True, new_ket=None, **kwargs):
        raise NotImplementedError

    def copy(self, deep=True):
        new_solver = super().copy(deep=deep)
        # new_solver = self.create_like(copy=deep)
        # new_solver.err = self.err
        # new_solver.is_conv = self.is_conv
        new_solver.select_inds_l = self.select_inds_l
        new_solver.select_inds_r = self.select_inds_r
        return new_solver

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        super().reinitialize_envs()
        self.select_inds_r = {}
        self.select_inds_l = {}
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
        self.select_inds_r.pop(i)
        self.select_inds_l.pop(i)
        # for env in self.left_envs:
        #     if i == 0:  continue
        #     if env is not None:  env.reinitialize_i(i)
        #
        # for env in self.right_envs:
        #     if i == self.L - 1:  continue
        #     if env is not None:  env.reinitialize_i(i)

    # def get_ket_to_bra_inds(self, i) -> dict[str, str]:
    #     """ inds mapping inds on self.bra to corresponding inds on self.ket
    #     """
    #     ind1 = self.mps_inds[i]
    #     inds_dict = {self.ket.site_ind(ind1): self.bra.site_ind(ind1)}
    #
    #     # print('self.mps inds', self.L, self.mps_inds, i)
    #     b1 = self.bra.select_tensors(self.bra.site_tag_id.format(ind1))[0]
    #     k1 = self.ket.select_tensors(self.ket.site_tag_id.format(ind1))[0]
    #
    #     ## right bond
    #     if i < self.L - 1:
    #         # inds_dict[self.ket.bond(i,i+1)] = self.bra.bond(i,i+1)
    #         ind2 = self.mps_inds[i + 1]
    #         b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
    #         k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]
    #
    #         bbond = next(iter(b1.bonds(b2)))
    #         kbond = next(iter(k1.bonds(k2)))
    #         inds_dict[kbond] = bbond
    #
    #     ## left bond
    #     if i > 0:
    #         # inds_dict[self.ket.bond(i, i - 1)] = self.bra.bond(i, i - 1)
    #         ind2 = self.mps_inds[i - 1]
    #         b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
    #         k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]
    #
    #         bbond = next(iter(b1.bonds(b2)))
    #         kbond = next(iter(k1.bonds(k2)))
    #         inds_dict[kbond] = bbond
    #
    #     ## ancilla bonds (L-1)
    #     if i == self.L - 1 or i == 0:
    #         items = [item for k, item in inds_dict.items()]
    #         anc_b = [ind for ind in self.bra[ind1].inds if ind not in items]
    #         anc_k = [ind for ind in self.ket[ind1].inds if ind not in inds_dict.keys()]
    #         for (ab, ak) in zip(anc_b, anc_k):
    #             inds_dict[ak] = ab
    #
    #     # print('inds dict', i, inds_dict, ind1)
    #
    #     return inds_dict

    # def get_bra_to_ket_inds(self, i) -> dict[str, str]:
    #     ket_to_bra = self.get_ket_to_bra_inds(i)
    #     bra_to_ket = {item: k for k, item in ket_to_bra.items()}
    #     return bra_to_ket

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
            self.select_inds_l[updated_inds_l[i]] = sel_inds_list_l[i]
        for i in range(len(sel_inds_list_r)):
            self.select_inds_r[updated_inds_r[-1-i]] = sel_inds_list_r[i]

        # print('updated inds', updated_inds)
        # self.set_bra_from_ket(sites=updated_inds_l + updated_inds_r)


    def left_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i + 1)
        canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id], [ind1, ind2], inplace=True)

    def right_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i - 1)
        canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id], [ind1, ind2], inplace=True)

    def add_rand(self, canon_direction, strength=0.01):
        """ canon direction: desired end canon direction
                             sweep direction left/right -> right/left canon
        """
        raise NotImplementedError

    def get_mps_ind(self, i):
        return self.mps_inds[i]

    #########################

    # def get_b_eff(self, sites: Union[int, slice]) -> Optional[qtn.Tensor]:
    #     raise NotImplementedError
    #
    # def get_A_effs(self, sites: Union[int, slice], return_sum=False) \
    #         -> Optional[Union[qtn.Tensor, Sequence[qtn.TensorNetwork]]]:
    #     raise NotImplementedError

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
                self.select_inds_l[ind1] = select_inds[0]

        else:
            ## update k
            ind1 = self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = self.mps_inds[i - 1]
                _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                self.select_inds_r[ind1] = select_inds[0]

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
            self.select_inds_l[ind1] = select_inds

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
            self.select_inds_r[ind1] = select_inds

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
            self.select_inds_l[ind1] = select_inds[0]

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                [ind1, ind2], inplace=True)
            self.select_inds_r[ind1] = select_inds[0]

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    # def build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
    #     """ compute all right envs for <Ax|b>
    #     """
    #     end = nsites - 1 if end is None else end
    #
    #     if canonize:
    #         self.canonize(end)
    #
    #     for env in self.right_envs:
    #         if env is not None:
    #             for i in range(self.L - 1, end, -1):
    #                 env.extend_env(i) #, select_inds=self.select_inds_r[i])
    #
    #     self.canon_direction = EnvironmentSide.RIGHT
    #     return

    # def build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
    #     """ compute all left envs for <Ax|b>
    #     """
    #     end = self.L - nsites if end is None else end
    #
    #     if canonize:
    #         self.canonize(end)
    #
    #     for env in self.left_envs:
    #         if env is not None:
    #             for i in range(end):
    #                 env.extend_env(i) # , select_inds=self.select_inds_l[i])
    #
    #     self.canon_direction = EnvironmentSide.LEFT
    #     return

    # def _update_envs_left(self, pos: int): #, canonize=False):
    #     # if canonize:
    #     #     self.canonize(pos + 1)
    #         # helper.canonize(self.ket, i=pos + 1)
    #         # self.set_bra_from_ket()
    #     for env in self.left_envs:
    #         # inds = self.select_inds_l[pos]
    #         if env is not None:  env.extend_env(pos) #, select_inds=inds)
    #     return

    # def _update_envs_right(self, pos: int):  #, canonize=False):
    #     # if canonize:
    #     #     self.canonize(pos - 1)
    #         # helper.canonize(self.ket, i=pos - 1)
    #         # self.set_bra_from_ket()
    #     for env in self.right_envs:
    #         # inds = self.select_inds_r[pos]
    #         if env is not None:  env.extend_env(pos) # , select_inds=inds)
    #     return

    # def solve(self, *args, **kwargs):
    #     raise NotImplementedError
    #
    # def check_err(self, *args, **kwargs) -> 'float':
    #     """ also updates self.err, self.is_conv
    #     """
    #     raise NotImplementedError


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


class CrossAdder(CrossLocalSolver):
    def __init__(self, ket_state: 'qtn.MatrixProductState'=None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 ket_env0_L: 'qtn.Tensor' = None,
                 ket_env0_R: 'qtn.Tensor' = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, enrich_rank=0, func=None):

        # operators = [] if operators is None else list(operators)
        # self.A_envsLs, self.A_envsRs = [], []
        # self.b_envsLs, self.b_envsRs = [], []
        # self.k_envL, self.k_envR = None, None

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.target_env0_Ls = target_env0_Ls
        self.target_env0_Rs = target_env0_Rs

        if ket_state is None:
            init_state = qtn.MPS_rand_state(targets[0].L, 2, targets[0].phys_dim())
        else:
            init_state = ket_state

        self.canon_direction = 0
        self.func = func

        super().__init__(init_state, targets=targets, operators=None,
                         mps_inds=mps_inds, virtual=virtual, conv_tol=conv_tol, enrich_rank=enrich_rank,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter)

        # self.k_envL = CrossEnvironment(self.L, EnvironmentSide.LEFT, self.ket,
        #                                init_env=ket_env0_L,
        #                                select_inds_dict=self.select_inds_l,
        #                                mps_inds=self.mps_inds)
        # self.k_envR = CrossEnvironment(self.L, EnvironmentSide.RIGHT, self.ket,
        #                                init_env=ket_env0_R,
        #                                select_inds_dict=self.select_inds_r,
        #                                mps_inds=self.mps_inds)

        # for ib in range(self.num_targets):
        #     b_envL = CrossEnvironment(self.L, EnvironmentSide.LEFT, self.targets[ib], None,
        #                               init_env=target_env0_Ls[ib] if target_env0_Ls is not None else None,
        #                               select_inds_dict=self.select_inds_l,
        #                               bra_horizontal_bonds=self.k_envL.bra_horizontal_bonds,
        #                               mps_inds=self.mps_inds, )
        #     b_envR = CrossEnvironment(self.L, EnvironmentSide.RIGHT, self.targets[ib], None,
        #                               init_env=target_env0_Rs[ib] if target_env0_Rs is not None else None,
        #                               select_inds_dict=self.select_inds_r,
        #                               bra_horizontal_bonds=self.k_envR.bra_horizontal_bonds,
        #                               mps_inds=self.mps_inds, )
        #     self.b_envsLs += [b_envL]
        #     self.b_envsRs += [b_envR]


        # if enrich_rank > 0:
        #     self.Zb_envsLs, self.Zb_envsRs = [], []
        #     for ib in range(self.num_targets):
        #         b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, None,
        #                              mps_inds=self.mps_inds,
        #                              init_env=target_env0_Ls[ib] if target_env0_Ls is not None else None)
        #         b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, None,
        #                              mps_inds=self.mps_inds,
        #                              init_env=target_env0_Rs[ib] if target_env0_Rs is not None else None)
        #         self.Zb_envsLs += [b_envL]
        #         self.Zb_envsRs += [b_envR]
        #
        #     self.Zk_envL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra,  # init_norm=...
        #                               mps_inds=self.mps_inds)
        #     self.Zk_envR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra,  # init_norm=...
        #                               mps_inds=self.mps_inds)


    @property
    def k_envL(self):
        return self.A_envsLs[0]

    @property
    def k_envR(self):
        return self.A_envsRs[0]

    # @property
    # def left_envs(self):
    #     return self.b_envsLs + [self.k_envL]
    #
    # @property
    # def right_envs(self):
    #     return self.b_envsRs + [self.k_envR]

    # def reinitialize_envs(self):
    #     """ if boundary envs are None, uses old values
    #     """
    #     self.select_inds_r = {}
    #     self.select_inds_l = {}
    #
    #     for env in self.left_envs:
    #         if env is not None:
    #             env.reinitialize()
    #
    #     for env in self.right_envs:
    #         if env is not None:
    #             env.reinitialize()
    #
    #     self.err = np.inf
    #     self.is_conv = False
    #     return

    def create_like(self, copy=True, new_ket=None, **kwargs):
        ket_env0_L = self.ket_env0_L
        ket_env0_R = self.ket_env0_R
        target_env0_Ls = self.target_env0_Ls
        target_env0_Rs = self.target_env0_Rs

        new_solver = self.__class__((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                    targets=kwargs.get('targets', [t.copy() if copy else t for t in self.targets]),
                                    mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                    ket_env0_L=kwargs.get('ket_env0_L', ket_env0_L),
                                    ket_env0_R=kwargs.get('ket_env0_R', ket_env0_R),
                                    target_env0_Ls=kwargs.get('target_env0_Ls', target_env0_Ls),
                                    target_env0_Rs=kwargs.get('target_env0_Rs', target_env0_Rs),
                                    conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                    max_iter=kwargs.get('max_iter', self.max_iter),
                                    max_bond=kwargs.get('max_bond', self.max_bond),
                                    max_tot_iter=kwargs.get('max_tot_iter', self.max_tot_iter),
                                    func=kwargs.get('func', self.func),
                                    virtual=kwargs.get('virtual', False)
                                    )
        return new_solver

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)

        new_solver.select_inds_r = self.select_inds_r.copy()
        new_solver.select_inds_l = self.select_inds_l.copy()

        new_solver.b_envsLs = [bL.copy() if bL is not None else None for bL in self.b_envsLs]
        new_solver.b_envsRs = [bR.copy() if bR is not None else None for bR in self.b_envsRs]
        new_solver.A_envsLs = [AL.copy() if AL is not None else None for AL in self.A_envsLs]
        new_solver.A_envsRs = [AR.copy() if AR is not None else None for AR in self.A_envsRs]
        # new_solver.k_envL = self.k_envL.copy()
        # new_solver.k_envR = self.k_envR.copy()
        for env in new_solver.right_envs:
            env.select_inds_dict = new_solver.select_inds_r
        for env in new_solver.left_envs:
            env.select_inds_dict = new_solver.select_inds_l

        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        new_solver.canon_direction = self.canon_direction
        return new_solver


    # def _combine_b_effs(self, old_b_eff: Optional['qtn.Tensor'], next_b_eff: 'qtn.Tensor', func=None, inplace=True):
    #     """
    #     combine b_effs so that we can compute f(x) + f(y) + f(z) + ...
    #     :param b_effs:  effective bs
    #     :return: b_eff + f(y)
    #     """
    #     b_eff = old_b_eff if (inplace or old_b_eff is None) else old_b_eff.copy()
    #
    #     if func is not None:
    #         next_b_eff.modify(apply=lambda x: func(x))
    #
    #     if b_eff is None:
    #         b_eff = next_b_eff
    #     else:
    #         next_b_eff.transpose_like(b_eff, inplace=True)
    #         b_eff.modify(apply=lambda x: x + next_b_eff.data)
    #
    #     return b_eff


    # def get_b_eff(self, sites: Union[int, slice]) -> Optional[qtn.Tensor]:
    #     """ d/dT*[s] <x_o|b_j>
    #         sites: int or slice(start, stop, step)
    #     """
    #     site_pos = list(range(self.L))[sites]
    #     site_inds = self.mps_inds[sites]
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     b_eff: Optional['qtn.Tensor'] = None
    #     for ti in range(self.num_targets):
    #         if self.b_envsLs[ti] is None:  continue
    #
    #         b_left = self.b_envsLs[ti][site_pos[0]]
    #         b_right = self.b_envsRs[ti][site_pos[-1]]
    #
    #         # print('b left', b_left)
    #         # print('b right', b_right)
    #
    #         b_eff_t = qtn.TensorNetwork([])
    #         if b_left is not None:
    #             b_eff_t.add(b_left)
    #         if b_right is not None:
    #             b_eff_t.add(b_right)
    #
    #         if isinstance(site_inds, (list, tuple)):
    #             for si in site_inds:
    #                 b_eff_t.add(self.targets[ti][si])
    #         else:
    #             b_eff_t.add(self.targets[ti][site_inds])
    #
    #         # print('b_eff TN', b_eff_t)
    #
    #         b_eff_exponent = self.b_envsLs[ti].exponent + b_eff_t.exponent
    #         b_eff_t = b_eff_t.contract_tags(all)
    #         b_eff_t.modify(apply=lambda x: 10 ** b_eff_exponent * x)
    #
    #         func = self.func[ti] if isinstance(self.func, (list, tuple)) else self.func
    #         b_eff = self._combine_b_effs(b_eff, b_eff_t, func=func, inplace=True)
    #
    #         # if self.func is not None:
    #         #     b_eff_t.modify(apply=lambda x: self.func(x))
    #         #
    #         # if b_eff is None:
    #         #     b_eff = b_eff_t
    #         # else:
    #         #     b_eff_t.transpose_like(b_eff, inplace=True)
    #         #     b_eff.modify(apply=lambda x: x + b_eff_t.data)
    #
    #     return b_eff

    # @profile
    def _site_solve(self, left_site_pos: int, nsites: int, **kwargs) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        # print('site solve', self.solve_type)
        site_pos = range(left_site_pos, left_site_pos + nsites)
        site_inds = [self.mps_inds[ix] for ix in site_pos]
        site_l = left_site_pos
        site_r = left_site_pos + nsites - 1
        inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L-1 else []

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm(), current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        ### this is also the selected submatrix if self is in cross-canonical form
        b_eff = self.get_b_eff(left_site_pos, nsites)

        ## solve for T[i]
        x_eff = b_eff.copy()
        x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))

        ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
        ## but it should be canonical?
        tensors = [x_eff]
        if site_l > 0:
            inv_env_l = self.k_envL[site_l].copy()
            inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
            tensors += [inv_env_l]
        if site_r < self.L-1:
            inv_env_r = self.k_envR[site_r].copy()
            inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
            tensors += [inv_env_r]

        x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
        x_eff.transpose(*(inds_l + [self.ket.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)
        # print('x eff data', x_eff.norm(), x_eff.data)

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm( x_eff.data - current_x.data )

        return x_eff, site_err


    # @profile
    def solve(self, nsites: int, init_direction=SweepDirection.RIGHT, conv_tol=DEFAULT_CONV_TOL,
              verbose=False, **kwargs):
        """ iterative solver for entire MPS
            minimize || Ax-b ||_2 = <Ax|Ax> + <b|b> - <Ax|b> - <b|Ax>
            sweep through sites i:
                d/dT*[i] () = d/dT*[i] <Ax|Ax> - <Ax|b> = 0
                    A_eff = d/dT*[i] d/dT[i] <Ax|Ax>
                    b_eff = d/dT*[i] <Ax|b>
                --> A_eff T[i] = b_eff

            A can be the identity (A is None) --> optimizing || x - b ||_2
            with proper canonicalization of x, A_left, A_right should be identity

            A: self.operators:
                can be composed of a series of MPOs (A1, ..., Am) order from ket to bra
            b: self.targets:
                can be written as a sum of MPSs (scaling included in the MPS; not kept track of separately)
            x: self.ket (updated in place)
        """
        # verbose = True
        conv_tol = self.conv_tol if conv_tol == 0 else conv_tol
        L = self.ket.L
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)
        if canon_site == 0:
            print('check orthog', check_right_orthog(self.ket, self.select_inds_r, [self.ket.site_ind_id]))
        else:
            print('check orthog', check_left_orthog(self.ket, self.select_inds_l, [self.ket.site_ind_id]))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        if self.Z_enrich:
            self.initialize_Z(direction=(SweepDirection.LEFT if canon_site==0 else SweepDirection.RIGHT))

        # ## ovlp <f(x)|b>, <f(x)|f(x)> init envs
        if init_direction == SweepDirection.RIGHT:
            self.build_all_envs_right(nsites, canonize=False)
        else:
            self.build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = init_direction
        while err > conv_tol and conv_it < self.max_iter \
                and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter:

            it += 1
            tot_err = 0.0

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites + 1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i, i + nsites)

                    site_i, site_err = self._site_solve(inds)
                    # print('site i', site_i.norm(), site_err)
                    if site_err > 10**5:
                        print('site error too large')
                        exit()
                    tot_err += site_err

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site_with_Z(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site_with_Z(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    if i < L - 1:
                        self._update_envs_left(i)

                    # print('check L orthog', i)
                    # print(check_left_orthog(self.ket, self.select_inds_l, [self.ket.site_ind_id]))

                self.canon_direction = EnvironmentSide.LEFT
                err = tot_err  # self.check_err(dense=True)
                self.select_inds_r.clear()

            else:
                ## left sweep
                for i in range(L - 1, nsites - 2, -1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i - nsites + 1, i + 1)
                    # print('update', i, inds)

                    # print('update R', inds)
                    site_i, site_err = self._site_solve(inds)
                    # print('site err', i, site_err, site_i.norm())
                    if site_err > 10**5:
                        print('site err too large')
                        exit()
                    tot_err += site_err

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site_with_Z(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site_with_Z(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    if i > 0:
                        self._update_envs_right(i)

                    # print('check R orthog', i)
                    # print(check_right_orthog(self.ket, self.select_inds_r, [self.ket.site_ind_id]))

                self.canon_direction = EnvironmentSide.RIGHT
                err = tot_err  # self.check_err(dense=True)
                self.select_inds_l.clear()

            self.err = err
            self.is_conv = err < CONV_TOL

            if verbose:
                print('err', it, self.max_iter, err)
                # print('err np', self.check_err_np())
            direction *= -1  ## swaps sweep direction

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err:
                min_solver = self.copy(deep=True)
                # min_ket = self.ket.copy()
                min_err = err
                num_wrong_it = 0
            else:
                print('Warning: solve error went up', err, min_err)
                num_wrong_it += 1
                print('num wrong', num_wrong_it, self.max_wrong_iter)

        # conv = min_err < conv_tol

        ## revert to optimal results
        if err > min_err:
            self._ket = min_solver.ket
            self.b_envsLs = min_solver.b_envsLs
            self.b_envsRs = min_solver.b_envsRs
            self.k_envL = min_solver.k_envL
            self.k_envR = min_solver.k_envR
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            self.canon_direction = min_solver.canon_direction

        print('num iter', it)

        return self.ket, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)


    def _update_1site_with_Z(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
            todo: cap total bond dimension (after rank enrichment)
        """
        i = self.mps_inds[i]
        self._update_1site(i, site_i, direction)

        if not self.Z_enrich:
            return

        if direction == SweepDirection.RIGHT and i == self.L - 1:
            return

        if direction == SweepDirection.LEFT and i == 0:
            return

        x_eff = site_i

        ## func(target with Z envs)
        zzb_eff = self.Z_enrich.get_b_eff(slice(i,i+1))  # sum over all targets
        # zzb_eff.modify(apply=lambda x: self.func(x))   ## now included in zzb_eff

        ## func( target with Y (sweep start side) / Z (far side) envs )
        zyb_eff = None
        for ti in range(len(self.targets)):
            tensors = [self.targets[ti][i]]
            if direction == SweepDirection.RIGHT:
                if i < self.L - 1:
                    tensors += [self.Z_enrich.b_envsRs[ti][i]]
                if i > 0:
                    tensors += [self.b_envsLs[ti][i]]
            elif direction == SweepDirection.LEFT:
                if i > 0:
                    tensors += [self.Z_enrich.b_envsLs[ti][i]]
                if i < self.L - 1:
                    tensors += [self.b_envsRs[ti][i]]
            yzb_eff_ = qtn.tensor_contract(*tensors)

            b_eff_exponent = self.b_envsLs[ti].exponent
            yzb_eff_.modify(apply=lambda x: 10 ** b_eff_exponent * x)

            func = self.func[ti] if isinstance(self.func, (list,tuple)) else self.func
            zyb_eff = self._combine_b_effs(zyb_eff, yzb_eff_, func=func)

            # if zyb_eff is None:
            #     zyb_eff = yzb_eff_
            # else:
            #     yzb_eff_.transpose_like(zyb_eff, inplace=True)
            #     zyb_eff.modify(apply=lambda x: x + yzb_eff_.data)
            #
            # zyb_eff.modify(apply=lambda x: self.func(x))

        ## project new core onto Z envs on side not yet reached by sweep side
        ## already in Y envs on the sweep side side (env should be identity)
        zy_eff = x_eff.copy()
        if direction == SweepDirection.RIGHT:
            if i < self.L - 1:
                env_r = self.Z_enrich.k_envR[i]     ## should be iden
                zy_eff = qtn.tensor_contract(zy_eff, env_r)
            if i > 0:
                env_l = self.Z_enrich.k_envL[i]
                zz_eff = qtn.tensor_contract(zy_eff, env_l)
            else:
                zz_eff = zy_eff.copy()
        elif direction == SweepDirection.LEFT:
            if i > 0:
                env_l = self.Z_enrich.k_envL[i]     ## should be iden
                zy_eff = qtn.tensor_contract(zy_eff, env_l)
            if i < self.L - 1:
                env_r = self.Z_enrich.k_envR[i]
                zz_eff = qtn.tensor_contract(zy_eff, env_r)
            else:
                zz_eff = zy_eff.copy()

        ## todo: can add random noise to Z here too by adding columns/rows

        # ket_to_bra_inds = self.get_ket_to_bra_inds(i)
        ket_to_bra_inds = {}
        if i > 0:
            k_ind = self.ket.bond(i, i - 1)
            ket_to_bra_inds[k_ind] = k_ind + '_'
        elif self.ket_env0_L is not None:
            k_ind, b_ind = self.ket[0].filter_bonds(self.ket_env0_L)  ## shared, not shared
            ket_to_bra_inds[k_ind] = b_ind

        if i < self.L - 1:
            k_ind = self.ket.bond(i, i + 1)
            ket_to_bra_inds[k_ind] = k_ind + '_'
        elif self.ket_env0_R is not None:
            k_ind, b_ind = self.ket[-1].filter_bonds(self.ket_env0_R)  ## shared, not shared
            ket_to_bra_inds[k_ind] = b_ind

        zy_eff.reindex(ket_to_bra_inds, inplace=True)

        ## Z is defined as error from self.ket (trial) and func(target)
        zy_eff = zyb_eff - zy_eff
        bra_to_ket_inds = {v: k for k, v in ket_to_bra_inds.items()}
        zy_eff.reindex(bra_to_ket_inds, inplace=True)

        zz_eff = zzb_eff - zz_eff
        zz_eff.reindex(bra_to_ket_inds, inplace=True)

        ## reduce zz_eff, zy_eff via cross interpolation (or svd) ?  (apply interpolation matrices)

        bond_p = self.ket.site_ind_id.format(i)
        u = self.ket[i]  ## freshly canonicalized from normal update_1site

        if direction == SweepDirection.RIGHT:

            v = self.ket[i + 1]
            ## ideally, would be the part before contracting into the next site

            bonds_l = [self.ket.bond(i, i - 1)] if i > 0 else []
            bonds_r, bonds_v = v.filter_bonds(u)
            bond_r = next(iter(bonds_r))

            bonds_u = [bond_p] + bonds_l
            u0_shape = [u.ind_size(bv) for bv in bonds_u]
            v1_shape = [v.ind_size(bv) for bv in bonds_v]

            u = u.fuse({'eff': bonds_u})
            zy_eff = zy_eff.fuse({'eff': bonds_u})
            zz_eff = zz_eff.fuse({'eff': bonds_u})
            v = v.fuse({'eff_v': bonds_v})

            u.transpose('eff', bond_r, inplace=True)
            zy_eff.transpose_like(u, inplace=True)
            zz_eff.transpose_like(u, inplace=True)
            v.transpose(bond_r, 'eff_v', inplace=True)

        elif direction == SweepDirection.LEFT:

            v = self.ket[i - 1]

            bonds_r = [self.ket.bond(i, i + 1)] if i < self.L - 1 else []
            bonds_l, bonds_v = v.filter_bonds(u)
            bond_l = next(iter(bonds_l))

            bonds_u = [bond_p] + bonds_r
            u0_shape = [u.ind_size(bv) for bv in bonds_u]
            v1_shape = [v.ind_size(bv) for bv in bonds_v]

            u = u.fuse({'eff': bonds_u})
            zy_eff = zy_eff.fuse({'eff': bonds_u})
            zz_eff = zz_eff.fuse({'eff': bonds_u})
            v = v.fuse({'eff_v': bonds_v})

            u.transpose('eff', bond_l, inplace=True)
            zy_eff.transpose_like(u, inplace=True)
            zz_eff.transpose_like(u, inplace=True)
            v.transpose(bond_l, 'eff_v', inplace=True)

        else:
            raise ValueError

        u0,u1 = u.shape
        z0,z1 = zy_eff.shape
        v0,v1 = v.shape

        expand_u = np.empty((u0, u1 + z1), dtype=u.data.dtype)
        expand_u[:, :u1] = u.data
        expand_u[:, u1:] = zy_eff.data

        expand_v = np.zeros((u1 + z1, v1), dtype=v.data.dtype)
        expand_v[:v0, :] = v.data

        expand_u, rv = np.linalg.qr(expand_u)
        expand_v = np.dot(rv, expand_v)

        ## canonize expanded u, v
        u.modify(data=expand_u)
        v.modify(data=expand_v)

        u = u.unfuse({'eff': bonds_u}, {'eff': u0_shape})
        v = v.unfuse({'eff_v': bonds_v}, {'eff_v': v1_shape})

        u.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=u.data)

        if direction == SweepDirection.RIGHT:
            v.transpose_like(self.ket[i + 1], inplace=True)
            self.ket[i + 1].modify(data=v.data)
            _, sel_inds = canonize_tens_list([self.ket[i], self.ket[i + 1]], [self.ket.site_ind_id],
                                             [i, i + 1], inplace=True)
        else:
            v.transpose_like(self.ket[i - 1], inplace=True)
            self.ket[i - 1].modify(data=v.data)
            _, sel_inds = canonize_tens_list([self.ket[i], self.ket[i - 1]], [self.ket.site_ind_id],
                                             [i, i - 1], inplace=True)


        ### use zz_eff to get new select_inds for Z
        z_inds = maxvol_inds(zz_eff.data, do_qr=True)  # n x r

        if direction == SweepDirection.RIGHT:
            self.select_inds_l[i] = sel_inds[0]
            self.Z_enrich.select_inds_l[i] = z_inds
            self.Z_enrich._update_envs_left(i)
            ## self envs are updated outside of this function
        elif direction == SweepDirection.LEFT:
            self.select_inds_r[i] = sel_inds[0]
            self.Z_enrich.select_inds_r[i] = z_inds
            self.Z_enrich._update_envs_right(i)
            ## self envs are updated outside of this function

        return


    def _update_2site_with_Z(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        self._update_2site(i, site_i, direction=direction)
        return



class ElemMultCrossSolver(CrossSolver):
    """ computes f(x) * g(y) * h(z) * ...
    """
    def __init__(self, ket_state: 'qtn.MatrixProductState'=None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 mps_inds: Sequence[int] = None,
                 ket_env0_L: 'qtn.Tensor' = None,
                 ket_env0_R: 'qtn.Tensor' = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, enrich_rank=0, func=None):

        # operators = [] if operators is None else list(operators)
        # self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []
        self.k_envL, self.k_envR = None, None

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.target_env0_Ls = target_env0_Ls
        self.target_env0_Rs = target_env0_Rs

        if ket_state is None:
            init_state = qtn.MPS_rand_state(targets[0].L, 2, targets[0].phys_dim())
        else:
            init_state = ket_state

        self.canon_direction = 0
        if not isinstance(func, list):
            self.func = [func] * len(targets)
        else:
            self.func = func

        super().__init__(init_state, targets=targets,
                         mps_inds=mps_inds, virtual=virtual, conv_tol=conv_tol, enrich_rank=enrich_rank, func=self.func,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter)

        # self.k_envL = Environment(self.L, EnvironmentSide.LEFT, self.ket,
        #                           init_env=ket_env0_L,
        #                           select_inds_dict=self.select_inds_l,
        #                           mps_inds=self.mps_inds)
        # self.k_envR = Environment(self.L, EnvironmentSide.RIGHT, self.ket,
        #                           init_env=ket_env0_R,
        #                           select_inds_dict=self.select_inds_r,
        #                           mps_inds=self.mps_inds)
        #
        # for ib in range(self.num_targets):
        #     b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], None,
        #                          init_env=target_env0_Ls[ib] if target_env0_Ls is not None else None,
        #                          select_inds_dict=self.select_inds_l,
        #                          bra_horizontal_bonds=self.k_envL.bra_horizontal_bonds,
        #                          mps_inds=self.mps_inds, )
        #     b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], None,
        #                          init_env=target_env0_Rs[ib] if target_env0_Rs is not None else None,
        #                          select_inds_dict=self.select_inds_r,
        #                          bra_horizontal_bonds=self.k_envR.bra_horizontal_bonds,
        #                          mps_inds=self.mps_inds,)
        #     self.b_envsLs += [b_envL]
        #     self.b_envsRs += [b_envR]


    @property
    def left_envs(self):
        return self.b_envsLs + [self.k_envL]

    @property
    def right_envs(self):
        return self.b_envsRs + [self.k_envR]

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        self.select_inds_r = {}
        self.select_inds_l = {}

        for env in self.left_envs:
            if env is not None:
                env.reinitialize()

        for env in self.right_envs:
            if env is not None:
                env.reinitialize()

        self.err = np.inf
        self.is_conv = False
        return

    def create_like(self, copy=True, new_ket=None, **kwargs):
        ket_env0_L = self.ket_env0_L
        ket_env0_R = self.ket_env0_R
        target_env0_Ls = self.target_env0_Ls
        target_env0_Rs = self.target_env0_Rs

        new_solver = self.__class__((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                    targets=kwargs.get('targets', [t.copy() if copy else t for t in self.targets]),
                                    mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                    ket_env0_L=kwargs.get('ket_env0_L', ket_env0_L),
                                    ket_env0_R=kwargs.get('ket_env0_R', ket_env0_R),
                                    target_env0_Ls=kwargs.get('target_env0_Ls', target_env0_Ls),
                                    target_env0_Rs=kwargs.get('target_env0_Rs', target_env0_Rs),
                                    conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                    max_iter=kwargs.get('max_iter', self.max_iter),
                                    max_bond=kwargs.get('max_bond', self.max_bond),
                                    max_tot_iter=kwargs.get('max_tot_iter', self.max_tot_iter),
                                    func=kwargs.get('func', self.func),
                                    virtual=kwargs.get('virtual', False)
                                    )
        return new_solver

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)

        new_solver.select_inds_r = self.select_inds_r.copy()
        new_solver.select_inds_l = self.select_inds_l.copy()

        new_solver.b_envsLs = [bL.copy() if bL is not None else None for bL in self.b_envsLs]
        new_solver.b_envsRs = [bR.copy() if bR is not None else None for bR in self.b_envsRs]
        new_solver.k_envL = self.k_envL.copy()
        new_solver.k_envR = self.k_envR.copy()
        for env in new_solver.right_envs:
            env.select_inds_dict = new_solver.select_inds_r
        for env in new_solver.left_envs:
            env.select_inds_dict = new_solver.select_inds_l

        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        new_solver.canon_direction = self.canon_direction
        return new_solver

    def _combine_b_effs(self, old_b_eff: 'qtn.Tensor', next_b_eff: 'qtn.Tensor', func=None, inplace=True):
        """
        combine b_effs so that we can compute f(x) * g(y) * h(z) + ...
        :param b_effs:  effective bs
        :return: b_eff * f(y)
        """
        b_eff = old_b_eff if inplace else old_b_eff.copy()

        if func is not None:
            next_b_eff.modify(apply=lambda x: func(x))

        if b_eff is None:
            b_eff = next_b_eff
        else:
            next_b_eff.transpose_like(b_eff, inplace=True)
            b_eff.modify(apply=lambda x: x * next_b_eff.data)

        return b_eff

    # def get_b_eff(self, sites: Union[int, slice]) -> Optional[qtn.Tensor]:
    #     """ d/dT*[s] <x_o|b_j>
    #         sites: int or slice(start, stop, step)
    #     """
    #     site_pos = list(range(self.L))[sites]
    #     site_inds = self.mps_inds[sites]
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     b_eff: Optional['qtn.Tensor'] = None
    #     for ti in range(self.num_targets):
    #         if self.b_envsLs[ti] is None:  continue
    #
    #         target_func = self.funcs[ti]
    #
    #         b_left = self.b_envsLs[ti][site_pos[0]]
    #         b_right = self.b_envsRs[ti][site_pos[-1]]
    #
    #         # print('b left', b_left)
    #         # print('b right', b_right)
    #
    #         b_eff_t = qtn.TensorNetwork([])
    #         if b_left is not None:
    #             b_eff_t.add(b_left)
    #         if b_right is not None:
    #             b_eff_t.add(b_right)
    #
    #         if isinstance(site_inds, (list, tuple)):
    #             for si in site_inds:
    #                 b_eff_t.add(self.targets[ti][si])
    #         else:
    #             b_eff_t.add(self.targets[ti][site_inds])
    #
    #         # print('b_eff TN', b_eff_t)
    #
    #         b_eff_exponent = self.b_envsLs[ti].exponent + b_eff_t.exponent
    #         b_eff_t = b_eff_t.contract_tags(all)
    #         b_eff_t.modify(apply=lambda x: 10 ** b_eff_exponent * x)
    #
    #         if target_func is not None:
    #             b_eff_t.modify(apply=lambda x: target_func(x))
    #
    #         if b_eff is None:
    #             b_eff = b_eff_t
    #         else:
    #             b_eff_t.transpose_like(b_eff, inplace=True)
    #             b_eff.modify(apply=lambda x: x * b_eff_t.data)
    #
    #     return b_eff

    # # @profile
    # def _site_solve(self, sites: Union[int, slice]) -> tuple[qtn.Tensor, Numeric]:
    #     """
    #     sites: int or slice(start, stop, step)
    #     """
    #     # print('site solve', self.solve_type)
    #     site_pos = list(range(self.L))[sites]
    #     site_inds = self.mps_inds[sites]
    #     site_l = np.min(site_inds)
    #     site_r = np.max(site_inds)
    #     inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
    #     inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L-1 else []
    #
    #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
    #     current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
    #     # print('current_x', self.ket.exponent, current_x.norm(), current_x.data)
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     ### this is also the selected submatrix if self is in cross-canonical form
    #     b_eff = self.get_b_eff(sites)
    #
    #     ## change indices from bra (missing T*[i]) to ket
    #     # bra_to_ket_inds = {}  # self.get_bra_to_ket_inds(site_inds[0])
    #     # for site_p in site_pos:
    #     #     bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
    #
    #     ## solve for T[i]
    #     # x_eff = b_eff.reindex(bra_to_ket_inds)   ### ket is not orthogonal
    #     x_eff = b_eff.copy()
    #     # if self.func is not None:
    #     #     x_eff.modify(apply=lambda x: self.func(x))
    #     x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
    #     ## func(x) and b_eff_exponent already in b_eff tensor
    #
    #     ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
    #     ## but it should be canonical?
    #     tensors = [x_eff]
    #     if site_l > 0:
    #         inv_env_l = self.k_envL[site_l].copy()
    #         inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
    #         tensors += [inv_env_l]
    #     if site_r < self.L-1:
    #         inv_env_r = self.k_envR[site_r].copy()
    #         inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
    #         tensors += [inv_env_r]
    #
    #     x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
    #     x_eff.transpose(*(inds_l + [self.ket.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)
    #     # print('x eff data', x_eff.norm(), x_eff.data)
    #
    #     current_x.transpose_like(x_eff, inplace=True)
    #     site_err = np.linalg.norm( x_eff.data - current_x.data )
    #
    #     return x_eff, site_err


    # # @profile
    # def solve(self, nsites: int, init_direction=SweepDirection.RIGHT, conv_tol=DEFAULT_CONV_TOL,
    #           verbose=False, **kwargs):
    #     """ iterative solver for entire MPS
    #         minimize || Ax-b ||_2 = <Ax|Ax> + <b|b> - <Ax|b> - <b|Ax>
    #         sweep through sites i:
    #             d/dT*[i] () = d/dT*[i] <Ax|Ax> - <Ax|b> = 0
    #                 A_eff = d/dT*[i] d/dT[i] <Ax|Ax>
    #                 b_eff = d/dT*[i] <Ax|b>
    #             --> A_eff T[i] = b_eff
    #
    #         A can be the identity (A is None) --> optimizing || x - b ||_2
    #         with proper canonicalization of x, A_left, A_right should be identity
    #
    #         A: self.operators:
    #             can be composed of a series of MPOs (A1, ..., Am) order from ket to bra
    #         b: self.targets:
    #             can be written as a sum of MPSs (scaling included in the MPS; not kept track of separately)
    #         x: self.ket (updated in place)
    #     """
    #     # verbose = True
    #     conv_tol = self.conv_tol if conv_tol == 0 else conv_tol
    #     L = self.ket.L
    #     canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
    #     self.canonize(canon_site)
    #     if canon_site == 0:
    #         print('check orthog', check_right_orthog(self.ket, self.select_inds_r, [self.ket.site_ind_id]))
    #     else:
    #         print('check orthog', check_left_orthog(self.ket, self.select_inds_l, [self.ket.site_ind_id]))
    #
    #     # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])
    #
    #     if self.Z_enrich:
    #         self.initialize_Z(direction=(SweepDirection.LEFT if canon_site==0 else SweepDirection.RIGHT))
    #
    #     # ## ovlp <f(x)|b>, <f(x)|f(x)> init envs
    #     if init_direction == SweepDirection.RIGHT:
    #         self.build_all_envs_right(nsites, canonize=False)
    #     else:
    #         self.build_all_envs_left(nsites, canonize=False)
    #
    #     ## iterative solver
    #     it = 0
    #     err = self.err
    #
    #     conv_it, prev_err = 0, err
    #     num_wrong_it = 0
    #
    #     # min_ket, min_err = self.ket, err
    #     min_solver, min_err = self.copy(), err
    #     direction = init_direction
    #     while err > conv_tol and conv_it < self.max_iter \
    #             and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter:
    #
    #         it += 1
    #         tot_err = 0.0
    #
    #         ## right sweep
    #         if direction == SweepDirection.RIGHT:
    #
    #             for i in range(L - nsites + 1):
    #                 ix = self.get_mps_ind(i)
    #                 inds = slice(i, i + nsites)
    #
    #                 site_i, site_err = self._site_solve(inds)
    #                 if site_err > 10**5:
    #                     print('site error too large')
    #                     exit()
    #                 tot_err += site_err
    #
    #                 ## update ket (+ canonicalization)
    #                 if nsites == 1:
    #                     self._update_1site_with_Z(i, site_i, direction)
    #                 elif nsites == 2:
    #                     self._update_2site_with_Z(i, site_i, direction)
    #                 else:
    #                     raise NotImplementedError
    #
    #                 ## update A, b envs (left)
    #                 if i < L - 1:
    #                     self._update_envs_left(i)
    #
    #                 # print('check L orthog', i)
    #                 # print(check_left_orthog(self.ket, self.select_inds_l, [self.ket.site_ind_id]))
    #
    #             self.canon_direction = EnvironmentSide.LEFT
    #             err = tot_err  # self.check_err(dense=True)
    #             self.select_inds_r.clear()
    #
    #         else:
    #             ## left sweep
    #             for i in range(L - 1, nsites - 2, -1):
    #                 ix = self.get_mps_ind(i)
    #                 inds = slice(i - nsites + 1, i + 1)
    #                 # print('update', i, inds)
    #
    #                 # print('update R', inds)
    #                 site_i, site_err = self._site_solve(inds)
    #                 # print('site err', i, site_err, site_i.norm())
    #                 if site_err > 10**5:
    #                     print('site err too large')
    #                     exit()
    #                 tot_err += site_err
    #
    #                 ## update ket, bra
    #                 if nsites == 1:
    #                     self._update_1site_with_Z(i, site_i, direction)
    #                 elif nsites == 2:
    #                     self._update_2site_with_Z(i, site_i, direction)
    #                 else:
    #                     raise NotImplementedError
    #
    #                 ## update A, b envs (right)
    #                 if i > 0:
    #                     self._update_envs_right(i)
    #
    #                 # print('check R orthog', i)
    #                 # print(check_right_orthog(self.ket, self.select_inds_r, [self.ket.site_ind_id]))
    #
    #             self.canon_direction = EnvironmentSide.RIGHT
    #             err = tot_err  # self.check_err(dense=True)
    #             self.select_inds_l.clear()
    #
    #         self.err = err
    #         self.is_conv = err < CONV_TOL
    #
    #         if verbose:
    #             print('err', it, self.max_iter, err)
    #             # print('err np', self.check_err_np())
    #         direction *= -1  ## swaps sweep direction
    #
    #         if np.abs((prev_err - err) / err) < 1.0e-4:
    #             conv_it += 1
    #
    #         prev_err = err
    #
    #         if err < min_err:
    #             min_solver = self.copy(deep=True)
    #             # min_ket = self.ket.copy()
    #             min_err = err
    #             num_wrong_it = 0
    #         else:
    #             print('Warning: solve error went up', err, min_err)
    #             num_wrong_it += 1
    #             print('num wrong', num_wrong_it, self.max_wrong_iter)
    #
    #     # conv = min_err < conv_tol
    #
    #     ## revert to optimal results
    #     if err > min_err:
    #         self._ket = min_solver.ket
    #         self.b_envsLs = min_solver.b_envsLs
    #         self.b_envsRs = min_solver.b_envsRs
    #         self.k_envL = min_solver.k_envL
    #         self.k_envR = min_solver.k_envR
    #         self.err = min_solver.err
    #         self.is_conv = min_solver.is_conv
    #         self.canon_direction = min_solver.canon_direction
    #
    #     return self.ket, self.err, self.is_conv


    # def _update_1site_with_Z(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: int of mps site
    #         todo: cap total bond dimension (after rank enrichment)
    #     """
    #     i = self.mps_inds[i]
    #     self._update_1site(i, site_i, direction)
    #
    #     if not self.Z_enrich:
    #         return
    #
    #     if direction == SweepDirection.RIGHT and i == self.L - 1:
    #         return
    #
    #     if direction == SweepDirection.LEFT and i == 0:
    #         return
    #
    #     x_eff = site_i
    #
    #     ## func(target with Z envs)
    #     zzb_eff = self.Z_enrich.get_b_eff(slice(i,i+1))  # sum over all targets
    #     # zzb_eff.modify(apply=lambda x: self.func(x))
    #
    #     ## func( target with Y (sweep start side) / Z (far side) envs )
    #     zyb_eff = None
    #     for ti in range(len(self.targets)):
    #         tensors = [self.targets[ti][i]]
    #         if direction == SweepDirection.RIGHT:
    #             if i < self.L - 1:
    #                 tensors += [self.Z_enrich.b_envsRs[ti][i]]
    #             if i > 0:
    #                 tensors += [self.b_envsLs[ti][i]]
    #         elif direction == SweepDirection.LEFT:
    #             if i > 0:
    #                 tensors += [self.Z_enrich.b_envsLs[ti][i]]
    #             if i < self.L - 1:
    #                 tensors += [self.b_envsRs[ti][i]]
    #         yzb_eff_ = qtn.tensor_contract(*tensors)
    #
    #         b_eff_exponent = self.b_envsLs[ti].exponent
    #         yzb_eff_.modify(apply=lambda x: 10 ** b_eff_exponent * x)
    #
    #         if zyb_eff is None:
    #             zyb_eff = yzb_eff_
    #         else:
    #             yzb_eff_.transpose_like(zyb_eff, inplace=True)
    #             zyb_eff.modify(apply=lambda x: x + yzb_eff_.data)
    #
    #         zyb_eff.modify(apply=lambda x: self.func(x))
    #
    #     ## project new core onto Z envs on side not yet reached by sweep side
    #     ## already in Y envs on the sweep side side (env should be identity)
    #     zy_eff = x_eff.copy()
    #     if direction == SweepDirection.RIGHT:
    #         if i < self.L - 1:
    #             env_r = self.Z_enrich.k_envR[i]     ## should be iden
    #             zy_eff = qtn.tensor_contract(zy_eff, env_r)
    #         if i > 0:
    #             env_l = self.Z_enrich.k_envL[i]
    #             zz_eff = qtn.tensor_contract(zy_eff, env_l)
    #         else:
    #             zz_eff = zy_eff.copy()
    #     elif direction == SweepDirection.LEFT:
    #         if i > 0:
    #             env_l = self.Z_enrich.k_envL[i]     ## should be iden
    #             zy_eff = qtn.tensor_contract(zy_eff, env_l)
    #         if i < self.L - 1:
    #             env_r = self.Z_enrich.k_envR[i]
    #             zz_eff = qtn.tensor_contract(zy_eff, env_r)
    #         else:
    #             zz_eff = zy_eff.copy()
    #
    #     ## todo: can add random noise to Z here too by adding columns/rows
    #
    #     # ket_to_bra_inds = self.get_ket_to_bra_inds(i)
    #     ket_to_bra_inds = {}
    #     if i > 0:
    #         k_ind = self.ket.bond(i, i - 1)
    #         ket_to_bra_inds[k_ind] = k_ind + '_'
    #     elif self.ket_env0_L is not None:
    #         k_ind, b_ind = self.ket[0].filter_bonds(self.ket_env0_L)  ## shared, not shared
    #         ket_to_bra_inds[k_ind] = b_ind
    #
    #     if i < self.L - 1:
    #         k_ind = self.ket.bond(i, i + 1)
    #         ket_to_bra_inds[k_ind] = k_ind + '_'
    #     elif self.ket_env0_R is not None:
    #         k_ind, b_ind = self.ket[-1].filter_bonds(self.ket_env0_R)  ## shared, not shared
    #         ket_to_bra_inds[k_ind] = b_ind
    #
    #     zy_eff.reindex(ket_to_bra_inds, inplace=True)
    #
    #     ## Z is defined as error from self.ket (trial) and func(target)
    #     zy_eff = zyb_eff - zy_eff
    #     bra_to_ket_inds = {v: k for k, v in ket_to_bra_inds.items()}
    #     zy_eff.reindex(bra_to_ket_inds, inplace=True)
    #
    #     zz_eff = zzb_eff - zz_eff
    #     zz_eff.reindex(bra_to_ket_inds, inplace=True)
    #
    #     ## reduce zz_eff, zy_eff via cross interpolation (or svd) ?  (apply interpolation matrices)
    #
    #     bond_p = self.ket.site_ind_id.format(i)
    #     u = self.ket[i]  ## freshly canonicalized from normal update_1site
    #
    #     if direction == SweepDirection.RIGHT:
    #
    #         v = self.ket[i + 1]
    #         ## ideally, would be the part before contracting into the next site
    #
    #         bonds_l = [self.ket.bond(i, i - 1)] if i > 0 else []
    #         bonds_r, bonds_v = v.filter_bonds(u)
    #         bond_r = next(iter(bonds_r))
    #
    #         bonds_u = [bond_p] + bonds_l
    #         u0_shape = [u.ind_size(bv) for bv in bonds_u]
    #         v1_shape = [v.ind_size(bv) for bv in bonds_v]
    #
    #         u = u.fuse({'eff': bonds_u})
    #         zy_eff = zy_eff.fuse({'eff': bonds_u})
    #         zz_eff = zz_eff.fuse({'eff': bonds_u})
    #         v = v.fuse({'eff_v': bonds_v})
    #
    #         u.transpose('eff', bond_r, inplace=True)
    #         zy_eff.transpose_like(u, inplace=True)
    #         zz_eff.transpose_like(u, inplace=True)
    #         v.transpose(bond_r, 'eff_v', inplace=True)
    #
    #     elif direction == SweepDirection.LEFT:
    #
    #         v = self.ket[i - 1]
    #
    #         bonds_r = [self.ket.bond(i, i + 1)] if i < self.L - 1 else []
    #         bonds_l, bonds_v = v.filter_bonds(u)
    #         bond_l = next(iter(bonds_l))
    #
    #         bonds_u = [bond_p] + bonds_r
    #         u0_shape = [u.ind_size(bv) for bv in bonds_u]
    #         v1_shape = [v.ind_size(bv) for bv in bonds_v]
    #
    #         u = u.fuse({'eff': bonds_u})
    #         zy_eff = zy_eff.fuse({'eff': bonds_u})
    #         zz_eff = zz_eff.fuse({'eff': bonds_u})
    #         v = v.fuse({'eff_v': bonds_v})
    #
    #         u.transpose('eff', bond_l, inplace=True)
    #         zy_eff.transpose_like(u, inplace=True)
    #         zz_eff.transpose_like(u, inplace=True)
    #         v.transpose(bond_l, 'eff_v', inplace=True)
    #
    #     else:
    #         raise ValueError
    #
    #     u0,u1 = u.shape
    #     z0,z1 = zy_eff.shape
    #     v0,v1 = v.shape
    #
    #     expand_u = np.empty((u0, u1 + z1), dtype=u.data.dtype)
    #     expand_u[:, :u1] = u.data
    #     expand_u[:, u1:] = zy_eff.data
    #
    #     expand_v = np.zeros((u1 + z1, v1), dtype=v.data.dtype)
    #     expand_v[:v0, :] = v.data
    #
    #     expand_u, rv = np.linalg.qr(expand_u)
    #     expand_v = np.dot(rv, expand_v)
    #
    #     ## canonize expanded u, v
    #     u.modify(data=expand_u)
    #     v.modify(data=expand_v)
    #
    #     u = u.unfuse({'eff': bonds_u}, {'eff': u0_shape})
    #     v = v.unfuse({'eff_v': bonds_v}, {'eff_v': v1_shape})
    #
    #     u.transpose_like(self.ket[i], inplace=True)
    #     self.ket[i].modify(data=u.data)
    #
    #     if direction == SweepDirection.RIGHT:
    #         v.transpose_like(self.ket[i + 1], inplace=True)
    #         self.ket[i + 1].modify(data=v.data)
    #         _, sel_inds = canonize_tens_list([self.ket[i], self.ket[i + 1]], [self.ket.site_ind_id],
    #                                          [i, i + 1], inplace=True)
    #     else:
    #         v.transpose_like(self.ket[i - 1], inplace=True)
    #         self.ket[i - 1].modify(data=v.data)
    #         _, sel_inds = canonize_tens_list([self.ket[i], self.ket[i - 1]], [self.ket.site_ind_id],
    #                                          [i, i - 1], inplace=True)
    #
    #
    #     ### use zz_eff to get new select_inds for Z
    #     z_inds = maxvol_inds(zz_eff.data, do_qr=True)  # n x r
    #
    #     if direction == SweepDirection.RIGHT:
    #         self.select_inds_l[i] = sel_inds[0]
    #         self.Z_enrich.select_inds_l[i] = z_inds
    #         self.Z_enrich._update_envs_left(i)
    #         ## self envs are updated outside of this function
    #     elif direction == SweepDirection.LEFT:
    #         self.select_inds_r[i] = sel_inds[0]
    #         self.Z_enrich.select_inds_r[i] = z_inds
    #         self.Z_enrich._update_envs_right(i)
    #         ## self envs are updated outside of this function
    #
    #     return
    #
    #
    # def _update_2site_with_Z(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: int of mps site
    #     """
    #     self._update_2site(i, site_i, direction=direction)
    #     return
    #
