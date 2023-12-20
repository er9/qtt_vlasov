import numpy as np

from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

from local_solvers.local_solver import LocalSolver, MPS_type, MPO_type
from local_solvers.local_solver import DEFAULT_MAX_TOT_ITER, DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DEFAULT_MAX_WRONG_ITER
from local_solvers.environments_v2 import TargetEnv_DMRG, OperatorEnv_DMRG

"""
General structure:
e.g. LinearSolverDMRG( DMRGSolver, LinearSolver )
DMRGSolver:  DMRG initializiation, Env updates...
LinearSolver: actual solving
"""

class DMRGSolver(LocalSolver):
    def __init__(self, ket_state: Optional['qtn.MatrixProductState'] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 operators_H: Sequence['MPO_type'] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['OperatorEnv_DMRG']] = None,
                 A_envsRs: Optional[Sequence['OperatorEnv_DMRG']] = None,
                 b_envsLs: Optional[Sequence['TargetEnv_DMRG']] = None,
                 b_envsRs: Optional[Sequence['TargetEnv_DMRG']] = None,
                 # normAA_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # normAA_env0_Rs: Sequence['qtn.Tensor'] = None,
                 mps_inds: Optional[tuple[int]] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None):

        super().__init__(ket_state, targets=targets, operators=operators, operators_H=operators_H, bra_state=bra_state,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
                         op_env0_Rs=op_env0_Rs, op_env0_Ls=op_env0_Ls, A_envsRs=A_envsRs, A_envsLs=A_envsLs,
                         target_env0_Rs=target_env0_Rs, target_env0_Ls=target_env0_Ls,
                         b_envsLs=b_envsLs, b_envsRs=b_envsRs,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         max_bond=max_bond)

        # norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        # norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        # self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Ls[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]
        # self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Rs[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]

    @property
    def left_envs(self):
        return self.A_envsLs

    @property
    def right_envs(self):
        return self.A_envsRs

    def check_eigval(self):
        """ also updates self.eigval
        """
        eigval = 0
        for i in range(self.num_operators):
            eigval += helper.expectation_value(self.ket, self.operators[i], self.bra)
        self.eigval = eigval
        return eigval

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.A_envsLs:
            if env is not None:  env.modify(ket=self.ket, bra=self.bra)

        for env in self.A_envsRs:
            if env is not None:  env.modify(ket=self.ket, bra=self.bra)

        self.err = np.inf
        self.is_conv = False

    def create_like(self, copy=True, new_ket=None, **kwargs):
        op_env0_Ls = [A_envL[0] if A_envL is not None else None for A_envL in self.A_envsLs]
        op_env0_Rs = [A_envR[self.L - 1] if A_envR is not None else None for A_envR in self.A_envsRs]
        target_env0_Ls = [b_envL[0] if b_envL is not None else None for b_envL in self.b_envsLs]
        target_env0_Rs = [b_envR[self.L - 1] if b_envR is not None else None for b_envR in self.b_envsRs]

        new_solver = DMRGSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                operators_H=kwargs.get('operators_H',
                                                       [o.copy() if copy else o for o in self.operators_H]),
                                bra_state=kwargs.get('bra_state', self.bra.copy() if copy else self.bra),
                                in_ind=kwargs.get('in_ind', self.in_ind),
                                out_ind=kwargs.get('out_ind', self.out_ind),
                                op_env0_Ls=kwargs.get('op_env0_Ls', op_env0_Ls),
                                op_env0_Rs=kwargs.get('op_env0_Rs', op_env0_Rs),
                                target_env0_Ls=kwargs.get('target_env0_Ls', target_env0_Ls),
                                target_env0_Rs=kwargs.get('target_env0_Rs', target_env0_Rs),
                                # normAA_env0_Ls=kwargs.get('normAA_env0_Ls', self.normAA_env0_Ls),
                                # normAA_env0_Rs=kwargs.get('normAA_env0_Rs', self.normAA_env0_Rs),
                                mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                max_iter=kwargs.get('max_iter', self.max_iter),
                                max_bond=kwargs.get('max_bond', self.max_bond),
                                max_tot_iter=kwargs.get('max_tot_iter', self.max_tot_iter))
        return new_solver

    def copy(self, deep=True):
        new_solver = super().copy(deep=deep)  # self.create_like()
        new_solver.A_envsLs = [env.copy() if deep and env is not None else env for env in self.A_envsLs]
        new_solver.A_envsRs = [env.copy() if deep and env is not None else env for env in self.A_envsRs]
        new_solver.b_envsLs = [env.copy() if deep and env is not None else env for env in self.b_envsLs]
        new_solver.b_envsRs = [env.copy() if deep and env is not None else env for env in self.b_envsRs]
        # new_solver.eigval = self.eigval
        return new_solver



class EigenSolver(LocalSolver):

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 operators: Sequence['MPO_type'],
                 operators_H: Sequence['MPO_type'] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['OperatorEnv']] = None,
                 A_envsRs: Optional[Sequence['OperatorEnv']] = None,
                 normAA_env0_Ls: Sequence['qtn.Tensor'] = None,
                 normAA_env0_Rs: Sequence['qtn.Tensor'] = None,
                 mps_inds: Optional[tuple[int]] = None,
                 method = 'DMRG',
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None):

        super().__init__(ket_state, targets=None, operators=operators, operators_H=operators_H, bra_state=bra_state,
                         in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
                         op_env0_Rs=op_env0_Rs, op_env0_Ls=op_env0_Ls, A_envsRs=A_envsRs, A_envsLs=A_envsLs,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         max_bond=max_bond)



        self.normAA_env0_Ls = [None] * (self.num_operators ** 2) if normAA_env0_Ls is None else normAA_env0_Ls
        self.normAA_env0_Rs = [None] * (self.num_operators ** 2) if normAA_env0_Rs is None else normAA_env0_Rs

        # self.A2_envsRs, self.A2_envsLs = [], []
        # for i, j in np.ndindex(self.num_operators, self.num_operators):
        #     A, AT = self.operators[i], self.operators_H[j]
        #     if isinstance(A, qtn.MatrixProductOperator):
        #         AAT = MatrixProductOperatorTN([A,AT])
        #     else:
        #         AAT = A.apply(AT)
        #
        #     self.A2_envsLs += [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Ls[i])]
        #     self.A2_envsRs += [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Rs[i])]
        #     ## init env only works if MPOs don't have ancillas
        #
        # # self.A2_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Ls[i]) for i in range(self.num_operators)]
        # # self.A2_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Rs[i]) for i in range(self.num_operators)]
        #
        # # self.left_envs = [self.A_envsL, self.A2_envsL]
        # # self.right_envs = [self.A_envsR, self.A2_envsR]

        # self.left_envs = self.A_envsLs # + self.A2_envsLs
        # self.right_envs = self.A_envsRs # + self.A2_envsRs

        ### initial results
        self.eigval = 0
        # self.update_results()


    def _site_solve(self, left_site_pos: int, nsites: int, **kwargs) -> qtn.Tensor:
        """
        sites: int or slice(start, stop, step)
        """
        # print('site solve', self.solve_type)
        site_pos = list(range(left_site_pos, left_site_pos + nsites))
        site_inds = [self.mps_inds[ix] for ix in site_pos]

        ### BL * B * BR = d/dT*[i] <x|b>
        b_eff = self.get_b_eff(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        bra_to_ket_inds = {}  # self.get_bra_to_ket_inds(site_inds[0])
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        # print('self.ket', self.ket)
        # print('bra to ket inds', bra_to_ket_inds)
        bra_inds = b_eff.inds if b_eff is not None else []
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        ## solve for T[i]
        if self.num_operators == 0:
            x_eff = b_eff.reindex(bra_to_ket_inds)
            x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type is SolveMethod.CGD:
                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [self.ket[si] for si in site_inds]
                else:
                    ket_tensors = self.ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*ket_inds, inplace=True)

                ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
                A_effs: Sequence['qtn.TensorNetwork'] = self.get_A_effs(left_site_pos, nsites, return_combined=False)

                # x_tens.modify(apply=lambda x: x * 0.)
                if self.is_H:
                    x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
                else:
                    x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
                # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))

            else:

                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [self.ket[si] for si in site_inds]
                else:
                    ket_tensors = self.ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*ket_inds, inplace=True)

                ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
                A_eff: 'qtn.Tensor' = self.get_A_effs(left_site_pos, nsites, return_combined=True)
                A_eff.transpose(*bra_inds, *ket_inds, inplace=True)

                # A_eff_exponent = A_eff.exponent
                out_shape = A_eff.shape[:len(bra_inds)]
                in_shape = A_eff.shape[len(bra_inds):]
                A_eff_data = A_eff.data.reshape(np.prod(out_shape), -1)

                if b_eff is not None:
                    b_eff.transpose(*bra_inds, inplace=True)
                    b_eff_data = b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                if self.solve_type is SolveMethod.CGDx:
                    if self.is_H:
                        x_eff_data, error = conjugate_gradient_descent(A_eff_data, b_eff_data,
                                                                       x=x_tens.data.reshape(-1))
                    else:
                        x_eff_data, error = conjugate_gradient_squared(A_eff_data, b_eff_data,
                                                                       x=x_tens.data.reshape(-1))

                else:  # if self.solve_type is SolveMethod.LSQ:
                    x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
                    # x_eff_data = x_eff_data.reshape(in_shape)

                x_eff_data = x_eff_data.reshape(in_shape)
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

        return x_eff


