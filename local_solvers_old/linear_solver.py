import numpy as np
from abc import ABC

from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

from local_solvers_old.local_solver import LocalSolver, MPS_type, MPO_type, SweepDirection
from local_solvers_old.local_solver import DEFAULT_MAX_TOT_ITER, DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DEFAULT_MAX_WRONG_ITER
from local_solvers_old.environments_v2 import TargetEnv_DMRG, OperatorEnv_DMRG

"""
General structure:
e.g. LinearSolverDMRG( DMRGSolver, LinearSolver )
DMRGSolver:  DMRG initializiation, Env updates...
LinearSolver: actual solving
"""

class SolveMethod(Enum):
    CGD = 'CGD'  # local CGD using quimb tensors
    CGDx = 'CGDx'  # local CGD using numpy
    LSQ = 'LSQ'  # least squares regression




class LinearSolver(LocalSolver, ABC):

    solve_type: SolveMethod = SolveMethod.CGD
    is_H: bool = False

    # def __init__(self, ket_state: Optional['qtn.MatrixProductState'] = None,
    #              targets: Optional[Sequence['MPS_type']] = None,
    #              operators: Optional[Sequence['MPO_type']] = None,
    #              operators_H: Sequence['MPO_type'] = None,
    #              bra_state: Optional['qtn.MatrixProductState'] = None,
    #              in_ind: int = 0, out_ind: int = 0,
    #              op_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              op_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              target_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              target_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              A_envsLs: Optional[Sequence['OperatorEnv_DMRG']] = None,
    #              A_envsRs: Optional[Sequence['OperatorEnv_DMRG']] = None,
    #              b_envsLs: Optional[Sequence['TargetEnv_DMRG']] = None,
    #              b_envsRs: Optional[Sequence['TargetEnv_DMRG']] = None,
    #              # normAA_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              # normAA_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              mps_inds: Optional[tuple[int]] = None,
    #              conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
    #              max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None):
    #
    #     super().__init__(ket_state, targets=targets, operators=operators, operators_H=operators_H, bra_state=bra_state,
    #                      in_ind=in_ind, out_ind=out_ind, mps_inds=mps_inds,
    #                      op_env0_Rs=op_env0_Rs, op_env0_Ls=op_env0_Ls, A_envsRs=A_envsRs, A_envsLs=A_envsLs,
    #                      target_env0_Rs=target_env0_Rs, target_env0_Ls=target_env0_Ls,
    #                      b_envsLs=b_envsLs, b_envsRs=b_envsRs,
    #                      conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
    #                      max_bond=max_bond)

        # norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        # norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        # self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Ls[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]
        # self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Rs[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]


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

    def check_err_np(self):
        """ check error of Ax=b using numpy
        """
        if self.num_operators > 0:
            A = np.zeros((2 ** self.L, 2 ** self.L))
            for op in self.operators:
                if op is None:  continue
                op_tens = op.contract(all) * 10 ** op.exponent
                op_tens.transpose(*[op.upper_ind_id.format(i) for i in self.mps_inds],
                                  *[op.lower_ind_id.format(i) for i in self.mps_inds],
                                  inplace=True)
                op_mat = op_tens.data.reshape(2 ** self.L, 2 ** self.L)
                # A = np.dot(op_mat, A)
                A += op_mat
        else:
            A = np.eye(2 ** self.L)

        b = np.zeros(2 ** self.L)
        for target in self.targets:
            target_tens = target.contract(all) * 10 ** target.exponent
            # print('target tens?', target_tens)
            target_tens.transpose(*[target.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
            target_vec = target_tens.data.reshape(-1)
            b = b + target_vec

        x_tens = self.ket.contract(all) * 10 ** self.ket.exponent
        x_tens.transpose(*[self.ket.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
        x_vec = x_tens.data.reshape(-1)

        # Ax = np.dot(A, x_vec)
        # print('<xA|Ax>', np.dot(Ax, Ax), '<b|Ax>', np.dot(b, Ax), '<b|b>', self.targets_norm**2)

        err = np.linalg.norm(np.dot(A, x_vec) - b)
        return err / self.targets_norm

    def check_err_exact(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
        """
        if dense:

            ## combine err envs
            ### L, R:  <xA|Ax>, <b|Ax>, <b|b> ancilla envs; ordered by bra_ind, ket_ind
            def combine_err_envs(err_envs: Sequence['qtn.Tensor']):
                xAAx, bAx, bb = err_envs
                bb = bb.reindex({bb.inds[i]: xAAx.inds[i] for i in range(len(bb.inds))})
                tot_env = qtn.tensor_direct_product(xAAx, bb, inplace=False)

                b_size, Ax_size = bb.shape[0], xAAx.shape[0]
                tot_size = tot_env.shape[0]
                bAx = bAx.copy()
                for ind in bAx.inds:
                    bAx.expand_ind(ind, tot_size)

                    if ind == 0:  ## move b inds to correct position
                        reorder_tens = np.zeros(tot_size)
                        reorder_tens[-b_size - 1:, :b_size] = np.eye(b_size)
                        bAx.modify(apply=lambda x: np.tensordot(reorder_tens, x, axes=1), inplace=True)
                tot_env.modify(apply=lambda x: x + bAx.data)
                return tot_env

            err_env_L = None if self.err_env0_Ls is None else combine_err_envs(self.err_env0_Ls)
            err_env_R = None if self.err_env0_Rs is None else combine_err_envs(self.err_env0_Rs)

            ## sum |Ax>
            if self.num_operators > 0:
                ket_copy = self.ket.copy()
                Ax = None
                for i in range(self.num_operators):
                    if self.operators[i] is None:  continue
                    Ax_i = helper.apply(self.operators[i], ket_copy)
                    if Ax is None:
                        Ax = Ax_i
                    else:
                        Ax = helper.add_MPS(Ax, Ax_i, inplace=True)
            else:
                Ax = self.ket.copy()

            ## sum |b>
            total_b = self.targets[0].copy()
            # print('total b', total_b)
            for ib in range(1, self.num_targets):
                total_b = helper.add_MPS(total_b, self.targets[ib], inplace=True)

            Ax.site_ind_id = total_b.site_ind_id
            ## method=overlap method generally does not give as accurate results
            if err_env_L is None and err_env_R is None:
                err = helper.distance(Ax, total_b, method='auto') / self.targets_norm
            else:
                helper.match_inner_inds(total_b, Ax, inplace=True)
                total_b.site_ind_id = Ax.site_ind_id
                Ax.distribute_exponent()
                total_b.distribute_exponent()

                tens_list = []
                if err_env_L is not None:
                    tens_list += [err_env_L]
                    shared_, ancL_b = total_b[0].filter_bonds(Ax[0])
                    shared_, ancL_Ax = Ax[0].filter_bonds(total_b[0])
                    total_b[0].reindex({anc1: anc2 for anc1, anc2 in zip(ancL_b, ancL_Ax)}, inplace=True)

                if err_env_R is not None:
                    tens_list += [err_env_R]
                    shared_, ancR_b = total_b[-1].filter_bonds(Ax[-1])
                    shared_, ancR_Ax = Ax[-1].filter_bonds(total_b[-1])
                    total_b[-1].reindex({anc1: anc2 for anc1, anc2 in zip(ancR_b, ancR_Ax)}, inplace=True)

                diff = helper.add_MPS(Ax, total_b.scalar_multiply(-1, inplace=True))
                diff_conj = diff.conj()
                if err_env_L is not None:
                    diff_conj[0].reindex({anc1: anc1 + '_' for anc1 in ancL_Ax}, inplace=True)
                if err_env_L is not None:
                    diff_conj[-1].reindex({anc1: anc1 + '_' for anc1 in ancR_Ax}, inplace=True)

                tens_list += [diff, diff_conj]
                err = qtn.tensor_contract(tens_list)

        else:  ## contract tags doesn't give as small errors as helper.distance()

            old_site_ind_id = self.bra.site_ind_id

            if self.num_operators > 0:
                self.bra.site_ind_id = self.operators_H[-1].upper_ind_id

            ## <xA|Ax>
            if self.num_operators > 0:
                exp_AA_tot = 0.0
                ket_copy = self.ket.copy()
                bra_copy = self.bra.copy()
                for i, j in np.ndindex(self.num_operators, self.num_operators):
                    if self.operators[i] is None or self.operators_H[j] is None:  continue
                    # norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
                    #                             [mpo for mpo in self.operators_H], virtual=True)
                    norm_Ax = qtn.TensorNetwork([ket_copy, bra_copy, self.operators[i], self.operators_H[j]],
                                                virtual=False)  ## can mangle indices
                    A_eff_exponent = norm_Ax.exponent
                    # print('norm Ax', norm_Ax)
                    exp_AA = norm_Ax.contract_tags(all)
                    exp_AA *= 10 ** A_eff_exponent
                    exp_AA_tot += exp_AA
                norm_Ax = exp_AA_tot
            else:
                norm_Ax = helper.norm(self.ket) ** 2

            ## <xA|b>
            ovlp_b_tot = 0.
            if self.num_operators > 0:
                bra_copy = self.bra.copy()
                for i, ib in np.ndindex(self.num_operators, self.num_targets):
                    if self.operators_H[i] is None:   continue
                    ovlp_b = qtn.TensorNetwork([self.targets[ib], bra_copy, self.operators_H[i].conj()], virtual=False)
                    # [mpo for mpo in self.operators_H], virtual=True)
                    # [helper.mpo_transpose(mpo) for mpo in self.operators_H], virtual=True)

                    # b_eff_exponent = ovlp_b.exponent
                    b_eff_exponent = self.operators_H[i].exponent + self.bra.exponent + self.targets[ib].exponent
                    ovlp_b = ovlp_b.contract_tags(all)
                    ovlp_b_tot = ovlp_b_tot + ovlp_b * 10 ** b_eff_exponent
            else:
                for ib in range(self.num_targets):
                    ovlp_b = qtn.TensorNetwork([self.targets[ib], self.bra], virtual=False)
                    b_eff_exponent = self.bra.exponent + self.targets[ib].exponent
                    ovlp_b = ovlp_b.contract_tags(all)
                    ovlp_b_tot = ovlp_b_tot + ovlp_b * 10 ** b_eff_exponent

            self.bra.site_ind_id = old_site_ind_id

            err2 = (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot))
            err2 = np.abs(err2)
            err = np.sqrt(err2) / self.targets_norm

        self.err = err
        self.is_conv = err < self.conv_tol

        return err

    def check_err_approx(self) -> float:
        """ calculates <y|Ax-b> where y is a random MPS
            assume no ancilla bonds
        """
        y = qtn.MPS_rand_state(self.ket.L, self.max_bond)


        tot_yb_ovlp = 0.0
        for target in self.targets:
            yb_ovlp = helper.ovlp(y, target)
            tot_yb_ovlp = tot_yb_ovlp + yb_ovlp

        tot_yAx_ovlp = 0.0
        for operator in self.operators:
            ovlp = helper.expectation_value(self.ket, operator, y)
            tot_yAx_ovlp = tot_yAx_ovlp + ovlp

        err = np.abs(tot_yAx_ovlp - tot_yb_ovlp)
        return err


################################
### numpy optimzation method ###
################################

def conjugate_gradient_descent(A, b, x=None, max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False):
    """ solve Ax = b via conjugate gradient descent
        using normal matrices and vectors
        assumes A is psd and symmetric
    """
    norm_b = np.linalg.norm(b)
    if x is None:
        x, r = 0, b
    else:
        r = b - np.dot(A, x)
    d = r

    it = 0
    err = np.linalg.norm(r)
    min_x, min_err = x, err
    while it < max_iter and err > conv_tol * norm_b:
        # x_old = x
        alpha = np.linalg.norm(r) ** 2 / np.dot(d.conj(), np.dot(A, d))

        x_ = x + alpha * d
        r_ = r - alpha * np.dot(A, d)
        beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        d_ = r_ + beta_ * d

        x, r, d, = x_, r_, d_
        err = np.linalg.norm(r)
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

        it += 1

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')

    return x, err / norm_b


def conjugate_gradient_squared(A, b, x=None, max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False):
    """ solve Ax = b via conjugate gradient squared algorithm
        using numpy matrices and vectors
        assumes A has positive eigenvalues but is not necessarily symmetric
    """
    norm_b = np.linalg.norm(b)
    if x is None:
        x, r = 0, b
        rT0 = b
    else:
        r = b - np.dot(A, x)
        rT0 = b - np.dot(x, A)

    it = 0
    q, p, rho = 0, 0, 1
    err = np.linalg.norm(r)
    min_x, min_err = x, err
    while it < max_iter and err > conv_tol * norm_b:

        rho_ = np.dot(rT0, r)
        beta = rho_ / rho

        u = r + beta * q
        p_ = u + beta * (q + beta * p)
        v = np.dot(A, p_)

        sigma = np.dot(rT0, v)
        alpha = rho_ / sigma

        q_ = u - alpha * v
        r_ = r - alpha * np.dot(A, u + q_)
        x_ = x + alpha * (u + q_)

        p, q, r, x = p_, q_, r_, x_
        rho = rho_
        it += 1

        err = np.linalg.norm(r)
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')

    return x, err / norm_b

########################
### helper functions ###
########################
def _contract_x_with_As(As: Sequence[qtn.TensorNetwork], *x_tens: qtn.Tensor, out_inds, scale=True, precomp_Ax=None):
    Ax_tot = None
    for A in As:
        # if A is None:  ## identity; need to reindex inds though
        #     Ax = qtn.TensorNetwork(x_tens, virtual=True, check_collisions=False)
        if precomp_Ax is None:  ## A with first x_tens
            Ax = qtn.TensorNetwork([A, *x_tens], virtual=True, check_collisions=False)
        else:
            Ax = qtn.TensorNetwork([precomp_Ax, *x_tens[1:]], virtual=True, check_collisions=False)
            ## Ax exponent = 0.0
        out = Ax.contract_tags(all)

        if isinstance(out, qtn.Tensor):
            # out.transpose_like(r, inplace=True)
            out.transpose(*out_inds, inplace=True)
            out.modify(apply=lambda data: data * 10 ** Ax.exponent)
            if Ax_tot is None:
                Ax_tot = out
            else:
                Ax_tot.modify(apply=lambda data: data + out.data)
        else:
            if scale:
                out *= 10 ** Ax.exponent
            if Ax_tot is None:
                Ax_tot = out
            else:
                Ax_tot += out
    return Ax_tot


# @profile
def qtn_conjugate_gradient_descent_1site(As: Sequence[qtn.TensorNetwork], b: qtn.Tensor, x: qtn.Tensor,
                                         output_to_input_inds: dict[str, str],
                                         max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False,
                                         contract_Ax=_contract_x_with_As,
                                         contract_xAx=_contract_x_with_As) \
        -> tuple[qtn.Tensor, float]:
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
        A: matrix, b: target, x: solution
        r: residual = b - Ax
        d: search direction
        contract_with_As:  function with arguments:
            As: data structure of TNs, *xs: qtn.Tensors, output_inds: list[str]
            contracts As with input xs
    """
    norm_b = b.norm()
    input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.reindex(input_to_output_inds, inplace=False)  ## residual

    if x is not None:
        # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        # Ax_tens = Ax.contract_tags(all)
        # Ax_tens.transpose_like(r, inplace=True)
        # Ax_tens = contract_with_As(x)
        Ax_tens = contract_Ax(As, x, out_inds=r.inds)
        r.modify(apply=lambda data: data - Ax_tens.data)

    d = r.copy()  ## d: search r
    norm_r = r.norm()

    it = 0
    x_min, min_err = x, np.inf
    err = r.norm()
    while it < max_iter and err > conv_tol * norm_b:

        ## alpha = np.linalg.norm(r) ** 2 / np.dot(d.conj(), np.dot(A, d))
        dT = d.conj()
        d.reindex(output_to_input_inds, inplace=True)
        # proj = qtn.TensorNetwork([dT, A, d], virtual=True, check_collisions=False)
        # alpha = err ** 2 / proj.contract_tags(all)
        # proj = contract_with_As(dT, d)
        r_proj = contract_Ax(As, d, out_inds=r.inds)
        proj = contract_xAx(As, d, dT, out_inds=r.inds, precomp_Ax=r_proj)  # op, ket, bra
        alpha = err ** 2 / proj

        ## update x:  x_ = x + alpha * d
        if x is None:
            x = d.copy()
            x.modify(apply=lambda data: data * alpha)
            # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        else:
            x.modify(apply=lambda data: data + alpha * d.data)

        ## update residual r:  r_ = r - alpha * np.dot(A, d)
        # r_proj = qtn.tensor_contract(*A.tensors, d)
        # r_proj.transpose_like(r, inplace=True)
        # r_proj = contract_with_As(d)
        # r_proj = contract_Ax(As, d, out_inds=r.inds)  ## now computed above
        r.modify(apply=lambda data: data - alpha * r_proj.data)
        new_norm_r = r.norm()

        ## update search direction d:
        ## beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        ## d_ = r_ + beta_ * d
        beta_ = new_norm_r ** 2 / norm_r ** 2
        d.reindex(input_to_output_inds, inplace=True)
        d.transpose_like(r, inplace=True)
        d.modify(apply=lambda data: data * beta_ + r.data)

        err = new_norm_r
        norm_r = new_norm_r

        if err < min_err:
            min_err = err
            x_min = x.copy()
        # else:
        #     ## convergence not necessarily monotonic
        #     print('error went up', err, min_err)

        it += 1

        # print('it', it, err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')
            # print('using min x')
            x = x_min

    return x, err / norm_b


# @profile
def qtn_conjugate_gradient_squared_1site(As: Sequence[qtn.TensorNetwork], b: qtn.Tensor, x: qtn.Tensor,
                                         output_to_input_inds: dict[str, str],
                                         max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False,
                                         contract_Ax=_contract_x_with_As,
                                         contract_xAx=_contract_x_with_As,
                                         ) \
        -> tuple[qtn.Tensor, float]:
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
        A: matrix, b: target, x: solution
        r: residual = b - Ax
        d: search direction
    """
    norm_b = b.norm()
    input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.reindex(input_to_output_inds, inplace=False)  ## ket inds
    rT0 = r.copy(deep=True)

    bra_inds = r.inds  ## out inds
    ket_inds = tuple([output_to_input_inds[k] for k in r.inds])  ## in inds

    # def contract_with_As(*x_tens, out_inds=bra_inds, scale=True):
    #     Ax_tot = None
    #     for A in As:
    #         Ax = qtn.TensorNetwork([A, *x_tens], virtual=True, check_collisions=False)
    #         out = Ax.contract_tags(all)
    #
    #         if isinstance(out, qtn.Tensor):
    #             # out.transpose_like(r, inplace=True)
    #             out.transpose(*out_inds, inplace=True)
    #             if scale:
    #                 out.modify(apply=lambda data: data * 10 ** Ax.exponent)
    #             if Ax_tot is None:
    #                 Ax_tot = out
    #             else:
    #                 Ax_tot.modify(apply=lambda data: data + out.data)
    #         else:
    #             if scale:
    #                 out *= 10 ** Ax.exponent
    #             if Ax_tot is None:
    #                 Ax_tot = out
    #             else:
    #                 Ax_tot += out
    #     return Ax_tot

    if x is not None:
        # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        # Ax_tens = Ax.contract_tags(all)
        # Ax_tens.transpose_like(r, inplace=True)
        Ax_tens = contract_Ax(As, x, out_inds=bra_inds)
        # Ax_tens = contract_with_As(x)
        r.modify(apply=lambda data: data - Ax_tens.data)  ## out inds

        xT = x.reindex(input_to_output_inds, inplace=False)  ## out inds
        # AxT = qtn.TensorNetwork([As[0], xT], virtual=True, check_collisions=False)
        # AxT_tens = AxT.contract_tags(all)   ## in inds
        AxT_tens = contract_Ax(As, xT, out_inds=ket_inds, scale=True)
        # AxT_tens = contract_with_As(xT, out_inds=ket_inds, scale=True)
        # print('AxT inds', AxT_tens.inds, ket_inds)
        AxT_tens.reindex(input_to_output_inds, inplace=True)
        AxT_tens.transpose(*bra_inds, inplace=True)
        rT0.modify(apply=lambda data: data - AxT_tens.data)

    # rT0 = r.copy()

    zero_tens = qtn.Tensor(np.zeros(r.shape), inds=bra_inds)
    zero_tens_cc = qtn.Tensor(np.zeros(r.shape), inds=ket_inds)

    it = 0
    q, p, rho = zero_tens_cc.copy(), zero_tens_cc.copy(), 1
    # err = qtn.tensor_contract(r, rT0)
    err = r.norm()
    min_x, min_err = x, err
    while it < max_iter and np.abs(err) > conv_tol * norm_b:

        rho_ = qtn.tensor_contract(r, rT0)
        beta = rho_ / rho

        u = qtn.Tensor(data=r.data + beta * q.data, inds=ket_inds)  ## in inds
        p_ = qtn.Tensor(data=u.data + beta * (q.data + beta * p.data), inds=ket_inds)  ## in inds
        # v = qtn.tensor_contract(*A.tensors, p_)     ## out inds
        # v.transpose_like(r, inplace=True)
        v = contract_Ax(As, p_, out_inds=bra_inds)
        # v = contract_with_As(p_)

        sigma = qtn.tensor_contract(rT0, v)  ## out inds
        alpha = rho_ / sigma

        q_ = qtn.Tensor(data=u.data - alpha * v.data, inds=ket_inds)  ## in inds
        tmp_uq = qtn.Tensor(data=u.data + q_.data, inds=ket_inds)  ## in_inds
        # tmp_A = qtn.tensor_contract(*A.tensors, tmp_uq)     ## out inds
        # tmp_A.transpose_like(r, inplace=True)
        tmp_A = contract_Ax(As, tmp_uq, out_inds=bra_inds)
        # tmp_A = contract_with_As(tmp_uq)
        # r_ = r - alpha * np.dot(A, u + q_)
        # x_ = x + alpha * (u + q_)
        r.modify(data=r.data - alpha * tmp_A.data)
        x.modify(data=x.data + alpha * tmp_uq.data)

        p, q = p_, q_
        rho = rho_
        it += 1

        err = r.norm()
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')
            # print('using min x')
            x = min_x

    return x, min_err / norm_b
