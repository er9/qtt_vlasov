import numpy as np
import itertools
import quimb

from setup.defaults import *
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2

from helper_tdvp import *
from helper_tdvp import _sum_eff_TNs
from helper_dmrg import *
from helper_block_dmrg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gridTN import GridTN

class SolveMethod(Enum):
    RK4 = 'RK4'
    CN = 'CN'
    EX = 'EX'


def block_tddmrg(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                 x_vecs: dict[int, 'qtn.MatrixProductState'],
                 b_vecs: dict[int, Sequence['MPS_type']] = None,
                 verbose_output=False,
                 init_direction=SweepDirection.RIGHT,
                 **solver_kwargs):
    print('BLOCK DMRG 2')
    solver = BlockedTDDMRGSolver(ncomps, x_vecs, b_vecs, A_mats, **solver_kwargs)
    solver.take_time_step(dt, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets



def _exactTE_site(dt: float,
                  x_dict: dict[int,'qtn.Tensor'], A_effs_dict: dict[int, Sequence['qtn.TensorNetwork']],
                  b_eff_dict: dict[int,'qtn.Tensor'], ket_inds: Sequence[str], bra_inds: Sequence[str],
                  **kwargs):
    """ Time evolves dx/dt = Ax + b
    """
    # site_pos = list(range(self.L))[sites]
    # site_inds = self.get_mps_ind(sites)
    #
    # ## combine As, bs
    # start = np.min(site_pos) if isinstance(site_pos, list) else site_pos
    # end = np.max(site_pos) if isinstance(site_pos, list) else site_pos
    #
    # bra_inds = []  # [self.component_ind + '_']
    # if start > 0:
    #     bra_inds += [self.bra.bond(start - 1, start)]
    # if end < self.L - 1:
    #     bra_inds += [self.bra.bond(end, end + 1)]
    #
    # bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
    # for site_p in site_pos:
    #     bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
    # ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

    bra_to_ket_inds = {b: k for b, k in zip(bra_inds, ket_inds)}

    A_tens_dict = {}
    for k, A_effs in A_effs_dict.items():
        A_eff = _sum_eff_TNs(A_effs, transpose_bonds=[*bra_inds,*ket_inds])
        A_tens_dict[k] = A_eff

    for k, b_eff in b_eff_dict.items():
        b_eff.transpose(*bra_inds, inplace=True)

    for k, x_eff in x_dict.items():
        x_eff.transpose(*ket_inds, inplace=True)

    ## solve for T[i]
    new_x_dict = {}
    if len(A_effs_dict) == 0:
        for ik in b_eff_dict.keys():
            ## still need to make compatible with ket exponent
            dxdt_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
            dxdt_eff.modify(apply=lambda x: x * dt)

            new_x = x_dict.get(ik, None)
            if new_x is None:
                new_x_dict[ik] = dxdt_eff
            else:
                new_x_dict[ik] = helper.add_tensors(new_x, dxdt_eff, inplace=False)
    else:

        in_keys = list(x_dict.keys())
        out_keys = list(b_eff_dict.keys())
        in_keys.sort(), out_keys.sort()
        in_shape = [x_dict[k].size for k in in_keys]
        out_shape = [b_eff_dict[k].size for k in out_keys]

        if len(b_eff_dict) == 0:
            x_vec = combine_vec_tens(x_dict, keys=in_keys)
            A_mat = combine_mat_tens(A_tens_dict, out_keys, in_keys, out_shape, in_shape)

            x_shape = x_vec.shape
            x_size = int(np.prod(x_shape))
            exp_A = A_mat.data.reshape(x_size, x_size)
            exp_A = scipy.linalg.expm(exp_A * dt)

            x_eff_data = np.dot(exp_A, x_vec.reshape(x_size)).reshape(*x_shape)
            new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)
        else:
            raise NotImplementedError

    return new_x_dict


# def explicit_TE(dt: float,
#                 xeff_dict: dict[int,'qtn.Tensor'], Aeff_dict: dict[int,Sequence['qtn.TensorNetwork']],
#                 beff_dict: dict[int,'qtn.Tensor'], ket_inds: Sequence[str], bra_inds: Sequence[str],
#                 te_order: int = 4, **kwargs):
#     """ time integration of specific site, as dictated by te_order. limited to RK TE methods
#     """
#
#     bonds_i = ket_inds
#     bonds_o = bra_inds
#
#     if dt == 0:
#         return xeff_dict
#
#     def deriv_func(xeff_dict_, dt=1, **kwargs) -> dict[int, 'qtn.Tensor']:
#         eff_Ax = {}
#         for (oo, ii), A_effs in Aeff_dict.items():
#             eff_Ax = []
#             xeff = xeff_dict_[ii]
#             for deriv_tn in A_effs:
#                 eff_Ax += [qtn.TensorNetwork([deriv_tn, xeff])]
#                 ## dist_submpx exponent should equal 0
#
#             eff_Ax += [qtn.TensorNetwork([beff_dict[oo]])]
#
#             if oo not in eff_Ax:
#                 eff_Ax[oo] = eff_Ax
#             else:
#                 eff_Ax[oo] += eff_Ax
#
#         out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
#         out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
#         out.modify(apply=lambda x: x * dt)
#         return out
#
#     def add_func(tens1_dict, tens2_dict, inplace=False, **kwargs) -> dict[int, 'qtn.Tensor']:
#         out_dict = tens1_dict if inplace else {k: tens.copy() for k, tens in tens1_dict.items()}
#         tot_keys = set(tens1_dict.keys()).union(tens2_dict.keys())
#         for k in tot_keys:
#             try:
#                 tens2 = tens2_dict[k]
#                 try:
#                     tens1 = out_dict[k]
#                     out_dict[k] = helper.add_tensors(tens1, tens2, inplace=True)
#                 except IndexError:
#                     out_dict[k] = tens2.copy()
#             except IndexError:
#                 pass
#         return out_dict
#
#     def scale_func(tens_dict, scale_val, inplace=False, **kwargs) -> dict[int, 'qtn.Tensor']:
#         out_dict = tens_dict if inplace else {k: tens.copy() for k, tens in tens_dict.items()}
#         for k, tens in out_dict.items():
#             tens.modify(apply=lambda x: x * scale_val)
#         return out_dict
#
#     def euler_func(state, dt, deriv0, **kwargs) -> dict[int, 'qtn.Tensor']:
#         return add_func(state, scale_func(deriv0, dt))
#
#     if te_order in [1,2,3,4]:
#         out = helper_TE.time_integration(xeff_dict, dt, euler_func, deriv_func, add_func, helper.scale_tensors,
#                                          te_order=te_order)
#     else:
#         raise NotImplementedError
#
#     return out




class BlockedTDDMRGSolver(BlockedLinearSolver):
    def __init__(self, ncomps, init_state: dict[int, 'qtn.MatrixProductState'],
                 sources: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 is_H=False,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 norm_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Ls: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Rs: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 err_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 err_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solve_type: SolveMethod = DEFAULT_SOLVE,
                 solvers=None,
                 ):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        super(BlockedLinearSolver, self).__init__(ncomps, init_state, sources, operators,
                                                  mps_inds=mps_inds, conv_tol=conv_tol,
                                                  max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter,
                                                  base_solver=BaseSolver.TDDMRG,
                                                  norm_env0_Rs=norm_env0_Rs, norm_env0_Ls=norm_env0_Ls,
                                                  ovlp_env0_Rs=ovlp_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
                                                  err_env0_Rs=err_env0_Rs, err_env0_Ls=err_env0_Ls,
                                                  solvers=solvers)


        for k, solver in self.solvers.items():
            solver.is_H = False  # is_H
            # solver.solve_type = solve_type

        # self.solve_type = solve_type
        self.is_H = False

    combined_ind_o = 'out_ind'
    combined_ind_i = 'in_ind'

    def canonize(self, i, cur_orthog=None):
        ### relies on fact that canonicalization is done in the td-DMRG site solve
        return

    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        ket_to_bra_inds = self.get_ket_to_bra_inds(i)

        for (oo,ii), solver in self.solvers.items():
            new_ket_tens = site_i_dict[ii].copy()
            new_bra_tens = site_i_dict[oo].conj().reindex(ket_to_bra_inds)

            solver.update_bra_tens([new_bra_tens], sites=[i]) # , new_exponent=ref_b_solver.bra.exponent)
            solver.update_ket_tens([new_ket_tens], sites=[i]) # , new_exponent=ref_k_solver.ket.exponent)

            solver.reinitialize_envs_i(i)

        return

    def _update_2site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        sites_1, sites_2 = {}, {}
        i2 = i + 1 if direction == SweepDirection.RIGHT else i - 1

        for k, tens in site_i_dict.items():
            ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i2)
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            site1, site2 = tens.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            sites_1[k] = site1
            sites_2[k] = site2

        self._update_1site(i, sites_1, direction=direction)
        self._update_1site(i2, sites_2, direction=direction)

        return

    def _modify_solver_ket_and_bra(self, ix, func, mod_sites: Union[Sequence[int], int], *args, keyed_kwargs=None,
                                   **kwargs):

        for k, solver in self.solvers.items():
            keyed_kwarg = {kw: keyed_dict[k] for kw, keyed_dict in keyed_kwargs.items()
                           } if keyed_kwargs is not None else {}

        ## diag solvers
        for k, solver in self.solvers.items():
            if k[0] == k[1]:
                keyed_kwarg = {kw: keyed_dict[k] for kw, keyed_dict in keyed_kwargs.items()
                               } if keyed_kwargs is not None else {}
                func(solver, ix, *args, **keyed_kwarg, **kwargs)

        for k, solver in self.solvers.items():
            out_ind, in_ind = k
            if out_ind == in_ind:
                continue

            ref_k_solver = self.solvers[in_ind, in_ind]
            ref_b_solver = self.solvers[out_ind, out_ind]
            # print('ref k exp', ref_k_solver.ket.exponent, ref_k_solver.bra.exponent)
            # print('ref b exp', ref_b_solver.ket.exponent, ref_b_solver.bra.exponent)
            if ix is all:
                solver._ket = ref_k_solver.ket
                solver._bra = ref_b_solver.bra
                solver.reinitialize_envs()
            else:
                inds = self.get_mps_ind(mod_sites)
                if isinstance(inds, int):  inds = [inds]
                solver.update_bra_tens([ref_b_solver.bra[ind] for ind in inds], sites=mod_sites,
                                       new_exponent=ref_b_solver.bra.exponent)
                solver.update_ket_tens([ref_k_solver.ket[ind] for ind in inds], sites=mod_sites,
                                       new_exponent=ref_k_solver.ket.exponent)

                if isinstance(mod_sites, int):
                    solver.reinitialize_envs_i(mod_sites)
                else:
                    for ix in mod_sites:
                        solver.reinitialize_envs_i(ix)

                # print('sovler exp', solver.ket.exponent, solver.bra.exponent)
                # print([env.ket is solver.ket for env in solver.left_envs])
                # print([env.ket is solver.ket for env in solver.right_envs])
                # print([env.bra is solver.bra for env in solver.left_envs])
                # print([env.bra is solver.bra for env in solver.right_envs])
        return



    def cgd(self, sites, x_dict, A_effs_dict, b_eff_dict, **kwargs):

        site_pos = list(range(self.L))[sites]
        site_inds = self.get_mps_ind(sites)

        ## combine As, bs
        start = np.min(site_pos) if isinstance(site_pos, list) else site_pos
        end = np.max(site_pos) if isinstance(site_pos, list) else site_pos

        bra_inds = []  # [self.component_ind + '_']
        if start > 0:
            bra_inds += [self.bra.bond(start - 1, start)]
        if end < self.L - 1:
            bra_inds += [self.bra.bond(end, end + 1)]

        ## JANK fix for comb branch
        bra_inds += [self.bra.site_ind_id.format(i) for i in site_inds
                     if self.bra.site_ind_id.format(i) in self.bra[i].inds]

        bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        if len(A_effs_dict) == 0:
            new_x_dict = {}
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
                new_x_dict[ik] = x_eff
        else:
            in_keys = list(x_dict.keys())
            out_keys = list(b_eff_dict.keys())
            in_keys.sort(), out_keys.sort()
            # in_keys = out_keys = list(range(6))
            in_shape = [x_dict[k].size for k in in_keys]
            out_shape = [b_eff_dict[k].size for k in out_keys]

            x_vec = combine_vec_tens(x_dict, keys=in_keys)
            b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

            combined_ind_i, combined_ind_o = self.__class__.combined_ind_i, self.__class__.combined_ind_o
            x_vec_tens = qtn.Tensor(data=x_vec, inds=(combined_ind_i,))
            b_vec_tens = qtn.Tensor(data=b_vec, inds=(combined_ind_o,))

            def local_compute_Ax(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                 x_vec_tens_: 'qtn.Tensor', out_inds, **kwargs) -> 'qtn.Tensor':

                return self.__class__.compute_Ax(A_effs_dict_, x_vec_tens_,
                                                 ref_x_dict=x_dict, ref_b_eff_dict=b_eff_dict,
                                                 out_keys=out_keys, bra_inds=bra_inds, out_shape=out_shape,
                                                 in_keys=in_keys, ket_inds=ket_inds, in_shape=in_shape,
                                                 out_inds=out_inds, **kwargs)

            def local_compute_xAx(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                  x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor', out_inds,
                                  precomp_Ax=None, **kwargs) -> 'qtn.Tensor':

                return self.__class__.compute_xAx(A_effs_dict_, x_vec_tens_i, x_vec_tens_o,
                                                  ref_x_dict=x_dict, ref_b_eff_dict=b_eff_dict,
                                                  out_keys=out_keys, bra_inds=bra_inds, out_shape=out_shape,
                                                  in_keys=in_keys, ket_inds=ket_inds, in_shape=in_shape,
                                                  out_inds=out_inds, precomp_Ax=precomp_Ax, **kwargs)

            bra_to_ket_inds_tmp = {**bra_to_ket_inds, combined_ind_o: combined_ind_i}
            if self.is_H:
                x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                    bra_to_ket_inds_tmp,
                                                                    contract_Ax=local_compute_Ax,
                                                                    contract_xAx=local_compute_xAx)
            else:
                x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                    bra_to_ket_inds_tmp,
                                                                    contract_Ax=local_compute_Ax,
                                                                    contract_xAx=local_compute_xAx)
            # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
            # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))
            new_x_dict = extract_vec_tens(x_eff.data, in_keys, in_shape, x_dict)
        return new_x_dict


    def lsq(self, sites, x_dict: dict[int,'qtn.Tensor'], A_effs_dict: dict[int,'qtn.Tensor'],
            b_eff_dict: dict[int,'qtn.Tensor'], **kwargs):
        """ solve A x = b
        """

        site_pos = list(range(self.L))[sites]
        site_inds = self.get_mps_ind(sites)

        ## combine As, bs
        start = np.min(site_pos) if isinstance(site_pos, list) else site_pos
        end = np.max(site_pos) if isinstance(site_pos, list) else site_pos

        bra_inds = []  # [self.component_ind + '_']
        if start > 0:
            bra_inds += [self.bra.bond(start - 1, start)]
        if end < self.L - 1:
            bra_inds += [self.bra.bond(end, end + 1)]

        bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        for k, A_eff in A_effs_dict.items():
            A_eff.transpose(*bra_inds, *ket_inds, inplace=True)

        for k, b_eff in b_eff_dict.items():
            b_eff.transpose(*bra_inds, inplace=True)

        for k, x_eff in x_dict.items():
            x_eff.transpose(*ket_inds, inplace=True)

        # A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
        # for key, solver in self.solvers.items():
        #     ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
        #     A_effs_dict[key] = solver._get_A_effs(sites, return_sum=(not self.solve_type is SolveMethod.CGD))
        #     if self.solve_type is not SolveMethod.CGD:
        #         A_effs_dict[key].transpose(*bra_inds, *ket_inds, inplace=True)
        #
        # b_eff_dict = self._get_b_eff_dict_(sites, bra_inds)

        # x_dict = {}
        # for k, ket in self.kets.items():
        #     if isinstance(site_inds, (list, tuple)):
        #         ket_tensors = [ket[si] for si in site_inds]
        #     else:
        #         ket_tensors = ket[site_inds]
        #     x_tens = qtn.tensor_contract(*ket_tensors)
        #     x_tens.transpose(*ket_inds, inplace=True)
        #     x_dict[k] = x_tens

        ## solve for T[i]
        new_x_dict = {}
        if len(A_effs_dict) == 0:
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
                new_x_dict[ik] = x_eff
        else:

            in_keys = list(x_dict.keys())
            out_keys = list(b_eff_dict.keys())
            in_keys.sort(), out_keys.sort()
            in_shape = [x_dict[k].size for k in in_keys]
            out_shape = [b_eff_dict[k].size for k in out_keys]

            x_vec = combine_vec_tens(x_dict, keys=in_keys)
            b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

            # print('in keys', in_keys, 'out keys', out_keys)
            A_mat = combine_mat_tens(A_effs_dict, out_keys, in_keys, out_shape, in_shape)
            # print('A is H?', np.linalg.norm(A_mat - A_mat.conj().T))

            solve_type = SolveMethod.LSQ
            if solve_type is SolveMethod.CGDx:
                if self.is_H:
                    x_eff_data, error = conjugate_gradient_descent(A_mat, b_vec, x=x_vec)
                else:
                    x_eff_data, error = conjugate_gradient_squared(A_mat, b_vec, x=x_vec)

                new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)
                # exit()
            else:  ## ALS optimzation
                x_eff_data = np.linalg.solve(A_mat, b_vec)
                new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)

        return new_x_dict


    def get_time_evolved_site(self, dt, sites, xeff_dict, Aeff_dict, beff_dict, bonds_i, bonds_o,
                              te_order=4) -> dict[int, 'qtn.Tensor']:
        """ dx/dt = Ax + b
        """
        if dt == 0:
            return xeff_dict

        def deriv_func(xeff_dict_, dt=1, **kwargs) -> dict[int,'qtn.Tensor']:
            eff_Ax_dict = {}
            for (oo, ii), A_effs in Aeff_dict.items():
                eff_Ax = []
                xeff = xeff_dict_[ii]
                for deriv_tn in A_effs:
                    eff_Ax += [qtn.TensorNetwork([deriv_tn, xeff])]
                    # print('eff Ax', oo, ii, eff_Ax[-1])
                    # print('self.ket', self.kets[ii])
                    # print('self.bra', self.kets[oo])
                    ## dist_submpx exponent should equal 0

                beff = beff_dict.get(oo, None)
                if beff is not None:
                    eff_Ax += [qtn.TensorNetwork([beff])]

                if oo not in eff_Ax_dict:
                    eff_Ax_dict[oo] = eff_Ax
                else:
                    eff_Ax_dict[oo] += eff_Ax

            out_dict = {}
            for oo, eff_Axs in eff_Ax_dict.items():
                out = _sum_eff_TNs(eff_Axs, transpose_bonds=bonds_o)
                # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
                out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
                out.modify(apply=lambda x: x * dt)
                out_dict[oo] = out
            return out_dict

        def add_func(tens1_dict, tens2_dict, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens1_dict if inplace else {k: tens.copy() for k, tens in tens1_dict.items()}
            tot_keys = set(tens1_dict.keys()).union(tens2_dict.keys())
            for k in tot_keys:
                try:
                    tens2 = tens2_dict[k]
                    try:
                        tens1 = out_dict[k]
                        out_dict[k] = helper.add_tensors(tens1, tens2, inplace=True)
                    except IndexError:
                        out_dict[k] = tens2.copy()
                except IndexError:
                    pass
            return out_dict

        def scale_func(tens_dict, scale_val, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens_dict if inplace else {k: tens.copy() for k, tens in tens_dict.items()}
            for k, tens in out_dict.items():
                tens.modify(apply=lambda x: x * scale_val)
            return out_dict

        def euler_func(state, dt, deriv0, **kwargs) -> dict[int,'qtn.Tensor']:
            return add_func(state, scale_func(deriv0, dt))

        ### implicit funcs
        def scale_tn_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.TensorNetwork':
            tens = helper.scalar_multiply(tens, scale_val, inplace=inplace)
            return tens

        def solve_func_cgd(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()}

            out = self.cgd(sites, init_guess, deriv_ops, explicit_contribution,
                           max_iter=max_iter, conv_tol=conv_tol, **kwargs)
            # print('solve err', err)
            return out

        def solve_func_lsq(deriv_ops: dict[int,Sequence['qtn.TensorNetwork']],
                           explicit_contribution: dict[int,'qtn.Tensor'],
                           init_guess=None, max_iter=None, conv_tol=None, **kwargs):

            # shape_ = explicit_contribution.shape
            # size_ = int(np.prod(shape_))

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            deriv_ops_ = {k: _sum_eff_TNs(d_ops, [*bonds_o, *bonds_i]) for k, d_ops in deriv_ops.items()}
            # explicit_contribution = explicit_contribution.transpose(*bonds_i)
            # print('A eff tens', A_eff_tens, explicit_contribution)

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()}

            out = self.lsq(sites, init_guess, deriv_ops_, explicit_contribution,
                           max_iter=max_iter, conv_tol=conv_tol, **kwargs)
            return out

        def identity_func(xeff_dict: dict[int,Union['qtn.Tensor', 'qtn.TensorNetwork']]):
            iden_dict = {}
            for i, state in xeff_dict.items():
                shape_ = state.shape
                size_ = int(np.prod(shape_))
                iden = qtn.Tensor( np.eye(size_).reshape(*shape_, *shape_),
                                   inds=(*bonds_o,*bonds_i) )
                iden_dict[i] = qtn.TensorNetwork([iden])
            return iden_dict

        ### perform actual TE
        if te_order == 4:
            print('dt', dt)
            out = helper_TE.rk4(xeff_dict, dt, euler_func, deriv_func, add_func, scale_func)
        elif te_order == 22:
            out = helper_TE.solve_crank_nicolson(xeff_dict, euler_func, deriv_func, add_func, scale_tn_func,
                                                 Aeff_dict, identity_func, solve_func_lsq, dt)
        else:
            raise NotImplementedError

        return out


    def get_site_total_rdm(self, dt, left_site_pos, x_bond, nsites=1, **kwargs) -> dict[int,'qtn.Tensor']:
        """
        """
        site_inds = self.get_mps_ind(slice(left_site_pos,left_site_pos + nsites,1))
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
        for key, solver in self.solvers.items():
            ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
            A_effs_dict[key] = solver._get_A_effs(left_site_pos, nsites)
            # if self.solve_type is not SolveMethod.CGD:
            #     A_effs_dict[key].transpose(*bonds_o, *bonds_i, inplace=True)

        b_eff_dict = self._get_b_eff_dict_(site_inds, bonds_o)

        x_dict = {}
        for k, ket in self.kets.items():
            if isinstance(site_inds, (list, tuple)):
                ket_tensors = [ket[si] for si in site_inds]
            else:
                ket_tensors = ket[site_inds]
            x_tens = qtn.tensor_contract(*ket_tensors)
            x_tens.transpose(*bonds_i, inplace=True)
            x_dict[k] = x_tens

        ## bond to contract over
        def compute_denmat(ket_dict: dict[int,'qtn.Tensor']):
            denmat_dict = {}
            for k_, ket_ in ket_dict.items():
                bra_ = ket_.conj()
                bra_.reindex({kk: b for kk, b in ket_to_bra_inds.items() if kk != x_bond}, inplace=True)
                denmat_dict[k_] = qtn.tensor_contract(bra_, ket_)
            return denmat_dict

        rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
        rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]

        target_times = [0, dt/3, 2*dt/3, dt]
        weights = [1./3, 1./6, 1./6, 1./3]
        # target_times = [0, dt/3, 2*dt/3, 4*dt/3, dt]
        # weights = [1./3, 1./9, 1./9., 1./9, 1./3]
        # target_times = [0, (1 - np.sqrt(3)/2) * dt , dt, (1 + np.sqrt(3)/2) * dt ]
        # weights = [1./4, 1./4, 1./4, 1./4]
        # target_times = [dt]
        # weights = [1.]

        ## for accumulated time steps
        sort_inds = np.argsort(target_times)
        target_times = [0] + [target_times[i + 1] - target_times[i] for i in sort_inds[:-1]]
        weights = [weights[i] for i in sort_inds]

        tot_denmat = {}
        for w, tt in zip(weights, target_times):
            new_xeff_dict = self.get_time_evolved_site(tt, site_inds, x_dict, A_effs_dict, b_eff_dict,
                                                       bonds_i, bonds_o)
            x_dict = new_xeff_dict       ## for accumulated time steps

            rho0_dict = compute_denmat(x_dict)
            for k, rho0 in rho0_dict.items():
                rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|
                rho0.modify(apply=lambda x: x * w)

            for k in rho0_dict.keys():
                if k not in tot_denmat:
                    tot_denmat[k] = rho0_dict[k].copy()
                else:
                    tot_denmat[k].modify(apply=lambda x: x + rho0_dict[k].data)

        return tot_denmat


    # @profile
    def _site_time_evolution(self, dt,left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT, ):
        """
        sites: int or slice(start, stop, step)
        solve local A' x = b',  (1/2?) <x|Ax> - <x|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A|x> = d/dT[i] d/dT[i]^* \sum_ij <x_j|A_ji|x_i>
            where b' = d/dT[i]^* <x|b> = d/dT[i] \sum_ij <x_j|b_i>
        """
        # print("SITE SOLVE", self.solve_type, sites)
        sites = list( range(left_site_pos, left_site_pos + nsites) )
        site_pos = sites
        site_inds = self.get_mps_ind(sites)

        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        if compress_direction == CompressDirection.RIGHT:
            sweep_dir = SweepDirection.RIGHT
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            sweep_dir = SweepDirection.LEFT
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        # print('self ket', self.ket)
        # print('self bra', self.bra)

        bra_inds = [self.bra.site_ind_id.format(i) for i in site_inds]
        min_ind, max_ind = min(site_inds), max(site_inds)
        if min_ind > 0:
            bra_inds += [self.bra.bond(min_ind - 1, min_ind)]
        if max_ind < self.L - 1:
            bra_inds += [self.bra.bond(max_ind, max_ind + 1)]

        bra_to_ket_inds = {} # self.bra.site_ind_id.format(i): self.ket.site_ind_id.format(i) for i in site_inds}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        # print('left site pos', left_site_pos, nsites, compress_direction)
        # print('ind1', ind1, ind2, at_end)
        # print('bra inds', bra_inds)
        # print('bra-to-ket', bra_to_ket_inds)

        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        bonds_o = bra_inds
        bonds_i = ket_inds

        ## targeted time steps
        if not at_end:

            ## bond to contract over
            if compress_direction == CompressDirection.RIGHT:
                x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
            else:
                x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

            ## solve for T[i]
            tot_denmat_dict = self.get_site_total_rdm(dt, left_site_pos, x_bond, nsites=nsites)

            updated_ket_1 = {}
            updated_ket_2 = {}
            for k, tot_denmat in tot_denmat_dict.items():

                rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
                rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
                nb = len(rho_bonds_o)

                # print('rho bonds i', rho_bonds_i)
                # print('rho bonds o', rho_bonds_o)
                # print('tot denmat')
                # print(k, tot_denmat)

                tot_denmat.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)

                denmat_shape = tot_denmat.shape
                tens_shape = denmat_shape[:nb]
                sq_shape = np.prod(denmat_shape[:nb])

                eigval, eigvec = np.linalg.eigh(tot_denmat.data.reshape(sq_shape, sq_shape))

                sort_inds = np.argsort(np.abs(eigval))[::-1]
                ev_max = eigval[sort_inds[0]]
                # eigval_inds = np.where(np.abs(eigval) > 1.0e-12 * ev_max)
                new_sort_inds = []
                for si in sort_inds:
                    if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
                        new_sort_inds += [si]
                    else:
                        break
                sort_inds = new_sort_inds

                if self.max_bond is not None:
                    if self.max_bond // nsites > 0:
                        eigvec = eigvec[:, sort_inds[:self.max_bond // nsites]]

                eigvec = eigvec.reshape(*tens_shape, -1)
                eigvec = qtn.Tensor(data=eigvec, inds=tuple(rho_bonds_i) + (x_bond + '_',))

                ket = self.ket
                if nsites == 1:
                    eigvec.transpose_like(ket[ind1], inplace=True)
                    next_site = qtn.tensor_contract(eigvec.conj(), ket[ind1], ket[ind2])
                    next_site.transpose_like(ket[ind2], inplace=True)
                    # ket[ind1].modify(data=eigvec.data)
                    # ket[ind2].modify(data=next_site.data)

                    eigvec.modify(inds = ket[ind1].inds)
                    next_site.modify(inds = ket[ind2].inds)
                    updated_ket_1[k] = eigvec  # ket[ind1]
                    updated_ket_2[k] = next_site  # ket[ind2]

                elif nsites == 2:
                    ## decompose eigvec into two sites, tensor contract ind1 site
                    bond_ms, left_inds = ket[ind1].filter_bonds(ket[ind2])
                    bond_m = next(iter(bond_ms))
                    ##
                    ev1, ev2 = qtn.tensor_split(eigvec, left_inds, absorb='right', bond_ind=bond_m + '_',
                                                max_bond=self.max_bond)
                    next_site = qtn.tensor_contract(ev1.conj(), ket[ind1], ket[ind2])
                    ev1.transpose_like(ket[ind1], inplace=True)
                    next_site.transpose_like(ket[ind2], inplace=True)
                    ev1.modify(inds = ket[ind1].inds)
                    next_site.modify(inds=ket[ind2].inds)
                    # ket[ind1].modify(data=ev1.data)
                    # ket[ind2].modify(data=next_site.data)
                    # print('ind1', ind1, ind2)
                    # helper.check_orthog(self.ket)

                    updated_ket_1[k] = ev1  # ket[ind1]
                    updated_ket_2[k] = next_site  # ket[ind2]
                else:
                    raise NotImplementedError

            ## i think this is already done in canonize
            self._update_1site(ind1, updated_ket_1, direction=sweep_dir)
            self._update_1site(ind2, updated_ket_2, direction=sweep_dir)

        else:

            A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
            for key, solver in self.solvers.items():
                ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
                A_effs_dict[key] = solver._get_A_effs(left_site_pos, nsites)
                # if self.solve_type is not SolveMethod.CGD:
                #     A_effs_dict[key].transpose(*bonds_o, *bonds_i, inplace=True)

            b_eff_dict = self._get_b_eff_dict_(site_inds, bonds_o)

            x_dict = {}
            for k, ket in self.kets.items():
                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [ket[si] for si in site_inds]
                else:
                    ket_tensors = ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*bonds_i, inplace=True)
                x_dict[k] = x_tens

            new_xeff_dict = self.get_time_evolved_site(dt, site_inds, x_dict, A_effs_dict, b_eff_dict,
                                                       bonds_i, bonds_o)

            # i think this is already done in canonize
            if nsites == 1:
                self._update_1site(left_site_pos, new_xeff_dict, direction=sweep_dir)
            elif nsites == 2:
                self._update_2site(left_site_pos, new_xeff_dict, direction=sweep_dir)
            else:
                raise NotImplementedError


        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            ## canonicalization is not needed here, but is used by subclass TDDMRGSolver_v3 for ket0, bra0
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2):
                self._update_envs_left(i, canonize=False)

            for k, solver in self.solvers.items():
                print('solver', k, ind2)
                helper.check_orthog(solver.ket)
                helper.check_orthog(solver.bra)

        else:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2, -1):
                self._update_envs_right(i, canonize=False)

            for k, solver in self.solvers.items():
                print('solver', k, ind2)
                helper.check_orthog(solver.ket)
                helper.check_orthog(solver.bra)

        return


    # @profile
    def take_time_step(self, dt, grid=None, do_adapt=False, init_direction=SweepDirection.RIGHT, **kwargs):
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
        L = self.ket.L
        # print('L', self.L, self.ket.L)
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(1, canonize=False)

            ### left to right sweep
            self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### right to left sweep
            self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        else:
            self._build_all_envs_left(1, canonize=False)

            ### right to left sweep
            self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### left to right sweep
            self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done sweep')

        return self.kets

    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r block tddmrg', dt, 'do adapt', do_adapt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = CompressDirection.RIGHT
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                if verbose:  print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = max_bond  # min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                max_bond_ = min(max_bond, 2**(site_ind+1), 2**(L-site_ind-1))
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind += 1
                    continue

            nsites = 2 if adapt else 1

            ## forward propagation
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
            self._site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        # do_adapt = False
        max_bond = self.max_bond
        print('r2l tdvp', dt, 'do adapt', do_adapt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                if verbose:   print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                 max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                 adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind -= 1
                    continue

            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
            self._site_time_evolution(dt, cur_orthog, nsites=nsites, compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites

        print('done right to left sweep')
        return self.ket


