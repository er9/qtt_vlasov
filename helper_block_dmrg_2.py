import numpy as np
import quimb

from setup.defaults import *
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2

from helper_dmrg import *
from helper_block_dmrg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gridTN import GridTN

""" implicit solver for Ax = b, where x, b are composed of multiple components
"""


class BaseSolver(Enum):
    LOCAL = LocalSolver
    DMRG = DMRGSolver
    LINEAR = LinearSolver
    LINEAR2 = helper_dmrg_2.LinearSolver


def implicit_solver(ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                    b_vecs: dict[int, Sequence['MPS_type']],
                    init_guess: dict[int, Sequence['MPS_type']] = None, nsites=2, verbose_output=False,
                    **solver_kwargs):
    solver = BlockedLinearSolver(ncomps, init_guess, b_vecs, A_mats, **solver_kwargs)
    solver.solve(nsites)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets


class BlockedLinearSolver2(BlockedLinearSolver):
    def __init__(self, ncomps, trial_state: dict[int, 'qtn.MatrixProductState'],
                 targets: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 is_H=True,
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

        ## key (i,j)
        ## -> operators   : [A_kj for all k]
        ## -> operators_H : [A.H_ik for all i]
        out_keys = targets.keys()
        operators2 = {key: [operators[ok, key[1]] for ok in out_keys] for key in operators.keys()}

        init_state = trial_state
        super(BlockedLinearSolver, self).__init__(ncomps, init_state, targets=targets, operators=operators2,
                                                  mps_inds=mps_inds, conv_tol=conv_tol,
                                                  max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter,
                                                  base_solver=BaseSolver.LINEAR2,
                                                  norm_env0_Rs=norm_env0_Rs, norm_env0_Ls=norm_env0_Ls,
                                                  ovlp_env0_Rs=ovlp_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
                                                  err_env0_Rs=err_env0_Rs, err_env0_Ls=err_env0_Ls,
                                                  solvers=solvers)

        for k, solver in self.solvers.items():
            solver.is_H = True  # is_H
            solver.solve_type = SolveMethod.CGDx  # solve_type

        self.solve_type = solve_type
        self.is_H = is_H

    # @profile
    def _site_solve(self, sites: Union[int, slice], ) -> dict[int, qtn.Tensor]:
        """
        sites: int or slice(start, stop, step)
        solve local A' x = b',  <xA|Ax> - <xA|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A^T A|x> = d/dT[i] d/dT[i]^* \sum_ijk <x_j|A.T_kj A_ki|x_i>
            where b' = d/dT[i]^* <x|A|b> = d/dT[i] \sum_ij <x_j|A.T_ij| b_i>
        """

        # print("SITE SOLVE", self.solve_type, sites)
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
        bra_inds += [self.bra.site_ind_id.format(i) for i in site_inds]

        bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        old_x_shape = {}
        for ki, ket in self.kets.items():
            ind_size_dict = {}
            if start > 0:
                ind_size_dict[ket.bond(start - 1, start)] = ket.bond_size(start - 1, start)
            if end < self.L - 1:
                ind_size_dict[ket.bond(end, end + 1)] = ket.bond_size(end, end + 1)
            old_x_shape[ki] = ind_size_dict

        A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
        for key, solver in self.solvers.items():
            ### AL * A * AR = d/dT*[i] d/dT[i] \sum_k <x_j|A.T_kj A_ki|x_i>
            A_effs_dict[key] = solver._get_A_effs(sites, return_sum=(not self.solve_type is SolveMethod.CGD))
            try:
                A_effs_dict[key].transpose(*bra_inds, *ket_inds, inplace=True)
            except AttributeError:
                pass  ## a list of tensor netowkrs

            ### BL * B * BR = d/dT*[s] \sum_i \sum_y <x_j |A.T_ji| b_i,y>
            ### solver (i,o) computes d/dT*[s] \sum_y <x_i |A.T_oi| b_o,y>
            b_eff_ij = solver._get_b_eff(sites)
            b_eff_ij.transpose(*bra_inds, inplace=True)
            if key[0] not in b_eff_dict:
                b_eff_dict[key[0]] = b_eff_ij
            else:
                b_eff_dict[key[0]].modify(apply=lambda x: x + b_eff_ij.data)
            # print('b eff size', key[0], b_eff_dict[key[0]].shape)

        def combine_vec_tens(vec_dict: dict[int, qtn.Tensor], keys=None):
            """ all tensors are transposed to a certain order
            """
            if keys is None:
                keys = vec_dict.keys()

            tot_vec = None
            for k in keys:
                if tot_vec is None:
                    tmp = vec_dict[k].data
                    tot_vec = np.reshape(tmp, -1)
                else:
                    tot_vec = np.append(tot_vec, np.reshape(vec_dict[k].data, -1))

            return tot_vec

        # @profile
        def combine_mat_tens(mat_dict: dict[(int, int), qtn.Tensor], keys_o: Sequence, keys_i: Sequence,
                             shapes_o: Sequence[int], shapes_i: Sequence[int]):
            """ all tensors are transposed to a certain order
            """
            tot_npts_o = np.sum(shapes_o)
            tot_npts_i = np.sum(shapes_i)

            tot_mat = np.zeros((tot_npts_o, tot_npts_i))

            for k, mat in mat_dict.items():
                # print('combine mat', k, mat.shape)
                o_idx, i_idx = keys_o.index(k[0]), keys_i.index(k[1])
                start_ind_o, end_ind_o = int(np.sum(shapes_o[:o_idx])), int(np.sum(shapes_o[:o_idx + 1]))
                start_ind_i, end_ind_i = int(np.sum(shapes_i[:i_idx])), int(np.sum(shapes_i[:i_idx + 1]))
                # print('o idx', o_idx, 'i_idx', i_idx)
                # print('shapes o', shapes_o, 'shapes i', shapes_i)
                # print('inds o', start_ind_o, end_ind_o)
                # print('inds i', start_ind_i, end_ind_i)

                mat_data = mat.data.reshape(shapes_o[o_idx], shapes_i[i_idx])  # <x|A|x>
                tot_mat[start_ind_o: end_ind_o, start_ind_i: end_ind_i] += mat_data
                ### APPARENTLY IS SLOW STEP
                # plt.figure()
                # plt.imshow(mat_data)
                # plt.colorbar()
                # plt.show()

            return tot_mat

        def extract_vec_tens(vec, keys, npts, ref_dict: dict[int, qtn.Tensor]):
            """ vector to dict of tensors
            """
            vec_dict = {}
            for i in range(len(keys)):
                start_ind, end_ind = int(np.sum(npts[:i])), int(np.sum(npts[:i + 1]))
                new_x = ref_dict[keys[i]].copy()
                new_x_data = vec[start_ind:end_ind].reshape(new_x.shape)
                new_x.modify(data=new_x_data)
                vec_dict[keys[i]] = new_x

            return vec_dict

        x_dict = {}
        for k, ket in self.kets.items():
            if isinstance(site_inds, (list, tuple)):
                ket_tensors = [ket[si] for si in site_inds]
            else:
                ket_tensors = ket[site_inds]
            x_tens = qtn.tensor_contract(*ket_tensors)
            x_tens.transpose(*ket_inds, inplace=True)
            x_dict[k] = x_tens

        for k, x_eff in x_dict.items():
            x_eff.transpose(*ket_inds, inplace=True)

        ## solve for T[i]
        new_x_dict = {}
        if len(A_effs_dict) == 0:
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
                new_x_dict[ik] = x_eff
            # x_eff = combined_b_eff.reindex(bra_to_ket_inds)
            # x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type is SolveMethod.CGD:

                in_keys = list(x_dict.keys())
                out_keys = list(b_eff_dict.keys())
                in_keys.sort(), out_keys.sort()
                # in_keys = out_keys = list(range(6))
                in_shape = [x_dict[k].size for k in in_keys]
                out_shape = [b_eff_dict[k].size for k in out_keys]

                x_vec = combine_vec_tens(x_dict, keys=in_keys)
                b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

                combined_ind_i, combined_ind_o = 'in_ind', 'out_ind'
                x_vec_tens = qtn.Tensor(data=x_vec, inds=(combined_ind_i,))
                b_vec_tens = qtn.Tensor(data=b_vec, inds=(combined_ind_o,))

                # @profile
                def compute_Ax(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                               x_vec_tens_: 'qtn.Tensor', out_inds=(combined_ind_o,), **kwargs) -> 'qtn.Tensor':

                    Ax_dict = {}
                    input_is_ket = out_inds == (combined_ind_o,)
                    if input_is_ket:
                        out_keys_ = out_keys  # b key ordering
                        bra_inds_ = bra_inds  # out = output inds
                        x_dict_ = extract_vec_tens(x_vec_tens_.data, in_keys, in_shape, x_dict)
                        ## tens have ket_inds
                    else:
                        out_keys_ = out_keys  # b key ordering
                        bra_inds_ = ket_inds  # out = input inds
                        x_dict_ = extract_vec_tens(x_vec_tens_.data, in_keys, in_shape, b_eff_dict)
                        ## tens have bra_inds

                    for (ooo, iii), A_eff_tns in A_effs_dict_.items():
                        tot_A_eff = None
                        ix_, ix_c = (ooo, iii) if input_is_ket else (iii, ooo)  ## out_index, contracted index

                        ## sum over A_eff|x> for A_eff in list of A_eff_tns
                        for A_eff_tn in A_eff_tns:
                            Ax_tens = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_[ix_c])
                            Ax_tens.modify(apply=lambda x: x * 10 ** A_eff_tn.exponent)
                            if tot_A_eff is None:
                                tot_A_eff = Ax_tens.transpose(*bra_inds_)
                            else:
                                Ax_tens.transpose_like(tot_A_eff, inplace=True)
                                tot_A_eff.modify(apply=lambda x: x + Ax_tens.data)

                        ### sum over ix_c
                        if ix_ not in Ax_dict:
                            Ax_dict[ix_] = tot_A_eff
                        else:
                            Ax_dict[ix_].modify(apply=lambda x: x + tot_A_eff.data)

                    Ax_vec = combine_vec_tens(Ax_dict, keys=out_keys_)
                    Ax_vec_tens = qtn.Tensor(data=Ax_vec, inds=out_inds)
                    return Ax_vec_tens

                def compute_xAx(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor',
                                **kwargs) -> Numeric:

                    x_dict_i = extract_vec_tens(x_vec_tens_i.data, in_keys, in_shape, x_dict)  # in inds
                    x_dict_o = extract_vec_tens(x_vec_tens_o.data, out_keys, out_shape, b_eff_dict)  # out inds
                    tot_xAx = 0.0
                    for (ooo, iii), A_eff_tns in A_effs_dict_.items():
                        for A_eff_tn in A_eff_tns:
                            # print('x', x_dict_i[iii], x_dict_o[ooo])
                            out = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_i[iii], x_dict_o[ooo])
                            out *= 10 ** A_eff_tn.exponent
                            tot_xAx += out
                    return tot_xAx

                bra_to_ket_inds_tmp = {**bra_to_ket_inds, 'out_ind': 'in_ind'}
                if False:  # self.is_H:
                    x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                        bra_to_ket_inds_tmp,
                                                                        contract_Ax=compute_Ax,
                                                                        contract_xAx=compute_xAx)
                else:
                    x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                        bra_to_ket_inds_tmp,
                                                                        contract_Ax=compute_Ax)
                # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))
                new_x_dict = extract_vec_tens(x_eff.data, in_keys, in_shape, x_dict)

            elif self.solve_type is SolveMethod.CGDx:

                in_keys = list(x_dict.keys())
                out_keys = list(b_eff_dict.keys())
                in_keys.sort(), out_keys.sort()
                in_shape = [x_dict[k].size for k in in_keys]
                out_shape = [b_eff_dict[k].size for k in out_keys]

                x_vec = combine_vec_tens(x_dict, keys=in_keys)
                b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

                A_mat = combine_mat_tens(A_effs_dict, out_keys, in_keys, out_shape, in_shape)

                if False:  # self.is_H:
                    x_eff_data, error = conjugate_gradient_descent(A_mat, b_vec, x=x_vec)
                else:
                    x_eff_data, error = conjugate_gradient_squared(A_mat, b_vec, x=x_vec)

                new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)

            else:  ## ALS optimzation
                in_keys = list(x_dict.keys())
                out_keys = list(b_eff_dict.keys())
                in_keys.sort(), out_keys.sort()
                in_shape = [x_dict[k].size for k in in_keys]
                out_shape = [b_eff_dict[k].size for k in out_keys]

                x_vec = combine_vec_tens(x_dict, keys=in_keys)
                b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

                A_mat = combine_mat_tens(A_effs_dict, keys_o=out_keys, keys_i=in_keys, shapes_o=out_shape,
                                         shapes_i=in_shape)

                x_eff_data = np.linalg.solve(A_mat, b_vec)
                new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)

        return new_x_dict

    # @profile
    def solve(self, nsites: int, init_direction=SweepDirection.RIGHT, conv_tol=0,
              skip_inds: set = None, opt_inds: set = None, verbose=False, **kwargs):
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
        # print('L', self.L, self.ket.L)
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(nsites, canonize=False)
        else:
            self._build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        print('dmrg self.max_bond', self.max_bond, self.ket.max_bond())
        if self.max_bond is None or self.ket.max_bond() < self.max_bond:
            print('getting error')
            err = self.check_err(dense=True)  ## true is actually faster for large MPO bond dim?
            print('init error dmrg', err, self.err)
        else:
            err = self.err

        conv_it, prev_err = 0, err

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = init_direction
        while err > conv_tol and conv_it < self.max_iter and it < self.max_tot_iter:

            it += 1

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites + 1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i, i + nsites)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({ix + iix for iix in range(nsites)}):
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds, self.L)
                        # print(opt_inds.intersection({ix + iix for iix in range(nsites)}))
                        if not opt_inds.intersection({self.get_mps_ind(i + iix) for iix in range(nsites)}):
                            # print('opt', i, ix, {self.get_mps_ind(i + iix) for iix in range(nsites)}, opt_inds)
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    # print('update L', inds)
                    site_i = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        # print('LR one site', site_i)
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        # print('LR two site', site_i)
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)
                    # print('err', i, self.check_err())

                self.canon_direction = EnvironmentSide.LEFT
                err = self.check_err(dense=True)

            else:
                ## left sweep
                for i in range(L - 1, nsites - 2, -1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i - nsites + 1, i + 1)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({self.get_mps_ind(i - iix) for iix in range(nsites)}):
                            self.right_canonize_site(i)  # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds)
                        # print(opt_inds.intersection({ix - iix for iix in range(nsites)}))
                        if not opt_inds.intersection({ix - iix for iix in range(nsites)}):
                            self.right_canonize_site(i)  # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    # print('update R', inds)
                    site_i = self._site_solve(inds)

                    ## update ket, bra
                    if nsites == 1:
                        # print('RL one site', site_i)
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        # print('RL two site', site_i)
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)
                    # print('err', i, self.check_err())

                self.canon_direction = EnvironmentSide.RIGHT
                err = self.check_err(dense=True)

            if verbose:
                print('err', it, err)
                # print('err np', self.check_err_np())
            direction *= -1  ## swaps sweep direction

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err:
                min_solver = self.copy(deep=True)
                ### COPY IS KIND OF EXPENSIVE
                # min_ket = self.ket.copy()
                min_err = err
            else:
                print('Warning: solve error went up', err, min_err)
                # helper.replace_mps(self._ket, old_solver.ket)
                # helper.replace_mps(self._bra, old_solver.bra)
                # # self.ket = old_solver.ket
                # # self.reinitialize_envs()
                # self.A_envsL = old_solver.A_envsL
                # self.A_envsR = old_solver.A_envsR
                # self.b_envsRs = old_solver.b_envsRs
                # self.b_envsLs = old_solver.b_envsLs
                # self.left_envs = old_solver.left_envs
                # self.right_envs = old_solver.right_envs
                #
                # print('back to err', self.check_err())
                # print('numpy err', self.check_err_np())
                # print('fail direction', direction*-1)
                # # return old_solver.ket
                # break
                ## if not breaking, would need to recanonicalize, recalc envs

        # conv = min_err < conv_tol

        ## revert to optimal results
        if err > min_err:
            self.solvers = {k: solver.copy() for k, solver in min_solver.solvers.items()}
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            # self._ket = min_solver.ket
            # self._bra = min_solver.bra
            # self.A_envsRs = min_solver.A_envsRs
            # self.A_envsLs = min_solver.A_envsLs
            # self.b_envsLs = min_solver.b_envsLs
            # self.b_envsRs = min_solver.b_envsRs
            # self.err = min_solver.err
            # self.is_conv = min_solver.is_conv
            # self.canon_direction = min_solver.canon_direction

        # print(self.kets)
        return self.kets, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)

    def check_err_np(self):
        """ check err using numpy """
        As, bs, xs = {}, {}, {}
        iis, oos = [], []
        for (oo, ii), solver in self.solvers.items():
            if self.num_operators > 0:
                A = np.zeros((2 ** self.L, 2 ** self.L))
                for op in solver.operators:
                    op_tens = op.contract(all) * 10 ** op.exponent
                    op_tens.transpose(*[op.upper_ind_id.format(i) for i in solver.mps_inds],
                                      *[op.lower_ind_id.format(i) for i in solver.mps_inds],
                                      inplace=True)
                    op_mat = op_tens.data.reshape(2 ** self.L, 2 ** self.L)
                    # A = np.dot(op_mat, A)
                    A += op_mat
            else:
                A = np.eye(2 ** self.L)
            As[oo, ii] = A

            if oo == ii:
                b = np.zeros(2 ** self.L)
                for target in solver.targets:
                    target_tens = target.contract(all) * 10 ** target.exponent
                    # print('target tens?', target_tens)
                    target_tens.transpose(*[target.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                    target_vec = target_tens.data.reshape(-1)
                    b = b + target_vec

                bs[oo] = b
                oos += [oo]

                x_tens = solver.ket.contract(all) * 10 ** solver.ket.exponent
                x_tens.transpose(*[solver.ket.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                x_vec = x_tens.data.reshape(-1)
                xs[ii] = x_vec
                iis += [ii]

        errors2 = {}
        for oo in oos:
            # Ax_i = {ii: np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis}
            # xAAx_ii = {(iii, ii): np.dot(Ax_i[iii], Ax_i[ii]) for ii, iii in np.ndindex(6, 6)}
            Ax = np.sum([np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis], axis=0)
            # print('xAAx', oo, iis, [np.linalg.norm(np.dot(As.get((oo, ii), 0), xs[ii])) ** 2 for ii in iis])
            errors2[oo] = np.linalg.norm(bs[oo] - Ax) ** 2
        # print('err2s', errors2)

        b2norms = {oo: np.linalg.norm(bs[oo]) ** 2 for oo in oos}
        err2 = np.sum([errors2[oo] for oo in oos]) / np.sum([b2norms[oo] for oo in oos])
        err = np.sqrt(err2)
        return err

    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
            for each output component:
                0 = \sum_i A_oi |x_i> - |b_o>
                \sum_i <A_oi x_i | A_oi x_i> - 2Re[<b_o|A_oi|x_i>] + <b_o|b_o>
        """
        # print('np err', self.check_err_np())
        dense = False
        if dense:
            raise NotImplementedError

        else:
            ## || |bo> - sum_i Aoi |xi> ||^2 =
            ## <bo|bo> - sum_i <bo | Aoi | xi> * 2 + sum_i,i' <xi'|Aoi'.T Aoi|xi>
            normAAs = {}  # key o: list [<A_oi x_i | A_oi x_i>]
            ovlps = {}  # key o: list [<b_o|A_oi|x_i>]
            b2_norms = {}  # key o

            if self.num_operators > 1:
                raise NotImplementedError

            def compute_cross_Ax(solver1: LocalSolver, solver2: LocalSolver):
                ### <solver 2 | solver 1>
                A1, x1 = solver1.operators[0].copy(), solver1.ket.copy()
                A2, x2 = solver2.operators_H[0].copy(), solver2.ket.conj()
                x2.site_ind_id = A2.upper_ind_id
                ovlp = qtn.TensorNetwork([A1, x1, A2, x2])
                exponent = ovlp.exponent
                ovlp = ovlp.contract_tags(all)
                ovlp = ovlp * 10 ** exponent
                return ovlp

            for (oo, ii), solver in self.solvers.items():

                # norm_Ax = solver._get_xAAx_()  ## includes exponent
                ovlp_b_tot = solver._get_xAb_()  ## includes exponent
                b_norm = solver.targets_norm  ## includes exponent

                if oo not in normAAs:
                    normAAs[oo] = {}

                ### \sum_i <A_oj x_j | A_oi x_i>
                norm_Ax = 0.0
                for (ooo, iii), solver_x in self.solvers.items():
                    if oo == ooo:  # and ii != iii:  # and oo != iii:
                        # print('iii:', oo, iii)
                        ## <solver | solver x>
                        cross_AA = compute_cross_Ax(solver_x, solver)  ## includes exponent
                        norm_Ax += cross_AA
                normAAs[oo][ii] = norm_Ax  # norm Ax summed over i (iii); ii is i'
                # print('saved oo', oo, 'ii', ii, normAAs[oo])

                ### \sum_i <A_oi x_i | b_o>
                if oo not in ovlps.keys():
                    ovlps[oo] = 2 * np.real(ovlp_b_tot)
                else:
                    ovlps[oo] += 2 * np.real(ovlp_b_tot)

                if oo not in b2_norms:
                    b2_norms[oo] = b_norm ** 2

            # for oo, normAAs_o in normAAs.items():
            #     print('norm AA ', oo, normAAs[oo].keys(), [norm_AA_i for ii, norm_AA_i in normAAs_o.items()])
            normAAs = {oo: np.sum([norm_AA_i for ii, norm_AA_i in normAAs_o.items()]) for oo, normAAs_o in
                       normAAs.items()}
            # print('norm AAs', normAAs)
            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err2_o = {oo: (normAAs[oo] + b2_norms[oo] - ovlps[oo]) for oo in b2_norms.keys()}
            b2_norm = np.sum([np.abs(b2norm) for oo, b2norm in b2_norms.items()])

            err2 = np.sum([np.abs(val) for oo, val in err2_o.items()]) / b2_norm
            err = np.sqrt(np.abs(err2))

        # print('err2s', err2_o)
        print('err', err, self.conv_tol)
        self.err = err
        self.is_conv = err < self.conv_tol
        return err
