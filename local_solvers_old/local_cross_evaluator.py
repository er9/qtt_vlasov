import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from local_solvers_old.environments_v2 import Environment, EnvironmentSide
from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
from local_solvers_old.local_solver import SweepDirection
from local_solvers_old.blocks import EnvironmentType, init_block, Block, BlockOperator, BlockVector, BlockVector_Cross, BlockOperator_Cross

from gridTN_1D import GridTN1D

from local_solvers_old.local_evaluator import get_mps_matching_inds
from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

from local_solvers_old.local_solver_v2 import *  # LocalSolver
from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
from local_solvers_old.helper_cross_v3 import compress_tens_list


def local_cross_evaluator(*terms: tuple[ Sequence[Union[GridTN1D, Callable[[np.ndarray], np.ndarray]]],
                                         GridTN1D ],
                          nsites=1, direction: SweepDirection.LEFT, max_bond=None,
                          combine_terms_func:Callable=None):
    """ terms are sequences of ((A, B, ...), x)
        where A, B are operators or functions that are applied to x in
        the order A(B(x))

        algorithm: compute rdm for each term
        sum together terms by summing rdms together (state-averaging method)
        projecting next site onto new rdm
    """
    opx, ref_x = terms[0]
    L = ref_x.L

    # ref_x_data: 'qtn.MatrixProductState' = ref_x.data
    # ref_x_conj: 'qtn.MatrixProductState' = ref_x.data.conj(mangle_inner=False)
    # for ix in range(L):
    #     tens = ref_x_conj[ix]
    #     tens.reindex({ind: ind + '_' for ind in tens.inds if ind != ref_x_conj.site_ind(ix)}, inplace=True)
    ## also need to adjust ancilliary bonds

    mps_terms = []
    for ops, x in terms:
        new_ops = [op.data if isinstance(op, GridTN1D) else op for op in ops]
        mps_terms += [(new_ops, x.data.copy())]

    solver = CrossEvaluator(mps_terms, combine_terms_func=combine_terms_func,
                            init_direction=direction, max_bond=max_bond)
    solver.solve(nsites)

    out_gtn = GridTN1D(ref_x.grid, solver.ket)
    return out_gtn

def apply_zipup(mps: 'GridTN1D', mpo: 'GridTN1D', max_bond=None) -> 'GridTN1D':

    new_mps = []

    site_ind_id = mps.data.site_ind_id
    mpo.data.lower_ind_id = site_ind_id + '_'
    mps.data.site_ind_id = site_ind_id + '_'
    mpo.data.upper_ind_id = site_ind_id

    new_bond, T2 = None, None
    new_ind_id = 'v{}'
    select_inds = {}

    ## init build
    for i in range(mps.L):

        mps_tens = mps[i]
        mpo_tens = mpo[i]
        mps_h_bond = mps.data.bond(i, i + 1)
        mpo_h_bond = mpo.data.bond(i, i + 1)
        phys_bond = site_ind_id.format(i)

        tens_list = [mps_tens, mpo_tens]
        inds = [phys_bond, mps_h_bond, mpo_h_bond]
        if T2 is not None:
            tens_list += [T2]
            inds = [new_bond] + inds
        new_tens = qtn.tensor_contract(*tens_list, output_inds=inds)

        # new_tens.transpose(*inds, inplace=True)
        if new_bond is None:
            shape1, shape2 = new_tens.shape[:1], new_tens.shape[1:]
            size1, size2 = new_tens.shape[0], np.prod(shape2)
        else:
            shape1, shape2 = new_tens.shape[:2], new_tens.shape[2:]
            size1, size2 = np.prod(shape1), np.prod(shape2)

        ## canonize new_tens / split new tens
        tens_data = new_tens.data.reshape(size1, size2)
        T1, T2, sel_inds = cross_compress(tens_data, max_bond=max_bond, return_inds=True)

        old_bond = new_bond
        new_bond = new_ind_id.format(i)
        inds = [phys_bond, new_bond]
        if old_bond is not None:
            inds = [old_bond] + inds
        T1 = qtn.Tensor(T1.reshape(*shape1, -1), inds=tuple(inds), tags=(mps.data.site_tag_id.format(i)))
        inds = [new_bond, mps_h_bond, mpo_h_bond]
        T2 = qtn.Tensor(T2.reshape(-1, *shape2), inds=tuple(inds))

        new_mps += [T1]
        select_inds[i] = sel_inds

    ## CRITICAL:  how to measure error to measure convergence though?
    ## if first sweep is without compression, then back sweep can be done as usual (which is...)
    ## iterate sweeps until convergence

    new_mps = qtn.TensorNetwork(new_mps, virtual=True)
    new_mps.view_as(qtn.MatrixProductState, inplace=True, L=mps.L, cyclic=False, site_ind_id=site_ind_id,
                    site_tag_id=mps.data.site_tag_id)
    new_gtn = mps.create_like(new_mps)

    return new_gtn


class CrossEvaluator:

    def __init__(self,
                 terms: Sequence[tuple[Sequence[Union['MPOType','Callable','str']], 'MPSType']],
                 ket_state: 'qtn.MatrixProductState' = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Optional[Sequence[int]] = None,
                 # op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 # target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 # A_blocks: Optional[Sequence['BlockOperator_Cross']] = None,
                 # b_blocks: Optional[Sequence['BlockTarget_Cross']] = None,
                 # combine_operators_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 combine_terms_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 # left_envs: Optional[Sequence['Environment']] = None,
                 # right_envs: Optional[Sequence['Environment']] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, is_H=False,
                 init_direction=SweepDirection.RIGHT,
                 cur_orthog = None,
                 copy_obj: 'CrossEvaluator' = None,
                 enrich_rank=0):

        if copy_obj is not None:

            self.in_ind, self.out_ind = copy_obj.in_ind, copy_obj.out_ind
            self.L = copy_obj.L
            self.cur_orthog = copy_obj.cur_orthog
            self.ket = copy_obj.ket.copy()
            self.ket_select_inds = {k: list(v) for k, v in copy_obj.ket_select_inds.items()}

            # terms = []
            # for ops_block, target in terms:
            #     ops = [op.operator if isinstance(op, BlockOperator_Cross) else op for op in ops_block]
            #     terms += [ops, target.ket]

            new_blocks = []
            for (ops, target) in copy_obj.term_blocks:
                ops_new = []
                target = target.bra.copy()
                target_select_inds = None
                for op in ops:
                    if isinstance(op, BlockOperator_Cross):
                        if target_select_inds is None:
                            target_select_inds = op.select_inds_ket
                        ops_new += [BlockOperator_Cross(op.L, target, None, operator=op.operator,
                                                       anc_env_left=op.anc_env_L, anc_env_right=op.anc_env_R,
                                                       select_inds_bra=self.ket_select_inds,
                                                       select_inds_ket=target_select_inds,
                                                       cur_orthog=self.cur_orthog)]
                    else:
                        ops_new += [op]
                new_blocks = [ops_new, target]
            self.term_blocks = new_blocks

            self.combine_terms_func = copy_obj.combine_terms_func
            self.direction: SweepDirection = copy_obj.direction

            self.is_H = copy_obj.is_H
            self.max_bond = copy_obj.max_bond

            self.mps_inds = copy_obj.mps_inds
            self.err = copy_obj.err
            self.is_conv = copy_obj.is_conv
            self.max_iter = copy_obj.max_iter
            self.max_tot_iter = copy_obj.max_tot_iter
            self.max_wrong_iter = copy_obj.max_wrong_iter
            self.conv_tol = copy_obj.conv_tol

        else:

            self.in_ind, self.out_ind = in_ind, out_ind
            # self.bra_select_inds = {}

            ref_target = terms[0][1]
            self.L = ref_target.L

            if cur_orthog is None:
                start = 0 if init_direction == SweepDirection.RIGHT else self.L - 1
                self.cur_orthog = None
            else:
                start = cur_orthog
                self.cur_orthog = cur_orthog

            self.ket = ket_state        ## an initial guess
            if ket_state is not None:
                self.ket, self.ket_select_inds = canonize(ket_state, start)
            # else:
            #     # self.ket = target.copy()
            #     self.ket = qtn.MPS_rand_state(self.L, 16,
            #                                   site_ind_id=ref_target.site_ind_id, site_tag_id=ref_target.site_tag_id)
            #     helper.match_inner_inds(self.ket, ref_target, inplace=True)
            #     _, self.ket_select_inds = canonize(self.ket, start)

            term_blocks: list[list[Union[BlockOperator_Cross, str, Callable]], TargetEnv_Cross] = []
            for ops, target in terms:
                op_blocks = []
                target, target_select_inds = canonize(target, start)
                helper.match_inner_inds(target, ref_target, inplace=True)

                target_use_ket_inds = True

                if self.ket is None:
                    self.ket = target.copy()
                    self.ket_select_inds = {k: list(v) for k, v in target_select_inds.items()}

                for op in ops:
                    if isinstance(op, qtn.MatrixProductOperator):
                        op_blocks += [BlockOperator_Cross(op.L, target, None, operator=op,
                                                          anc_env_left=None, anc_env_right=None,
                                                          select_inds_bra=self.ket_select_inds,
                                                          select_inds_ket=target_select_inds,
                                                          cur_orthog=start)]
                        target_use_ket_inds = False
                    else:
                        op_blocks += [op]

                sel_inds_dict = self.ket_select_inds if target_use_ket_inds else target_select_inds
                target_block = BlockVector_Cross(target.L, target, None,
                                                 anc_env_left=None, anc_env_right=None,
                                                 select_inds_bra=sel_inds_dict,
                                                 cur_orthog = start,
                                                 )
                term_blocks += [(op_blocks, target_block)]

            # print('get ket block')
            # self.ket_block = BlockTarget_Cross(self.ket.L, self.ket, None,
            #                                    anc_env_left=None, anc_env_right=None,
            #                                    select_inds_bra=self.ket_select_inds,
            #                                    cur_orthog = start
            #                                    )
            self.term_blocks = term_blocks
            self.combine_terms_func = helper_tn.sum_tens if combine_terms_func is None else combine_terms_func
            self.direction : SweepDirection = init_direction
            self.cur_orthog = start

            self.is_H = is_H
            self.max_bond = max_bond

            self.mps_inds = list(range(self.L)) if mps_inds is None else mps_inds
            self.err = np.inf
            self.is_conv = False
            self.max_iter = max_iter
            self.max_tot_iter = max_tot_iter
            self.max_wrong_iter = max_wrong_iter
            self.conv_tol = conv_tol


    # def canonize(self, i, cur_orthog=None):
    #     raise NotImplementedError


    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        # print('site solve', self.solve_type)
        # site_pos = list(range(self.L))[sites]
        site_inds = self.mps_inds[left_site_pos: left_site_pos + nsites]
        site_l = np.min(site_inds)
        site_r = np.max(site_inds)
        inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L-1 else []

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm()) #, current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        ### this is also the selected submatrix if self is in cross-canonical form
        term_site_tens = []
        for (ops, target) in self.term_blocks:
            site_tens = target.get_projected(left_site_pos, nsites, return_combined=True)
            site_tens.reindex(target.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            for op in ops:
                if isinstance(op, BlockOperator_Cross):
                    # print('target sel ket', target.select_inds_bra)
                    # print('left site', left_site_pos)
                    eff_op = op.get_projected(left_site_pos, nsites, return_combined=False)

                    # b_to_k = op.projected_bra_to_ket(left_site_pos, nsites)
                    # b_inds = b_to_k.keys()
                    # b_inds = [b for b in b_inds if b is not None]
                    # k_inds = [b_to_k[b] for b in b_inds]
                    # eff_op_tens = eff_op.contract(all, output_inds = list(b_inds) + k_inds)
                    # sq_shape = np.prod(eff_op_tens.shape[:eff_op_tens.ndim//2])
                    # plt.figure()
                    # plt.imshow(eff_op_tens.data.reshape(sq_shape, sq_shape))
                    # plt.show()

                    site_tens = qtn.tensor_contract(site_tens, *eff_op.tensors)
                    site_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
                    site_tens.reindex(op.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

                elif op is not None:       ## perform elemental operation
                    site_data = site_tens.data
                    # site_data = site_data * 10**x.exponent
                    site_data = op(site_data)
                    # site_data = site_data * 10**(-x.exponent)
                    site_tens = site_tens.copy()
                    site_tens.modify(data=site_data)
                    # site_tens.reindex(target.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
                else:
                    pass
                    # site_tens.reindex(target.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

            # if len(ops) == 0:
            #     site_tens.reindex(target.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

            term_site_tens += [site_tens]

        site_tens = self.combine_terms_func(term_site_tens)

        ## solve for T[i]
        # x_eff = b_eff.reindex(bra_to_ket_inds)   ### ket is not orthogonal
        x_eff = site_tens.copy()
        x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
        ## b_eff_exponent already in b_eff tensor

        ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
        ## but it should be canonical?
        tensors = [x_eff]
        # if site_l > 0:
        #     inv_env_l = self.ket_block.envs[site_l-1]
        #     if inv_env_l is not None:
        #         ## indeed, these terms aren't needed (except for ket to bra reindex)
        #         print('skipping inv env L')
        #         # inv_env_l = inv_env_l.copy()
        #         # inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
        #         tensors += [inv_env_l]
        # if site_r < self.L-1:
        #     inv_env_r = self.ket_block.envs[site_r+1]
        #     # print('inv_env_r', inv_env_r)
        #     if inv_env_r is not None:
        #         ## indeed, these lines aren't needed (except for ket to bra reindex)
        #         print('skipping inv env R')
        #         # inv_env_r = inv_env_r.copy()
        #         # inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
        #         tensors += [inv_env_r]

        x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
        x_eff.transpose(*(inds_l + [self.ket.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)
        # print('x eff data', x_eff.norm(), x_eff.data)

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm( x_eff.data - current_x.data * 10 ** (-self.ket.exponent) )
        # self.err += site_err

        return x_eff, site_err


    # @profile
    def solve(self, nsites: int, conv_tol=DEFAULT_CONV_TOL, verbose=False, **kwargs):
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
        verbose = True
        conv_tol = self.conv_tol if conv_tol == 0 else conv_tol
        L = self.ket.L
        # canon_site = 0 if self.direction == SweepDirection.RIGHT else L - 1
        # self.canonize(canon_site)
        if self.cur_orthog == 0:
            print('check orthog', check_right_orthog(self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        else:
            print('check orthog', check_left_orthog(self.ket, self.ket_select_inds, [self.ket.site_ind_id]))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        # if self.Z_enrich:
        #     self.initialize_Z(direction=(SweepDirection.LEFT if canon_site==0 else SweepDirection.RIGHT))

        # ## ovlp <f(x)|b>, <f(x)|f(x)> init envs
        # if self.direction == SweepDirection.RIGHT:
        #     self.build_all_envs_right(nsites, canonize=False)
        # else:
        #     self.build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        self.max_tot_iter = 1

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = self.direction
        while err > conv_tol and conv_it < self.max_iter \
                and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter:

            it += 1
            tot_err = 0.0

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites + 1):
                    # ix = self.get_mps_ind(i)
                    # inds = slice(i, i + nsites)

                    site_i, site_err = self._site_solve(i, nsites)
                    # print('site i', site_i.norm(), site_err)
                    if site_err > 10**5:
                        print('site error too large')
                        exit()
                    tot_err += site_err

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                        # self._update_1site_with_Z(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                        # self._update_2site_with_Z(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b blocks (left)
                    if i < L - 1:
                        self.update_blocks(i, direction)

                    # print('check L orthog', i)
                    # print(check_left_orthog(self.ket, self.select_inds_l, [self.ket.site_ind_id]))

                err = tot_err  # self.check_err(dense=True)
                # self.select_inds_r.clear()

            else:
                ## left sweep
                for i in range(L - 1, nsites - 2, -1):
                    # ix = self.get_mps_ind(i)
                    # inds = slice(i - nsites + 1, i + 1)
                    # print('update', i, inds)

                    # print('update R', inds)
                    left_site_pos = i - nsites + 1
                    site_i, site_err = self._site_solve(left_site_pos, nsites)
                    # print('site err', i, site_err, site_i.norm())
                    if site_err > 10**5:
                        print('site err too large')
                        exit()
                    tot_err += site_err

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                        # self._update_1site_with_Z(i, nsites, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                        # self._update_2site_with_Z(i, nsites, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    if i > 0:
                        self.update_blocks(i, direction)

                    # print('check R orthog', i)
                    # print(check_right_orthog(self.ket, self.select_inds_r, [self.ket.site_ind_id]))

                # self.canon_direction = EnvironmentSide.RIGHT
                err = tot_err  # self.check_err(dense=True)
                # self.select_inds_l.clear()

            self.err = err
            self.is_conv = err < self.conv_tol

            if verbose:
                print('err', it, self.max_iter, err)
                # print('err np', self.check_err_np())

            direction *= -1  ## swaps sweep direction
            self.direction = direction

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err:
                min_solver = self.copy()
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
            pass
            # self.ket = min_solver.ket
            # self.term_blocks = min_solver.term_blocks
            # self.err = min_solver.err
            # self.is_conv = min_solver.is_conv
            # self.cur_orthog = min_solver.cur_orthog

        print('num iter', it)

        return self.ket, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)

    def get_mps_ind(self, i):
        return self.mps_inds[i]

    def update_blocks(self, i: int, direction: SweepDirection):

        for op_blocks, target_block in self.term_blocks:
            for op_block in op_blocks:
                if isinstance(op_block, BlockOperator):
                    op_block.extend_env(i, direction=direction)
            target_block.extend_env(i, direction=direction)

        # self.ket_block.extend_env(i, direction=direction)

    def copy(self):
        out = self.__class__(None, copy_obj=self)
        return out

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
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
                # _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                #                                     [ind1, ind2], inplace=True)
                _, select_inds = compress_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                    [ind1, ind2], inplace=True, max_bond=self.max_bond)
                self.ket_select_inds[ind1] = select_inds[0]

                for ops, target_block in self.term_blocks:
                    # for op in ops:
                    #     print('op')
                    #     if isinstance(op, BlockOperator_Cross):
                    #         print('op ket inds', op.select_inds_ket)
                    #         print('target', target_block.select_inds_ket)

                    target = target_block.bra
                    # print('target', target)
                    # print('canonize', ind1, ind2)
                    # _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                    #                                     [ind1, ind2], inplace=True)
                    _, select_inds = compress_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                        [ind1, ind2], inplace=True, max_bond=self.max_bond)
                    # target_block.select_inds_ket[ind1] = select_inds[0]
                    for op in ops:
                        if isinstance(op, BlockOperator_Cross):
                            op.select_inds_ket[ind1] = select_inds[0]
                            # op.select_inds_ket[ind1 + 1] = []
                            if ind1 < op.L - 1:
                                op.select_inds_ket[ind1 + 1] = []
                            # print('check orthog')
                            # print('op', op.select_inds_ket)
                            # check_orthog(target, op.select_inds_ket, [target.site_ind_id])
                            # print('op ket = target', op.ket is target)
                            # print('op ket inds', op.select_inds_ket)
                            # print('target', target_block.select_inds_ket)

                self.cur_orthog = ind1 + direction

        else:
            ## update k
            ind1 = self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = self.mps_inds[i - 1]
                # _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                #                                     [ind1, ind2], inplace=True)
                _, select_inds = compress_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                    [ind1, ind2], inplace=True, max_bond=self.max_bond)
                self.ket_select_inds[ind1] = select_inds[0]

                for ops, target_block in self.term_blocks:
                    target = target_block.bra
                    # _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                    #                                     [ind1, ind2], inplace=True)
                    _, select_inds = compress_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                        [ind1, ind2], inplace=True, max_bond=self.max_bond)
                    # target_block.select_inds_ket[ind1] = select_inds[0]
                    for op in ops:
                        if isinstance(op, BlockOperator_Cross):
                            op.select_inds_ket[ind1] = select_inds[0]
                            if ind1 > 0:
                                op.select_inds_ket[ind1 - 1] = []
                            # print('check orthog', ind1)
                            # print('op', op.select_inds_ket)
                            # check_orthog(target, op.select_inds_ket, [target.site_ind_id])
                            # print('op ket = target', op.ket is target)

                self.cur_orthog = ind1 + direction

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
            self.ket_select_inds[ind1] = select_inds

            for ops, target_block in self.term_blocks:
                target = target_block.bra
                _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                # target_block.ket_select_inds[ind1] = select_inds[0]
                target_block.select_inds_ket[ind1] = select_inds[0]
                # target_block.bra_select_inds[ind1] = select_inds[0]

                for op in ops:
                    if isinstance(op, BlockOperator_Cross):
                        op.select_inds_ket[ind1] = select_inds[0]

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
            self.ket_select_inds[ind1] = select_inds

            for ops, target_block in self.term_blocks:
                target = target_block.bra
                _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                # target_block.ket_select_inds[ind1] = select_inds[0]
                target_block.select_inds_ket[ind1] = select_inds[0]

                for op in ops:
                    if isinstance(op, BlockOperator_Cross):
                        op.select_inds_ket[ind1] = select_inds[0]

        self.cur_orthog = ind2

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
            self.ket_select_inds[ind1] = select_inds[0]

            for ops, target_block in self.term_blocks:
                target = target_block.bra
                _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                target_block.ket_select_inds[ind1] = select_inds[0]

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                                                [ind1, ind2], inplace=True)
            self.ket_select_inds[ind1] = select_inds[0]

            for ops, target_block in self.term_blocks:
                target = target_block.bra
                _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                                                    [ind1, ind2], inplace=True)
                target_block.ket_select_inds[ind1] = select_inds[0]

        self.cur_orthog = ind2

        return




