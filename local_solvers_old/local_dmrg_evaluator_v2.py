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
# from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
from local_solvers_old.local_solver import SweepDirection
# from local_solvers_old.blocks import EnvironmentType, init_block, Block, BlockOperator, BlockTarget

from gridTN_1D import GridTN1D

from local_solvers_old.local_evaluator import get_mps_matching_inds
from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress
from local_solvers_old.blocks import BlockDiagKet_DMRG, BlockVector_DMRG, BlockOperator_DMRG
from local_solvers_old.local_solver_v2 import *  # LocalSolver
from local_solvers_old.local_solver_v2 import DMRGSolver  #, canonize, canonize_tens_list, cross_compress

import helper_quimb

def canonize(mps: Union['MPSType', 'MPOType'], i=None, scale=True, form='right', cur_orthog=None, bra=None):
    helper_quimb.canonize(mps, i=i, scale=scale, form=form, cur_orthog=None, bra=None)
    return mps

def local_dmrg_evaluator(*terms: tuple[ Sequence[Union[GridTN1D, Callable[[np.ndarray], np.ndarray]]],
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

    mps_terms = []
    for ops, x in terms:
        new_ops = [op.data if isinstance(op, GridTN1D) else op for op in ops]
        mps_terms += [(new_ops, x.data.copy())]

    solver = DMRGEvaluator(L, mps_terms, combine_terms_func=combine_terms_func,
                           init_direction=direction, max_bond=max_bond)
    solver.solve(nsites)

    out_gtn = GridTN1D(ref_x.grid, solver.ket)
    return out_gtn


class DMRGEvaluator:

    def __init__(self, L: int,
                 terms: Sequence[tuple[Sequence[Union['MPOType','Callable','str']], 'MPSType']],
                 ket_state: 'qtn.MatrixProductState' = None,
                 in_ind: int = 0, out_ind: int = 0,
                 combine_terms_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 virtual: bool = False,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 mps_inds: Optional[Sequence[int]] = None,
                 max_bond=None, is_H=False,
                 init_direction=SweepDirection.RIGHT,
                 cur_orthog = None,
                 copy_obj: 'DMRGEvaluator' = None,
                 enrich_rank=0):

        if copy_obj is not None:

            self.in_ind, self.out_ind = copy_obj.in_ind, copy_obj.out_ind
            self.L = copy_obj.L
            self.cur_orthog = copy_obj.cur_orthog
            start = self.cur_orthog
            self.ket = copy_obj.ket.copy()
            self.bra = copy_obj.bra.copy()
            self.vecs = [trg.copy() for trg in copy_obj.vecs]
            self.direction: SweepDirection = copy_obj.direction

            terms = []
            for (op_blocks, trg_block) in copy_obj.term_blocks:
                ops = []
                for op_block in op_blocks:
                    if isinstance(op_block, BlockOperator):
                        ops += [op_block.operator]
                    elif isinstance(op_block, tuple):
                        ops += [op_block[1]]   ## polynomial coeffs
                    else:
                        ops += [op_block]
                terms += [(ops, trg_block.bra)]


            # self.is_H = copy_obj.is_H
            # self.max_bond = copy_obj.max_bond

            # self.mps_inds = copy_obj.mps_inds
            self.err = copy_obj.err
            self.is_conv = copy_obj.is_conv
            self.max_iter = copy_obj.max_iter
            self.max_tot_iter = copy_obj.max_tot_iter
            self.max_wrong_iter = copy_obj.max_wrong_iter
            self.conv_tol = copy_obj.conv_tol

        else:
            self.L = L
            self.in_ind, self.out_ind = in_ind, out_ind

            start = (0 if init_direction == SweepDirection.RIGHT else L - 1) if cur_orthog is None else cur_orthog
            self.cur_orthog = cur_orthog

            self.combine_terms_func = helper_tn.sum_tens  # if combine_terms_func is None else combine_terms_func
            self.direction: SweepDirection = init_direction
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

            ref_target = None
            self.vecs = []  # , self.targets_conj = [], []
            for ops, vec in terms:
                vec = vec if virtual else vec.copy()
                canonize(vec, start)
                # vec[start].modify(apply=lambda data: data * 10 ** vec.exponent)
                # vec.exponent = 0.0

                # vec_conj = vec.conj()

                self.vecs += [vec]

                if ref_target is None:
                    ref_target = vec
                else:
                    helper.match_inner_inds(vec, ref_target, inplace=True)

            ## define output ket and bra (an initial guess)
            if ket_state is not None:
                self.ket = ket_state
                self.ket = canonize(ket_state, start)
                # self.ket[start].modify(apply=lambda data: data * 10 ** self.ket.exponent)
                # self.ket.exponent = 0.0
            else:
                self.ket = ref_target.copy()
                # print('ket exp', self.ket.exponent)
                # self.ket = qtn.MPS_rand_state(self.L, 16,
                #                               site_ind_id=ref_target.site_ind_id, site_tag_id=ref_target.site_tag_id)
                # helper.match_inner_inds(self.ket, ref_target, inplace=True)
                # canonize(self.ket, start)

            self.bra = self.ket.conj()
            for tens in self.bra:
                tens.modify(inds=tuple([ind + '_' for ind in tens.inds]))
            self.bra._site_ind_id = self.bra.site_ind_id + '_'

        term_blocks: list[list[Union[BlockOperator_DMRG, str, Callable]], BlockVector_DMRG] = []
        for vec, (ops, _) in zip(self.vecs, terms):
            op_blocks: Sequence[Optional['Block']] = []

            target_use_tot_bra = True

            # target_conj = target.conj()
            # for ix in range(self.L):
            #     tens = target_conj[ix]
            #     tens.reindex({ind: ind + '_' for ind in tens.inds if ind != target_conj.site_ind(ix)}, inplace=True)

            # targets += [target]
            # target_conjs += [target_conj]

            for op in ops:    ### will be summed together
                if isinstance(op, qtn.MatrixProductOperator):
                    op_blocks += [BlockOperator_DMRG(op.L, vec, self.bra, operator=op,
                                                     anc_env_left=None, anc_env_right=None,
                                                     # mps_inds=self.mps_inds,
                                                     cur_orthog=start)]
                    target_use_tot_bra = False
                elif isinstance(op, (tuple, list, np.ndarray)):
                    op_blocks += [(BlockDiagKet_DMRG(self.L, vec, bra=self.bra, cur_orthog=start),
                                   BlockDiagKet_DMRG(self.L, vec, cur_orthog=start), op)]
                    target_use_tot_bra = False
                else:
                    op_blocks += [op]

            vec_conj = self.bra if target_use_tot_bra else None
            target_block = BlockVector_DMRG(vec.L, vec, vec_conj,
                                            anc_env_left=None, anc_env_right=None,
                                            cur_orthog=start,
                                            )
            term_blocks += [(op_blocks, target_block)]

        # print('term blocks', term_blocks)
        self.term_blocks: Sequence[tuple[Sequence[BlockOperator], BlockVector]] = term_blocks
        # exit()


    # def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:
    #
    #     site_inds = self.mps_inds[left_site_pos: left_site_pos + nsites]
    #     direction = self.direction
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #
    #     ref_x_data = self.vecs[0]
    #
    #     if nsites == 1:
    #         ix = left_site_pos
    #         ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1
    #     elif nsites == 2:
    #         if direction == SweepDirection.RIGHT:
    #             ix, ix2 = left_site_pos, left_site_pos + 1
    #         else:
    #             ix2, ix = left_site_pos, left_site_pos + 1
    #     else:
    #         raise ValueError
    #
    #     print('ix, ix2', ix, ix2)
    #
    #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
    #     current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
    #
    #     ### compute new site
    #     tot_rdm = None
    #     tot_site = None
    #     for (op_blocks, vec_block) in self.term_blocks:
    #         # site_tens = x.data[ix]
    #         site_tens = vec_block.get_projected(ix, nsites, return_combined=True)
    #         b_to_k = vec_block.projected_bra_to_ket(ix, nsites)
    #         site_tens.reindex(b_to_k, inplace=True)
    #         print('site tens', site_tens)
    #         # site_tens = qtn.tensor_contract(*[vec_block.ket[i] for i in site_inds])
    #         # site_tens.modify(apply=lambda data: data * 10**vec_block.exponent)
    #
    #         for env in op_blocks:
    #             if isinstance(env, BlockOperator_DMRG):  ## operator is not None, apply projected operator
    #                 eff_op = env.get_projected(ix, nsites, return_combined=False)
    #                 site_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
    #                 site_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
    #                 site_tens.reindex(env.projected_bra_to_ket(ix, nsites), inplace=True)
    #             elif isinstance(env, (list, tuple, np.ndarray)):  ## polynomial function
    #                 env, coeffs = env
    #                 eff_op = env.get_projected(ix, nsites, return_combined=False)
    #                 pow_terms = []
    #                 # ## zero
    #                 # if op[0] != 0:
    #                 #     pow_term = qtn.Tensor(np.ones(site_tens.shape) * op[0], inds=site_tens.inds)
    #                 #     pow_terms += [pow_term]
    #                 ## power of 1+
    #                 for power, coeff in enumerate(coeffs):
    #                     pow_term = site_tens.copy()
    #                     eff_op = env.get_projected(ix, nsites, return_combined=False)
    #                     if coeff != 0:
    #                         it = 0
    #                         while it < power:
    #                             it += 1
    #                             pow_term = qtn.tensor_contract(pow_term, *eff_op.tensors)
    #                             pow_term.modify(apply=lambda x: x * 10 ** eff_op.exponent)
    #                             pow_term.reindex(env.projected_bra_to_ket(ix, nsites), inplace=True)
    #                         pow_coeff = pow_term.copy()
    #                         pow_coeff.modify(apply=lambda data: data * coeff)
    #                         pow_terms += [pow_coeff]
    #                 site_tens = helper_tn.sum_tens(pow_terms)
    #                 # site_tens.reindex(env.projected_bra_to_ket(ix, 1), inplace=True)
    #             elif env is not None:  ## perform elemental operation
    #                 site_data = site_tens.data
    #                 site_data = env(site_data)
    #                 site_tens = site_tens.copy()
    #                 site_tens.modify(data=site_data)
    #             else:
    #                 pass
    #
    #         if tot_site is None:
    #             tot_site = site_tens
    #         else:
    #             tot_site = helper_quimb.add_tensors(tot_site, site_tens, inplace=True)
    #
    #     if not at_end:
    #         # horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
    #         if nsites == 1:
    #             x_bond = ref_x_data.bond(ix, ix2) if not at_end else None
    #         else:
    #             x_bond = ref_x_data.bond(ix2, ix2 + direction) if not at_end else None
    #
    #         ## bonds to not contract over
    #         # ket_iso = [ind for ind in ref_x_data[ix].inds if ind != x_bond]
    #         ket_iso = [ind for ind in tot_site.inds if ind != x_bond]
    #         new_bra_iso = [k + '_' for k in ket_iso]
    #
    #         # print('x bond', x_bond)
    #         # print('ket iso', ket_iso)
    #         # print('bra iso', new_bra_iso)
    #
    #         ## compute rdm
    #         site_conj = tot_site.conj()
    #         site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #
    #         tot_rdm = qtn.tensor_contract(site_conj, tot_site)
    #         # rdm = qtn.tensor_contract(site_conj, site_tens)
    #         # if tot_rdm is None:
    #         #     tot_rdm = rdm
    #         # else:
    #         #     tot_rdm = helper_quimb.add_tensors(tot_rdm, rdm)
    #
    #     if not at_end:
    #         print('tot rdm', tot_rdm)
    #
    #         ## eigendecomposition to obtain canonical form
    #         tot_rdm.transpose(*ket_iso, *new_bra_iso, inplace=True)
    #         nb = len(ket_iso)
    #
    #         denmat_shape = tot_rdm.shape
    #         tens_shape = denmat_shape[:nb]
    #         sq_shape = np.prod(denmat_shape[:nb])
    #
    #         eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))
    #
    #         sort_inds = np.argsort(np.abs(eigval))[::-1]
    #         ev_max = eigval[sort_inds[0]]
    #         new_sort_inds = []
    #         for si in sort_inds:
    #             if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
    #                 new_sort_inds += [si]
    #             else:
    #                 break
    #         sort_inds = new_sort_inds
    #
    #         if self.max_bond is not None:
    #             eigvec = eigvec[:, sort_inds[:self.max_bond]]
    #         else:
    #             eigvec = eigvec[:, sort_inds]
    #
    #         eigvec = eigvec.reshape(*tens_shape, -1)
    #         eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (x_bond + '_',))
    #
    #         x_eff = eigvec
    #         site_err = 0.0
    #
    #     else:
    #         print('at end', at_end)
    #         x_eff = tot_site
    #         site_err = np.linalg.norm(tot_site.data - current_x.data)
    #
    #     print('current x', current_x)
    #     print('x_eff', x_eff)
    #     current_x.transpose_like(x_eff, inplace=True)
    #     # site_err = np.nan
    #     # site_err = np.linalg.norm( x_eff.data - current_x.data * 10 ** (-self.ket.exponent) )
    #     # self.err += site_err
    #
    #     return x_eff, site_err



    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:

        site_inds = self.mps_inds[left_site_pos: left_site_pos + nsites]
        direction = self.direction
        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        ref_x_data = self.vecs[0]

        if nsites == 1:
            ix = left_site_pos
            ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1
        elif nsites == 2:
            if direction == SweepDirection.RIGHT:
                ix, ix2 = left_site_pos, left_site_pos + 1
            else:
                ix2, ix = left_site_pos, left_site_pos + 1
        else:
            raise ValueError

        # print('site solve', site_inds, ix, ix2, at_end)

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)

        # ### compute new site
        # if nsites == 1:
        #     x_bond = ref_x_data.bond(ix, ix2) if not at_end else None
        # else:
        #     x_bond = ref_x_data.bond(ix2, ix2 + direction) if not at_end else None
        #
        # ## bonds to not contract over
        # ket_iso = [ind for ind in current_x.inds if ind != x_bond]
        # new_bra_iso = [k + '_' for k in ket_iso]
        #
        # ## compute rdm
        # conj_x = current_x.conj()
        # conj_x.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
        # tot_rdm = qtn.tensor_contract(current_x, conj_x)

        tot_rdm = None
        tot_site = None
        for (op_blocks, vec_block) in self.term_blocks:
            # site_tens = x.data[ix]
            # site_tens = qtn.tensor_contract(*[vec_block.ket[i] for i in site_inds])
            # site_tens.modify(apply=lambda data: data * 10**vec_block.exponent)

            site_tens = vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            b_to_k = vec_block.projected_bra_to_ket(left_site_pos, nsites)
            site_tens.reindex(b_to_k, inplace=True)
            # print('site tens', site_tens)

            for env in op_blocks:
                if isinstance(env, BlockOperator_DMRG):  ## operator is not None, apply projected operator
                    eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
                    site_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
                    site_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
                    site_tens.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
                elif isinstance(env, (list, tuple, np.ndarray)):  ## polynomial function
                    envB, env, coeffs = env
                    eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
                    eff_op_B = envB.get_projected(left_site_pos, nsites, return_combined=False)
                    # print('envB bra is bra', envB.bra is self.bra)
                    # print('env bra is None', env.bra is None)
                    # print('env ket is target ket', env.ket is vec_block.ket)
                    # print('env ket is target ket', envB.ket is vec_block.ket)
                    # print('vec block ket is self ket', self.ket is vec_block.ket)
                    # print('vec block bra is None', vec_block.bra is None)
                    pow_terms = []
                    # ## zero
                    # if op[0] != 0:
                    #     pow_term = qtn.Tensor(np.ones(site_tens.shape) * op[0], inds=site_tens.inds)
                    #     pow_terms += [pow_term]
                    ## power of 1+
                    for power, coeff in enumerate(coeffs):
                        pow_term = site_tens.copy()
                        # eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
                        if coeff != 0:
                            it = 0
                            while it < power - 1:
                                print('it', it)
                                it += 1
                                pow_term = qtn.tensor_contract(pow_term, *eff_op.tensors)
                                pow_term.modify(apply=lambda x: x * 10 ** eff_op.exponent)
                                pow_term.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
                            ### last power
                            print('last power', it)
                            pow_term = qtn.tensor_contract(pow_term, *eff_op_B.tensors)
                            pow_term.modify(apply=lambda x: x * 10 ** eff_op_B.exponent)
                            pow_term.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

                            pow_coeff = pow_term.copy()
                            pow_coeff.modify(apply=lambda data: data * coeff)
                            pow_terms += [pow_coeff]
                    site_tens = helper_tn.sum_tens(pow_terms)
                    # site_tens.reindex(env.projected_bra_to_ket(ix, 1), inplace=True)
                elif env is not None:  ## perform elemental operation
                    site_data = site_tens.data
                    site_data = env(site_data)
                    site_tens = site_tens.copy()
                    site_tens.modify(data=site_data)
                else:
                    pass

            site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)

            ### decimation
            if at_end:
                if tot_site is None:
                    tot_site = site_tens
                else:
                    print('add tot_site')
                    tot_site = helper_quimb.add_tensors(tot_site, site_tens, inplace=True)

            else:  # if not at_end:
                # horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
                if nsites == 1:
                    x_bond = ref_x_data.bond(ix, ix2) if not at_end else None
                else:
                    x_bond = ref_x_data.bond(ix2, ix2 + direction) if not at_end else None

                # if direction == SweepDirection.RIGHT:
                #     x_bond = ref_x_data.bond(ix, ix + 1) if not at_end else None
                # else:
                #     x_bond = ref_x_data.bond(ix, ix - 1) if not at_end else None

                ## bonds to not contract over
                ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
                new_bra_iso = [k + '_' for k in ket_iso]

                ## compute rdm
                site_conj = site_tens.conj()
                site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)

                rdm = qtn.tensor_contract(site_conj, site_tens)
                if tot_rdm is None:
                    tot_rdm = rdm
                else:
                    print('add tot rdm')
                    tot_rdm = helper_quimb.add_tensors(tot_rdm, rdm)

                if tot_site is None:
                    tot_site = site_tens
                else:
                    print('add tot site')
                    tot_site = helper_quimb.add_tensors(tot_site, site_tens, inplace=True)

        if not at_end:
            # print('tot rdm', tot_rdm)

            ## eigendecomposition to obtain canonical form
            tot_rdm.transpose(*ket_iso, *new_bra_iso, inplace=True)
            nb = len(ket_iso)

            denmat_shape = tot_rdm.shape
            tens_shape = denmat_shape[:nb]
            sq_shape = np.prod(denmat_shape[:nb])

            # test = tot_denmat.data.reshape(sq_shape, sq_shape)
            # print('rho is H', np.linalg.norm(test - test.T.conj()))
            eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

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
                eigvec = eigvec[:, sort_inds[:self.max_bond]]
            else:
                eigvec = eigvec[:, sort_inds]

            eigvec = eigvec.reshape(*tens_shape, -1)
            eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (x_bond + '_',))

            x_eff = eigvec
            # site_err = 0.0
            current_x.transpose_like(tot_site, inplace=True)
            # current_x.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
            print('site err', site_err)

            # ## update next site on ket, bra
            # eigvec.transpose_like(self.ket[ix], inplace=True)
            # next_site = qtn.tensor_contract(eigvec.conj(), self.ket[ix], self.ket[ix2])
            # next_site.transpose_like(self.ket[ix2], inplace=True)
            # self.ket[ix].modify(data=eigvec.data)
            # self.ket[ix2].modify(data=next_site.data)
            #
            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix)
            # self.bra[ix].transpose(*[ket_to_bra[ind] for ind in self.ket[ix].inds], inplace=True)
            # self.bra[ix].modify(data=eigvec.data.conj())
            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix2)
            # self.bra[ix2].transpose(*[ket_to_bra[ind] for ind in self.ket[ix2].inds], inplace=True)
            # self.bra[ix2].modify(data=next_site.data.conj())
            #
            # ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
            # for (op_blocks, vec_block) in self.term_blocks:
            #     vec_block.decimate(ix, direction=direction) # , new_ket_site=eigvec)
            #     vec_block.extend_env(ix, direction)
            #
            #     for env in op_blocks:
            #         if isinstance(env, BlockOperator):
            #             env.extend_env(ix, direction)
            #         elif isinstance(env, tuple):
            #             env, coeffs = env
            #             env.extend_env(ix, direction)
        else:
            print('at end', at_end)
            current_x.transpose_like(tot_site, inplace=True)
            # current_x.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
            print('site err', site_err)
            x_eff = tot_site

            # if nsites == 1:
            #     x_ket = self.ket
            #     tot_site.transpose_like(x_ket[ix], inplace=True)
            #     self.ket[ix].modify(data=tot_site.data)
            # else:
            #     raise NotImplementedError

        # print('current x', current_x)
        # print('x_eff', x_eff)
        # current_x.transpose_like(x_eff, inplace=True)
        # site_err = np.nan
        # site_err = np.linalg.norm( x_eff.data - current_x.data * 10 ** (-self.ket.exponent) )
        # self.err += site_err

        return x_eff, site_err

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
            print('check orthog', helper_quimb.check_right_orthog(self.ket))
        else:
            print('check orthog', helper_quimb.check_left_orthog(self.ket))

        ## iterative solver
        it = 0
        err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        self.max_tot_iter = 10

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

                    print(L - nsites + 1, L, nsites)
                    print('L2R', i, nsites)
                    site_i, site_err = self._site_solve(i, nsites)
                    # print('site i', site_i.norm(), site_err)
                    if site_err > 10 ** 5:
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
                    # print('update R', inds)
                    left_site_pos = i - nsites + 1
                    print('R2L', i, nsites, left_site_pos)
                    site_i, site_err = self._site_solve(left_site_pos, nsites)
                    # print('site err', i, site_err, site_i.norm())
                    if site_err > 10 ** 5:
                        print('site err too large')
                        exit()
                    tot_err += site_err

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
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

            # plt.figure()
            # for (op_blocks, vec_block) in self.term_blocks:
            #     print(type(op_blocks[0]), type(vec_block))
            #     print('op ket is vec ket', op_blocks[0][0].ket is vec_block.ket)
            #     print('self ket is vec ket', self.ket is vec_block.ket)
            #     print('vec bra is self bra', vec_block.bra is self.bra)

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
            # pass
            self.ket = min_solver.ket
            self.term_blocks = min_solver.term_blocks
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            self.cur_orthog = min_solver.cur_orthog

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
        out = self.__class__(self.L, [], copy_obj=self)
        return out

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('update 1 site', site_i.norm(), direction)
        if direction == SweepDirection.RIGHT:
            ind1 = self.mps_inds[i]
            ind2 = self.mps_inds[i + 1] if i < self.L - 1 else None
            ## canonicalize ket to next site
            if i < self.L - 1:
                self.cur_orthog = ind1 + direction

        else:
            ind1 = self.mps_inds[i]
            ind2 = self.mps_inds[i - 1] if i > 0 else None
            ## canonicalize ket to next site
            if i > 0:
                self.cur_orthog = ind1 + direction

        if ind2 is None:   ## is at_end
            print('update 1 site at_end')
            ## update ket
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
            site_i = site_i.reindex(ket_to_bra)
            site_i.transpose_like(self.bra[ind1], inplace=True)
            self.bra[ind1].modify(data=site_i.data.conj())

        else:
            eigvec = site_i
            ix = ind1
            ix2 = ind2
            ## update next site on ket, bra
            eigvec.transpose_like(self.ket[ix], inplace=True)
            next_site = qtn.tensor_contract(eigvec.conj(), self.ket[ix], self.ket[ix2])
            next_site.transpose_like(self.ket[ix2], inplace=True)
            self.ket[ix].modify(data=eigvec.data)
            self.ket[ix2].modify(data=next_site.data)

            ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix)
            self.bra[ix].transpose(*[ket_to_bra[ind] for ind in self.ket[ix].inds], inplace=True)
            self.bra[ix].modify(data=eigvec.data.conj())
            ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix2)
            self.bra[ix2].transpose(*[ket_to_bra[ind] for ind in self.ket[ix2].inds], inplace=True)
            self.bra[ix2].modify(data=next_site.data.conj())

            ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
            for (op_blocks, vec_block) in self.term_blocks:
                vec_block.decimate(ix, direction=direction, new_ket_site=eigvec)
                vec_block.extend_env(ix, direction)

                for env in op_blocks:
                    if isinstance(env, BlockOperator):
                        env.extend_env(ix, direction)
                    elif isinstance(env, tuple):
                        envB, env, coeffs = env
                        envB.extend_env(ix, direction)
                        env.extend_env(ix, direction)

            self.cur_orthog = ind2

        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        print('update site', i, site_i)

        if direction == SweepDirection.RIGHT:
            ind1, ind2 = i, i + 1
            at_end = (ind2 == self.L - 1)
            # x_bond = self.ket.bond(ind1, ind1 - 1) if ind1 > 0 else None
        else:
            ind1, ind2 = i, i - 1
            at_end = (ind2 == 0)
            # x_bond = self.ket.bond(ind1, ind1 + 1) if ind1 < self.L-1 else None

        print('check ket orthog')
        helper_quimb.check_orthog(self.ket)

        ## decimation to get new site_i
        if not at_end:
            ind3 = ind2 + direction
            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
            # ket_to_bra.update(get_mps_matching_inds(self.ket, self.bra, ind2))
            # if x_bond is not None:
            #     ket_to_bra.pop(x_bond)
            # print('ket to bra', ket_to_bra)
            site_conj = site_i.conj()
            # site_conj = site_conj.reindex(ket_to_bra)
            tens3 = qtn.tensor_contract(site_conj, self.ket[ind1], self.ket[ind2], self.ket[ind3])
            helper_quimb.canonize_tens_list(tens3, site_i, inplace=True)
            # print('site_i', site_i, 'R', R)
            # site_i = qtn.tensor_contract(site_i, R)
            # print('new ket site', site_i)

        ## decompose eigvec into two sites, tensor contract ind1 site
        bond_ms, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
        bond_m = next(iter(bond_ms))
        ##
        tens1, tens2 = qtn.tensor_split(site_i, left_inds, absorb='right', bond_ind=bond_m,
                                        cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, max_bond=self.max_bond)

        ## update self.ket, self.bra
        tens1.transpose_like(self.ket[ind1], inplace=True)
        tens2.transpose_like(self.ket[ind2], inplace=True)
        self.ket[ind1].modify(data=tens1.data)
        self.ket[ind2].modify(data=tens2.data)

        ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
        self.bra[ind1].transpose(*[ket_to_bra[ind] for ind in self.ket[ind1].inds], inplace=True)
        self.bra[ind1].modify(data=tens1.data.conj())
        ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind2)
        self.bra[ind2].transpose(*[ket_to_bra[ind] for ind in self.ket[ind2].inds], inplace=True)
        self.bra[ind2].modify(data=tens2.data.conj())

        if not at_end:
            tens3.transpose_like(self.ket[ind3], inplace=True)
            self.ket[ind3].modify(data=tens3.data)

            ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind3)
            self.bra[ind3].transpose(*[ket_to_bra[ind] for ind in self.ket[ind3].inds], inplace=True)
            self.bra[ind3].modify(data=tens3.data.conj())

        self.cur_orthog = ind2

        ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
        if not at_end:
            for (op_blocks, vec_block) in self.term_blocks:
                vec_block.decimate(ind1, direction=direction, new_ket_site=tens1)
                vec_block.extend_env(ind1, direction)

                for env in op_blocks:
                    if isinstance(env, BlockOperator):
                        env.extend_env(ind1, direction)
                    elif isinstance(env, tuple):
                        envB, env, coeffs = env
                        env.extend_env(ind1, direction)
                        envB.extend_env(ind1, direction)

        return




    # def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: mps_site
    #     """
    #     print('update site', i, site_i)
    #
    #     if direction == SweepDirection.RIGHT:
    #         ind1, ind2 = i, i + 1
    #         at_end = (ind2 == self.L - 1)
    #         # x_bond = self.ket.bond(ind1, ind1 - 1) if ind1 > 0 else None
    #     else:
    #         ind1, ind2 = i, i - 1
    #         at_end = (ind2 == 0)
    #         # x_bond = self.ket.bond(ind1, ind1 + 1) if ind1 < self.L-1 else None
    #
    #     print('check ket orthog')
    #     helper_quimb.check_orthog(self.ket)
    #
    #     ## decimation to get new site_i
    #     if not at_end:
    #         # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
    #         # ket_to_bra.update(get_mps_matching_inds(self.ket, self.bra, ind2))
    #         # if x_bond is not None:
    #         #     ket_to_bra.pop(x_bond)
    #         # print('ket to bra', ket_to_bra)
    #         site_conj = site_i.conj()
    #         # site_conj = site_conj.reindex(ket_to_bra)
    #         R = qtn.tensor_contract(site_conj, self.ket[ind1], self.ket[ind2])
    #         # print('site_i', site_i, 'R', R)
    #         site_i = qtn.tensor_contract(site_i, R)
    #         # print('new ket site', site_i)
    #
    #     ## decompose eigvec into two sites, tensor contract ind1 site
    #     bond_ms, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
    #     bond_m = next(iter(bond_ms))
    #     ##
    #     tens1, tens2 = qtn.tensor_split(site_i, left_inds, absorb='right', bond_ind=bond_m + '_',
    #                                     cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, max_bond=self.max_bond)
    #
    #     ## update self.ket, self.bra
    #     tens1.transpose_like(self.ket[ind1], inplace=True)
    #     tens2.transpose_like(self.ket[ind2], inplace=True)
    #     self.ket[ind1].modify(data=tens1.data)
    #     self.ket[ind2].modify(data=tens2.data)
    #
    #     ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
    #     self.bra[ind1].transpose(*[ket_to_bra[ind] for ind in self.ket[ind1].inds], inplace=True)
    #     self.bra[ind1].modify(data=tens1.data.conj())
    #     ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind2)
    #     self.bra[ind2].transpose(*[ket_to_bra[ind] for ind in self.ket[ind2].inds], inplace=True)
    #     self.bra[ind2].modify(data=tens2.data.conj())
    #
    #     self.cur_orthog = ind2
    #
    #     ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
    #     if not at_end:
    #         for (op_blocks, vec_block) in self.term_blocks:
    #             vec_block.decimate(ind1, direction=direction, new_ket_site=tens1)
    #             vec_block.extend_env(ind1, direction)
    #
    #             for env in op_blocks:
    #                 if isinstance(env, BlockOperator):
    #                     env.extend_env(ind1, direction)
    #                 elif isinstance(env, tuple):
    #                     envB, env, coeffs = env
    #                     env.extend_env(ind1, direction)
    #                     envB.extend_env(ind1, direction)
    #
    #     return


    # def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: mps_site
    #     """
    #     print('update site', i, site_i)
    #
    #     if direction == SweepDirection.RIGHT:
    #         ind1, ind2 = i, i + 1
    #         at_end = (ind2 == self.L - 1)
    #     else:
    #         ind1, ind2 = i, i - 1
    #         at_end = (ind2 == 0)
    #
    #     print('ind1, ind2', ind1, ind2)
    #     print('at end?', at_end)
    #
    #     ## decompose eigvec into two sites, tensor contract ind1 site
    #     bond_ms, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
    #     bond_m = next(iter(bond_ms))
    #     ##
    #     ev1, ev2 = qtn.tensor_split(site_i, left_inds, absorb='right', bond_ind=bond_m + '_',
    #                                 cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, max_bond=self.max_bond)
    #     if not at_end:
    #         ev2 = qtn.tensor_contract(ev1.conj(), self.ket[ind1], self.ket[ind2])
    #     # ev2 = qtn.tensor_contract(ev1.conj(), self.ket[ind1], self.ket[ind2])
    #
    #     ## update self.ket, self.bra
    #     ev1.transpose_like(self.ket[ind1], inplace=True)
    #     ev2.transpose_like(self.ket[ind2], inplace=True)
    #     self.ket[ind1].modify(data=ev1.data)
    #     self.ket[ind2].modify(data=ev2.data)
    #
    #     ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
    #     self.bra[ind1].transpose(*[ket_to_bra[ind] for ind in self.ket[ind1].inds], inplace=True)
    #     self.bra[ind1].modify(data=ev1.data.conj())
    #     ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind2)
    #     self.bra[ind2].transpose(*[ket_to_bra[ind] for ind in self.ket[ind2].inds], inplace=True)
    #     self.bra[ind2].modify(data=ev2.data.conj())
    #
    #     self.cur_orthog = ind2
    #
    #     ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
    #     if not at_end:
    #         for (op_blocks, vec_block) in self.term_blocks:
    #             vec_block.decimate(ind1, direction=direction, new_ket_site=ev1)
    #             vec_block.extend_env(ind1, direction)
    #
    #             for env in op_blocks:
    #                 if isinstance(env, BlockOperator):
    #                     env.extend_env(ind1, direction)
    #                 elif isinstance(env, tuple):
    #                     envB, env, coeffs = env
    #                     env.extend_env(ind1, direction)
    #                     envB.extend_env(ind1, direction)
    #
    #     return


