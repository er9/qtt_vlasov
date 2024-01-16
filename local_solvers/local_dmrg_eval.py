import numpy as np
from abc import ABC

import helper_quimb
from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
# from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# from local_solvers_old.environments_v2 import Environment, EnvironmentSide
# from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
# from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
# from local_solvers_old.local_solver import SweepDirection
# from local_solvers_old.blocks import EnvironmentType, init_block, Block, BlockOperator, BlockVector, BlockVector_Cross, BlockOperator_Cross

# from gridTN_1D import GridTN1D
# import local_solvers.helper_tn as helper_tn

# from local_solvers_old.local_evaluator import get_mps_matching_inds
# from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

# from local_solvers_old.local_solver_v2 import *  # LocalSolver
# from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
# from local_solvers_old.helper_cross_v3 import compress_tens_list
from local_solvers.terms import Term, Term_DMRG, Term_Cross
from local_solvers.local_evaluator import *



def local_dmrg_evaluator(terms: Sequence['Term_DMRG'],
                          init_guess: Optional['qtn.MatrixProductState'] = None,
                          max_bond=None, combine_terms_func:Callable=None,
                          nsites=1, direction: SweepDirection = SweepDirection.RIGHT,
                          ):
    """ terms are sequences of ((A, B, ...), x)
        where A, B are operators or functions that are applied to x in
        the order A(B(x))

        algorithm: compute rdm for each term
        sum together terms by summing rdms together (state-averaging method)
        projecting next site onto new rdm
    """
    init_guess = terms[0].ket.copy() if init_guess is None else init_guess

    solver = DMRGEvaluator(init_guess, terms, combine_terms_func=combine_terms_func,
                            direction=direction, max_bond=max_bond)
    solver.solve(nsites)

    # out_gtn = GridTN1D(ref_x.grid, solver.ket)
    # return out_gtn
    return solver.ket


class DMRGEvaluator(LocalEvaluator):

    # def __init__(self,
    #              ket_state: 'qtn.MatrixProductState',
    #              # bra_state: 'qtn.MatrixProductState',
    #              terms: Sequence[Union['Term', tuple[tuple, dict]]],
    #              max_bond: int =None,
    #              conv_tol: float=DEFAULT_CONV_TOL, max_iter: int=DEFAULT_MAX_ITER,
    #              max_tot_iter: int=DEFAULT_MAX_TOT_ITER, max_wrong_iter: int=DEFAULT_MAX_WRONG_ITER,
    #              copy_obj: 'LocalEvaluator' = None,
    #              combine_terms_func: 'Callable' = None,
    #              ):
    #
    #     if copy_obj:
    #         self.L = copy_obj.L
    #         self.ket = copy_obj.ket.copy()
    #         self.bra = copy_obj.bra.copy()
    #         new_terms = []
    #         for term in copy_obj.terms:
    #             new_term = term.copy()
    #             if term.bra is copy_obj.bra:
    #                 new_term.bra = self.bra
    #             new_terms += [new_term]
    #         self._terms = new_terms
    #
    #         self.conv_tol = copy_obj.conv_tol
    #         self.max_tot_iter = copy_obj.max_tot_iter
    #         self.max_iter = copy_obj.max_iter
    #         self.max_wrong_iter = copy_obj.max_wrong_iter
    #         self.max_bond = copy_obj.max_bond
    #
    #         self.combine_terms_func = combine_terms_func
    #
    #         ## results
    #         self.err = copy_obj.err
    #         self.is_conv = copy_obj.is_conv
    #         self.cur_orthog = copy_obj.cur_orthog
    #
    #     else:
    #         self.L = ket_state.L
    #         self.ket = ket_state
    #         self.bra = ket_state.conj(mangle_inner=True)
    #         self._terms = terms
    #         self.terms = terms      ## updates bra in the terms?
    #
    #         ## convergence params
    #         self.conv_tol = conv_tol
    #         self.max_iter = max_iter
    #         self.max_tot_iter = max_tot_iter
    #         self.max_wrong_iter = max_wrong_iter
    #         self.max_bond = max_bond
    #
    #         ## callables for combining operators, targets. defaults to summation
    #         self.combine_terms_func = combine_terms_func
    #
    #         ## results
    #         self.err = np.inf  # self.check_err()
    #         self.is_conv = False  # np.abs(self.err) < conv_tol
    #         self.cur_orthog = None


    @property
    def term_class(self) -> Union[Type['Term_DMRG'], Type['Term_Cross']]:
        return Term_DMRG

    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # site_l = left_site_pos
        # site_r = left_site_pos + nsites - 1
        # inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        # inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        direction = self.direction
        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # ref_x_data = self.vecs[0]

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
        #     x_bond = self.ket.bond(ix, ix2) if not at_end else None
        # else:
        #     x_bond = self.ket.bond(ix2, ix2 + direction) if not at_end else None
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
        for term in self.terms:

            # site_tens = vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # b_to_k = vec_block.projected_bra_to_ket(left_site_pos, nsites)
            # site_tens.reindex(b_to_k, inplace=True)
            #
            # for env in op_blocks:
            #     if isinstance(env, BlockOperator_DMRG):  ## operator is not None, apply projected operator
            #         eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
            #         site_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
            #         site_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
            #         site_tens.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            #     elif isinstance(env, (list, tuple, np.ndarray)):  ## polynomial function
            #         envB, env, coeffs = env
            #         eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
            #         eff_op_B = envB.get_projected(left_site_pos, nsites, return_combined=False)
            #         # print('envB bra is bra', envB.bra is self.bra)
            #         # print('env bra is None', env.bra is None)
            #         # print('env ket is target ket', env.ket is vec_block.ket)
            #         # print('env ket is target ket', envB.ket is vec_block.ket)
            #         # print('vec block ket is self ket', self.ket is vec_block.ket)
            #         # print('vec block bra is None', vec_block.bra is None)
            #         pow_terms = []
            #         # ## zero
            #         # if op[0] != 0:
            #         #     pow_term = qtn.Tensor(np.ones(site_tens.shape) * op[0], inds=site_tens.inds)
            #         #     pow_terms += [pow_term]
            #         ## power of 1+
            #         for power, coeff in enumerate(coeffs):
            #             pow_term = site_tens.copy()
            #             # eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
            #             if coeff != 0:
            #                 it = 0
            #                 while it < power - 1:
            #                     print('it', it)
            #                     it += 1
            #                     pow_term = qtn.tensor_contract(pow_term, *eff_op.tensors)
            #                     pow_term.modify(apply=lambda x: x * 10 ** eff_op.exponent)
            #                     pow_term.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            #                 ### last power
            #                 print('last power', it)
            #                 pow_term = qtn.tensor_contract(pow_term, *eff_op_B.tensors)
            #                 pow_term.modify(apply=lambda x: x * 10 ** eff_op_B.exponent)
            #                 pow_term.reindex(env.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            #
            #                 pow_coeff = pow_term.copy()
            #                 pow_coeff.modify(apply=lambda data: data * coeff)
            #                 pow_terms += [pow_coeff]
            #         site_tens = helper_tn.sum_tens(pow_terms)
            #         # site_tens.reindex(env.projected_bra_to_ket(ix, 1), inplace=True)
            #     elif env is not None:  ## perform elemental operation
            #         site_data = site_tens.data
            #         site_data = env(site_data)
            #         site_tens = site_tens.copy()
            #         site_tens.modify(data=site_data)
            #     else:
            #         pass

            site_tens = term.get_evaluated_site(left_site_pos, nsites)
            site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)

            ### decimation
            if at_end:

                if tot_site is None:
                    tot_site = site_tens
                else:
                    tot_site = helper_quimb.add_tensors(tot_site, site_tens, inplace=True)

            else:  # if not at_end:
                # rdm = term.get_evaluated_rdm(left_site_pos, nsites, direction=direction)

                ref_ket = term.vec_block.ket
                # horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
                if nsites == 1:
                    x_bond = ref_ket.bond(ix, ix2) if not at_end else None
                else:
                    x_bond = ref_ket.bond(ix2, ix2 + direction) if not at_end else None

                ## bonds to not contract over
                ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
                new_bra_iso = [k + '_' for k in ket_iso]
                #
                ## compute rdm of evaluated site
                site_conj = site_tens.conj()
                site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)

                # rdm = qtn.tensor_contract(site_conj, site_tens)
                # print('rdm original', rdm)

                rdm = term.get_evaluated_rdm(left_site_pos, nsites, direction,
                                             transpose_inds=[*ket_iso, *new_bra_iso],
                                             site_tens=site_tens)
                # print('rdm new', rdm)

                if tot_rdm is None:
                    tot_rdm = rdm
                else:
                    tot_rdm = helper_quimb.add_tensors(tot_rdm, rdm)

                if tot_site is None:
                    tot_site = site_tens
                else:
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
            current_x.transpose_like(tot_site, inplace=True)
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
            print('site err', site_err)

        else:
            # print('at end', at_end)
            current_x.transpose_like(tot_site, inplace=True)
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
            print('site err', site_err)
            x_eff = tot_site

        return x_eff, site_err


    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('update 1 site', site_i.norm(), direction)
        if direction == SweepDirection.RIGHT:
            ind1 = i
            ind2 = i + 1 if i < self.L - 1 else None
            ## canonicalize ket to next site
            if i < self.L - 1:
                self.ket.cut_orthog = ind1 + direction
                # self.cur_orthog = ind1 + direction

        else:
            ind1 = i
            ind2 = (i - 1) if i > 0 else None
            ## canonicalize ket to next site
            if i > 0:
                # self.cur_orthog = ind1 + direction
                self.ket.cut_orthog = ind1 + direction

        if ind2 is None:  ## is at_end
            ## update ket
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ind1)
            # site_i = site_i.reindex(ket_to_bra)
            # site_i.transpose_like(self.bra[ind1], inplace=True)
            # self.bra[ind1].modify(data=site_i.data.conj())

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

            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix)
            # self.bra[ix].transpose(*[ket_to_bra[ind] for ind in self.ket[ix].inds], inplace=True)
            # self.bra[ix].modify(data=eigvec.data.conj())
            # ket_to_bra = get_mps_matching_inds(self.ket, self.bra, ix2)
            # self.bra[ix2].transpose(*[ket_to_bra[ind] for ind in self.ket[ix2].inds], inplace=True)
            # self.bra[ix2].modify(data=next_site.data.conj())

            ## update next site in blocks; only target blocks need to be adjusted. can do directly with decimate
            self.update_blocks(ix, direction=direction)
            # for (op_blocks, vec_block) in self.term_blocks:
            #     vec_block.decimate(ix, direction=direction, new_ket_site=eigvec)
            #     vec_block.extend_env(ix, direction)
            #
            #     for env in op_blocks:
            #         if isinstance(env, BlockOperator):
            #             env.extend_env(ix, direction)
            #         elif isinstance(env, tuple):
            #             envB, env, coeffs = env
            #             envB.extend_env(ix, direction)
            #             env.extend_env(ix, direction)
            # self.cur_orthog = ind2
        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        raise NotImplementedError




