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

from gridTN_1D import GridTN1D
import local_solvers.helper_tn as helper_tn

# from local_solvers_old.local_evaluator import get_mps_matching_inds
# from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

# from local_solvers_old.local_solver_v2 import *  # LocalSolver
# from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
# from local_solvers_old.helper_cross_v3 import compress_tens_list
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
from local_solvers.local_evaluator import *
from local_solvers import helper_cross




def local_cross_evaluator(terms: Sequence['Term_Cross'],
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

    solver = CrossEvaluator(init_guess, terms, combine_terms_func=combine_terms_func,
                            direction=direction, max_bond=max_bond)
    solver.solve(nsites)

    # out_gtn = GridTN1D(ref_x.grid, solver.ket)
    # return out_gtn
    return solver.ket


class CrossEvaluator(LocalEvaluator):

    def __init__(self,
                 ket_state: 'MPS',  # 'qtn.MatrixProductState',
                 # bra_state: 'qtn.MatrixProductState',
                 terms: Sequence['Term_Cross'],
                 # cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int =None,
                 conv_tol: float=DEFAULT_CONV_TOL, max_iter: int=DEFAULT_MAX_ITER,
                 max_tot_iter: int=DEFAULT_MAX_TOT_ITER, max_wrong_iter: int=DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'CrossEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 # ket_select_inds = None
                 ):

        super().__init__(ket_state, terms,
                         # cur_orthog=cur_orthog,
                         direction=direction,
                         max_bond=max_bond, conv_tol=conv_tol,
                         max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter, max_iter=max_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func)

        # if copy_obj:
        #     self.ket_select_inds = copy_obj.ket_select_inds
        # else:
        #     self.ket_select_inds = {} if ket_select_inds is None else ket_select_inds
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


    # @property
    # def terms(self) -> Sequence['Term_Cross']:
    #     return super().terms

    @property
    def term_class(self) -> Type['Term_Cross']:
        return Term_Cross

    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        site_l = left_site_pos
        site_r = left_site_pos + nsites - 1
        inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm()) #, current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        term_site_tens = []
        for term in self.terms:
            term_site_tens += [term.get_evaluated_site(left_site_pos, nsites)]

        # print('term site tens', term_site_tens, term_site_tens[0].norm())

        site_tens = self.combine_terms_func(term_site_tens)

        ## solve for T[i]
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

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm(x_eff.data - current_x.data * 10 ** (-self.ket.exponent))
        # print('site err', site_err)
        return x_eff, site_err

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('update 1 site', site_i.norm(), direction)
        if direction == SweepDirection.RIGHT:
            ## update ket
            ind1 = i  # self.mps_inds[i]
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            ## canonicalize ket to next site
            if i < self.L - 1:
                ind2 = i + 1  # self.mps_inds[i + 1]
                # _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                #                                     [ind1, ind2], inplace=True)
                _, select_inds = self.term_class.compress_tens_list([self.ket[ind1], self.ket[ind2]],
                                                                    site_ind_ids=[self.ket.site_ind_id], site_inds=[ind1, ind2],
                                                                    inplace=True, max_bond=self.max_bond)
                # self.ket_select_inds[ind1] = select_inds[0]
                self.ket.select_inds[ind1] = select_inds[0]

                for term in self.terms:
                    term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

                # for term in self.terms:
                #     target = term.vec_block.ket
                #     _, select_inds = term.compress_tens_list([target[ind1], target[ind2]],
                #                                              site_ind_ids =[target.site_ind_id], site_inds=[ind1, ind2],
                #                                              inplace=True, max_bond=self.max_bond)
                #
                #     for op_block in term.op_blocks:
                #         op_block.ket_select_inds[ind1] = select_inds[0]
                #         if ind1 < term.L - 1:
                #             op_block.ket_select_inds[ind1 + 1] = []


                # for ops, target_block in self.term_blocks:
                #     # for op in ops:
                #     #     print('op')
                #     #     if isinstance(op, BlockOperator_Cross):
                #     #         print('op ket inds', op.select_inds_ket)
                #     #         print('target', target_block.select_inds_ket)
                #
                #     target = target_block.ket
                #     # print('target', target)
                #     # print('canonize', ind1, ind2)
                #     # _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                #     #                                     [ind1, ind2], inplace=True)
                #     _, select_inds = self.term_class.compress_tens_list([target[ind1], target[ind2]],
                #                                                         [target.site_ind_id], [ind1, ind2],
                #                                                         inplace=True, max_bond=self.max_bond)
                #     # target_block.select_inds_ket[ind1] = select_inds[0]
                #     for op_block in ops:
                #         if isinstance(op_block, BlockOperator_Cross):
                #             op_block.ket_select_inds[ind1] = select_inds[0]
                #             # op.select_inds_ket[ind1 + 1] = []
                #             if ind1 < op_block.L - 1:
                #                 op_block.ket_select_inds[ind1 + 1] = []
                #             # print('check orthog')
                #             # print('op', op.select_inds_ket)
                #             # check_orthog(target, op.select_inds_ket, [target.site_ind_id])
                #             # print('op ket = target', op.ket is target)
                #             # print('op ket inds', op.select_inds_ket)
                #             # print('target', target_block.select_inds_ket)

                # self.cur_orthog = ind1 + direction

        else:
            ## update k
            ind1 = i   # self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = i - 1  # self.mps_inds[i - 1]
                # _, select_inds = canonize_tens_list([self.ket[ind1], self.ket[ind2]], [self.ket.site_ind_id],
                #                                     [ind1, ind2], inplace=True)
                _, select_inds = self.term_class.compress_tens_list([self.ket[ind1], self.ket[ind2]],
                                                                    site_ind_ids=[self.ket.site_ind_id],
                                                                    site_inds=[ind1, ind2],
                                                                    inplace=True, max_bond=self.max_bond)
                # self.ket_select_inds[ind1] = select_inds[0]
                self.ket.select_inds[ind1] = select_inds[0]

                for term in self.terms:
                    term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

                # for ops, target_block in self.term_blocks:
                #     target = target_block.ket
                #     # _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                #     #                                     [ind1, ind2], inplace=True)
                #     _, select_inds = compress_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
                #                                         [ind1, ind2], inplace=True, max_bond=self.max_bond)
                #     # target_block.select_inds_ket[ind1] = select_inds[0]
                #     for op_block in ops:
                #         if isinstance(op_block, BlockOperator_Cross):
                #             op_block.ket_select_inds[ind1] = select_inds[0]
                #             if ind1 > 0:
                #                 op_block.ket_select_inds[ind1 - 1] = []
                #             # print('check orthog', ind1)
                #             # print('op', op.select_inds_ket)
                #             # check_orthog(target, op.select_inds_ket, [target.site_ind_id])
                #             # print('op ket = target', op.ket is target)

                # self.cur_orthog = ind1 + direction

        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = i, i + 1   # self.mps_inds[i], self.mps_inds[i + 1]
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
            mat1, mat2, select_inds = helper_cross.cross_compress(site_i_mat, self.max_bond, do_qr=True,
                                                                  return_inds=True)
            # print('self.ket', self.ket[ind1], self.ket[ind2])
            # print('ind1, ind2', ind1, ind2, select_inds)
            # print('site i mat', site_i_mat.shape)

            mat1 = mat1.reshape(shape_l + (-1,))
            mat2 = mat2.reshape((-1,) + shape_r)

            ## update ket
            self.ket[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
            self.ket[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
            # self.ket_select_inds[ind1] = select_inds
            self.ket.select_inds[ind1] = select_inds

            for term in self.terms:
                term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

            # for ops, target_block in self.term_blocks:
            #     target = target_block.ket
            #     _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
            #                                         [ind1, ind2], inplace=True)
            #     target_block.ket_select_inds[ind1] = select_inds[0]
            #
            #     for op in ops:
            #         if isinstance(op, BlockOperator_Cross):
            #             op.ket_select_inds[ind1] = select_inds[0]

        else:
            ind1, ind2 = i, i - 1   # self.mps_inds[i], self.mps_inds[i - 1]
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
            mat1, mat2, select_inds = helper_cross.cross_compress(site_i_mat, self.max_bond, do_qr=True,
                                                                  return_inds=True)

            mat1 = mat1.reshape(shape_l + (-1,))
            mat2 = mat2.reshape((-1,) + shape_r)

            ## update ket
            self.ket[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
            self.ket[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
            # self.ket_select_inds[ind1] = select_inds
            self.ket.select_inds[ind1] = select_inds

            for term in self.terms:
                term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

            # for ops, target_block in self.term_blocks:
            #     target = target_block.ket
            #     _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
            #                                         [ind1, ind2], inplace=True)
            #     target_block.ket_select_inds[ind1] = select_inds[0]
            #
            #     for op in ops:
            #         if isinstance(op, BlockOperator_Cross):
            #             op.ket_select_inds[ind1] = select_inds[0]

        # self.cur_orthog = ind2

        return


    def _update_2site_svd(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = i, i + 1  # self.mps_inds[i], self.mps_inds[i + 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site1.data, inds=site1.inds)
            self.ket[ind2].modify(data=site2.data, inds=site2.inds)

            _, select_inds = self.term_class.canonize_tens_list([self.ket[ind1], self.ket[ind2]],
                                                                site_ind_ids=[self.ket.site_ind_id],
                                                                site_inds=[ind1, ind2], inplace=True)
            self.ket_select_inds[ind1] = select_inds[0]

            for term in self.terms:
                term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

            # for ops, target_block in self.term_blocks:
            #     target = target_block.ket
            #     _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
            #                                         [ind1, ind2], inplace=True)
            #     target_block.ket_select_inds[ind1] = select_inds[0]

        else:
            ind1, ind2 = i, i - 1   # self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            _, select_inds = self.term_class.canonize_tens_list([self.ket[ind1], self.ket[ind2]],
                                                                site_ind_ids=[self.ket.site_ind_id],
                                                                site_inds=[ind1, ind2], inplace=True)
            self.ket_select_inds[ind1] = select_inds[0]

            for term in self.terms:
                term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

            # for ops, target_block in self.term_blocks:
            #     target = target_block.ket
            #     _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
            #                                         [ind1, ind2], inplace=True)
            #     target_block.ket_select_inds[ind1] = select_inds[0]

        self.cur_orthog = ind2

        return




