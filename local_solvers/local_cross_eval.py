import numpy as np
from abc import ABC

import helper_quimb
from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn

from gridTN_1D import GridTN1D
import local_solvers.helper_tn as helper_tn

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
    return solver.solution


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


    @classmethod
    def term_class(cls) -> Type['Term_Cross']:
        return Term_Cross

    @property
    def solution(self) -> 'MPS':
        return self.out

    def _site_solve(self, left_site_pos: int, nsites: int, site_tens=None) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        site_l = left_site_pos
        site_r = left_site_pos + nsites - 1
        inds_l = [self.ket.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.ket.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        if site_tens is None:
            current_x = qtn.tensor_contract(*[self.solution[i] for i in site_inds])
            # if left_site_pos not in self._current_ket_sites:
            #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            # else:
            #     current_x = self._current_ket_sites[left_site_pos]
        else:
            current_x = site_tens.copy()
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm()) #, current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        solve_func = self._local_solve_func
        if solve_func is None:
            term_site_tens = []
            for term in self.terms:
                term_site_tens += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
                term_site_tens[-1].modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent
            site_tens = self.combine_terms_func(term_site_tens)
            site_tens.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))  ## remove self.ket exponent

        else:
            site_tens = solve_func(left_site_pos, nsites, return_intermediates=False) #, site_tens=site_tens)

        ## solve for T[i]
        x_eff = site_tens.copy()
        # x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
        ### rescale moved to get_evaluated_site
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

        self._new_ket_site = (left_site_pos, nsites, self.direction, [x_eff])
        # self._current_ket_sites[left_site_pos] = x_eff

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm(x_eff.data - current_x.data) # * 10 ** (-self.ket.exponent))
        # print('site err', site_err)
        return x_eff, site_err


    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        helper_cross.update_1site(self.out, i, site_i, direction, max_bond=self.max_bond)
        ## for this solver, this is redundant

        at_end = (i == 0) if direction == SweepDirection.LEFT else (i == self.L - 1)

        helper_cross.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond, decimate_only=True)
        # if not at_end:
        #     helper_cross.decimate_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)

        for term in self.terms:
            term.update_intermediate_kets(i, 1, direction)

        if not at_end:

            for term in self.terms:
                term.canonize_ket_tens(i, 1, direction, max_bond=None)

            self.update_blocks(i, direction)

        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        raise NotImplementedError
        left_site_pos = i if direction == SweepDirection.RIGHT else i - 1

        # at_end = (left_site_pos == 0) if direction == SweepDirection.LEFT else (left_site_pos == self.L - 2)

        helper_cross.update_2site(self.out, left_site_pos, site_i, direction, max_bond=self.max_bond)
        helper_cross.decimate_2site(self.ket, left_site_pos, site_i, direction, max_bond=self.max_bond)

        for term in self.terms:
            term.update_intermediate_kets(i, 2, direction)

        for term in self.terms:
            term.canonize_ket_tens(i, 1, direction, max_bond=None)

            # helper_cross.check_orthog(term.vec_block.bra, term.vec_block.bra.select_inds,
            #                           [term.vec_block.bra.site_ind_id])

        self.update_blocks(i, direction)

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
            self.ket.select_inds[ind1] = select_inds[0]

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
            self.ket.select_inds[ind1] = select_inds[0]

            for term in self.terms:
                term.canonize_ket_tens(ind1, 1, direction, max_bond=None)

            # for ops, target_block in self.term_blocks:
            #     target = target_block.ket
            #     _, select_inds = canonize_tens_list([target[ind1], target[ind2]], [target.site_ind_id],
            #                                         [ind1, ind2], inplace=True)
            #     target_block.ket_select_inds[ind1] = select_inds[0]

        self.ket.cur_orthog = ind2

        return




