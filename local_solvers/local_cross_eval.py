import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn

from gridTN_1D import GridTN1D
import local_solvers.helper_tn as helper_tn

from local_solvers.local_evaluator import *
from local_solvers import helper_cross as helper_cross




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

    @classmethod
    def term_class(cls) -> Type['Term_Cross']:
        return Term_Cross

    @property
    def terms(self) -> Sequence[Union['Term_DMRG', 'Term_Cross']]:
        return self._terms

    @terms.setter
    def terms(self, terms_list: list[Term]):
        terms = []
        ref_term = None
        for term_ in terms_list:
            # if not isinstance(term_, Term):
            #     args, kwargs = term_
            #     term_ = self.term_class(*args, **kwargs)
            #     term_.bra = self.bra

            if ref_term is None:
                ref_term = term_
                if term_.bra is not None:
                    self.out = term_.bra.copy()
            else:
                term_.match_inds(ref_term)

            # term_.bra = self.ket
            if term_.bra is None:
                term_.bra = self.out  # self.ket.copy()

            # term_.bra = self.out  # self.ket.copy()

            term_._direction = self.direction
            term_.max_bond = self.max_bond
            term_.initialize()
            # print('term_ setter cur orthog', self.direction, term_.cur_orthog)
            terms += [term_]

        self._terms = terms

    @property
    def solver_type(self) -> LocalSolverType:
        return LocalSolverType.Cross

    @property
    def solution(self) -> 'MPS':
        return self.out

    def copy(self):
        out = super().copy()
        # if self.terms[0].ket is self.out:
        #     out.out = out.terms[0].out
        return out


    def check_orthog(self):
        for term in self.terms:
            term.check_orthog()
        # for term in self.terms:
        #     bra_canon = helper_cross.check_orthog(term.vec_block.bra, term.vec_block.bra.select_inds,
        #                                           [term.vec_block.bra.site_ind_id])
        #     # ket_canon = helper_cross.check_orthog(term.vec_block.ket, term.vec_block.ket.select_inds,
        #     #                                       [term.vec_block.ket.site_ind_id])
        return

    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        print('in CROSS site solve')
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        site_l = left_site_pos
        site_r = left_site_pos + nsites - 1
        inds_l = [self.out.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.out.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        current_x = qtn.tensor_contract(*[self.out[i] for i in site_inds]).copy()
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # if site_tens is None:
        #     current_x = qtn.tensor_contract(*[self.solution[i] for i in site_inds])
        #     # current_x = self.terms[0].get_evaluated_site(left_site_pos, nsites)     ## self term
        #     # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        #     # if left_site_pos not in self._current_ket_sites:
        #     #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        #     # else:
        #     #     current_x = self._current_ket_sites[left_site_pos]
        # else:
        #     current_x = site_tens.copy()
        #     # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        # print('current_x', self.ket.exponent, current_x.norm()) #, current_x.data)

        ### BL * B * BR = d/dT*[i] <x|b>
        solve_func = self._local_solve_func
        print('solve func', solve_func)
        if solve_func is None:
            term_site_tens = []
            for term in self.terms:  # [1:]:
                tmp = term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)

                tmp.modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent
                # helper_cross.plot_submat(self.ket, left_site_pos, nsites, tmp, select_inds=self.ket.select_inds,
                #                          ref_ket=term.ket)

                term_site_tens += [tmp]

            site_tens = self.combine_terms_func(term_site_tens)

            # k1 = self.terms[0].ket.to_dense() * 10**self.terms[0].ket.exponent
            # k2 = self.terms[1].ket.to_dense() * 10**self.terms[1].ket.exponent
            # out = k1 * k2
            # # out = self.out
            # helper_cross.plot_submat(self.ket, left_site_pos, nsites, site_tens, select_inds=self.ket.select_inds,
            #                          ref_kets=[k1 * k2])

            site_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))  ## remove self.ket exponent
            out_tensors = [site_tens]

        else:
            out_tensors = solve_func(left_site_pos, nsites, return_intermediates=return_intermediates) #, site_tens=site_tens)
            ## first tensor is tensor of interest. other tensors are for targeting

            # term_site_tens = [current_x]
            # print('getting evaluated site of terms (2)')
            # for term in self.terms[1:]:
            #     term_site_tens += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
            #     term_site_tens[-1].modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent


        ## solve for T[i]
        x_eff = out_tensors[0].copy()
        # x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
        ### rescale moved to get_evaluated_site
        ## b_eff_exponent already in b_eff tensor

        # ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
        # ## but it should be canonical?
        # tensors = [x_eff]
        # # if site_l > 0:
        # #     inv_env_l = self.ket_block.envs[site_l-1]
        # #     if inv_env_l is not None:
        # #         ## indeed, these terms aren't needed (except for ket to bra reindex)
        # #         print('skipping inv env L')
        # #         # inv_env_l = inv_env_l.copy()
        # #         # inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
        # #         tensors += [inv_env_l]
        # # if site_r < self.L-1:
        # #     inv_env_r = self.ket_block.envs[site_r+1]
        # #     # print('inv_env_r', inv_env_r)
        # #     if inv_env_r is not None:
        # #         ## indeed, these lines aren't needed (except for ket to bra reindex)
        # #         print('skipping inv env R')
        # #         # inv_env_r = inv_env_r.copy()
        # #         # inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
        # #         tensors += [inv_env_r]
        #
        # x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
        x_eff.transpose(*(inds_l + [self.out.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)

        self._new_ket_site = (left_site_pos, nsites, self.direction, [x_eff])
        # self._current_ket_sites[left_site_pos] = x_eff

        current_x.transpose_like(x_eff, inplace=True)
        site_err = np.linalg.norm(x_eff.data - current_x.data) # * 10 ** (-self.ket.exponent))
        print('site err', site_err)

        # ### plot out
        # coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # i = left_site_pos
        # if i > 0:
        #     inds += [self.out.bond(i, i - 1)]
        # inds += [self.out.site_ind(i + ix) for ix in range(nsites)]
        # if i + nsites < self.out.L:
        #     inds += [self.out.bond(i + nsites - 1, i + nsites)]
        #
        # for tens in term_site_tens + [x_eff]:
        #     tens.transpose(*inds, inplace=True)
        #     plt.plot(selectors, tens.data.reshape(-1), 'x')
        # plt.title('cross eval')
        # plt.show()

        return out_tensors, site_err


    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None):
        """ update ket, bra with new_site
            i: int of mps site
        """
        at_end = (i == 0) if direction == SweepDirection.LEFT else (i == self.L - 1)

        kets, tensors = [self.out], [site_i]
        # kets, tensors = [self.ket, self.out], [self.terms[0].proj_vec_targets, site_i]
        for term in self.terms:
            if term.num_tiers > 1:
                kets += [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
                tensors += [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]

        helper_cross.update_kets(kets, tensors, i, 1, direction=direction, max_bond=self.max_bond)

        # #######################
        # ### update + canonicalize only self.ket = self.out = term.bra
        # ### "decimation only" by specifying selection inds may lead to bad canonicalization
        # ### bc the submatrix to be inverted is singular.
        # # site_i[0].transpose_like(self.out[i], inplace=True)
        # # self.out[i].modify(data=site_i[0].data)
        # # helper_cross.canonize(self.out, i + direction, cur_orthog=i)
        # # helper_cross.update_kets(kets, tensors, i, 1, direction=direction, max_bond=self.max_bond)
        # helper_cross.update_kets([self.out], [site_i], i, 1, direction=direction, max_bond=self.max_bond)
        # for term in self.terms:
        #     term.update_intermediate_kets(i, 1, direction)
        # self.ket = self.ket
        #
        # for term in self.terms:
        #     print('term bra', term.bra is self.ket, term.bra is self.out)
        #
        # # helper_cross.plot_submat(self.out, i+direction, 1, self.out[i+direction],
        # #                          ref_kets=[self.out], plt_title='out_canon nxt')
        # # helper_cross.plot_submat(self.out, i, 1, site_i[0],
        # #                          ref_kets=[self.out], plt_title='out_canon i')
        #
        # # print('self.out')
        # # helper_cross.check_orthog(self.out)
        # # helper_cross.check_orthog(self.terms[-1].get_intermediate_ket(0))

        #######################
        ### original
        # select_inds = helper_cross.update_1site(self.out, i, site_i, direction, max_bond=self.max_bond,
        #                                         decimate_only=(not at_end))
        # ## for this solver, this is redundant
        # if not at_end:
        #     # helper_cross.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond, decimate_only=True)
        #     helper_cross.decimate_1site(self.ket, i, select_inds, direction) #, max_bond=self.max_bond)
        #
        #
        # for term in self.terms:
        #     print('term bra is out?', term.bra is self.out)
        #     # if not at_end:
        #     #     helper_cross.decimate_1site(term.bra, i, select_inds, direction) #, max_bond=self.max_bond)
        #         # helper_cross.decimate_1site(term.ket, i, select_inds, direction, max_bond=self.max_bond)
        #     term.update_intermediate_kets(i, 1, direction)

        if not at_end:

            # for term in self.terms:
            #     term.canonize_ket_tens(i, 1, direction, max_bond=None)

            self.update_blocks(i, direction)

        print('updated ket', i, self.out.cur_orthog )
        # helper_cross.check_orthog(self.ket)
        helper_cross.check_orthog(self.out)

        return

    def _update_2site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        # raise NotImplementedError
        # left_site_pos = i if direction == SweepDirection.RIGHT else i - 1

        # at_end = (left_site_pos == 0) if direction == SweepDirection.LEFT else (left_site_pos == self.L - 2)

        # for term in self.terms:
        #     print('term ket', term.ket is self.ket)
        #     print('term bra', term.bra is self.out)

        print('UPDATE2', i)
        print('self.out select inds', self.out.select_inds)

        # coords = helper_cross.get_selectors(self.out, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # if left_site_pos > 0:
        #     inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
        # inds += [self.out.site_ind(left_site_pos), self.out.site_ind(left_site_pos + 1)]
        # if left_site_pos + 2 < self.out.L:
        #     inds += [self.out.bond(left_site_pos + 1, left_site_pos + 2)]
        #
        # plt.figure()
        # site_i[0].transpose(*inds, inplace=True)
        # print('site i', inds, site_i)
        # plt.plot(selectors, site_i[0].data.reshape(-1), 'x')

        # plt.figure()
        # term = self.terms[-1]
        #
        # int_ket = term.get_intermediate_ket(0)
        # out_ket = term.get_intermediate_ket(1)
        # print('out ket = out', out_ket is self.out)
        #
        # print('term intermediate sites', term.intermediate_sites)
        # print('proj vec', term.proj_vec_targets)

        # ### out ket ###
        # coords = helper_cross.get_selectors(self.out, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # tens = term.evaluated_site
        # print('tens', tens)
        # tens.transpose(*inds, inplace=True)
        # plt.plot(selectors, tens.data.reshape(-1), 'o')
        #
        # ### intermediate ket
        # coords = helper_cross.get_selectors(out_ket, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # tens = term.intermediate_sites[0]
        # print('tens', tens)
        # tens[0].transpose(*inds, inplace=True)
        # plt.plot(selectors, tens[0].data.reshape(-1), 'x')
        #
        # plt.title('nonlinear term')
        # plt.show()
        #######
        print('exponent', self.out.exponent, [[t_op.exponent for t_op in t.operators] for t in self.terms])
        kets, tensors = [self.out], [site_i]
        # print('out site_i', site_i)
        # print('terms 0 proj vec targets', self.terms[0].proj_vec_targets)
        # kets, tensors = [self.ket, self.out], [self.terms[0].proj_vec_targets, site_i]
        for term in self.terms:
            if term.num_tiers > 1:
                kets += [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
                tensors += [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]

        helper_cross.update_kets(kets, tensors, i, 2, direction=direction)

        #### original version
        # inds_r, inds_c = helper_cross.update_2site(self.out, left_site_pos, site_i, direction, max_bond=self.max_bond,)
        #
        #
        # # print('select inds', i, select_inds)
        # # helper_cross.decimate_2site(self.ket, left_site_pos, inds_r, inds_c, direction)
        # for term in self.terms:
        #     print('ket is bra', self.ket is term.bra)
        #     print('out is bra', self.out is term.bra)
        #     # helper_cross.update_2site(term.bra, left_site_pos, [term.evaluated_site], direction,
        #     #                           inds_r=inds_r, inds_c=inds_c)
        #     # helper_cross.decimate_2site(term.ket, left_site_pos, inds_r, inds_c, direction)
        #     term.update_intermediate_kets(i, 2, direction) #, select_inds=(inds_r, inds_c))
        #     print('updated terms')
        #
        # print('check ket')
        # chk1 = helper_cross.check_orthog(self.ket)
        # print('chk1', left_site_pos, chk1)
        # if chk1[0] != chk1[1]:
        #     exit()
        #
        # chk1 = helper_cross.check_orthog(self.out)
        # print('check out', left_site_pos, chk1)
        # if chk1[0] != chk1[1]:
        #     exit()
        #
        # for term in self.terms:
        #     chk1 = helper_cross.check_orthog(term.ket)
        #     print('term check ket', left_site_pos, chk1)
        #     if chk1[0] != chk1[1]:
        #         exit()
        #
        #     if True:  # left_site_pos == self.ket.L - 2:
        #         print(term._intermediate_kets.keys(), term.num_tiers)
        #         plt.figure()
        #         plt.plot(helper_quimb.to_dense(term.bra, [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
        #                  label='bra')
        #         if term.num_tiers > 1:
        #             plt.plot(helper_quimb.to_dense(term._intermediate_kets[0],
        #                                            [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
        #                  label='ket0')
        #         plt.plot(helper_quimb.to_dense(term.ket, [term.ket.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1), '--',
        #                  label='ket')
        #         plt.title(f'updated term {left_site_pos}')
        #         plt.legend()
        #         plt.show()
        #
        # # for term in self.terms:
        # #     term.update_intermediate_kets(i, 2, direction)
        #
        # # for term in self.terms:
        # #     term.canonize_ket_tens(i, 1, direction, max_bond=None)
        #
        #     # helper_cross.check_orthog(term.vec_block.bra, term.vec_block.bra.select_inds,
        #     #                           [term.vec_block.bra.site_ind_id])
        #

        self.update_blocks(i, direction)

        # plt.figure()
        # term = self.terms[-1]
        # ket_data = term.ket.to_dense()
        # int_data = term.get_intermediate_ket(0).to_dense()
        # bra_data = term.bra.to_dense()
        #
        # plt.plot(ket_data, label='ket')
        # plt.plot(int_data, label='int', ls='--')
        # plt.plot(bra_data, label='bra', ls=':')
        # plt.legend()
        # plt.title(f'2-site updated term final {i}')
        # plt.show()

        # for chk in kets:
        #     print('chk', chk)
        # exit()

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




