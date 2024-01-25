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
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.blocks import BlockPowerKet_DMRG
# from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
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

    @property
    def term_class(self) -> Type['Term_DMRG']:
        return Term_DMRG

    # def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[Sequence[qtn.Tensor], Numeric]:
    #     """
    #     sites: int or slice(start, stop, step)
    #     """
    #     site_inds = list(range(left_site_pos, left_site_pos + nsites))
    #
    #     direction = self.direction
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #
    #     # ref_x_data = self.vecs[0]
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
    #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
    #     # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
    #
    #     # tot_rdm = None
    #     # tot_site = None
    #     site_tens_list = []
    #     site_rdm_list = []
    #     for term in self.terms:
    #
    #         site_tens = term.get_evaluated_site(left_site_pos, nsites)
    #
    #         # if isinstance(site_tens, list):
    #         #     site_tens = [t.modify(apply=lambda x: x * 10 ** -self.ket.exponent) for t in site_tens]
    #         #     site_tens_list += [site_tens]
    #         # else:
    #         #     site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
    #         #     site_tens_list += [site_tens]
    #
    #         # site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
    #         ### moved to get_evaluated_site
    #         site_tens_list += [site_tens]
    #
    #         vec_block = term.vec_block
    #         if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.ket is vec_block.bra:
    #             # vec_tens = term.proj_vec
    #             # site_rdm_list += [vec_tens]
    #             # pow_list = term.proj_vec
    #             pow_list = [t.copy() for t in term.proj_vec] #vec_block.get_pows(left_site_pos, nsites)]
    #             # for t in pow_list:
    #             #     t.modify(apply=lambda x: x / t.norm())
    #             site_rdm_list += pow_list
    #
    #     # tot_site = helper_tn.sum_tens(site_tens_list)
    #     tot_site = self.combine_terms_func(site_tens_list)
    #
    #     current_x.transpose_like(tot_site, inplace=True)
    #     try:
    #         # print('tot site', tot_site)
    #         # print('current x', current_x)
    #         # print('self,max bond', self.max_bond)
    #         site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
    #     except ValueError:  ## shape mismatch
    #         site_err = np.nan
    #
    #     if at_end:
    #         return [tot_site], site_err
    #
    #     else:
    #         return site_tens_list + site_rdm_list, site_err


    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        site_inds = list(range(left_site_pos, left_site_pos + 1))

        direction = self.direction
        at_end = (left_site_pos == self.L - 1) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        ix = left_site_pos
        ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1

        if site_tens is None:
            raise NotImplementedError
            # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        else:
            current_x = site_tens.copy()
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)

        # tot_rdm = None
        # tot_site = None
        site_tens_list = []
        site_rdm_list = []

        func = self._local_solve_func
        if func is not None:
            if not at_end:
                tot_site, site_rdm_list = func(left_site_pos, 0, return_intermediates=True)
            else:
                tot_site = func(left_site_pos, 0, return_intermediates=False)

        else:
            for term in self.terms:
                site_tens = term.get_evaluated_site(left_site_pos, 0)
                site_tens_list += [site_tens]

                vec_block = term.vec_block
                if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.ket is vec_block.bra:
                    pow_list = [t.copy() for t in term.proj_vec] #vec_block.get_pows(left_site_pos, nsites)]
                    site_rdm_list += pow_list

            # tot_site = helper_tn.sum_tens(site_tens_list)
            tot_site = self.combine_terms_func(site_tens_list)

        current_x.transpose_like(tot_site, inplace=True)
        try:
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
        except ValueError:  ## shape mismatch
            site_err = np.nan

        if at_end:
            return [tot_site], site_err

        else:
            return site_tens_list + site_rdm_list, site_err


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor=None
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))

        direction = self.direction
        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

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

        if site_tens is None:
            current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        else:
            current_x = site_tens.copy()
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)

        # tot_rdm = None
        # tot_site = None
        site_tens_list = []
        site_rdm_list = []

        func = self._local_solve_func
        if func is not None:
            if not at_end:
                tot_site, site_rdm_list = func(left_site_pos, nsites, return_intermediates=True)
            else:
                tot_site = func(left_site_pos, nsites, return_intermediates=False)

        else:
            for term in self.terms:
                site_tens = term.get_evaluated_site(left_site_pos, nsites)
                site_tens_list += [site_tens]

                vec_block = term.vec_block
                if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.ket is vec_block.bra:
                    pow_list = [t.copy() for t in term.proj_vec] #vec_block.get_pows(left_site_pos, nsites)]
                    site_rdm_list += pow_list

            # tot_site = helper_tn.sum_tens(site_tens_list)
            tot_site = self.combine_terms_func(site_tens_list)

        current_x.transpose_like(tot_site, inplace=True)
        try:
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
        except ValueError:  ## shape mismatch
            site_err = np.nan

        if at_end:
            return [tot_site], site_err

        else:
            return site_tens_list + site_rdm_list, site_err


    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('update i', i, site_i)
        # print([t.norm() for t in site_i])
        helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)

        at_end = (i == 0 if direction==SweepDirection.LEFT else i == self.L-1)

        # print('update vecblock')
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        # print('update blocks')
        if not at_end:
            self.update_blocks(i, direction=direction)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)

        if not at_end:
            raise NotImplementedError

        x_ind = self.ket.bond(i, i + direction)
        left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
        q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=x_ind)

        q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=q.data)

        r.transpose_like(self.ket[i + direction], inplace=True)
        self.ket[i + direction].modify(data=r.data)

        return

