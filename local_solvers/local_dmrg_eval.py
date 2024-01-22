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

    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))

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

        current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)

        # tot_rdm = None
        # tot_site = None
        site_tens_list = []
        site_rdm_list = []
        for term in self.terms:

            site_tens = term.get_evaluated_site(left_site_pos, nsites)

            # if isinstance(site_tens, list):
            #     site_tens = [t.modify(apply=lambda x: x * 10 ** -self.ket.exponent) for t in site_tens]
            #     site_tens_list += [site_tens]
            # else:
            #     site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
            #     site_tens_list += [site_tens]

            # site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
            ### moved to get_evaluated_site
            site_tens_list += [site_tens]

            vec_block = term.vec_block
            if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.ket is vec_block.bra:
                vec_tens = term.proj_vec
                # vec_tens = qtn.tensor_contract(*[term.vec_block.ket[si]
                #                                  for si in range(left_site_pos, left_site_pos + nsites)])
                # vec_tens.modify(apply=lambda x: x * 10 ** (vec_block.ket.exponent-self.ket.exponent))
                # ### scaling moved to get_evaluated_site

                site_rdm_list += [vec_tens]

        tot_site = helper_tn.sum_tens(site_tens_list)

        current_x.transpose_like(tot_site, inplace=True)
        site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)

        if at_end:
            return [tot_site], site_err

        else:
            return site_tens_list + site_rdm_list, site_err

        # return x_eff, site_err


    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """

        helper_dmrg.update_1site(self.ket, i, site_i, direction)

        at_end = (i == 0 if direction==SweepDirection.LEFT else i == self.L-1)

        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        raise NotImplementedError




