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

from local_solvers_old.local_evaluator import get_mps_matching_inds
from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

# from local_solvers_old.local_solver_v2 import *  # LocalSolver
# from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
# from local_solvers_old.helper_cross_v3 import compress_tens_list
from local_solvers.mps_classes import MPS
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross



class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

class SolveMethod(Enum):
    CGD = 'CGD'  # local CGD using quimb tensors
    CGDx = 'CGDx'  # local CGD using numpy
    LSQ = 'LSQ'  # least squares regression


DEFAULT_MAX_ITER = 20  # 10
DEFAULT_MAX_TOT_ITER = 50
DEFAULT_CONV_TOL = 1.0e-6
DEFAULT_SOLVE = SolveMethod.CGD
DEFAULT_MAX_WRONG_ITER = 10



# def local_cross_evaluator(*terms: tuple[ Sequence[Union[GridTN1D, Callable[[np.ndarray], np.ndarray]]],
#                                          GridTN1D ],
#                           nsites=1, direction: SweepDirection.LEFT, max_bond=None,
#                           combine_terms_func:Callable=None):
#     """ terms are sequences of ((A, B, ...), x)
#         where A, B are operators or functions that are applied to x in
#         the order A(B(x))
#
#         algorithm: compute rdm for each term
#         sum together terms by summing rdms together (state-averaging method)
#         projecting next site onto new rdm
#     """
#     opx, ref_x = terms[0]
#     L = ref_x.L
#
#     mps_terms = []
#     for ops, x in terms:
#         new_ops = [op.data if isinstance(op, GridTN1D) else op for op in ops]
#         mps_terms += [(new_ops, x.data.copy())]
#
#     solver = CrossEvaluator(mps_terms, combine_terms_func=combine_terms_func,
#                             init_direction=direction, max_bond=max_bond)
#     solver.solve(nsites)
#
#     out_gtn = GridTN1D(ref_x.grid, solver.ket)
#     return out_gtn


class LocalEvaluator:

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 # bra_state: 'qtn.MatrixProductState',
                 terms: Sequence[Union['Term_DMRG','Term_Cross']],
                 # cur_orthog: int=None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int =None,
                 conv_tol: float=DEFAULT_CONV_TOL, max_iter: int=DEFAULT_MAX_ITER,
                 max_tot_iter: int=DEFAULT_MAX_TOT_ITER, max_wrong_iter: int=DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'LocalEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 ):

        if copy_obj:
            # self.L = copy_obj.L
            self.ket = copy_obj.ket.copy()
            # self.bra = copy_obj.bra.copy()
            new_terms = []
            for term in copy_obj.terms:
                new_term = term.copy()
                if term.bra is copy_obj.ket:
                    new_term.bra = self.ket
                new_terms += [new_term]
            self._terms = new_terms

            # self.cur_orthog = copy_obj.cur_orthog
            self.direction = copy_obj.direction

            self.conv_tol = copy_obj.conv_tol
            self.max_tot_iter = copy_obj.max_tot_iter
            self.max_iter = copy_obj.max_iter
            self.max_wrong_iter = copy_obj.max_wrong_iter
            self.max_bond = copy_obj.max_bond

            self.combine_terms_func = copy_obj.combine_terms_func

            ## results
            self.err = copy_obj.err
            self.is_conv = copy_obj.is_conv
            # self.cur_orthog = copy_obj.cur_orthog

        else:
            self.L : int = ket_state.L
            # self.cur_orthog = cur_orthog
            self.direction = direction

            if not isinstance(ket_state, MPS):
                ket_state.view_as(MPS, inplace=True)
            self.ket = ket_state
            # self.bra = ket_state.conj(mangle_inner=True)
            self._terms = terms
            self.terms = terms      ## updates bra in the terms? and initialize

            ## convergence params
            self.conv_tol = conv_tol
            self.max_iter = max_iter
            self.max_tot_iter = max_tot_iter
            self.max_wrong_iter = max_wrong_iter
            self.max_bond = max_bond

            ## callables for combining operators, targets. defaults to summation
            if combine_terms_func is None:
                combine_terms_func = helper_tn.sum_tens
            self.combine_terms_func = combine_terms_func

            ## results
            self.err = np.inf  # self.check_err()
            self.is_conv = False  # np.abs(self.err) < conv_tol


    @property
    def conv_kwargs(self):
        out = {'conv_tol': self.conv_tol, 'max_iter': self.max_iter, 'max_tot_iter': self.max_tot_iter,
               'max_wrong_iter': self.max_wrong_iter,}
        return out

    # @classmethod
    @property
    def term_class(self):  # -> Type[Term]:
        raise NotImplementedError

    @property
    def terms(self) -> Sequence[Union['Term_DMRG', 'Term_Cross']]:
        return self._terms
        ## terms can be created, and then initialized separately

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
            else:
                term_.match_inds(ref_term)

            if term_.bra is None:
                term_.bra = self.ket

            term_.initialize()
            terms += [term_]

        self._terms = terms

    def copy(self):
        new_solver = self.__class__(self.ket, self.terms, copy_obj=self)
        return new_solver

    # def get_ket_to_bra_inds(self, i) -> dict[str, str]:
    #     """ inds mapping inds on self.bra to corresponding inds on self.ket
    #     """
    #     return helper_tn.get_mps_matching_inds(self.ket, self.bra, i)
    #
    # def get_bra_to_ket_inds(self, i) -> dict[str, str]:
    #     ket_to_bra = self.get_ket_to_bra_inds(i)
    #     bra_to_ket = {item: k for k, item in ket_to_bra.items()}
    #     return bra_to_ket

    # def update_bra_from_ket(self, sites=None):
    #
    #     if self.bra is None:
    #         pass
    #     else:  ## inplace update
    #         sites = range(self.L) if sites is None else sites
    #         for ix in sites:
    #             ket_to_bra_inds = self.get_ket_to_bra_inds(ix)
    #             ket_tens = self.ket[ix].conj()
    #             ket_tens = ket_tens.reindex(ket_to_bra_inds)
    #             self.bra[ix].modify(data=ket_tens.data, inds=ket_tens.inds)
    #     return

    @property
    def cur_orthog(self) -> int:
        cur_orthog = self.terms[0].cur_orthog
        print(self.terms[0], self.terms[0].cur_orthog)
        for term in self.terms[1:]:
            if cur_orthog != term.cur_orthog:
                raise ValueError('orthogonalities of terms not consistent')
        return cur_orthog


    def canonize(self, target_orthog: int):

        self.term_class.canonize(self.ket, target_orthog)
        # self.update_bra_from_ket()

        for term in self.terms:
            term.canonize_ket(target_orthog)
            if term.initialized:
                if target_orthog < self.cur_orthog:
                    for i in range(self.cur_orthog, target_orthog, -1):
                        term.extend_env(i, direction=SweepDirection.LEFT)
                else:
                    for i in range(self.cur_orthog, target_orthog):
                        term.extend_env(i, direction=SweepDirection.RIGHT)


    def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        raise NotImplementedError

    def solve_l2r(self, nsites: int, canonize=True, verbose=False, **kwargs):

        L = self.L

        if canonize:
            canon_site = 0
            self.canonize(canon_site)
            self.direction = SweepDirection.RIGHT

        ## right sweep
        assert(self.cur_orthog==0), f'cur_orthog should be 0 not {self.cur_orthog}'
        assert(self.direction == SweepDirection.RIGHT), f'should be SweepDirection.RIGHT, not {self.direction}'
        direction = self.direction

        # if self.cur_orthog == 0:
        #     print('check orthog', self.check_right_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # else:
        #     print('check orthog', self.check_left_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        ## one right to left sweep
        tot_err = 0.0
        for i in range(L - nsites + 1):
            site_i, site_err = self._site_solve(i, nsites)
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
            # if i < L - 1:
            #     self.update_blocks(i, direction)

        return self.ket, tot_err


    def solve_r2l(self, nsites: int, canonize=True, verbose=False, **kwargs):

        L = self.L

        if canonize:
            canon_site = L-1
            self.canonize(canon_site)
            self.direction = SweepDirection.RIGHT

        ## right sweep
        assert (self.cur_orthog == L-1), f'cur_orthog should be {L-1} not {self.cur_orthog}'
        assert (self.direction == SweepDirection.LEFT), f'direction should be SweepDirection.LEFT, not {self.direction}'
        direction = self.direction

        tot_err = 0.0
        for i in range(L - 1, nsites - 2, -1):
            left_site_pos = i - nsites + 1
            site_i, site_err = self._site_solve(left_site_pos, nsites)
            if site_err > 10 ** 5:
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
            # if i > 0:
            #     self.update_blocks(i, direction)

        return self.ket, tot_err


    def solve(self, nsites: int, conv_tol=0, verbose=False, **kwargs):
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
        canon_site = 0 if self.direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)
        # if self.cur_orthog == 0:
        #     print('check orthog', self.check_right_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # else:
        #     print('check orthog', self.check_left_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        ## iterative solver
        it = 0
        err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        self.max_tot_iter = 2

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = self.direction
        while err > conv_tol and conv_it < self.max_iter \
                and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter:

            it += 1
            # tot_err = 0.0

            ## right sweep
            if direction == SweepDirection.RIGHT:

                _, err = self.solve_l2r(nsites, canonize=False, verbose=verbose)

                # for i in range(L - nsites + 1):
                #     site_i, site_err = self._site_solve(i, nsites)
                #     if site_err > 10**5:
                #         print('site error too large')
                #         exit()
                #     tot_err += site_err
                #
                #     ## update ket (+ canonicalization)
                #     if nsites == 1:
                #         self._update_1site(i, site_i, direction)
                #         # self._update_1site_with_Z(i, site_i, direction)
                #     elif nsites == 2:
                #         self._update_2site(i, site_i, direction)
                #         # self._update_2site_with_Z(i, site_i, direction)
                #     else:
                #         raise NotImplementedError
                #
                #     ## update A, b blocks (left)
                #     if i < L - 1:
                #         self.update_blocks(i, direction)
                #
                # err = tot_err  # self.check_err(dense=True)

            else:

                ## left sweep
                _, err = self.solve_r2l(nsites, canonize=False, verbose=verbose)

                # ## left sweep
                # for i in range(L - 1, nsites - 2, -1):
                #     left_site_pos = i - nsites + 1
                #     site_i, site_err = self._site_solve(left_site_pos, nsites)
                #     if site_err > 10**5:
                #         print('site err too large')
                #         exit()
                #     tot_err += site_err
                #
                #     ## update ket, bra
                #     if nsites == 1:
                #         self._update_1site(i, site_i, direction)
                #         # self._update_1site_with_Z(i, nsites, direction)
                #     elif nsites == 2:
                #         self._update_2site(i, site_i, direction)
                #         # self._update_2site_with_Z(i, nsites, direction)
                #     else:
                #         raise NotImplementedError
                #
                #     ## update A, b envs (right)
                #     if i > 0:
                #         self.update_blocks(i, direction)
                #
                # err = tot_err  # self.check_err(dense=True)

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
            # self.bra = min_solver.bra
            self._terms = min_solver.terms
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            # self.cur_orthog = min_solver.cur_orthog

        print('num iter', it)

        return self.ket, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)

    def update_blocks(self, i: int, direction: SweepDirection, new_ket_site=None):
        for term in self.terms:
            term.extend_env(i, direction=direction, new_ket_site=new_ket_site)

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection') -> None:
        raise NotImplementedError

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection') -> None:
        raise NotImplementedError