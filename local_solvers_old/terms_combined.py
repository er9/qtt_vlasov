import numpy as np
from abc import ABC

# from sympy import *
import sympy.core.symbol
from sympy.parsing.sympy_parser import parse_expr

from setup_.defaults import *
from local_solvers.blocks import *
import local_solvers.helper_cross as helper_cross
from local_solvers.mps_classes import MPS, MPO
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
import tensor_callables as tc

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
SelectIndsType = dict[int, Sequence[int]]

class BlockType(Enum):
    DMRG = 'dmrg'
    CROSS = 'cross'

""" include multiple kets, bras.
"""



class CombinedTerms(ABC):

    def __init__(self,
                 terms: Sequence[Union[Term, tuple]],
                 combine_func: Callable,
                 bra: Union['MPS', qtn.MatrixProductState] = None,  ## bra shared across all terms
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # **canon_kwargs
                 term_type: Union[Type[Term_DMRG], Type[Term_Cross]] = Term_DMRG,
                 # copy_obj: 'CombinedTerm' = None,
                 ):

        # if copy_obj:
        #     # self.L = copy_obj.L
        #     self._bra = copy_obj.bra.copy()
        #     # self.bra = copy_obj.bra.copy()
        #     new_terms = []
        #     for term in copy_obj.terms:
        #         new_term = term.copy()
        #         if term.bra is copy_obj.bra:
        #             new_term.bra = self.bra
        #         new_terms += [new_term]
        #     self._terms = new_terms
        #
        #     self.combine_func = copy_obj.combine_func

        # else:
        if bra is not None:
            if not isinstance(bra, MPS):
                bra.view_as(MPS, inplace=True)
        self._bra = bra

        self._terms = []
        terms_list = []
        for term in terms:
            if not isinstance(term, term_type):
                terms_list += term_type(*term, cur_orthog=cur_orthog, direction=direction)
            else:
                terms_list += [term]
        self.terms = terms_list

        # self.initialized = False
        self.term_type = self.terms[0].__class__
        self._combine_func = combine_func


    @classmethod
    def from_str_expression(cls, expr: str, variables: dict[str, Union[MPS, MPO]]):
        sym_expr = parse_expr(expr, evaluate=False)

        combine_func = []

        sym_func = sym_expr.func

        def extract_terms(sym_expr):

            if all([(i in [sympy.core.symbol.Symbol, sympy.core.symbol.NumberKind]) for i in sym_expr.args]):
                if sym_func is sympy.core.add.Add:
                    func = term_sum
                elif sym_func is sympy.core.mul.Mul:
                    func =  term_elem_mult
                else:
                    pass
            else:
                args = sym_expr.args






    @property
    def combine_func(self):
        return self._combine_func

    @property
    def initialized(self):
        return all(*[term.initialized for term in self.terms])

    @property
    def direction(self):
        return self.terms[0].direction

    @property
    def cur_orthog(self):
        return self.terms[0].cur_orthog

    @property
    def L(self) -> int:
        return self.terms[0].L

    @property
    def ref_ket(self) -> 'MPS':
        return self.terms[0].ket

    @property
    def bra(self) -> Optional['MPS']:
        return self._bra

    @bra.setter
    def bra(self, bra: 'MPS'):
        if bra is not None and not isinstance(bra, MPS):
            bra.view_as(MPS, inplace=True)

        for term in self.terms:
            if term.bra is self._bra:
                term.bra = bra

        self._bra = bra

    @property
    def ket_ind_id(self):
        return self.terms[0].ket.site_ind_id

    @property
    def bra_ind_id(self):
        return self.ket_ind_id + '_'

    @property
    def term_class(self) -> Union[Type['Term_DMRG'], Type['Term_Cross']]:
        return self.terms[0].__class__

    @property
    def terms(self) -> Sequence[Union['Term_DMRG', 'Term_Cross']]:
        return self._terms
        ## terms can be created, and then initialized separately

    @terms.setter
    def terms(self, terms_list: list[Term]):
        terms = []
        ref_term = None
        for term_ in terms_list:

            if ref_term is None:
                ref_term = term_
            else:
                term_.match_inds(ref_term)

            if term_.bra is None:
                term_.bra = self.bra

            term_.initialize()
            terms += [term_]

        self._terms = terms

    def canonize(self, orthog: int):
        for term in self.terms:
            term.canonize(orthog)
            term.canonize_bra(orthog)
            term.cur_orthog = orthog

    def copy(self, bra_copy=None):
        if bra_copy is None:
            bra_copy = self.bra.copy() if self.bra is not None else None

        new_terms = [term.copy(bra_copy=bra_copy) for term in self.terms]
        out = self.__class__(new_terms, self.combine_func, bra=bra_copy)
        return out

    def match_inds(self, other: Union['Term','CombinedTerms']):

        for term in self.terms:
            term.ket.site_ind_id = other.ket_ind_id
            helper_quimb.match_inner_inds(term.ket, other.ref_ket if isinstance(other, CombinedTerms) else other.ket)

        if self.bra is not None and other.bra is not None:
            if not self.bra is other.bra:
                self.bra.site_ind_id = other.bra_ind_id
                helper_quimb.match_inner_inds(self.bra, other.bra)
                helper_quimb.match_inner_inds(self.bra, other.bra)

    def initialize(self):
        for term in self.terms:
            term.initialize()

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        return self.terms[0].projected_bra_to_ket(left_site_pos, nsites)


    def extend_env(self, i, direction, new_ket_site=None):
        for term in self.terms:
            term.extend_env(i, direction, new_ket_site=new_ket_site)


    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tensors: Sequence[qtn.Tensor]=None):

        if site_tensors is None:
            site_tensors = [None] * len(self.terms)

        if len(site_tensors) < 0:
            site_tensors = site_tensors + [None] * (len(self.terms) - len(site_tensors))

        eff_terms = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                     for (term, site_tens) in zip(self.terms, site_tensors)]
        ### already reindexed to ket inds

        return self.combine_func(eff_terms)


    def get_evaluated_rdm(self, left_site_pos: int, nsites: int, direction: SweepDirection,
                          transpose_inds: Sequence[str] = None, site_tens: 'qtn.Tensor' = None) -> 'qtn.Tensor':

        if site_tens is None:
            site_tens = self.get_evaluated_site(left_site_pos, nsites)  ## has ket inds

        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
        assert (not at_end), 'should not need to evaluate rdm for targeting if at end'

        ## horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
        if direction == SweepDirection.RIGHT:
            x_bond = self.ref_ket.bond(left_site_pos + nsites, left_site_pos + nsites - 1)
        else:
            x_bond = self.ref_ket.bond(left_site_pos, left_site_pos - 1)

        ## bonds to not contract over
        b_to_k = self.projected_bra_to_ket(left_site_pos, nsites)
        k_to_b = {v: k for k, v in b_to_k.items()}
        ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
        new_bra_iso = [k_to_b[k] for k in ket_iso]

        if transpose_inds is None:
            transpose_inds = [*ket_iso, *new_bra_iso]

        ## evaluated rdm
        site_conj = site_tens.conj()
        site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
        rdm = qtn.tensor_contract(site_conj, site_tens, output_inds=transpose_inds)

        ## self rdm; include only if term.bra is self.bra
        for term in self.terms:
            if term.bra is self.bra:
                term_rdm = term.get_evaluated_rdm(left_site_pos, nsites, direction, transpose_inds=transpose_inds)
                rdm = helper.add_tensors(rdm, term_rdm)

        return rdm


##############################
### combine term functions ###
##############################

def term_sum(terms: Sequence['Term'], left_site_pos: int, nsites: int, site_tensors: Sequence[qtn.Tensor]
             ) -> 'qtn.Tensor':

    if site_tensors is None:
        site_tensors = [None] * len(terms)

    if len(site_tensors) < 0:
        site_tensors = site_tensors + [None] * (len(terms) - len(site_tensors))

    eff_terms = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                 for (term, site_tens) in zip(terms, site_tensors)]
    return helper_tn.sum_tens(eff_terms)


def term_elem_mult(terms: Sequence['Term'], left_site_pos: int, nsites: int, site_tensors: Sequence[qtn.Tensor]
                   ) -> 'qtn.Tensor':
    if site_tensors is None:
        site_tensors = [None] * len(terms)

    if len(site_tensors) < 0:
        site_tensors = site_tensors + [None] * (len(terms) - len(site_tensors))

    eff_terms = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                 for (term, site_tens) in zip(terms, site_tensors)]
    return helper_tn.prod_tens(eff_terms)


def term_apply(terms: Sequence['Term'], left_site_pos: int, nsites: int, site_tens: Optional['qtn.Tensor'] = None
               ) -> 'qtn.Tensor':
    eff_ket = terms[0].get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)

    for term in terms[1:]:
        eff_ops = term.get_eff_operator(left_site_pos, nsites, return_combined=False)
        output_to_input_inds = terms[-1].projected_bra_to_ket(left_site_pos, nsites)
        eff_ket = tc.apply_effective_op(eff_ket, eff_ops, output_to_input_inds=output_to_input_inds)

    return eff_ket


def term_euler(terms: Sequence['Term'], left_site_pos: int, nsites: int, site_tens: Optional['qtn.Tensor'] = None
               ) -> 'qtn.Tensor':

    eff_ket = terms[0].get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
    deriv = terms[1].get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
    eff_ket = helper.add_tensors(eff_ket, deriv, )

