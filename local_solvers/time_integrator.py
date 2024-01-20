from abc import ABC

from setup.defaults import *

from local_solvers.mps_classes import MPS
from local_solvers.local_evaluator import LocalEvaluator
from local_solvers.local_cross_eval import CrossEvaluator
from local_solvers.local_dmrg_eval import DMRGEvaluator
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
from local_solvers.terms_combined import CombinedTerms


class TimeIntegrator(LocalEvaluator, ABC):

    @classmethod
    def euler_solver(cls, dt: Numeric,
                     state: Union[qtn.MatrixProductState,'MPS'],
                     linear_operators: Sequence['qtn.MatrixProductOperator'],
                     source_terms: Optional[Sequence[Union['Term','qtn.MatrixProductState',MPS]]] = None,
                     nonlinear_terms: Optional[Union['Term','CombinedTerms','Callable',
                                                     tuple['qtn.MatrixProductOperator', 'Callable']]] = None,
                     te_order: int = 4,
                     ):
        """ df/dt = Af + nonlinear terms + sources
        """
        out = state.copy()

        TermClass = cls.term_class

        lin_terms = TermClass(state, out, linear_operators, mpo_poly=[1, dt] )

        src_terms = []
        for term in source_terms:   ## include capability to deal with time-dependence
            if isinstance(term, qtn.MatrixProductState):
                term = TermClass(term, out)
            elif isinstance(term, Term):
                pass
            else:
                raise TypeError('invalid type for source terms')
            src_terms += [term]

        nonlin_terms = []
        if isinstance(nonlinear_terms, tuple):
            TermClass(state, out, )



        cls(state, terms)
