from abc import ABC

import helper_quimb
from setup.defaults import *
import local_solvers.helper_tn as helper_tn

from local_solvers.defaults import *
from local_solvers.mps_classes import MPS
from local_solvers.local_evaluator import LocalEvaluator
from local_solvers.local_dmrg_eval import DMRGEvaluator
from local_solvers.local_cross_eval import CrossEvaluator
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
import local_solvers.tensor_callables as tc
import helper_TE

if TYPE_CHECKING:
    from grid import Grid
    from setup.configs import DerivativeConfiguration


class TimeIntegMethod(IntEnum):
    Euler = 1
    RK4 = 4
    CN = 223
    CN6 = 226
    EXACT = 0



class TimeIntegrator(LocalEvaluator, ABC):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 linear_operators: Sequence['qtn.MatrixProductOperator'],
                 sources: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
                 nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']] = None,
                 # constraints: Sequence[Sequence[Union['Term_DMRG','Term_Cross']]] = None,
                 constraints: Sequence[Sequence[Union[qtn.MatrixProductState, qtn.MatrixProductOperator]]] = None,
                 constraint_vals: Sequence[Union[float, qtn.MatrixProductState, Term]] = None,
                 constraint_funcs: Sequence[Callable] = None,   ## expressions that should yield 0 / be minimized (via scipy)
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'TimeIntegrator' = None,
                 # combine_terms_func: 'Callable' = None,
                 dt = 0.1, time = None,
                 te_order_target = 4, te_order_final = 4,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                 ):

        if copy_obj is not None:
            super().__init__(ket_state, None, copy_obj=copy_obj)

            self.linear_operators = copy_obj.linear_operators
            self.sources = copy_obj.sources
            # self.nonlinear_terms = copy_obj.nonlinear_terms
            self.constraints = [c.copy() for c in copy_obj.constraints]
            self.constraint_vals = copy_obj.constraint_vals

            self.self_term = self.terms[0]
            num_lin = len(copy_obj.linear_terms)
            num_src = len(copy_obj.source_terms)
            num_nlin = len(copy_obj.nonlinear_terms)
            num_cons = len(self.constraints)
            self.linear_terms = self.terms[1: num_lin + 1]
            self.source_terms = self.terms[num_lin + 1: num_lin + 1 + num_src]
            self.nonlinear_terms = self.terms[num_lin + num_src + 1: num_lin + num_src + num_lin + 1]
            self.constraint_terms = self.terms[num_lin + num_src + num_nlin + 1:
                                               num_lin + num_src + num_nlin + num_cons + 1]
                                    # copy_obj.constraint_terms
            cons_val_terms = self.terms[num_lin + num_src + num_lin + num_cons + 1:]
            iter_val_terms = iter(cons_val_terms)
            self.constraint_vals = [next(iter_val_terms) if isinstance(c,Term) else c for c in copy_obj.constraint_vals]

            self.dt = copy_obj.dt
            self.time = copy_obj.time
            self.te_order_target = copy_obj.te_order_target
            if self.te_order_target == 0:
                self.te_order_target = 223
            self.te_order_final = copy_obj.te_order_final

        else:

            self.te_order_target = te_order_target if te_order_target != 0 else 223
            self.te_order_final = te_order_final

            self.linear_operators = linear_operators
            self.sources = sources if sources is not None else []
            self.nonlinear_terms = nonlinear_terms if nonlinear_terms is not None else []
            self.constraints = constraints if constraints is not None else []
            self.constraint_vals = [*constraint_vals] if constraint_vals is not None else []

            self.self_term: Term = None
            self.linear_terms: list[Term] = []
            self.source_terms: list[Term] = []
            self.constraint_terms: list[Term] = []
            self.constraint_val_terms = []

            terms = self.initialize_terms(ket_state)

            super().__init__(ket_state, terms, direction, max_bond, conv_tol=conv_tol, max_iter=max_iter,
                             max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter, copy_obj=copy_obj,
                             grid=grid, ax_deriv_configs=ax_deriv_configs)  # , combine_terms_func=combine_terms_func)

            self.dt = dt
            self.time = time


    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'],
                         **kwargs
                         # linear_operators: Sequence[qtn.MatrixProductOperator],
                         # sources: Optional[Sequence[Union['qtn.MatrixProductState','MPS']]],
                         # nonlinear_terms: Optional[Sequence[Union['Term_DMRG','Term_Cross']]]
                         ):

        LocalTerm = self.term_class()

        term_self = LocalTerm(ket_state)
        self.self_term = term_self

        if len(self.linear_operators) > 0:
            term_linear = [LocalTerm(ket_state, operators=[mpo for mpo in self.linear_operators])]
        else:
            term_linear = []
        self.linear_terms = term_linear

        term_sources = [LocalTerm(source) for source in self.sources]
        # term_sources = [LocalTerm(source, bra=self.ket) for source in self.sources]
        self.source_terms = term_sources
        print('self.source terms', self.source_terms)

        nonlinear_terms = self.nonlinear_terms

        term_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, qtn.MatrixProductState):
                term_constraints += [LocalTerm(constraint, bra=ket_state)]  ## conjugate of what it should be
            else:
                tmp = [constraint] if not isinstance(constraint, (list,tuple)) else constraint
                term_constraints += [LocalTerm(ket_state, operators=tmp)]
        self.constraint_terms = term_constraints

        constraint_val_all = []
        constraint_val_terms = []
        for cv in self.constraint_vals:
            if isinstance(cv, qtn.MatrixProductState):
                constraint_val_terms += [LocalTerm(cv, bra=ket_state)]  ## the conjugate of what it should be
                constraint_val_all += [constraint_val_terms[-1]]
            else:
                constraint_val_all += [cv]  # float or Term
        self.constraint_val_terms = constraint_val_all

        return [term_self, *term_linear, *term_sources, *nonlinear_terms, *term_constraints, *constraint_val_terms]


    def _set_local_solve_func(self, te_order: int):
        # te_order = 4
        # te_order = 223
        # te_order = 0
        print('te order', te_order)
        if te_order == TimeIntegMethod.RK4:
            func = self.local_rk4
        elif te_order == TimeIntegMethod.EXACT:
            func = self.local_exact_solve
        elif te_order == TimeIntegMethod.Euler:
            func = self.local_euler
        elif te_order == TimeIntegMethod.CN:
            func = self.local_implicit_solve
        elif te_order == TimeIntegMethod.CN6:
            def func(left_site_pos: int, nsites: int, return_intermediates=False):
                self.local_implicit_solve(left_site_pos, nsites, return_intermediates=return_intermediates,
                                          weight=0.6)
        elif te_order == TimeIntegMethod.EXACT:
            func = self.local_exact_solve
        else:
            raise ValueError(f'invalid te_order ({te_order}) provided')

        self._local_solve_func = func

    # @classmethod
    # def euler_solver(cls, dt: Numeric,
    #                  init_state: Union[qtn.MatrixProductState, 'MPS'],
    #                  linear_operators: Sequence['qtn.MatrixProductOperator'],
    #                  sources: Optional[Sequence[Union['Term', 'qtn.MatrixProductState', MPS]]] = None,
    #                  nonlinear_terms: Optional[Sequence['Term']] = None,
    #                  ):
    #     """ df/dt = Af + nonlinear terms + sources
    #     """
    #     TermClass = cls.term_class
    #
    #     lin_term = TermClass(init_state.copy(), operators=linear_operators)
    #
    #     src_terms = []
    #     for src in sources:   ## include capability to deal with time-dependence
    #         if isinstance(src, qtn.MatrixProductState):
    #             src = TermClass(src.copy())
    #         elif isinstance(src, Term):
    #             pass
    #         else:
    #             raise TypeError('invalid type for source terms')
    #         src_terms += [src]
    #
    #     nonlin_terms = [] if nonlinear_terms is None else nonlinear_terms
    #
    #     out = cls(init_state, [lin_term, src_terms, nonlin_terms], dt=dt, time=0., )


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None
                    ) -> tuple[qtn.Tensor, Numeric]:

        print('new TimeIntegrator site solve', left_site_pos)

        plot_partitions = False
        if plot_partitions:
            copy = self.ket.copy()
            l_partition = copy[:left_site_pos]
            r_partition = copy[left_site_pos + nsites:]

            if l_partition.num_tensors > 0:
                l_inds = [copy.site_ind_id.format(i) for i in range(left_site_pos)]
                xl_bond = copy.bond(left_site_pos, left_site_pos - 1)
                l_size = copy.bond_size(left_site_pos, left_site_pos - 1)

                l_tens = l_partition.contract()
                l_tens.transpose(xl_bond, *l_inds, inplace=True)
                plt.figure()
                plt.plot(l_tens.data.reshape(l_size, -1).T)
                plt.title(f'left bases, {left_site_pos}')

            if r_partition.num_tensors > 0:
                r_inds = [copy.site_ind_id.format(i) for i in range(left_site_pos + nsites, self.ket.L)]
                xr_bond = copy.bond(left_site_pos + nsites - 1, left_site_pos + nsites)
                r_size = copy.bond_size(left_site_pos + nsites - 1, left_site_pos + nsites)

                r_tens = r_partition.contract()
                r_tens.transpose(xr_bond, *r_inds, inplace=True)
                plt.figure()
                plt.plot(r_tens.data.reshape(r_size, -1).T)
                plt.title(f'right bases, {left_site_pos}')

            plt.show()

        at_end = (left_site_pos == self.L - nsites) if self.direction == SweepDirection.RIGHT else (left_site_pos == 0)

        if at_end:
            self._set_local_solve_func(self.te_order_final)
        else:
            self._set_local_solve_func(self.te_order_target)
        out = super()._site_solve(left_site_pos, nsites, site_tens=site_tens)   ## target_rdm_list, error

        # print('TE site solve', out)
        if not at_end:
            out_targets = out[0]
            nonlin_targets = []
            for nl_term in self.nonlinear_terms:
                nonlin_targets += [nl * out_targets[0].norm()/nl.norm() for nl in nl_term._proj_vec_targets]
                print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])

            out = (*out[0], *nonlin_targets), out[1]

        return out


    def deriv_func(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, time: Numeric = None
                   ) -> 'qtn.Tensor':
        """ compute df/dt = Af + sources + nonlinear terms O[f]
        """
        site_tens_list = []
        ## do something with time with nonlinear term
        for term in self.linear_terms:
            site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]

        for term in self.nonlinear_terms + self.sources:
            # data = term.ket.to_dense().reshape(-1)
            # plt.figure()
            # plt.plot(data * 10**(term.ket.exponent))
            # plt.title(f'source term, {term.ket.exponent}')
            # plt.show()
            # print('DERIV FUNC site tens', site_tens.norm() if site_tens is not None else None)
            # print('site_tens norm', site_tens.norm() if site_tens is not None else None)
            # tmp = term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # print('vec blcock norm', tmp.norm())
            site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
            ## doesn't perform well if incorporate site_tens


        # print('deriv func', site_tens_list)
        # print('DERIV FUNC OUT', [t.norm() for t in site_tens_list])
        tot_site = helper_tn.sum_tens(site_tens_list)
        return tot_site


    def euler_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                   deriv: 'qtn.Tensor' = None, site_tens: 'qtn.Tensor' = None):
        """ compute df/dt = Af + sources + nonlinear terms
                assumes the first term is the linear term, followed by sources and nonlinear terms
                (though those are effectively the same)
            for the given effective operators
        """
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # direction = self.direction
        # at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])

        # ket_x = self.terms[0].proj_vec if site_tens is None else site_tens
        if site_tens is None:
            # ket_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            ket_x = site_tens

        # print('site tens', site_tens.inds, site_tens.data)

        if deriv is None:
            deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)

        dt = self.dt if dt is None else dt
        # print('EULER DT', dt)
        deriv = deriv.copy()    ## already excludes self.ket.exponent
        deriv.modify(apply=lambda x: x * dt)
        # print('EULER deriv', deriv.norm())
        # print('EULER ket x', ket_x.norm())
        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10**(-self.ket.exponent))  ## remove exponent
        # print('EULER ket x', ket_x.norm())
        site_tens_list = [ket_x, deriv]

        # print(self.terms[0].bra is self.ket)
        # print(self.terms[0].vec_block.bra is self.ket)
        # print('ket_x', ket_x)
        # print('deriv', deriv)

        tot_site = helper_tn.sum_tens(site_tens_list)
        # print('euler tot site', tot_site.norm())
        tot_site.modify(apply=lambda x: x * 10 ** self.ket.exponent)  ## include exponent for input into bra
        return tot_site


    def exponential_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                         # deriv: 'qtn.Tensor' = None,
                         site_tens: 'qtn.Tensor' = None, return_intermediates=False):
        """ compute df/dt = Af + sources + nonlinear terms
                assumes the first term is the linear term, followed by sources and nonlinear terms
                (though those are effectively the same)
            ## here we focus on Euler time step (EXP1, ETD1)
            ## u_{n+1} = exp(dt J_n) u_n + dt phi_1(dt J_n) N_n  where J: effective operator, N: effective source terms
        """
        dt = self.dt if dt is None else dt
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # direction = self.direction
        # at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])

        # ket_x = self.terms[0].proj_vec if site_tens is None else site_tens
        if site_tens is None:
            # # ket_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            # ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            ket_x = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            ket_x.reindex(self.self_term.vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            ket_x = site_tens

        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))  ## remove exponent
        sq_shape = ket_x.size

        ## nonlinear operators
        sources = []
        for term in self.nonlinear_terms + self.source_terms:
            # data = term.ket.to_dense().reshape(-1)
            # plt.figure()
            # plt.plot(data * 10**(term.ket.exponent))
            # plt.title(f'source term, {term.ket.exponent}')
            # plt.show()
            # print('DERIV FUNC site tens', site_tens.norm() if site_tens is not None else None)
            sources += [term.get_evaluated_site(left_site_pos, nsites)]  # , site_tens=site_tens)]

        if len(sources) > 0:
            tot_sources = helper_tn.sum_eff_TNs(sources)
            tot_sources.transpose_like(ket_x, inplace=True)
            tot_sources.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))  ## remove exponent
        else:
            tot_sources = None


        ## effective operators
        eff_ops = []
        for term in self.linear_terms:
            eff_ops += [*term.get_eff_operator(left_site_pos, nsites)[0]]
            ## dict for each "level": list of effective operators

        if len(eff_ops) > 0:
            term = next(iter(self.linear_terms))
            b2k_inds = term.projected_bra_to_ket(left_site_pos, nsites)
            k2b_inds = {k: b for b, k in b2k_inds.items()}

            ket_inds = ket_x.inds
            bra_inds = [k2b_inds[k] for k in ket_inds]
            tot_eff_op = helper_tn.sum_eff_TNs(eff_ops)
            tot_eff_op.transpose(*bra_inds, *ket_inds, inplace=True)

            op_data = tot_eff_op.data.reshape(sq_shape,sq_shape)

            eff_evals, eff_evecs = np.linalg.eig(op_data)
            inv_eff_evecs = np.linalg.inv(eff_evecs)
            exp_eff_op = (eff_evecs * np.exp(dt * eff_evals)) @ inv_eff_evecs

            phi1 = None
            if tot_sources is not None:
                print('tot sources is not None')

                ## phi_1(J) = J^{-1} (exp(J) - I)
                ## phi_2(J) = J^{-2} (exp(J) - I - J)
                ## phi_3(J) = J^{-3} (exp(J) - I - J - J^2/2) ...
                ## phi_k(J) = J^{-1} (phi_{k-1}(J) - phi_{k-1}(J))

                ## here we focus on Euler time step (EXP1, ETD1)
                ## u_{n+1} = exp(dt J_n) u_n + dt pi_1(dt J_n) N_n
                ## where J: effective operator, N: effective source / nonlinear terms

                tmp = (np.exp(eff_evals * dt) - 1) / eff_evals / dt
                phi1_D = np.where(np.abs(eff_evals) > 10e-15 / dt, tmp, 1.)
                phi1 = (eff_evecs * phi1_D) @ inv_eff_evecs

        else:
            exp_eff_op = None
            phi1 = None         ## = 1

        ## here we focus on Euler time step (EXP1, ETD1)
        ## u_{n+1} = exp(dt J_n) u_n + dt phi_1(dt J_n) N_n  where J: effective operator, N: effective source terms

        ## linear term
        lin_term = exp_eff_op @ ket_x.data.reshape(-1) if exp_eff_op is not None else ket_x.data.reshape(-1)
        ## nonlinear/source term
        nl_term = phi1 @ tot_sources.data.reshape(-1) if phi1 is not None else \
                        (tot_sources.data.reshape(-1) if tot_sources is not None else 0.)

        out = lin_term + nl_term * dt
        tot_site = qtn.Tensor(out.reshape(ket_x.shape), inds = ket_x.inds)
        # tot_site.modify(apply=lambda x: x * 10 ** self.ket.exponent)  ## include exponent for input into bra

        # print('tot site', tot_site.norm(), ket_x.norm())

        if return_intermediates:
            return tot_site, [tot_site, ket_x]

        return tot_site



    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        # print('RK4 DT', dt, left_site_pos)
        self.terms[0].get_evaluated_site(left_site_pos, nsites)
        state0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        state0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        def deriv_func(site_tens: 'qtn.Tensor', time=None , **kwargs):
            return self.deriv_func(left_site_pos, nsites, site_tens=site_tens)

        def euler_func(site_tens: 'qtn.Tensor', dt, deriv0: qtn.Tensor=None, time=None, **kwargs):
            return self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens)

        def scale_func(site_tens: 'qtn.Tensor', coeff: Numeric, inplace: bool = False):
            site_tens = site_tens if inplace else site_tens.copy()
            site_tens.modify(apply=lambda x: x * coeff)
            return site_tens

        def add_func(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False, **kwargs):
            return helper_quimb.add_tensors(tens1, tens2, inplace=inplace)

        out = helper_TE.rk4(state0, dt, euler_func, deriv_func, add_func, scale_func,
                            return_intermediates=return_intermediates)

        ## remove exponent from result
        if return_intermediates:  ## new sites, intermediates
            out[0].modify(apply=lambda x: x * 10**(-self.ket.exponent))
            out[1][0].modify(apply=lambda x: x * 10**(-self.ket.exponent))

            # nonlin_targets = []
            # for nl_term in self.nonlinear_terms:
            #     nonlin_targets += [nl * state0.norm()/nl.norm() for nl in nl_term._proj_vec_targets]
            #     print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])
            #
            # intermediates = (*out[1], *nonlin_targets)
            # return out[0], intermediates

        else:
            out.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))

        return out


    def setup_crank_nicolson(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                             site_tens: qtn.Tensor = None, weight: Numeric = 0.5):
        """ implicit time differentiation:  dy/dt = F(y) + b + nonlinear(y)
            y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
            y_(n+1) = y_(n) + h * weight * F(y_(n+1)) + h * (1-weight) * F(y_(n))

            dy/dt = Ay + b, assuming b is constant in time
            y_(n+1) = y_(n) + h ( weight * (A(y_(n+1)) + b) + (1-weight) * (A(y_(n)) + b) )
            (I - A * h * weight) y_[n+1] = (I + (1 - weight) * h * A) y_[n] + h * b
        """
        dt = self.dt if dt is None else dt

        lin_ops = []
        for lin_term in self.linear_terms:
            op = lin_term.get_eff_operator(left_site_pos, nsites)[0]
            lin_ops += [op1.copy() for op1 in op]

        # lin_ops = self.linear_terms[0].get_eff_operator(left_site_pos, nsites)[0]
        # lin_ops = [op.copy() for op in lin_ops]
        # print('lin ops', lin_ops)

        for op in lin_ops:
            helper_quimb.scalar_multiply(op, - dt * weight, inplace=True)
            # op.exponent += [dt * weight]

        ## identity
        bra_to_ket_inds = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
        bra_inds = [b for b in bra_to_ket_inds.keys() if b is not None]
        ket_inds = [bra_to_ket_inds[b] for b in bra_inds]
        # sq_shape = [lin_ops[0].ind_size(b) for b in bra_inds]
        sq_shape = [self.terms[0].ket[left_site_pos].ind_size(k) for k in ket_inds]
        sq_size = int(np.prod(sq_shape))
        iden = np.eye(sq_size).reshape(*sq_shape,*sq_shape)
        iden_tens = qtn.Tensor(iden, inds=[*bra_inds, *ket_inds])
        lhs = lin_ops + [iden_tens]

        ## RHS
        ## the (1 - weight) * dt * F(f) term
        ### assumes site_tens is not scaled by ket (term bra) exponent (exponent is incorporated)
        # rhs = [self.self_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()]  ## linop

        # rhs = [self.linear_terms[0].get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()]  ## linop
        rhs = []
        for lin_term in self.linear_terms:
            op = lin_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()  ## linop
            rhs += [op.copy()]

        rhs += [term.get_evaluated_site(left_site_pos, nsites).copy()
                for term in self.source_terms + self.nonlinear_terms]    ## sources, nonlin
        for rhs_term in rhs:
            rhs_term.modify(apply=lambda x: x * (1 - weight) * dt)

        ## the I * y[n] term
        if site_tens is None:
            # rhs += [self.self_term.vec_block.projected_site.copy()]
            proj_vec = self.self_term.get_evaluated_site(left_site_pos, nsites)
            proj_vec.reindex(bra_to_ket_inds, inplace=True)
            rhs += [proj_vec]
        else:
            site_tens = site_tens.copy()
            site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
            rhs += [site_tens]
        # print('site tens norm', rhs[-1].norm())

        # print('rhs', rhs, [t.norm() for t in rhs])

        return lhs, rhs


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None, site_tens: qtn.Tensor = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        lhs, rhs = self.setup_crank_nicolson(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens,
                                             **solver_kwargs)
        init_guess = rhs[-1].copy()
        output_to_input_inds = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)

        rhs = helper_tn.sum_tens(rhs)
        solve_cgd = tc.get_cgd_func(has_constraints=(len(self.constraint_terms) > 0))

        constraint_vals = []
        for cv in self.constraint_vals:
            if isinstance(cv, Term):
                constraint_vals += [cv.get_evaluated_site(left_site_pos, nsites)]   ## <x|c>
                ## do not include site_tens because ket corresponds to reference vec
            else:
                constraint_vals += [qtn.Tensor(cv)]

        constraint_ops = []
        for ct in self.constraint_terms:
            if len(ct.operators) > 0:
                constraint_ops += [ct.get_eff_operator(left_site_pos, nsites)[0]]      ## <x|A|x>
            else:
                constraint_ops += [ct.get_evaluated_site(left_site_pos, nsites)[0]]    ## <x|c>


        out, err = solve_cgd(rhs, lhs, output_to_input_inds, init_guess=init_guess,
                             constraint_tns=constraint_ops, constraint_vals=constraint_vals)

        if return_intermediates:
            return out, []
        else:
            return out


    def local_exact_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                          dt: Numeric = None, time: Numeric = None, site_tens: qtn.Tensor = None,
                          **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        out = self.exponential_func(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens,
                                    return_intermediates=return_intermediates)
        return out



class TDDMRG(TimeIntegrator, DMRGEvaluator):
    """ RK4 follows Feiguin and White
        CN obtains three intermediate states by simply taking three time steps
    """

    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric = None, time: Numeric = None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # s0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        # s0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        # print('local RK4')
        # exit()

        dt = self.dt if dt is None else dt

        # if left_site_pos == 2:
        #     exit()

        if return_intermediates:
            out, rk_states = super().local_rk4(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time)

            # k0, = rk_states

            s0, k1, k2, k3, k4 = rk_states      ## initial x, stage 1, stage 2, stage 3
            # print('s0', s0.norm())
            # print('derive norms', k1.norm() * dt, k2.norm() * dt, k3.norm() * dt, k4.norm() * dt)

            k1.transpose_like(s0, inplace=True)
            k2.transpose_like(s0, inplace=True)
            k3.transpose_like(s0, inplace=True)
            k4.transpose_like(s0, inplace=True)

            ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            ## include weights here
            psi03 = s0.copy()
            psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            psi13 = s0.copy()
            psi13.modify(apply=lambda x: x + dt * (1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data)))
            psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi23 = s0.copy()
            psi23.modify(apply=lambda x: x + dt * (1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data)))
            psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi33 = s0.copy()
            psi33.modify(apply=lambda x: x + dt * (1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)))
            psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))

            ## out and psi33 are the same thing, but psi33 is scaled

            return out, (psi03, psi13, psi23, psi33)

        else:
            # exit()

            return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # print('local implicit')
        # exit()

        dt = self.dt if dt is None else dt
        # out = super().local_implicit_solve(left_site_pos, nsites, dt=dt, time=time, **solver_kwargs)
        out1 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, **solver_kwargs)
        out1_ = out1.copy()
        out1_.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        out2 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out1_, **solver_kwargs)
        out2_ = out2.copy()
        out2_.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        out3 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out2_, **solver_kwargs)
        out = out3.copy()
        #
        out0 = self.terms[0].vec_block.projected_site.copy()
        out0.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        out0.modify(apply=lambda x: x * np.sqrt(1. / 3))
        out1.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out2.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out3.modify(apply=lambda x: x * np.sqrt(1. / 3))

        if return_intermediates:
            # return out, (out)  # (out0, out1, out2)
            return out, (out0, out1, out2, out3)
        else:
            return out


class TDDMRG_mod(TimeIntegrator, ABC):
    """ RK4 follows Feiguin and White
        CN obtains three intermediate states by simply taking three time steps
    """
    _rdm_points = [0, 1]
    _rdm_weights = [1./2, 1./2]

    def set_rdm_locs(self, new_points: Sequence[Numeric], new_weights: Sequence[Numeric]):
        assert(np.sum(new_weights) == 1), f'new weights must sum to 1.0, not {np.sum(new_weights)}'
        assert(np.all([0 <= pt <= 1 for pt in new_points])), 'new points must be between 0 and 1'
        assert(len(new_points) == len(new_weights)), f'length of provided lists must be the same'

        self._rdm_points = new_points
        self._rdm_weights = new_weights

    def get_rdm_locs(self):
        return self._rdm_points, self._rdm_weights

    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric = None, time: Numeric = None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        if return_intermediates:
            out, rk_states = super().local_rk4(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time)

            k0, k1, k2, k3 = rk_states  ## initial x, stage 1, stage 2, stage 3
            k4 = out

            k1.transpose_like(k0, inplace=True)
            k2.transpose_like(k0, inplace=True)
            k3.transpose_like(k0, inplace=True)
            k4.transpose_like(k0, inplace=True)

            ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            ## include weights here
            psi03 = k0.copy()
            psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            psi13 = k0.copy()
            psi13.modify(apply=lambda x: x + 1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data))
            psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi23 = k0.copy()
            psi23.modify(apply=lambda x: x + 1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data))
            psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi33 = k0.copy()
            psi33.modify(apply=lambda x: x + 1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data))
            psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))

            ## out and psi33 are the same thing

            return out, (psi03, psi13, psi23)

    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        dt = self.dt if dt is None else dt
        out1 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, **solver_kwargs)
        out2 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out1,
                                            **solver_kwargs)
        out3 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out2,
                                            **solver_kwargs)

        out0 = self.terms[0].proj_vec.copy()
        out0.modify(apply=lambda x: x * np.sqrt(1. / 3))
        out1.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out2.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out3.modify(apply=lambda x: x * np.sqrt(1. / 3))

        return out3, (out0, out1, out2)


class TDVP_DMRG(TimeIntegrator, DMRGEvaluator):
    """ performs TDVP
    """
    def _site_time_evolution(self, site_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
        self.dt = -self.dt
        out, err = super()._site_solve(left_site_pos, 1, site_tens=site_tens)
        self.dt = -self.dt
        return out[0], err

    def _bond_time_evolution(self, bond_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
        self.dt = -self.dt
        print('bond tens', bond_tens)
        out, err = self._bond_solve(left_site_pos, site_tens=bond_tens)
        self.dt = - self.dt
        return out[0], err

    def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:

        self._set_local_solve_func(TimeIntegMethod.RK4)
        out, err = super()._bond_solve(left_site_pos, site_tens=site_tens)
        return out, err

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs=None):
        """ update ket, bra with new_site
            i: int of mps site
            canonicalize and then back-propagate "bond" (if not at end)
        """
        print('TDVP DMRG update 1 site', site_i, direction)
        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        print('at end?', at_end)
        if at_end:
            super()._update_1site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        print('update site i', site_i, self.L)

        x_ind = self.ket.bond(i, i + direction)
        left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
        q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=x_ind+'_tmp')

        if direction == SweepDirection.LEFT:
            bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
        else:  # direction is to the right
            bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
        r.reindex(bond_reindex_dict, inplace=True)

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=q.data)

        ## extend environments to include newly canonical site i
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        ## back-propagation of R
        print('back propagation of R', i, direction)
        print('self.cur_orthog', self.cur_orthog)
        back_i = i if direction == SweepDirection.RIGHT else i - 1
        new_r, err = self._bond_time_evolution(r, back_i, direction, -self.dt)
        new_r.reindex({v: k for k,v in bond_reindex_dict.items()},inplace=True)
        next_site = qtn.tensor_contract(new_r, self.ket[i + direction])
        next_site.transpose_like(self.ket[i + direction], inplace=True)
        self.ket[i + direction].modify(data=next_site.data)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        ## canonicalize and then back-propagate "site" (if not at end)
        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        if at_end:
            super()._update_2site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.ket.bond(i, i + direction)
        left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
        q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=x_ind)

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=q.data)

        ## extend environments to include newly canonical site i
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        ## back-propagation of R
        new_q = self._site_time_evolution(r, i, direction, -self.dt)
        next_site = qtn.tensor_contract(new_q, self.ket[i + direction])
        next_site.transpose_like(self.ket[i + direction], inplace=True)
        self.ket[i + direction].modify(data=next_site.data)

        return


class TDVP_Krylov_DMRG(TimeIntegrator, DMRGEvaluator):
    """ performs TDVP but with projection based back-propagation
    """
    _ovlp_projector = None

    # def _site_time_evolution(self, site_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
    #     self.dt = -self.dt
    #     out = super()._site_solve(left_site_pos, 1, site_tens=site_tens)
    #     self.dt = -self.dt
    #     return out
    #
    # def _bond_time_evolution(self, bond_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
    #     self.dt = -self.dt
    #     out = super()._bond_solve(left_site_pos, site_tens=bond_tens)
    #     self.dt = - self.dt
    #     return out
    #
    # def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None
    #                 ) -> tuple[qtn.Tensor, Numeric]:
    #
    #     self._set_local_solve_func(TimeIntegMethod.RK4)
    #     out = super()._bond_solve(left_site_pos, site_tens=site_tens)
    #     return out

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs=None):
        """ update ket, bra with new_site
            i: int of mps site
            canonicalize and then back-propagate "bond" (if not at end)
        """
        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        if at_end:
            super()._update_1site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.ket.bond(i, i + direction)
        left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
        b2k = self.terms[0].projected_bra_to_ket(i,1)
        k2b = {v: k for k,v in b2k.items()}
        # bra_left_inds = [k2b[k_ind] for k_ind in left_inds]

        old_Q, old_R = qtn.tensor_split(self.ket[i], left_inds, absorb='right',
                                        bond_ind=f'p_{x_ind}', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

        new_Q, _ = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=f'p_{x_ind}')

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        new_Q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=new_Q.data)

        new_Q_conj = new_Q.reindex({f'p_{x_ind}': f'p_{x_ind}_'}).conj()
        ovlp_projector = qtn.tensor_contract(new_Q_conj, old_Q)

        new_R = qtn.tensor_contract(ovlp_projector, old_R)
        new_M = qtn.tensor_contract(new_R, self.ket[i + direction])
        current_M = self.ket[i + direction]
        new_M.transpose_like(current_M, inplace=True)
        current_M.modify(data=new_M.data)

        ## i think this should just be the same as computing
        ## new_M = qtn.tensor_contract(self.ket[i], self.ket[i+1], new_Q_conj)

        ## extend environments to include newly canonical site i
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        ## canonicalize and then back-propagate "site" (if not at end)
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)
        if at_end:
            super()._update_2site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.ket.bond(i, i + direction)
        left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
        b2k = self.terms[0].projected_bra_to_ket(i, 1)
        k2b = {v: k for k, v in b2k.items()}
        # bra_left_inds = [k2b[k_ind] for k_ind in left_inds]

        # old_Q, old_R = qtn.tensor_split(self.ket[i], left_inds, absorb='right',
        #                                 bond_ind=f'p_{x_ind}', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
        # old_M = qtn.tensor_contract(old_R, self.ket[i + direction])
        old_M0 = self.ket[i].copy()
        old_M1 = self.ket[i + direction].copy()

        new_Q, new_M = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=f'p_{x_ind}')

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        new_Q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=new_Q.data)

        new_Q_conj = new_Q.reindex({f'p_{x_ind}': f'p_{x_ind}_'}).conj()
        ovlp_projector = qtn.tensor_contract(old_M0, new_Q_conj)

        new_M1 = qtn.tensor_contract(ovlp_projector, old_M1)
        new_M1.transpose_like(old_M1, inplace=True)
        self.ket[i + direction].modify(data=new_M1.data)

        ## i think this should just be the same as computing
        ## new_M1 = qtn.tensor_contract(self.ket[i], self.ket[i+1], new_Q_conj)

        ## extend environments to include newly canonical site i
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        return