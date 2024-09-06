import helper_quimb
import sample_helper_quimb
from setup_.defaults import *
import local_solvers.helper_tn as helper_tn

from local_solvers.defaults import *
from local_solvers.mps_classes import MPS
from local_solvers.local_evaluator import LocalEvaluator
from local_solvers.local_dmrg_eval import DMRGEvaluator
from local_solvers.local_cross_eval import CrossEvaluator
from local_solvers.terms_3 import Term_Cross
import local_solvers.tensor_callables as tc
import helper_TE
from local_solvers.local_cross_eval import local_cross_evaluator

from local_solvers.time_integrator import TimeIntegrator

class TDCross(TimeIntegrator, CrossEvaluator):
    """
    """

    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'], cur_orthog=None, **kwargs
                         # euler_state: Union['MPS', 'qtn.MatrixProductState'] = None,
                         # linear_operators: Sequence[qtn.MatrixProductOperator],
                         # sources: Sequence[Union['qtn.MatrixProductState', 'MPS']],
                         # nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']]
                         ):

        num_tiers = None   # if self.te_order_target in [4, 223, 226] else None
        # ket_state = ket_state.copy()

        LocalTerm = self.term_class()

        # LocalTerm = self.term_class()
        term_self = LocalTerm(ket_state) # if euler_state is None else LocalTerm(euler_state)
        self.self_term = term_self

        ## df/dt terms: linear terms, sources, nonlinear terms
        # terms = super()._initialize_terms(ket_state, linear_operators, sources, nonlinear_terms)
        term_linear = []
        if len(self.linear_operators) > 0:
            term_linear = [LocalTerm(ket_state, operators=[mpo for mpo in self.linear_operators],
                                     num_tiers=num_tiers)]
        self.linear_terms = term_linear

        term_sources = [LocalTerm(source) for source in self.sources]
        self.source_terms = term_sources

        nonlinear_terms = self.nonlinear_terms

        ###### extra terms ######
        ### time mpo list ###
        extra_terms_list = []
        if self.time_mpo_list is not None:
            for key, ops_list in self.time_mpo_list.items():
                self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo for mpo in ops_list])]
                extra_terms_list += self.extra_terms_dict[key]

        return [term_self, *term_linear, *term_sources, *nonlinear_terms, *extra_terms_list]


    # def deriv_func(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, time: Numeric = None
    #                ) -> 'qtn.Tensor':
    #     """ compute df/dt = Af + sources + nonlinear terms O[f]
    #     """
    #     site_tens_list = []
    #     ## do something with time with nonlinear term
    #     for term in self.terms[1:]:
    #         # print('DERIV FUNC site tens', site_tens.norm() if site_tens is not None else None)
    #         site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
    #     # print('deriv func', site_tens_list)
    #     # print('DERIV FUNC OUT', [t.norm() for t in site_tens_list])
    #     tot_site = helper_tn.sum_tens(site_tens_list)
    #     return tot_site


    # def euler_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
    #                deriv: 'qtn.Tensor' = None, site_tens: 'qtn.Tensor' = None):
    #     """ compute df/dt = Af + sources + nonlinear terms
    #             assumes the first term is the linear term, followed by sources and nonlinear terms
    #             (though those are effectively the same)
    #         for the given effective operators
    #         term0:  Identity * f
    #         term1:  Af
    #         to be implemented:
    #             term2:  nonlinear terms
    #             term3:  source terms
    #     """
    #     # if site_tens is None:
    #     #     # ket_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
    #     #     ket_x = self.terms[1].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #     #     ket_x.reindex(self.terms[1].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
    #     # else:
    #     #     ket_x = site_tens
    #
    #     # print('site tens', site_tens.inds, site_tens.data)
    #
    #     if deriv is None:
    #         deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)
    #
    #     dt = self.dt if dt is None else dt
    #     # print('EULER DT', dt)
    #     deriv = deriv.copy()    ## already excludes self.ket.exponent
    #     deriv.modify(apply=lambda x: x * dt)
    #     # print('EULER deriv', deriv.norm())
    #     # print('EULER ket x', ket_x.norm())
    #
    #     ## original state * I (with the same output indices as self.terms[1].vec_block)
    #     ket_x0 = self.terms[0].get_evaluated_site(left_site_pos, nsites).copy()
    #     ## ket exponent already removed
    #     # print('EULER ket x', ket_x.norm())
    #     site_tens_list = [ket_x0, deriv]
    #
    #     # print(self.terms[0].bra is self.ket)
    #     # print(self.terms[0].vec_block.bra is self.ket)
    #     # print('ket_x', ket_x)
    #     # print('deriv', deriv)
    #
    #     tot_site = helper_tn.sum_tens(site_tens_list)
    #     # print('euler tot site', tot_site.norm())
    #     tot_site.modify(apply=lambda x: x * 10 ** self.ket.exponent)  ## include exponent for input into bra
    #     return tot_site


    def local_euler(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        print('local EULER', self.time)
        dt = self.dt if dt is None else dt
        out = self.euler_func(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens)
        out.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
        return [out]


    # def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
    #               time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
    #
    #     dt = self.dt if dt is None else dt
    #     return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time,
    #                              site_tens=site_tens)
    #     # return self.local_euler(left_site_pos, nsites, dt=dt, time=time)

    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        print('local RK4', self.time)
        dt = self.dt if dt is None else dt

        if return_intermediates:
            out, rk_states = super().local_rk4(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time,
                                               site_tens=site_tens)

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

            print('TD-Cross with targeting')
            return out, psi03, psi13, psi23
            # print('only 0, dt target')
            # return out, (psi03, psi33)

        else:
            # exit()

            out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)
            return [out]


    # def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
    #               dt: Numeric=None, time: Numeric=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
    #     """
    #     state1 = state0 + deriv0(state0) * 0.5 * dt  --> sel inds x; sel inds 0 -> x
    #     state2 = state0 + deriv1(state1) * 0.5 * dt  --> sel inds x; sel inds x -> x
    #     state3 = state0 + deriv2(state2) * dt        --> sel inds x; sel inds x -> x
    #     out = state0 + (deriv0 + deriv1 * 2 + deriv2 * 2 + deriv3(state3))/6  --> sel inds x
    #
    #     state1 = state0 + deriv0(state0) * 0.5 * dt  --> sel inds 1; sel inds 0 -> 1
    #     state2 = state0 + deriv1(state1) * 0.5 * dt  --> sel inds 2; sel inds 1 -> 2
    #     state3 = state0 + deriv2(state2) * dt        --> sel inds 3; sel inds 2 -> 3
    #     out = state0 + (deriv0 + deriv1 * 2 + deriv2 * 2 + deriv3(state3))/6  --> sel inds x
    #     """
    #     dt = self.dt if dt is None else dt
    #     # print('RK4 DT', dt, left_site_pos)
    #     self.terms[0].get_evaluated_site(left_site_pos, nsites)
    #     # state0 = self.terms[0].proj_vec_targets[0]
    #     # print('vec block get projected')
    #     state0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #     state0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
    #     # print('local rk4 stat0', state0.norm())
    #     # try:
    #     #     bond_l = self.ket.bond(0,1)
    #     #     bond_r = self.ket.bond(1,2)
    #     #     print('bond l', bond_l, bond_r)
    #     #     state0.transpose(bond_r, 'i(1)', bond_l, inplace=True)
    #     # except:
    #     #     pass
    #     # print('state0', state0.inds)
    #     # print(state0.data)
    #
    #     def deriv_func(site_tens: 'qtn.Tensor', time=None , **kwargs):
    #         return self.deriv_func(left_site_pos, nsites, site_tens=site_tens)
    #
    #     def euler_func(site_tens: 'qtn.Tensor', dt, deriv0: qtn.Tensor=None, time=None, **kwargs):
    #         return self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens)
    #
    #     def scale_func(site_tens: 'qtn.Tensor', coeff: Numeric, inplace: bool = False):
    #         site_tens = site_tens if inplace else site_tens.copy()
    #         site_tens.modify(apply=lambda x: x * coeff)
    #         return site_tens
    #
    #     def add_func(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False, **kwargs):
    #         return helper_quimb.add_tensors(tens1, tens2, inplace=inplace)
    #
    #     out = helper_TE.rk4(state0, dt, euler_func, deriv_func, add_func, scale_func,
    #                         return_intermediates=return_intermediates)
    #
    #     ## remove exponent from result
    #     if return_intermediates:  ## new sites, intermediates
    #         out[0].modify(apply=lambda x: x * 10**(-self.ket.exponent))
    #         out[1][0].modify(apply=lambda x: x * 10**(-self.ket.exponent))
    #     else:
    #         out.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))
    #
    #     return out

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

        # lin_ops = self.linear_terms[0].get_eff_operator(left_site_pos, nsites)[0]
        # lin_ops = [op.copy() for op in lin_ops]
        # for op in lin_ops:
        #     helper_quimb.scalar_multiply(op, - dt * weight, inplace=True)
        #     # op.exponent += [dt * weight]

        lin_ops = []
        for lin_term in self.linear_terms:
            op = lin_term.get_eff_operator(left_site_pos, nsites)[0]
            lin_ops += [op1.copy() for op1 in op]

        for op in lin_ops:
            helper_quimb.scalar_multiply(op, - dt * weight, inplace=True)

        print('term ket is term bra', self.linear_terms[0].ket is self.linear_terms[0].bra)
        print('term ket is self ket', self.linear_terms[0].ket is self.ket)
        print('num tiers', self.linear_terms[0].num_tiers)

        ## identity
        bra_to_ket_inds = self.self_term.projected_bra_to_ket(left_site_pos, nsites)
        bra_inds = [b for b in bra_to_ket_inds.keys() if b is not None]
        ket_inds = [bra_to_ket_inds[b] for b in bra_inds]
        sq_shape = [lin_ops[0].ind_size(b) for b in bra_inds]
        sq_size = int(np.prod(sq_shape))
        iden = np.eye(sq_size).reshape(*sq_shape,*sq_shape)
        iden_tens = qtn.Tensor(iden, inds=[*bra_inds, *ket_inds])
        lhs = lin_ops + [iden_tens]

        ## the (1 - weight) * dt * F(f) term
        rhs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy() for term in self.terms[1:]]
        rhs[0].modify(apply=lambda x: x * (1 - weight) * dt)
        ## the I * y[n] term
        if site_tens is None:
            rhs += [self.self_term.get_evaluated_site(left_site_pos, nsites)]
        else:
            rhs += [site_tens]


        return lhs, rhs



    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        return super().local_implicit_solve(left_site_pos, nsites, dt=dt, time=time, **solver_kwargs)



def global_rk4_cross(dt, ket_state, linear_operators, sources=None, nonlinear_terms=None,
                     nsites: int = 1, max_bond=None):
    """
    state1 = state0 + deriv0(state0) * 0.5 * dt  --> sel inds x; sel inds 0 -> x
    state2 = state0 + deriv1(state1) * 0.5 * dt  --> sel inds x; sel inds x -> x
    state3 = state0 + deriv2(state2) * dt        --> sel inds x; sel inds x -> x
    out = state0 + (deriv0 + deriv1 * 2 + deriv2 * 2 + deriv3(state3))/6  --> sel inds x
    """

    nsites = 1
    state0 = ket_state.copy()

    source_terms = [Term_Cross(source, max_bond=max_bond) for source in sources] if sources is not None else []
    nonlinear_terms = [] if nonlinear_terms is None else nonlinear_terms

    def deriv_func(state, time=None, **kwargs):
        linop_terms = [Term_Cross(state0.copy(), operators=linear_operators, max_bond=max_bond)]
        nonlin_terms = [nl_term.create_like(ket=state) for nl_term in nonlinear_terms]
        deriv0 = local_cross_evaluator(linop_terms + source_terms + nonlin_terms, max_bond=max_bond, nsites=nsites)
        return deriv0

    def add_func(obj1, obj2, inplace=False, **kwargs):
        init_guess = obj1 if inplace else obj1.copy()
        out = local_cross_evaluator([Term_Cross(obj1.copy()), Term_Cross(obj2.copy())], init_guess=init_guess)
        return out

    def scale_func(obj1, val, inplace=True, **kwargs):
        return helper_quimb.scalar_multiply(obj1, val, inplace=inplace)

    def euler_func(state, dt, deriv0=None, **kwargs):
        if deriv0 is None:
            deriv0 = deriv_func(state)
        deriv0 = scale_func(deriv0, dt, inplace=False)
        state1 = local_cross_evaluator([Term_Cross(state.copy()), Term_Cross(deriv0)], max_bond=max_bond, nsites=nsites)
        return state1

    out = helper_TE.rk4(state0, dt, euler_func, deriv_func, add_func, scale_func)
    return out
