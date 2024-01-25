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
import tensor_callables as tc
import helper_TE


class TimeIntegMethod(IntEnum):
    RK4 = 4
    CN = 223
    CN6 = 226
    EXACT = 0



class TimeIntegrator(LocalEvaluator, ABC):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 terms: Sequence[Union['Term_DMRG', 'Term_Cross']],
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'TimeIntegrator' = None,
                 combine_terms_func: 'Callable' = None,
                 dt = 0.1, time = None,
                 te_order_target = 4, te_order_final = 4,
                 ):
        super().__init__(ket_state, terms, direction, max_bond,
                         conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func)

        if copy_obj is not None:
            self.dt = copy_obj.dt
            self.time = copy_obj.time
            self.te_order_target = copy_obj.te_order_target
            self.te_order_final = copy_obj.te_order_final
        else:
            self.dt = dt
            self.time = time
            self.te_order_target = te_order_target
            self.te_order_final = te_order_final


    def _set_local_solve_func(self, te_order: int):
        if te_order == TimeIntegMethod.RK4:
            func = self.local_rk4
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

        at_end = (left_site_pos == self.L - nsites) if self.direction == SweepDirection.RIGHT else (left_site_pos == 0)

        if at_end:
            self._set_local_solve_func(self.te_order_final)
        else:
            self._set_local_solve_func(self.te_order_target)
        out = super()._site_solve(left_site_pos, nsites, site_tens=site_tens)
        return out


    def deriv_func(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, time: Numeric = None
                   ) -> 'qtn.Tensor':
        """ compute df/dt = Af + sources + nonlinear terms O[f]
        """
        site_tens_list = []
        ## do something with time with nonlinear term
        for term in self.terms:
            site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
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

        ket_x = self.terms[0].proj_vec if site_tens is None else site_tens
        if deriv is None:
            deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)

        dt = self.dt if dt is None else dt
        deriv = deriv.copy()
        deriv.modify(apply=lambda x: x * dt)
        site_tens_list = [ket_x, deriv]
        tot_site = helper_tn.sum_tens(site_tens_list)
        return tot_site


    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

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

        dt = self.dt if dt is None else dt
        out = helper_TE.rk4(None, dt, euler_func, deriv_func, add_func, scale_func,
                            return_intermediates=return_intermediates)

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
        lin_ops = self.terms[0].get_eff_operator(left_site_pos, nsites)
        lin_ops = [op.copy() for op in lin_ops]
        for op in lin_ops:
            op.exponent += [dt * weight]
        lhs = lin_ops + [None]

        rhs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy() for term in self.terms]
        rhs[0].modify(apply=lambda x: x * (1 - weight) * dt)
        rhs += [self.terms[0].proj_vec]    ## the I * y[n] term

        return lhs, rhs


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None, site_tens: qtn.Tensor = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        lhs, rhs = self.setup_crank_nicolson(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens,
                                             **solver_kwargs)
        init_guess = rhs[-1].copy()
        output_to_input_inds = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)

        rhs = helper_tn.sum_tens(rhs)
        solve_cgd = tc.get_cgd_func()
        out = solve_cgd(rhs, lhs, init_guess, output_to_input_inds)
        return out


    def local_exact_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                          dt: Numeric = None, time: Numeric = None, site_tens: qtn.Tensor = None,
                          **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        raise NotImplementedError



class TDDMRG(TimeIntegrator, ABC):
    """ RK4 follows Feiguin and White
        CN obtains three intermediate states by simply taking three time steps
    """

    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric = None, time: Numeric = None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        if return_intermediates:
            out, rk_states = super().local_rk4(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time)

            k0, k1, k2, k3 = rk_states      ## initial x, stage 1, stage 2, stage 3
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

        else:
            return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        out1 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, **solver_kwargs)
        out2 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out1, **solver_kwargs)
        out3 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out2, **solver_kwargs)

        out0 = self.terms[0].proj_vec.copy()
        out0.modify(apply=lambda x: x * np.sqrt(1. / 3))
        out1.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out2.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out3.modify(apply=lambda x: x * np.sqrt(1. / 3))

        return out3, (out0, out1, out2)


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
        out = super()._site_solve(left_site_pos, 1, site_tens=site_tens)
        self.dt = -self.dt
        return out

    def _bond_time_evolution(self, bond_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
        self.dt = -self.dt
        out = super()._bond_solve(left_site_pos, site_tens=bond_tens)
        self.dt = - self.dt
        return out

    def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None
                    ) -> tuple[qtn.Tensor, Numeric]:

        self._set_local_solve_func(TimeIntegMethod.RK4)
        out = super()._bond_solve(left_site_pos, site_tens=site_tens)
        return out

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
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
        q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=x_ind+'_tmp')

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        q.transpose_like(self.ket[i], inplace=True)
        self.ket[i].modify(data=q.data)

        ## extend environments to include newly canonical site i
        for term in self.terms:
            term.update_vecblock_bra(i, 1, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        ## back-propagation of R
        new_r = self._bond_time_evolution(r, i, direction, -self.dt)
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

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
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