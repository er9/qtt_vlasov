import pdb

from setup_.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from helper_tdvp import TDVPSolver2
from field import Field, ScalarField
from pde_system import PDE_system

from helper_dmrg import SolveMethod, SweepDirection
import helper_block_dmrg as helper_block
import helper_block_dmrg_2 as helper_block_2

if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from gridTN import GridTN


class VectorPotential(PDE_system):
    """ Maxwell's equations solver
    """
    def __init__(self,
                 A: Optional[Field], dAdt: Optional[Field],
                 phi: Optional[ScalarField], dphidt: Optional[ScalarField]=None,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 background_A0: Optional['Field'] = None,
                 background_phi0: Optional['ScalarField'] = None,
                 is_electrostatic=False, # is_yee = False, clean=True,
                 normalize=False, conservative=False, upwind=False, zipup=True,
                 te_order=60, compress_levels=None,
                 ):
        """ solves Maxwell's equation using vector-potential with Lorenz gauge
            two methods:
            - solve for phi using the appropriate wave equation
            - solve for phi using the Lorenz gauge condition. **
            we opt for the latter to ensure gauge purity.
            in this case, dphidt is not retained

            future work: inhomogeneous media
            file:///C:/Users/erikaye/Downloads/14060904.pdf
            Vector Potential Electromagnetics with Generalized Gauge for Inhomogeneous Media: Formulation
            Weng Cho Chew
        """

        self.names = {}
        self.names['A'] = A.name if A is not None else 'A'
        self.names['dAdt'] = dAdt.name if dAdt is not None else 'dAdt'
        self.names['phi'] = phi.name if phi is not None else 'phi'
        self.names['dphidt'] = dphidt.name if dphidt is not None else 'dphidt'

        # field_names = [self.names[x] for x in ['A', 'dAdt', 'phi', 'dphidt']]
        field_names = ['A', 'dAdt', 'phi', 'dphidt']

        super().__init__(A, dAdt, phi, dphidt, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         conservative=conservative, upwind=upwind,
                         )

        self.coords_x = coords_x

        self.init_norm_A = None
        self.init_norm_phi = None

        if A is not None:
            self.grid_X = A.grid
        elif phi is not None:
            self.grid_X = phi.grid
        else:
            self.grid_X = grid_X

        ## background fields
        self.background_A0 = background_A0
        self.divA0 = None
        self.curlA0 = None

        self.background_phi0 = background_phi0
        self.gradphi0 = None

        ## species parameters
        self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
        self.is_electrostatic = is_electrostatic
        self.upwind = upwind
        self.zipup = zipup

        ## charge and current densities
        self.charge_density = None
        self.charge_density_deriv = None
        self.current_density = None
        self.current_density_deriv = None

        ## used if time integration like adam-bashforth is used
        ## history of vector potential (needed to compute E, Lorentz force)
        self.A_history: Sequence[Field] = [Field('A0', self.grid_X, {})]
        self.phi_history: Sequence[ScalarField] = [ScalarField('phi0', self.grid_X, None)]

        if te_order == 31:
            time_acc = 2
        else:
            time_acc = int(str(te_order)[-1])
            if time_acc > 4:
                time_acc -= 5
            if time_acc == 0:
                time_acc = 4
        self.history_length = time_acc
        self.staggered_phi = True if te_order == 31 else False
            ## leapfrog method

        self.E: Optional['Field'] = None
        self.B: Optional['Field'] = None


    @property
    def A(self) -> Optional['Field']:
        try:
            return self._fields['A']
        except KeyError:
            return None

    @A.setter
    def A(self, new_field: 'Field'):
        if len(self.A_history) == self.history_length:
            self.A_history = [self.A, *self.A_history[:-1]]
        else:
            self.A_history = [self.A, *self.A_history]

        self._fields['A'] = new_field

    @property
    def dAdt(self) -> Optional['Field']:
        try:
            return self._fields['dAdt']
        except KeyError:
            return None

    @dAdt.setter
    def dAdt(self, new_field: 'Field'):
        self._fields['dAdt'] = new_field

    @property
    def phi(self) -> Optional['ScalarField']:
        try:
            return self._fields['phi']
        except KeyError:
            return None

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        if len(self.phi_history) == self.history_length:
            self.phi_history = [self.phi, *self.phi_history[:-1]]
        else:
            self.phi_history = [self.phi, *self.phi_history]

        self._fields['phi'] = new_field

    @property
    def dphidt(self) -> Optional['ScalarField']:
        try:
            return self._fields['dphidt']
        except KeyError:
            return None

    @dphidt.setter
    def dphidt(self, new_field: 'ScalarField'):
        self._fields['dphidt'] = new_field

    # @property
    def compute_dAdt(self, dt: Numeric=None, compress_level=1):
        if dt is None:
            dt = self.dt
        if self.history_length == 1:
            A1 = self.A_history[-1]
            dAdt = self.A.add(A1.scalar_multiply(-1, inplace=False), inplace=False,
                               compress_level=compress_level)
            dAdt.scalar_multiply(1./ dt)

        elif self.history_length == 2:
            A0 = self.A_history[1].scalar_multiply(-1)
            dAdt = self.A.add(A0, inplace=False)
            dAdt.scalar_multiply(1./ 2 / dt)

        # elif self.history_length == 2:
        #     A0 = self.A.scalar_multiply(1.5, inplace=False)
        #     dt1, A1 = self.A_history[-1]
        #     A1 = A1.scalar_multiply(-2, inplace=False)
        #     dt2, A2 = self.A_history[-2]
        #     A2 = A2.scalar_multiply(0.5, inplace=False)
        #     dt = (dt1 + dt2) / 2        ## ideally roughly the same
        #
        #     dA_dt = A0.add(A1, inplace=True, compress_level=0)
        #     dA_dt = dA_dt.add(A2, inplace=True, compress_level=compress_level)
        else:
            raise NotImplementedError

        return dAdt

    def compute_dphidt(self, compress=1):
        """ Lorenz condition: div(A) + 1/c^2 d phi/dt = 0
            d phi / dt = - c^2 div(A)
        """
        divA = self.A.divergence(self.coords_x, compress_level=compress)
        if self.divA0 is not None:
            divA = divA.add(self.divA0)
        out = divA.scalar_multiply(-self.matl_params.c ** 2)
        out.name = 'dphidt'
        return out

    def compute_d2Adt2(self, compress=1):
        laplA = self.A.laplacian(self.coords_x, compress_level=compress)
        muJ = self.current_density.scalar_multiply(self.matl_params.mu0, inplace=False)

        out = laplA.add(muJ)
        out.scalar_multiply(self.matl_params.c ** 2, inplace=True)
        out.name = 'd2Adt2'
        return out

    def compute_d2phidt2(self, compress=1):
        lapl_phi = self.phi.laplacian(self.coords_x, compress_level=compress)
        eps_rho = self.charge_density.scalar_multiply(1./self.matl_params.eps0)

        out = lapl_phi.add(eps_rho)
        out.scalar_multiply(self.matl_params.c ** 2, inplace=True)
        out.name = 'd2phidt2'
        return out


    def compute_E(self, compress_level=1, inner_compress_level=4) -> Optional['Field']:
        """ E = -grad(phi) - dA/dt
        """
        ## dA/dt
        # print('self.dAdt', self.dAdt.components, self.dAdt.num_active)
        if self.dAdt is None or self.dAdt.num_active == 0:
            dAdt = self.compute_dAdt(self.dt)
        else:
            print('dAdt is not None')
            dAdt = self.dAdt
        dAdt = dAdt.scalar_multiply(-1, inplace=False)
        # print('dAdt', dAdt.components)

        if self.staggered_phi:
            phi0 = self.phi_history[0]
            phi = (phi0.add(self.phi)).scalar_multiply(0.5, inplace=True)
        else:
            phi = self.phi
        phi = phi.scalar_multiply(-1, inplace=False)
        grad_phi = phi.gradient(compress_level=inner_compress_level)

        out = grad_phi.add(dAdt, compress_level=compress_level, inplace=True)
        self.E = out
        return self.E


    def compute_B(self, compress_level=1, inner_compress_level=4) -> Optional['Field']:
        """ B = curl(A)
        """
        out = self.A.curl(self.coords_x, compress_level=compress_level, inner_compress_level=inner_compress_level)
        self.B = out
        return self.B


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 4:
            new_fields = new_fields + (None,) * (4 - len(new_fields))

        A, dAdt, phi, dphidt = new_fields[:4]
        if recalc:
            new_system = self.__class__(A, dAdt, phi, dphidt, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_A0=self.background_A0, background_phi0=self.background_phi0,
                                        normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
        else:
            new_system = self.__class__(A, dAdt, phi, dphidt, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_A0=self.background_A0, background_phi0=self.background_phi0,
                                        normalize=False, upwind=self.upwind, zipup=self.zipup,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_A = self.init_norm_A
            new_system.init_norm_phi = self.init_norm_phi
            new_system.dt = self.dt

            new_system.E = self.E
            new_system.B = self.B

        if A is None:           new_system.names['A'] = self.names['A']
        if dAdt is None:        new_system.names['dAdt'] = self.names['dAdt']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if dphidt is None:      new_system.names['dphidt'] = self.names['dphidt']

        new_system.divA0 = self.divA0
        new_system.curlA0 = self.curlA0
        new_system.gradphi0 = self.gradphi0

        new_system.charge_density = self.charge_density
        new_system.charge_density_deriv = self.charge_density_deriv
        new_system.current_density = self.current_density
        new_system.current_density_deriv = self.current_density_deriv
        new_system.A_history = self.A_history
        new_system.phi_history = self.phi_history

        return new_system

    def compute_poynting_vector(self):
        """ compute S = E x B*
        """
        B_conj = self.B.copy()
        for k, B in B_conj.components.items():
            B.conj(inplace=True)
        B_conj.scalar_multiply(1. / self.matl_params.mu0, inplace=True)
        return self.E.cross_product(B_conj, self.grid_X.axes[0].coord_sys)

    def electric_energy_density(self, compress_level=1):
        E_conj = self.E.copy()
        for k, E in E_conj.components.items():
            if E is not None and E.data is not None:
                E.conj(inplace=True)
        nrg = self.E.dot(E_conj, compress_level=compress_level, inner_compress_level=compress_level, zipup=True)
        nrg.scalar_multiply(self.matl_params.eps0 / 2, inplace=True)
        return nrg

    def magnetic_energy_density(self, compress_level=1):
        B_conj = self.B.copy()
        for k, B in B_conj.components.items():
            if B is not None and B.data is not None:
                B.conj(inplace=True)
        nrg = self.B.dot(B_conj, compress_level=compress_level, inner_compress_level=compress_level, zipup=True)
        nrg.scalar_multiply(1./2/self.matl_params.mu0, inplace=True)
        return nrg

    def global_momentum(self):
        """ compute S = 1/mu0 E x B*
        """
        B_conj = self.B.copy()
        for k, B in B_conj.components.items():
            if B.data is not None:
                B.data.conj(inplace=True)
        B_conj.scalar_multiply(1. / self.matl_params.mu0, inplace=True)
        E_field = self.E

        comps = self.coords_x.coords
        other_comps = comps

        dx3 = 1  # np.prod([ax.dx for ax in self.grid.axes])

        out = {}
        for i in range(3):
            c0 = comps[i]
            c1, c2 = comps[(i + 1) % 3], other_comps[(i + 2) % 3]

            try:
                out_comp_1 = E_field[c1].ovlp(B_conj[c2]) * dx3
            except KeyError:
                out_comp_1 = 0.0

            try:
                out_comp_2 = E_field[c2].ovlp(B_conj[c1]) * dx3
            except KeyError:
                out_comp_2 = 0.0

            out[c0] = (out_comp_1 - out_comp_2)

        return out

    def global_E_nrg(self):
        """ compute eps0 |E|^2 *
        """
        E_field = self.E.copy()
        comps = self.coords_x.coords
        out = {}
        for i in range(3):
            c0 = comps[i]
            out[c0] = E_field[c0].frobenius_norm()**2
        return out

    def global_B_nrg(self):
        """ compute eps0 |E|^2 *
        """
        B_field = self.B.copy()
        comps = self.coords_x.coords
        out = {}
        for i in range(3):
            c0 = comps[i]
            out[c0] = B_field[c0].frobenius_norm()**2
        return out

    def next_time_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False, compress_level: int = 1,
                       max_iter=100, err_tol=None, apply_constraints=False,
                       is_first_time_step=False, is_last_time_step=False,
                       do_postprocessing=False, process_kwargs=None, verbose_plot=False, **kwargs):
        """ convenience fct btwn different TE methods?
        """
        time = self.time
        self.dt = dt
        print('self.dt', self.dt)

        if self.te_order == 21:
            state_t = self.backwards_euler(dt, inplace=inplace, compress_level=compress_level)
        elif self.te_order==1:
            state_t = self.euler(dt, deriv0, inplace=inplace, compress_level=compress_level)
        elif self.te_order==2:
            state_t = self.rk2(dt, deriv0, compress_level=compress_level, )
        elif self.te_order==3:
            state_t = self.rk3(dt, deriv0, compress_level=compress_level, )
        elif self.te_order==4:
            state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
        elif str(self.te_order)[:2] == '22':
            if str(self.te_order)[-1] == '1':
                state_t = self.crank_nicolson(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '0':
                state_t = self.crank_nicolson(dt, use_A2=False, inplace=inplace, compress_level=compress_level, solver_kwargs={'adaptive_solve': False, 'opt_nsites':1})
            elif str(self.te_order)[-1] == '3':
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '4':
                state_t = self.crank_nicolson_block(dt, use_A2=True, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '6':
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, weight=0.6, compress_level=compress_level)
            elif str(self.te_order)[-1] == '7':     ## backward euler
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, weight=1.0, compress_level=compress_level)
            else:  # if str(self.te_order)[-1] == '2': ## default
                state_t = self.crank_nicolson(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
        elif str(self.te_order)[:2] in ['33', '44', '55', '66', '77']:

            do_cross = str(self.te_order)[:2] in ['44', '66']
            do_tdvp = str(self.te_order)[:2] in ['55', '66']
            do_mixed = str(self.te_order)[:2] in ['77']
            # if str(self.te_order)[:2] in ['44', '66']:
            #     do_cross = True
            # if str(self.te_order)[:2] in ['55', '66']:
            #     do_tdvp = True

            wk = None
            if self.te_order < 100:
                weight = 0.5
            else:
                wk = str(self.te_order)[2:]
                weight = np.round(int(wk) * 10 ** (-len(wk)), len(wk))
                if wk == '0':
                    weight = 0.5

            if weight < 0.5:
                te_order = int(wk)      ## rk1, 4, ...
            else:
                te_order = int(f'22{wk}') if wk is not None else 22

            print('33 weight', weight, te_order)

            print('is first time stpe', is_first_time_step)
            if is_first_time_step: # and self.rel_eps_inv is None:
                ## note: incorporation of mask is not included
                # state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
                state_t = self.rk4(dt, inplace=inplace, compress_level=compress_level)
            else:
                # te_order = 0 if wk == '0' else self.te_order

                print('block tddmrg te order', te_order)
                state_t = self.block_tddmrg(dt, te_order=te_order, use_A2=False, inplace=inplace,
                                            do_cross=do_cross, do_tdvp=do_tdvp, do_mixed=do_mixed,
                                            compress_level=compress_level, apply_constraints=apply_constraints,
                                            **kwargs)

                # try:
                #     state_t = self.block_tddmrg(dt, te_order=te_order, use_A2=False, inplace=inplace,
                #                                 compress_level=compress_level, apply_constraints=apply_constraints)
                #     # state_t = self.tddmrg(dt, te_order=4, use_A2=False, inplace=inplace, compress_level=compress_level)
                # except RuntimeError:  # effective A's all zero
                #     state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace,
                #                                         compress_level=compress_level)
        elif self.te_order == 31:
            state_t = self.leapfrog2(dt, inplace=inplace, verbose_plot=verbose_plot)
        else:
            state_t = super().next_time_step(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                                             max_iter=max_iter, err_tol=err_tol, is_first_time_step=is_first_time_step,
                                             is_last_time_step=is_last_time_step, do_postprocessing=do_postprocessing,
                                             process_kwargs=process_kwargs, verbose_plot=verbose_plot)

        self.time = time + dt

        return state_t

    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace: bool = False,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False, **deriv_kwargs) -> 'Maxwell':
        """ perform explicit Euler time evolution
            compress_opts are for compression after time evolution
        """
        state1 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = state1._get_compress_levels(compress_level, 5)

        if deriv0 is None:
            deriv0 = state1.calculate_time_derivative(time=state1.time, compress_level=comp4, compress_level1=comp5,
                                                      verbose_plot=verbose_plot, **deriv_kwargs)

        state1 = state1.add(deriv0 * dt, compress_level=0, inplace=True)

        if compress_level != 0:
            print('euler compressing')

            for k, comp in state1.A.components.items():
                print('k', k, 'rank', comp.max_bond())
                comp.info['internal_rank'] = comp.max_bond()

            for k, comp in state1.dAdt.components.items():
                print('k', k, 'rank', comp.max_bond())
                comp.info['internal_rank'] = comp.max_bond()

            state1.phi.component.info['internal_rank'] = state1.phi.max_bond()
            if state1.dphidt is not None:
                state1.dphidt.component.info['internal_rank'] = state1.dphidt.max_bond()


            # for k, comp in state1.E.components.items():
            #     print('k', k, 'rank', comp.max_bond())
            #     comp.info['internal_rank'] = comp.max_bond()
            #
            # for k, comp in state1.B.components.items():
            #     print('k', k, 'rank', comp.max_bond())
            #     comp.info['internal_rank'] = comp.max_bond()

            state1.compress(inplace=True, compress_level=compress_level, conservative=self.conservative,
                            use_rdm=False)

            print('end compress')

            # for fn, fn_data in state1.fields.items():
            #     for compID, comp_gtn in fn_data.components.items():
            #         if comp_gtn.data is not None:
            #             comp_gtn.data = helper_quimb.conservative_compress(comp_gtn.data)

        # if state1.do_normalization:
        #     state1.normalize()

        if state1.time is not None:
            state1.time += dt

        return state1


    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  do_evolve_B=True, do_evolve_E=True,
                                  verbose_plot=False, **kwargs) -> 'Maxwell':
        """ dA/dt = self.dAdt
            d/dt (dA/dt) = d^2 A / dt^2 = c^2 * (lapl(A) + mu0 * J)
            d/dt V = -c^2 * div(A)   [Lorenz gauge]
        """
        dAdt = self.dAdt.copy() if self.dAdt is not None else self.dAdt
        d2Adt2 = self.compute_d2Adt2( compress=compress_level )
        dphidt = self.compute_dphidt( compress=compress_level )

        dFdt = self.create_like(dAdt, d2Adt2, dphidt, None)
        return dFdt


    def evolve_A(self, dt, deriv0: Optional['VectorPotential']=None,
                 compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'Field':
        """ evolve A using standard second order method
            Ai[t+1] = 2 * Ai[t] - Ai[t-1] + c2dt2 * (lapl(Ai) + mu J)
        """
        print('evolve A')
        if deriv0 is None:
            d2Adt2 = self.compute_d2Adt2(compress=compress1)
        else:
            d2Adt2 = deriv0.dAdt
        print('d2Adt2 rank', d2Adt2.max_bonds())

        A_prev = self.A_history[0]
        # try:
        #     A_prev = self.A_history[0]
        # except IndexError:
        #     A_prev = Field('A0', grid=self.grid_X, data={})
        print('A prve rank', A_prev.max_bonds())

        tmp = self.A.scalar_multiply(2, inplace=False)
        tmp = tmp.add(A_prev.scalar_multiply(-1), inplace=True, compress_level=compress1)
        print('dA/dt rank', tmp.max_bonds(), compress1)
        new_A = tmp.add(d2Adt2.scalar_multiply(dt**2, inplace=False), compress_level=compress)
        print('new A', new_A.max_bonds())

        self.A = new_A
        return new_A

    def evolve_phi(self, dt, deriv0: Optional['VectorPotential']=None,
                   compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'ScalarField':
        """ evolve phi using the Lorenz gauge
            div(A) + 1/c^2 dphi/dt = 0
            dphi/dt = -c^2 div(A)
            phi is offset from A by a half time-step for second-order accuracy
        """
        print('evolve phi')
        if deriv0 is None:
            div_A_curr = self.A.divergence(self.coords_x, compress_level=compress1)
            dphi_dt = div_A_curr.scalar_multiply(-self.matl_params.c ** 2, inplace=True)
        else:
            dphi_dt = deriv0.phi
        print('dphi dt rank', compress1, dphi_dt.max_bond())

        phi_half = self.phi
        phi_half_new = phi_half.add(dphi_dt, inplace=False, compress_level=compress)
        print('new phi rank', compress, phi_half_new.max_bond())

        self.phi = phi_half_new
        return self.phi

    def leapfrog2(self, dt: Numeric, inplace=False,
                  compress_level=1, verbose_plot: bool = False) -> 'VectorPotential':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update -> FDTD-like
        """
        ## at initialization, need to evolve EM_sys with dt/2
        ## maybe include time as a part of self?
        state0 = self if inplace else self.copy()
        print('leapfrog TE')

        # print('is first time step?', is_first_time_step)

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        state0.evolve_phi(dt, compress=compress_level)
        ## phi(t-dt/2) --> phi(t+dt/2);  uses current A
        state0.evolve_A(dt, compress=compress_level)   ##
        ## A(t) --> A(t+dt)

        state0.time = self.time + dt if self.time is not None else None
        return state0


    def lw4(self, dt: Numeric, inplace=False,
                  compress_level=1, verbose_plot: bool = False) -> 'VectorPotential':
        """ Lax-Wendroff type
            u^(n+1) = 2u^n - u^(n-1) + (cΔt)²∇²u^n + (cΔt)⁴/12 ∇⁴u^n + O(Δt⁶)
            Fourth-order accurate in both space and time.
            lapl^2 is fourth order, lapl^4 is 2nd order spatially
        """
        raise NotImplementedError


    #### methods for DMRG-based time evolution ####
    def get_state_dict(self) -> dict[int, 'GridTN']:
        """ get Ex, Ey, Ez; Bx, By, Bz as dictionary with field "number" as key
            excludes fields that are None / 0
        """
        # print([compID.type for compID, comp in self.A.components.items()])
        A_fields_dict = {compID.type.value: comp.copy() for compID, comp in self.A.components.items()
                         if comp is not None and comp.data is not None}
        dAdt_fields_dict = {compID.type.value + 3: comp.copy() for compID, comp in self.dAdt.components.items()
                            if comp is not None and comp.data is not None}
        phi_fields_dict = {6: self.phi.component.copy()}   # if self.phi.component.data is not None
        all_fields_dict = {**A_fields_dict, **dAdt_fields_dict, **phi_fields_dict}
        return all_fields_dict

    def get_derivative_dict(self, dt=None, cleaning=False, get_backward_weights=False) -> dict[tuple[int, int], 'GridTN']:
        """ dA/dt = self.dAdt
            d/dt (dA/dt) = d^2 A / dt^2 = c^2 * (lapl(A) + mu0 * J)
            d/dt V = -c^2 * div(A)   [Lorenz gauge]

            get operations acting on A, dA/dt, phi to obtain dA/dt, d2A/dt2, dV/dt
            information returned as a dict with keys (output ind, input ind)
            get_backward_weights:  if True, return dict with coefficients with which to
                multiply MPO with when performing backwards time evolution
        """
        c2 = self.matl_params.c ** 2
        phi_ax_deriv_configs = self.phi.component.ax_deriv_configs

        all_mpos = {}
        ## d2A/dt2 = lapl(A) [[excludes + J]
        for compID, AC in self.A.components.items():
            ## note: only works for cartesian field
            lapl_mpo = self.grid_X.laplacian_mpo(ax_deriv_configs=AC.ax_deriv_configs)
            it = compID.type.value
            all_mpos[(it + 3, it)] = lapl_mpo.scalar_multiply(c2)

            all_mpos[(it, it + 3)] = self.grid_X.get_iden_mpo()

            ax = self.coords_x.get_axis(compID.type)
            print('ax', ax)
            if ax is not None:
                ddx_mpo = self.grid_X.get_firstderivative_mpo(ax, deriv_config=phi_ax_deriv_configs[ax])
                all_mpos[(6, it)] = ddx_mpo.scalar_multiply(-c2)

        return all_mpos

    def get_combined_derivative_mpo(self) -> 'GridTN':

        all_mpos = self.get_derivative_dict()

        ### build single MPS
        index_order = list(np.ndindex(7, 7))
        curl_op = self.grid_X.build_indexed_gtn(all_mpos, index_order=index_order)
        return curl_op

    def get_num_active_fields(self) -> int:
        num_fields = 0
        for C in self.coords_x.coords:
            AC = self.A.components.get(C, None)
            if AC is not None:
                num_fields += 1
            AC = self.dAdt.components.get(C, None)
            if AC is not None:
                num_fields += 1
        tmp = self.phi.component
        if tmp is not None:
            num_fields += 1

        return num_fields

    def get_combined_state(self) -> 'GridTN':
        """ get Ex, Ey, Ez; Bx, By, Bz as a single MPS
            excludes fields that are None / 0
        """
        index_order = np.arange(7)
        all_mps = self.get_state_dict()
        tot_state = self.grid_X.build_indexed_gtn(all_mps, index_order=index_order)
        return tot_state

    def crank_nicolson(self, dt, time: float = None, use_A2=True, inplace=False,
                       err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                       compress_level: int = 1, verbose_plot=False, solver_kwargs=None,
                       **deriv_kwargs):
        """
        d/dt A = dA/dt
        d/dt dA/dt = c^2 * lapl(A) + 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        d/dt V = - c^2 div(A) [SI]

        CN: (u_n+1 - u_n) / dt = 1/2( F_n+1(u) + F_n(u) )


        | 1/dt * I    -w * I       0    | | A^(n+1) | = | 1/dt * I      (1-w) * I      0    | | A^(n) |  - |    0    |
        |-c2 w lapl  1/dt * I      0    | |dA^(n+1) | = | c2 (1-w) lapl  1/dt * I      0    | |dA^(n) |  + |J^(n+1/2)|
        | c2 w div      0      1/dt * I | |phi^(n+1)| = | -c2 (1-w) div      0     1/dt * I | |phi^(n)}  + |    0    |
        where w = weight (default is 0.5)
        """
        # use_dmrg_new = True  # False
        # if use_dmrg_new:
        #     return self.crank_nicolson_dmrg(dt, time=time, err_tol=err_tol, inplace=inplace, max_iter=max_iter,
        #                                     compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs)

        new_sys = self if inplace else self.copy()
        use_dmrg = True
        if solver_kwargs is None:
            solver_kwargs = {}
        # try_adaptive_solve = False
        try_adaptive_solve = solver_kwargs.pop('adaptive_solve', True)  # False
        print('adaptive solve', try_adaptive_solve)

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        if use_dmrg:
            print("HERE, EM USE DMRG")

            num_active_fields = self.get_num_active_fields()
            all_fields_vec = self.get_combined_state()
            all_fields_time_deriv = self.get_combined_derivative_mpo()

            compress_opts = new_sys.A.compress_config.get_compress_opts(1).copy()
            max_bond = compress_opts.get('max_bond', None)
            if max_bond is not None:
                compress_opts['max_bond'] = max_bond * num_active_fields

            jF = None
            if j is not None:
                jF = self.grid_X.build_indexed_gtn(
                    {3 + compID.type.value: comp for compID, comp in j.components.items()},
                    index_order=list(range(7)), max_index=7)

            sourceF = jF

            def _implicit_solver(dt_, fields_vec, verbose_output=False, **solver_kwargs_):

                sourceF_dt = None
                if sourceF is not None and sourceF.data is not None:
                    sourceF_dt = sourceF.scalar_multiply(dt_, inplace=False)

                ## explicit contribution
                dFdt1 = fields_vec.apply(all_fields_time_deriv, compress=compress_level)  ## nans in compress
                dFdt1 = dFdt1.scalar_multiply(0.5 * dt_, inplace=False)
                if sourceF_dt is not None:
                    F1 = fields_vec.add(dFdt1, inplace=False)
                    F1 = F1.add(sourceF_dt, inplace=False, compress=compress_level, compress_opts=compress_opts)
                else:
                    F1 = fields_vec.add(dFdt1, inplace=False, compress=compress_level, compress_opts=compress_opts)
                # F1 = all_fields_vec.copy()
                # print('compress opts', compress_level, compress_opts)
                print('EM solve F1', F1.max_bond())

                # ########### tmp ##############
                # # reset fields
                # for i in range(3):
                #     compID = new_sys.coords_x.type_coords[i]
                #
                #     EC = new_sys.grid_X.select_indexed_gtn(F1, i).get_data()
                #     BC = new_sys.grid_X.select_indexed_gtn(F1, 3 + i).get_data()
                #
                #     if EC is not None:
                #         plt.figure()
                #         plt.plot(EC)
                #         plt.title(f'comp{i}')
                #
                #     if BC is not None:
                #         plt.figure()
                #         plt.plot(BC)
                #         plt.title(f'comp{i + 3}')
                # plt.show()
                # exit()

                ## implicit contribution operator
                dFdt_gtn = all_fields_time_deriv.scalar_multiply(-0.5 * dt_, inplace=False)
                iden_gtn = dFdt_gtn.get_like_iden()
                imp_gtn = dFdt_gtn.add(iden_gtn)
                print('EM implicit op', imp_gtn.max_bond())

                ## dmrg solve
                print('implicit dmrg solve', compress_opts)
                updated_F1, err, is_conv = F1.solve(imp_gtn, compress_type=CompressType.DMRG, use_A2=use_A2, is_H=False,
                                                    # init_guess=fields_vec,    ## actually gives wrong answer
                                                    compress_opts=compress_opts, verbose_output=True, **solver_kwargs_)
                # exit()
                if verbose_output:
                    return updated_F1, err, is_conv
                return updated_F1

            if all_fields_vec is None and sourceF is None:
                return new_sys
            elif all_fields_vec is None and sourceF is not None:
                sourceF_dt = sourceF.scalar_multiply(dt, inplace=False)
                new_F1 = sourceF_dt
                # print('new F1', new_F1.data)
                # new_F1.data.site_ind_id = new_F1.data.site_ind_id + '_j_'
            else:

                if try_adaptive_solve:
                    nt, max_nt, is_conv = self.nt, 4, False

                    old_fields_vec = all_fields_vec
                    while (not is_conv) and nt <= max_nt:
                        print('nt', nt)
                        new_F1 = old_fields_vec
                        for step in range(nt):
                            new_F1, err, is_conv = _implicit_solver(dt / nt, new_F1, verbose_output=True, **solver_kwargs)
                            if nt < max_nt and (not is_conv) and err > 1.0e-3:
                                nt = min(nt + 1, max_nt)
                                break

                            is_conv = (step == nt - 1)  ## reached the last step

                    self.nt = nt
                    # raise NotImplementedError
                else:
                    new_F1 = _implicit_solver(dt, all_fields_vec, **solver_kwargs)

            # reset fields
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]

                ref_AC = new_sys.A.components.get(compID, None)
                ax_deriv_configs = ref_AC.ax_deriv_configs if ref_AC is not None else None
                AC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
                AC.ax_deriv_configs = ax_deriv_configs
                if not (AC is None and compID not in new_sys.phi.componentIDs):
                    new_sys.A.components[compID] = AC
                    # print('EC', compID, EC.max_bond(), EC.canon_site)

                ref_dAdtC = new_sys.dAdt.components.get(compID, None)
                ax_deriv_configs = ref_dAdtC.ax_deriv_configs if ref_dAdtC is not None else None
                dAdtC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
                dAdtC.ax_deriv_configs = ax_deriv_configs
                if not (dAdtC is None and compID not in new_sys.dAdt.componentIDs):
                    new_sys.dAdt.components[compID] = dAdtC
                    # print('BC', compID, BC.max_bond(), BC.canon_site)

                ref_phi = new_sys.phi.component
                ax_deriv_configs = ref_phi.ax_deriv_configs if ref_phi is not None else None
                phiC = new_sys.grid_X.select_indexed_gtn(new_F1, 6)
                phiC.ax_deriv_configs = ax_deriv_configs
                if phiC is not None:
                    new_sys.phi.component = phiC


            new_sys.A.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
            new_sys.dAdt.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
            new_sys.phi.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

        else:  ## solve iteratively until convergence

            raise NotImplementedError

        return new_sys

    def crank_nicolson_block(self, dt, time: float = None, use_A2=True, inplace=False, weight=0.5,
                             err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                             compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
        """
        d/dt A = dA/dt
        d/dt dA/dt = c^2 * lapl(A) + 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        d/dt V = - c^2 div(A) [SI]

        CN: (u_n+1 - u_n) / dt = 1/2( F_n+1(u) + F_n(u) )


        | 1/dt * I    -w * I       0    | | A^(n+1) | = | 1/dt * I      (1-w) * I      0    | | A^(n) |  - |    0    |
        |-c2 w lapl  1/dt * I      0    | |dA^(n+1) | = | c2 (1-w) lapl  1/dt * I      0    | |dA^(n) |  + |J^(n+1/2)|
        | c2 w div      0      1/dt * I | |phi^(n+1)| = | -c2 (1-w) div      0     1/dt * I | |phi^(n)}  + |    0    |
        where w = weight (default is 0.5)
        """
        new_sys = self if inplace else self.copy()

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        print("HERE, EM USE BLOCK DMRG")

        all_fields_dict = self.get_state_dict()
        all_fields_time_deriv_dict = self.get_derivative_dict(dt)

        def add_gtn_list(list_gtn: Sequence['GridTN'], compress=1, **compress_opts):
            sum_out = list_gtn[0].copy()
            for gtn in list_gtn[1:]:
                sum_out.add(gtn, inplace=True)
            if compress:
                sum_out.compress(**compress_opts)
            return sum_out

        def apply_dict(fields_dict: dict[int, 'GridTN'], ops_dict: dict[tuple[int, int], 'GridTN'],
                       compress=1, **compress_opts) -> dict[int, 'GridTN']:

            outputs = {out: [] for out in range(6)}
            for k in ops_dict.keys():
                out_ind, in_ind = k
                op = ops_dict[k]
                vec = fields_dict.get(in_ind, None)
                if vec is not None and vec.data is not None:
                    outputs[out_ind] += [vec.apply(op, inplace=False)]

            output_gtns = {}
            for out, out_vecs in outputs.items():
                if len(out_vecs) == 0:
                    # output_gtns[out] = None  # self.grid_X.get_ones_mps()
                    continue
                output_gtns[out] = add_gtn_list(out_vecs, compress=compress, **compress_opts)
            return output_gtns

        def combine_field_dicts(*fields_dicts, add_gtns = False):
            combined_dict = {}
            for fdict in fields_dicts:
                for k, gtn in fdict.items():
                    if k not in combined_dict:
                        combined_dict[k] = [gtn]
                    else:
                        if add_gtns:
                            combined_dict[k] = [combined_dict[k][-1].add(gtn)]
                        else:
                            combined_dict[k] += [gtn]
                        # combined_dict[k] = [combined_dict[k][0].add(gtn, compress=5, inplace=False)]
            return combined_dict

        def convert_dict_to_dmrg(gtn_dict, is_mps=True):
            converted_dict = {}
            for k, gtns in gtn_dict.items():
                if isinstance(gtns, (list, tuple)):
                    dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
                    # for tmp in dmrg_vecs:
                    #     tmp.distribute_exponent()
                else:
                    dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
                    # dmrg_vecs.distribute_exponent()
                converted_dict[k] = dmrg_vecs
            return converted_dict

        compress_opts = new_sys.A.compress_config.get_compress_opts(1).copy()
        max_bond = compress_opts.get('max_bond', None)
        if max_bond is not None:
            compress_opts['max_bond'] = max_bond  # * num_active_fields

        jF_dict = {}
        if j is not None:
            jF_dict = {compID.type.value + 3: comp for compID, comp in j.components.items()}
            # jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
            #                                    index_order=list(range(7)))

        # curlF_bg = None
        background_comps = {}
        if self.background_A0 is not None:
            raise NotImplementedError

        if self.background_phi0 is not None:
            raise NotImplementedError


        def _implicit_solver(dt_, fields_dict, verbose_output=False, **solver_kwargs):

            jF_dt = {}
            if len(jF_dict) > 0:
                jF_dt = {k: mpo.scalar_multiply(dt_, inplace=False) if mpo is not None else None
                         for k, mpo in jF_dict.items()}
            bgF_dt = {}
            if len(background_comps) > 0:
                bgF_dt = {k: mpo.scalar_multiply(dt_, inplace=False) if mpo is not None else None
                          for k, mpo in background_comps.items()}

            #### explicit contribution ####
            dFdt1 = apply_dict(fields_dict, all_fields_time_deriv_dict, compress=compress_level)
            for k, gtn in dFdt1.items():
                dFdt1[k] = gtn.scalar_multiply((1.-weight) * dt_, inplace=False)
                # dFdt1[k] = gtn.scalar_multiply(0.5 * dt_, inplace=False)

            explicit_part = combine_field_dicts(fields_dict, dFdt1, jF_dt, bgF_dt)

            #### implicit contribution operator
            dFdt_imp_dict = {k: gtn.scalar_multiply(-weight * dt_, inplace=False)
                             for k, gtn in all_fields_time_deriv_dict.items()}
            iden_dict = {(k, k): self.grid_X.get_iden_mpo(upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='ID({})')
                         for k in range(7)}
            implicit_op = combine_field_dicts(dFdt_imp_dict, iden_dict, add_gtns=True)

            ## dmrg solve
            explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)
            implicit_part_mpo = convert_dict_to_dmrg(implicit_op, is_mps=False)
            # print('explicit exp', [[m.exponent for m in ms] for k,ms in explicit_part_mps.items()])
            # print('implicit exp', [[m.exponent for m in ms] for k,ms in implicit_part_mpo.items()])

            print('implicit block dmrg solve', compress_opts)
            if len(all_fields_dict) == 0:
                outputs = {}
                for k, mps_list in explicit_part.items():
                    tot_mps = None
                    for gtn_mps in mps_list:
                        tot_mps = helper.add_MPS(tot_mps, gtn_mps.data, compress=False)
                    helper.compress(tot_mps, compress_opts)
                    outputs[k] = tot_mps
            else:
                if use_A2:
                    outputs, err, is_conv = helper_block_2.implicit_solver_2(6, implicit_part_mpo, explicit_part_mps,
                                                                             # init_guess=init_guess,
                                                                             # init_direction=SweepDirection.LEFT,
                                                                             verbose_output=True,
                                                                             max_bond=max_bond, **solver_kwargs)
                else:
                    outputs, err, is_conv = helper_block.implicit_solver(6, implicit_part_mpo, explicit_part_mps,
                                                                         # init_guess=init_guess,
                                                                         is_H = False,
                                                                         verbose_output=True,
                                                                         solve_type=SolveMethod.CGD,
                                                                         max_bond=max_bond, **solver_kwargs)

            out_gtns = {}
            try:
                ref_gtn = fields_dict[next(iter(fields_dict))]
            except StopIteration:
                try:
                    ref_gtn = new_sys.E.components[next(iter(new_sys.E.components))]
                except StopIteration:
                    ref_gtn = new_sys.B.components[next(iter(new_sys.B.components))]

            for k, out_mps in outputs.items():
                out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)

            # for k in outputs.keys():
            #     out_mps = outputs[k]
            #     x_tens = out_mps.contract(all) * 10 ** out_mps.exponent
            #     # x_tens.transpose(*[out_mps.site_ind_id.format(i) for i in range(out_mps.L)], inplace=True)
            #     x_tens.transpose(*[out_mps.site_ind_id.format(i) for i in range(out_mps.L - 1, -1, -1)], inplace=True)
            #     x_vec = x_tens.data.reshape(-1)
            #
            #     gtn_data = out_gtns[k].get_data()
            #
            #     plt.figure()
            #     plt.plot(x_vec)
            #     plt.plot(gtn_data, '--')
            #
            #     plt.figure()
            #     plt.plot(np.abs(x_vec - gtn_data))
            #     plt.show()

            if verbose_output:
                return out_gtns, err, is_conv
            return out_gtns

        if all_fields_dict is None and len(jF_dict) == 0 and len(background_comps) == 0:
            return new_sys
        # elif all_fields_dict is None and (len(jF_dict) > 0 or len(background_comps) > 0):
        #     sourceF_dt = combine_field_dicts(jF_dict, background_comps)
        #     # sourceF.scalar_multiply(dt, inplace=False)
        #     # new_F1 = sourceF_dt
        else:
            new_F1 = _implicit_solver(dt, all_fields_dict)

        # reset fields
        for i in range(3):
            compID = new_sys.coords_x.type_coords[i]

            ref_AC = new_sys.A.components.get(compID, None)
            ax_deriv_configs = ref_AC.ax_deriv_configs if ref_AC is not None else {}
            # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
            AC = new_F1.get(i, self.grid_X.make_empty_gridTN())
            AC.ax_deriv_configs = ax_deriv_configs
            if not (AC is None and compID not in new_sys.A.componentIDs):
                new_sys.E.components[compID] = AC

            ref_AC = new_sys.dAdt.components.get(compID, None)
            ax_deriv_configs = ref_AC.ax_deriv_configs if ref_AC is not None else {}
            AC = new_F1.get(i + 3, self.grid_X.make_empty_gridTN())
            AC.ax_deriv_configs = ax_deriv_configs
            if not (AC is None and compID not in new_sys.dAdt.componentIDs):
                new_sys.dAdt.components[compID] = AC

        ref_phi = new_sys.phi.component
        ax_deriv_configs = ref_phi.ax_deriv_configs if ref_phi is not None else {}
        phi = new_F1.get(6, self.grid_X.make_empty_gridTN())
        phi.ax_deriv_configs = ax_deriv_configs
        if phi is not None:
            new_sys.phi.component = phi

        new_sys.A.compress(inplace=True, compress_level=1) #, norm_cutoff=1.0e-8)
        new_sys.dAdt.compress(inplace=True, compress_level=1) #, norm_cutoff=1.0e-8)
        new_sys.phi.compress(inplace=True, compress_level=1)

        return new_sys

    def block_tddmrg(self, dt, te_order=4, time: float = None, use_A2=False, inplace=False, weight=0.5,
                     err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                     compress_level: int = 1, verbose_plot=False, apply_constraints=False, direction=SweepDirection.RIGHT,
                     do_cross: bool = False, do_mixed: bool = False, do_tdvp: bool=False,
                     **deriv_kwargs):
        """
        """
        new_sys = self if inplace else self.copy()
        # try_adaptive_solve = True  # False

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        print(f"HERE, EM USE BLOCK TDDMRG, do cross {do_cross}")

        ## current state ##
        all_fields_dict = self.get_state_dict()

        ## time evolution operator ##
        all_fields_time_deriv_dict = self.get_derivative_dict(dt=dt)

        def combine_field_dicts(*fields_dicts):
            combined_dict = {}
            for fdict in fields_dicts:
                for k, gtn in fdict.items():
                    if k not in combined_dict:
                        combined_dict[k] = [gtn]
                    else:
                        combined_dict[k] += [gtn]
            return combined_dict

        def convert_dict_to_dmrg(gtn_dict, is_mps=True):
            converted_dict = {}
            for k, gtns in gtn_dict.items():
                if isinstance(gtns, (list, tuple)):
                    dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
                    # for tmp in dmrg_vecs:
                    #     tmp.distribute_exponent()
                else:
                    dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
                    # dmrg_vecs.distribute_exponent()

                if dmrg_vecs is not None:
                    converted_dict[k] = dmrg_vecs
            return converted_dict

        compress_opts = new_sys.A.compress_config.get_compress_opts(1).copy()
        max_bond = compress_opts.get('max_bond', None)
        cutoff = compress_opts.get('cutoff', CUTOFF)
        print('max bond', max_bond)
        print('cutoff', cutoff)
        # if max_bond is not None:
        #     compress_opts['max_bond'] = max_bond  # * num_active_fields

        ## current
        jF_dict = {}
        if j is not None:
            jF_dict = {compID.type.value + 3: comp for compID, comp in j.components.items()}

        ## background field components
        background_comps = {}
        if self.background_A0 is not None:
            raise NotImplementedError

        if self.background_phi0 is not None:
            raise NotImplementedError

        explicit_part = combine_field_dicts(jF_dict, background_comps)
        explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)

        if len(all_fields_dict) == 0:
            updated_field_mps = {}
            for k, val in explicit_part_mps.items():
                updated_field_mps[k] = helper.scalar_multiply(val, dt, inplace=False)
            info = None
        else:
            all_fields_mps = convert_dict_to_dmrg(all_fields_dict, is_mps=True)
            time_deriv_mpo = convert_dict_to_dmrg(all_fields_time_deriv_dict, is_mps=False)

            time_deriv_mpo_list = {}
            for k, item in time_deriv_mpo.items():
                time_deriv_mpo_list[k] = [item]

            if all_fields_dict is None and len(jF_dict) == 0 and len(background_comps) == 0:
                return new_sys

            site_ind_id = all_fields_mps[next(iter(all_fields_mps))].site_ind_id
            for k, v in all_fields_mps.items():
                if v is not None:
                    v.site_ind_id = site_ind_id
            for k,v in explicit_part_mps.items():
                for x in v:
                    if x is not None:
                        x.site_ind_id = site_ind_id

            ## weights for negative time evolution
            backward_weights = None
            # if do_tdvp:
            #     backward_weights = {}
            #     for k, vs in time_deriv_mpo_list.items():
            #         k1, k2 = k
            #         backward_weights[(k1, k2)] = [1.0] * len(vs) if k1 != k2 else [-1.0] * len(vs)
            #         ## upwind terms are the diagonals

            if do_cross and do_tdvp:
                from helper_block_tdvp_3 import block_tdvpx as block_tddmrg
                max_bond_ = max_bond
            elif do_tdvp:
                from helper_block_tdvp_3 import block_tdvp as block_tddmrg
                max_bond_ = max_bond
            elif do_cross:
                from helper_block_tddmrg_3 import block_tddmrgx as block_tddmrg
                max_bond_ = max_bond * (4 if te_order == 4 else 2) if max_bond is not None else None
            elif do_mixed:
                from helper_block_tddmrg_3 import block_tddmrgm as block_tddmrg
                max_bond_ = max_bond * (4 if te_order == 4 else 2) if max_bond is not None else None
            else:
                from helper_block_tddmrg_3 import block_tddmrg
                max_bond_ = max_bond * (4 if te_order == 4 else 2) if max_bond is not None else None

            print('all field mps', all_fields_mps)

            updated_field_mps, info = block_tddmrg(dt, 7, time_deriv_mpo_list, all_fields_mps,
                                                   b_vecs=explicit_part_mps, init_direction=direction,
                                                   max_bond=max_bond_, te_order=te_order, cutoff=cutoff,
                                                   backward_weights=backward_weights, do_adapt=do_tdvp,
                                                   grid=self.grid_X, return_info=True, verbose_plot=self.verbose_plot,
                                                   )

        out_gtns = {}
        ref_gtn = all_fields_dict[next(iter(all_fields_dict))]
        for k, out_mps in updated_field_mps.items():
            out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)
        new_F1 = out_gtns

        # reset fields A and dAdt
        for i in range(3):
            compID = new_sys.coords_x.type_coords[i]

            ref_AC = new_sys.A.components.get(compID, None)
            ax_deriv_configs = ref_AC.ax_deriv_configs if ref_AC is not None else {}
            # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
            AC = new_F1.get(i, self.grid_X.make_empty_gridTN())
            AC.ax_deriv_configs = ax_deriv_configs
            if not (AC is None and compID not in new_sys.A.componentIDs):
                new_sys.A.components[compID] = AC

            ref_AC = new_sys.dAdt.components.get(compID, None)
            ax_deriv_configs = ref_AC.ax_deriv_configs if ref_AC is not None else {}
            AC = new_F1.get(i + 3, self.grid_X.make_empty_gridTN())
            AC.ax_deriv_configs = ax_deriv_configs
            if not (AC is None and compID not in new_sys.dAdt.componentIDs):
                new_sys.dAdt.components[compID] = AC

        ref_phi = new_sys.phi.component
        ax_deriv_configs = ref_phi.ax_deriv_configs if ref_phi is not None else {}
        phi = new_F1.get(6, self.grid_X.make_empty_gridTN())
        phi.ax_deriv_configs = ax_deriv_configs
        if phi is not None:
            new_sys.phi.component = phi

        ### dmrg run info
        for it in range(3):
            compID = self.coords_x.coords[it]
            Acomp = new_sys.A[compID]
            if Acomp is not None:
                Acomp.info['num_evals'] = info['num_evals'].get(it, 0) if info is not None else 0
                Acomp.info['internal_rank'] = Acomp.max_bond()
                print('A internal rank', Acomp.max_bond())

            Acomp = new_sys.dAdt[compID]
            if Acomp is not None:
                Acomp.info['num_evals'] = info['num_evals'].get(it + 3, 0) if info is not None else 0
                Acomp.info['internal_rank'] = Acomp.max_bond()
                print('dAdt internal rank', Acomp.max_bond())

        phi = new_sys.phi
        if phi is not None:
            phi.component.info['num_evals'] = info['num_evals'].get(6, 0) if info is not None else 0
            phi.component.info['internal_rank'] = phi.max_bond()
            print('phi internal rank', phi.max_bond())


        new_sys.A.compress(inplace=True, compress_level=1,) # norm_cutoff=np.sqrt(cutoff))
        new_sys.dAdt.compress(inplace=True, compress_level=1,) # norm_cutoff=np.sqrt(cutoff))
        new_sys.phi.compress(inplace=True, compress_level=1)

        # print('new sys A', new_sys.A)
        # print('new sys dAdt', new_sys.dAdt)
        # print("new sys phi", new_sys.phi)
        #
        # for compID, comp in self.A.components.items():
        #     print('comp A', compID, comp)

        return new_sys
