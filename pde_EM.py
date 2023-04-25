from setup.configs import *
from helper_tdvp import TDVPSolver0
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from gridTN import GridTN


class Maxwell(PDE_system):
    """ Maxwell's equations solver
    """
    def __init__(self,
                 E: Optional[Field], B: Optional[Field],
                 phi: Optional[ScalarField] = None, psi: Optional[ScalarField] = None,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 background_E0: Optional['Field'] = None,
                 background_B0: Optional['Field'] = None,
                 is_electrostatic=False, is_yee = False, clean=True,
                 normalize=False, upwind=False, zipup=False,
                 te_order=2, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov.
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        self.names = {}
        self.names['E'] = E.name if E is not None else 'E'
        self.names['B'] = B.name if B is not None else 'B'
        self.names['phi'] = phi.name if phi is not None else 'phi'
        self.names['psi'] = psi.name if psi is not None else 'psi'

        field_names = [self.names[x] for x in ['E', 'B', 'phi', 'psi']]

        super().__init__(E, B, phi, psi, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         )

        self.coords_x: 'CoordinateSystem' = coords_x

        self.init_norm_E = None
        self.init_norm_B = None

        if E is not None:
            self.grid_X = E.grid
        elif B is not None:
            self.grid_X = B.grid
        else:
            self.grid_X = grid_X

        ## background fields
        self.background_E0 = background_E0
        self.divE0 = None
        self.curlE0 = None

        self.background_B0 = background_B0
        self.divB0 = None
        self.curlB0 = None

        ## species parameters
        self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
        self.is_electrostatic = is_electrostatic
        self.upwind = upwind
        self.zipup = zipup
        self.clean = clean
        self.is_yee = is_yee
        if is_yee:      ## hardcode derivative configurations
            # if self.B is not None:
            #     for compID, comp in self.B.components.items():
            #         for ax in self.grid_X.axes:
            #             deriv_config = comp.ax_deriv_configs[ax].copy()
            #             deriv_config.update(order=0, fd_type=FDType.BACKWARD)
            #             comp.ax_deriv_configs[ax] = deriv_config
            #
            # if self.E is not None:
            #     for compID, comp in self.E.components.items():
            #         for ax in self.grid_X.axes:
            #             deriv_config = comp.ax_deriv_configs[ax].copy()
            #             deriv_config.update(order=0, fd_type=FDType.FORWARD)
            #             comp.ax_deriv_configs[ax] = deriv_config

            self.te_order = 31      # split-step time evolution
            self.clean = False

        ## divergence cleaning params
        self.chi = 1
        self.gamma = 1

        ## charge and current densities
        self.charge_density = None
        self.current_density = None


    @property
    def E(self) -> Optional['Field']:
        try:
            return self._fields[self.names['E']]
        except KeyError:
            return None

    @E.setter
    def E(self, new_field: 'Field'):
        self._fields[self.names['E']] = new_field

    @property
    def B(self) -> Optional['Field']:
        try:
            return self._fields[self.names['B']]
        except KeyError:
            return None

    @B.setter
    def B(self, new_field: 'Field'):
        self._fields[self.names['B']] = new_field

    @property
    def phi(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['phi']]
        except KeyError:
            return None

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        self._fields[self.names['phi']] = new_field

    @property
    def psi(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['psi']]
        except KeyError:
            return None

    @psi.setter
    def psi(self, new_field: 'ScalarField'):
        self._fields[self.names['psi']] = new_field


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 4:
            new_fields = new_fields + (None,) * (4 - len(new_fields))

        E, B, phi, psi = new_fields[:4]
        if recalc:
            new_system = self.__class__(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_E0=self.background_E0, background_B0=self.background_B0,
                                        normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                        clean=self.clean, is_yee=self.is_yee, is_electrostatic=self.is_electrostatic,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
        else:
            new_system = self.__class__(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_E0=self.background_E0, background_B0=self.background_B0,
                                        normalize=False, upwind=self.upwind, zipup=self.zipup,
                                        clean=self.clean, is_yee=False,
                                        is_electrostatic=self.is_electrostatic,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_E = self.init_norm_E
            new_system.init_norm_B = self.init_norm_B
            new_system.is_yee = self.is_yee

        if E is None:           new_system.names['E'] = self.names['E']
        if B is None:           new_system.names['B'] = self.names['B']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if psi is None:         new_system.names['psi'] = self.names['psi']

        new_system.divE0 = self.divE0
        new_system.divB0 = self.divB0
        new_system.curlE0 = self.curlE0
        new_system.curlB0 = self.curlB0

        new_system.charge_density = self.charge_density
        new_system.current_density = self.current_density

        return new_system


    def normalize(self, target_B=None, target_E=None):
        """ normalized density field
            if normalize, would probably keep constant energy?
        """
        if self.E is not None:
            if target_E is None:
                target_E = self.init_norm_E if self.init_norm_E is not None else 1.0
            norm_E = self.E.norm()
            self.E.scalar_multiply(target_E / np.conj(norm_E), inplace=True)

        if self.B is not None:
            if target_B is None:
                target_B = self.init_norm_B if self.init_norm_B is not None else 1.0
            norm_B = self.B.norm()  # integrate(0)
            self.B.scalar_multiply(target_B / np.conj(norm_B), inplace=True)
        return


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError


    def evolve_E(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method=None,
                 inplace=False, compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'Maxwell':
        """ evolve E field in time via euler step
        """
        state0 = self if inplace else self.copy()

        if method is None:
            if deriv0 is None:
                dEdt = state0._calculate_time_derivative_E(compress=compress1, compress1=compress2, compress2=0,
                                                           verbose_plot=verbose_plot)

                # plt.figure()
                # for C in dEdt.componentIDs:
                #     dEdt_data = dEdt.get_comp_data(C)
                #     if dEdt_data is not None:
                #         plt.plot(np.real(dEdt_data), label=f'dEdt {C}')
                #         plt.plot(np.imag(dEdt_data), '--', label=f'dEdt {C}')
                # plt.legend()
                # plt.show()

                deriv0 = state0.create_like(dEdt, None, None, None)
            state0.euler(dt, deriv0=deriv0, inplace=True, compress_level=compress)

        # elif method[:4] == 'tdvp':
        #     ### this doesn't work at all. fundamentally, i think
        #     te_order = 0 # if len(method) == 4 else int(method[-1])
        #     if deriv0 is None:
        #         dEdt = state0._calculate_time_derivative_E(compress=compress1, compress1=compress2, compress2=0,
        #                                                    verbose_plot=verbose_plot)
        #     else:
        #         dEdt = deriv0.E
        #
        #     E_field = state0.E
        #     for C in dEdt.componentIDs:
        #         dEdt_comp = dEdt[C]
        #         E_comp = E_field.components.get(C, None)
        #         if dEdt_comp is not None and dEdt_comp.data is not None:
        #             if E_comp is None or E_comp.data is None:
        #                 E_comp = dEdt_comp.scalar_multiply(dt, inplace=False)
        #             else:
        #                 E_comp_copy = E_comp.copy()
        #                 E_comp.evolve_tdvp0(dt, [dEdt_comp], te_order=te_order, inplace=True,
        #                                     compress_config=E_field.compress_config)
        #                 print('E diff', E_comp.distance(E_comp_copy))
        #             E_field[C].data = E_comp.data

        elif method == 'rk4':
            print('evolve E rk4')
            state0.rk4(dt, deriv0=deriv0, inplace=True, compress_level=compress,
                       do_evolve_B=False)
        else:
            raise NotImplementedError

        return state0


    def evolve_B(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method=None,
                 inplace=False, compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'Maxwell':
        """ evolve E field in time via euler step
        """
        state0 = self if inplace else self.copy()

        if method is None or method == 'euler':
            if deriv0 is None:
                dBdt = state0._calculate_time_derivative_B(compress=compress1, compress1=compress2, compress2=0,
                                                           verbose_plot=verbose_plot)
                deriv0 = state0.create_like(None, dBdt, None, None)
            state0.euler(dt, deriv0=deriv0, inplace=True, compress_level=compress)

        # elif method[:4] == 'tdvp':
        #     ### this doesn't work at all. fundamentally, i think
        #     te_order = 0 #  if len(method) == 4 else int(method[-1])
        #     if deriv0 is None:
        #         dBdt = state0._calculate_time_derivative_B(compress=compress1, compress1=compress2, compress2=0,
        #                                                    verbose_plot=verbose_plot)
        #     else:
        #         dBdt = deriv0.B
        #
        #     B_field = state0.B
        #     for C in dBdt.componentIDs:
        #         dBdt_comp = dBdt[C]
        #         B_comp = B_field.components.get(C, None)
        #         if dBdt_comp is not None and dBdt_comp.data is not None:
        #             if B_comp is None or B_comp.data is None:
        #                 B_comp = dBdt_comp.scalar_multiply(dt, inplace=False)
        #             else:
        #                 B_comp_copy = B_comp.copy()
        #                 print('dBdt comp', dBdt_comp.norm(is_sqrt=True))
        #                 B_comp.evolve_tdvp0(dt, [dBdt_comp], te_order=te_order, inplace=True,
        #                                     compress_config=B_field.compress_config)
        #                 print('B diff', B_comp.distance(B_comp_copy))
        #             B_field[C].data = B_comp.data

        elif method == 'rk4':
            print('evolve B rk4')
            # print('B ax derivs', [(cID, c.ax_deriv_configs) for cID, c in state0.B.components.items()])
            new_state = state0.rk4(dt, deriv0=deriv0, compress_level=compress,
                                    do_evolve_E=False)
            state0.E = new_state.E
            state0.B = new_state.B
            state0.phi = new_state.phi
            state0.psi = new_state.psi
        else:
            raise NotImplementedError

        return state0


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_E = None, method_B = None,
                   is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'Maxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: include cleaning
        """
        ## at initialization, need to evolve EM_sys with dt/2
        ## maybe include time as a part of self?
        state0 = self if inplace else self.copy()
        print('split step','method E', method_E, 'method B', method_B, self.is_yee)
        # print('is first time step?', is_first_time_step)

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if is_first_time_step:
            state0 = state0.evolve_B(dt/2, inplace=True, method='rk4', # method=None,
                                     compress=comp1, compress1=comp4, compress2=comp5)

        else:
            state0 = state0.evolve_E(dt, inplace=True, method=method_E,
                                     compress=comp1, compress1=comp4, compress2=comp5)

            dt_ = dt/2 if is_last_time_step else dt
            method_B = 'rk4' if is_last_time_step else method_B
            state0 = state0.evolve_B(dt_, inplace=True, method=method_B,
                                     compress=comp1, compress1=comp4, compress2=comp5)

            state0.time = self.time + dt if self.time is not None else None

        return state0


    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  do_evolve_B = True, do_evolve_E = True,
                                  verbose_plot=False, **kwargs) -> 'Maxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        # print('do_evovlve E, B', do_evolve_E, do_evolve_B)
        if do_evolve_B:
            dFdt_B = self._calculate_time_derivative_B(compress=compress_level, compress1=compress_level1,
                                                       verbose_plot=verbose_plot)
        else:
            dFdt_B = None

        if do_evolve_E:
            dFdt_E = self._calculate_time_derivative_E(# current_density=self.current_density,
                                                       compress=compress_level, compress1=compress_level1,
                                                       verbose_plot=verbose_plot)
        else:
            dFdt_E = None

        # print('EM time deriv')
        # print('dBdt', dFdt_B.components)
        # print('dEdt', dFdt_E.components)

        if self.clean:
            dFdt_psi = self._calculate_time_derivative_psi(compress=compress_level, compress1=compress_level1)
            dFdt_phi = self._calculate_time_derivative_phi(# charge_density=self.charge_density,
                                                           compress=compress_level, compress1=compress_level1)
        else:
            dFdt_phi = None
            dFdt_psi = None

        dFdt = self.create_like(dFdt_E, dFdt_B, dFdt_phi, dFdt_psi)
        return dFdt


    # @profile
    def _calculate_time_derivative_E(self, current_density=None, compress=1, compress1=0, compress2=0,
                                     verbose_plot=False):
        """
        e0*mu0 dE/dt - curl(B) + chi*grad(phi) = -mu0 J
        J = sum_s qs ns vs
        """
        x_axes = self.coords_x.axes
        # verbose_plot = True

        ## current
        j = current_density if current_density is not None else self.current_density

        if j is not None:

            if self.is_electrostatic and len(x_axes) == 1:
                X = x_axes[0]
                # j_ghost_mps = helper.sum_tensornetwork(like_tn=j[X.coordinate].data)
                # j_ghost_mps.view_like(j[X.coordinate].data, inplace=True)
                jg_const = j.norm() * -1 / (X.npts * X.dx)  # norm integrates over all axes -> int
                # helper.scalar_multiply(j_ghost_mps,jg_const,inplace=True)
                # j_ghost = j.create_like({x_axes[0].coordinate:j_ghost_mps})
                # print('j',j[X.coordinate].data, j_ghost[X.coordinate].data)
                # j.add(j_ghost,inplace=True)
                j.scalar_add(jg_const, inplace=True)

                # plt.figure()
                # plt.plot(j.get_field_data(comps=0),label='total')
                # plt.plot(j_ghost.get_field_data(comps=0),label='ghost')
                # plt.legend()

            # plt.show()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        dEdt = j
        # print('dEdt', dEdt[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)

        if verbose_plot:
            if j is not None:
                plt.figure()
                for C in self.coords_x.coords:
                    j_data = j.get_comp_data(C)
                    if j_data is not None:
                        plt.plot(np.real(j_data), label=f'{C} re')
                        plt.plot(np.imag(j_data), '--', label=f'{C} im')
                plt.legend()
                plt.xlabel('x')
                plt.title('-j')
                # plt.show()
            else:
                print('j is None')

        if verbose_plot:
            X,Y,Z = self.coords_x.coords
            plt.figure()
            for C in self.coords_x.coords:
                B_data = self.B.get_comp_data(C)
                if B_data is not None:
                    plt.plot(np.real(B_data), label=f'{C} re')
                    plt.plot(np.imag(B_data), '--', label=f'{C} im')
            plt.xlabel('x')
            plt.legend()
            plt.title('B')
            plt.show()


        ## curl(B)
        curlB = None
        if self.B is not None:

            if self.is_yee:
                for compID, comp in self.B.components.items():
                    for ax in self.grid_X.axes:
                        if comp is not None:
                            deriv_config = comp.ax_deriv_configs[ax].copy()
                            deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                            comp.ax_deriv_configs[ax] = deriv_config

            # print('curl B')
            # for compID in self.B.componentIDs:
            #     try:
            #         print('self.B ax deriv confgs', self.B[compID].ax_deriv_configs)
            #     except AttributeError:
            #         print(f'no B{compID}')
            curlB = self.B.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            # print('curlB',curlB,curlB.components.keys())
            # curlB.scalar_multiply(self.params['c']**2, inplace=True)

            if self.matl_params.is_cgs:
                curlB.scalar_multiply(self.matl_params.c, inplace=True)
            else:
                curlB.scalar_multiply(self.matl_params.c**2, inplace=True)

            if verbose_plot:
                X,Y,Z = self.coords_x.coords
                plt.figure()
                for C in self.coords_x.coords:
                    B_data = self.B.get_comp_data(C)
                    if B_data is not None:
                        plt.plot(np.real(B_data), label=f'{C} re')
                        plt.plot(np.imag(B_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('B')
                plt.show()

                plt.figure()
                for C in self.coords_x.coords:
                    B_data = curlB.get_comp_data(C)
                    if B_data is not None:
                        plt.plot(np.real(B_data), label=f'{C} re')
                        plt.plot(np.imag(B_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('curl B')
                plt.show()

                # By_data = self.B.get_comp_data(Y)
                # if By_data is not None:
                #     plt.figure()
                #     plt.plot(np.real(By_data))
                #     plt.plot(np.imag(By_data))
                #     plt.title('By')
                #
                # curlB_z = curlB.get_comp_data(Z)
                # if curlB_z is not None:
                #     plt.figure()
                #     plt.plot(np.real(curlB_z))
                #     plt.plot(np.imag(curlB_z), '--')
                #     plt.title('curl B, Z')
                #     plt.show()

        if self.background_B0 is not None:
            if self.curlB0 is None:
                print('calc curl B0')
                curlB0 = self.background_B0.curl(self.coords_x, compress_level=compress1,
                                                 inner_compress_level=compress2)
                self.curlB0 = curlB0

            if curlB is not None:
                curlB.add(self.curlB0, inplace=True)
            else:
                curlB = self.curlB0.copy()

        if dEdt is not None:
            dEdt.add(curlB, inplace=True, compress_level=0)
        else:
            dEdt = curlB

        ## correction
        if self.phi is not None and self.phi.component is not None and self.phi.component.data is not None:
            grad_phi = self.phi.gradient(compress_level=compress1)
            grad_phi.scalar_multiply(-1 * self.chi, inplace=True)
            dEdt.add(grad_phi, inplace=True, compress_level=0)

        if dEdt is not None:
            if compress:    dEdt.compress(inplace=True, compress_level=compress)
            # if not self.matl_params.is_cgs:
            #     dEdt.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            dEdt.name = self.E.name if self.E is not None else 'E'

            if verbose_plot:
                X,Y,Z = self.coords_x.coords
                plt.figure()
                for C in self.coords_x.coords:
                    dEdt_data = dEdt.get_comp_data(C) # / self.matl_params.c ** 2
                    if dEdt_data is not None:
                        plt.plot(np.real(dEdt_data), label=f'{C} re')
                        plt.plot(np.imag(dEdt_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('dE/dt')
                plt.show()

        # print('dEdt', dEdt.components)

        return dEdt

    # @profile
    def _calculate_time_derivative_B(self, compress=1, compress1=0, compress2=0, verbose_plot=False):
        """
        dB/dt + curl(E) + gamma*grad(psi) = 0
        """
        dBdt = None
        # verbose_plot = True

        ## curl(E)
        if self.E is not None:

            # X, Y, Z = self.coords_x.coords
            #
            # Ey_data = self.E.get_comp_data(Y)
            # if Ey_data is not None:
            #     plt.figure()
            #     plt.plot(np.real(Ey_data))
            #     plt.plot(np.imag(Ey_data), '--')
            #     plt.title('Ey init')
            #
            # Ez_data = self.E.get_comp_data(Z)
            # if Ez_data is not None:
            #     plt.figure()
            #     plt.plot(np.real(Ez_data))
            #     plt.plot(np.imag(Ez_data), '--')
            #     plt.title('Ez init')
            #
            # plt.show()

            if self.is_yee:
                for compID, comp in self.E.components.items():
                    for ax in self.grid_X.axes:
                        deriv_config = comp.ax_deriv_configs[ax].copy()
                        deriv_config.update(order=0, fd_type=FDType.FORWARD)
                        comp.ax_deriv_configs[ax] = deriv_config

            # print('curl E')
            # for compID in self.E.componentIDs:
            #     try:
            #         print('self.E ax deriv confgs', self.E[compID].ax_deriv_configs)
            #     except AttributeError:
            #         print(f'no E{compID}')
            curlE = self.E.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            dBdt = curlE

            # if len(curlE._components) > 0:
            #     X, Y, Z = self.coords_x.coords
            #     ax_x = curlE.grid.axes[0]
            #     curlE_data = curlE.get_comp_data(Y)
            #     ax_x.basis.get_realspace_1D(curlE_data, 0)
            #     plt.figure()
            #     plt.plot(np.real(curlE_data))
            #     plt.plot(np.imag(curlE_data), '--')
            #     plt.title('curl E')
            #
            #     Ez_data = self.E.get_comp_data(Z)
            #     plt.figure()
            #     plt.plot(np.real(Ez_data))
            #     plt.plot(np.imag(Ez_data),'--')
            #     plt.title('self.Ez after')
            #
            #     Ey_data = self.E.get_comp_data(Y)
            #     plt.figure()
            #     plt.plot(np.real(Ey_data))
            #     plt.plot(np.imag(Ey_data),'--')
            #     plt.title('self.Ey after')
            #
            #     plt.show()

            if verbose_plot:
                X, Y, Z = self.coords_x.coords
                plt.figure()
                for C in self.coords_x.coords:
                    E_data = self.E.get_comp_data(C)
                    if E_data is not None:
                        plt.plot(np.real(E_data), label=f'{C} re')
                        plt.plot(np.imag(E_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('E')
                # plt.show()

                plt.figure()
                for C in self.coords_x.coords:
                    E_data = curlE.get_comp_data(C)
                    if E_data is not None:
                        plt.plot(np.real(E_data), label=f'{C} re')
                        plt.plot(np.imag(E_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('curl E')
                plt.show()


        if self.background_E0 is not None:
            if self.curlE0 is None:
                curlE0 = self.background_E0.curl(self.coords_x, compress_level=compress1,
                                                 inner_compress_level=compress2)
                self.curlE0 = curlE0

            if dBdt is not None:
                dBdt.add(self.curlE0, inplace=True, compress_level=0)
            else:
                dBdt = self.curlE0.copy()

        if self.psi is not None and self.psi.component is not None and self.psi.component.data is not None:
            grad_psi = self.psi.gradient(compress_level=compress1)
            grad_psi.scalar_multiply(self.gamma, inplace=True)

            # plt.figure()
            # plt.imshow(self.psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('psi')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('grad psi x')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=1))
            # plt.colorbar()
            # plt.title('grad psi y')
            # plt.show()

            # # plt.figure()
            # # plt.imshow(grad_psi.get_field_data(comps=2))
            # # plt.colorbar()
            # # plt.title('grad psi z')
            # # plt.show()

            dBdt = dBdt.add(grad_psi, inplace=True, compress_level=0) if dBdt is not None else grad_psi

        if dBdt is not None:
            if compress:
                dBdt.compress(inplace=True, compress_level=compress)

            if self.matl_params.is_cgs:
                dBdt.scalar_multiply(-self.matl_params.c, inplace=True)
            else:
                dBdt.scalar_multiply(-1, inplace=True)

            dBdt.name = self.B.name if self.B is not None else 'B'

            if verbose_plot:
                X,Y,Z = self.coords_x.coords
                # plt.figure()
                for C in self.coords_x.coords:
                    dBdt_data = dBdt.get_comp_data(C)
                    if dBdt_data is not None:
                        plt.figure()
                        plt.plot(np.real(dBdt_data))
                        plt.plot(np.imag(dBdt_data))
                        plt.title('dB/dt')
                        plt.show()

                        # plt.figure()
                        # plt.imshow(np.real(dBdt_data))
                        # plt.colorbar()
                        # plt.title('dB/dt re')
                        # plt.figure()
                        # plt.imshow(np.imag(dBdt_data))
                        # plt.colorbar()
                        # plt.title('dB/dt im')
                        # plt.show()
        # X, Y = dBdt.components.keys()
        # print('dBdt', dBdt.components)
        # exit()

        return dBdt


    def _calculate_time_derivative_phi(self, charge_density=None, compress=1, compress1=0, compress2=0,
                                       verbose_plot=False):
        """ dphi/dt + div(E) = rho/eps0
        """
        # charge_density = self.compute_charge_density(compress=compress1)
        charge_density = charge_density if charge_density is not None else self.charge_density

        if charge_density is not None:
            charge_density = charge_density.scalar_multiply(1. / self.matl_params.eps0, inplace=True)  # inplace=False)
        dPhidt = charge_density

        if self.E is not None:
            divE = self.E.divergence(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            if divE is not None:
                divE.scalar_multiply(-1, inplace=True)
            if dPhidt is not None:
                dPhidt.add(divE, inplace=True, compress_level=0)
            else:
                dPhidt = divE

        if self.background_E0 is not None:
            if self.divE0 is None:
                divE0 = self.E.divergence(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
                self.divE0 = divE0.copy()
            else:
                divE0 = self.divE0.copy()

            if divE0 is not None:
                divE0.scalar_multiply(-1, inplace=True)

            if dPhidt is not None:
                dPhidt.add(divE0, inplace=True, compress_level=0)
            else:
                dPhidt = divE0

        if dPhidt is not None:
            dPhidt.scalar_multiply(self.chi, inplace=True)
            if compress:
                dPhidt.compress(inplace=True, compress_level=compress)
            dPhidt.name = self.phi.name if self.phi is not None else 'phi'


        if verbose_plot:
            print('self.E', self.E.components)
            print('div E', divE.components)

            plt.figure()
            dPhidt_data = dPhidt.get_comp_data()
            plt.plot(np.real(dPhidt_data), label=f're')
            plt.plot(np.imag(dPhidt_data), label=f'im')
            plt.title('dphi/dt')
            plt.show()

        return dPhidt


    def _calculate_time_derivative_psi(self, compress=1, compress1=0, compress2=0, verbose_plot=False):
        """ eps0*mu0/gamma dpsi/dt + div(B) = 0
        """
        divB = None

        if self.B is not None:
            divB = self.B.divergence(self.coords_x, compress_level=compress, inner_compress_level=compress2)

        if self.background_B0 is not None:
            if self.divB0 is None:
                divB0 = self.B.divergence(self.coords_x, compress_level=compress, inner_compress_level=compress2)
                self.divB0 = divB0.copy()
            if divB is not None:
                divB.add(self.divB0, inplace=True, compress_level=0)
            else:
                divB = self.divB0.copy()

        if divB is not None:
            divB.scalar_multiply(-1./self.gamma * self.matl_params.c ** 2, inplace=True)
            divB.name = self.psi.name if self.psi is not None else 'psi'

        if verbose_plot:
            print('B', self.B.components)
            print('divB', divB.components)

            plt.figure()
            divB_data = divB.get_comp_data()
            plt.plot(np.real(divB_data), label=f're' )
            plt.plot(np.imag(divB_data), label=f'im')
            plt.legend()
            plt.title('dpsi/dt')
            plt.show()

        return divB


    def check_poisson(self, compress=5,) -> Optional['ScalarField']:
        divE = self.E.divergence(coord_sys=self.coords_x, compress_level=compress)
        divE.scalar_multiply(-1 * self.matl_params.eps0, inplace=True)
        return divE.add(self.charge_density, inplace=True)


    def check_divB(self, compress=5,) -> Optional['ScalarField']:
        divB = self.B.divergence(coord_sys=self.coords_x, compress_level=compress)
        return divB


    def get_combined_derivative_mpo(self) -> 'GridTN':

        ### get curl operator
        def _get_curl_op(comp_ax_deriv_configs):

            gr = self.grid_X
            gtn_comps = self.coords_x.coords

            dict_mpos = {}
            for compID in self.coords_x.coords:
                i = compID.type  # self.ax_coords[axID]
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                c1_ = self.coords_x.type_coords.get(i1, None)
                c2_ = self.coords_x.type_coords.get(i2, None)

                ax_deriv_configs = comp_ax_deriv_configs[compID]

                ax1_ = self.coords_x.coord_axes[c1_]
                ax2_ = self.coords_x.coord_axes[c2_]

                if c2_ in gtn_comps and ax1_ is not None:
                    mpo_1 = gr.get_firstderivative_mpo(ax1_, deriv_config=ax_deriv_configs.get(ax1_, None)).copy()
                else:
                    mpo_1 = None

                if c1_ in gtn_comps and ax2_ is not None:
                    mpo_2 = gr.get_firstderivative_mpo(ax2_, deriv_config=ax_deriv_configs.get(ax2_, None)).copy()
                    mpo_2 = mpo_2.scalar_multiply(-1, inplace=False)
                else:
                    mpo_2 = None  ## vector_field[i] = 0

                if mpo_1 is None and mpo_2 is None:
                    continue

                if mpo_2 is not None:
                    dict_mpos[(i.value, i1)] = mpo_2
                if mpo_1 is not None:
                    dict_mpos[(i.value, i2)] = mpo_1

            return dict_mpos

        ## dE/dt = curl(B) [[excludes + J]
        ## dB/dt = curl(E)
        E_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.E.components.items()}
        B_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.B.components.items()}
        dEdt_mpos = _get_curl_op(B_ax_deriv_configs)
        dBdt_mpos = _get_curl_op(E_ax_deriv_configs)

        ## adjust input, output coords accordingly
        all_mpos = {}
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        for (out_ind, in_ind), gtn in dEdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            all_mpos[(out_ind, in_ind + 3)] = gtn

        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
        for (out_ind, in_ind), gtn in dBdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(-self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(-1, inplace=True)
            all_mpos[(out_ind + 3, in_ind)] = gtn

        # print('all mpos')
        # print(all_mpos)

        ### build single MPS
        index_order = list(np.ndindex(6, 6))
        curl_op = self.grid_X.build_indexed_gtn(all_mpos, index_order=index_order)
        return curl_op


    def get_combined_state(self) -> 'GridTN':
        """ get Ex, Ey, Ez; Bx, By, Bz as a single MPS
            excludes fields that are None / 0
        """
        index_order = np.arange(6)
        all_mps = {}
        for C in self.coords_x.coords:
            EC = self.E.components.get(C, None)
            if EC is not None:
                all_mps[C.type] = EC.copy()
            BC = self.B.components.get(C, None)
            if BC is not None:
                all_mps[3 + C.type] = BC.copy()

        tot_state = self.grid_X.build_indexed_gtn(all_mps, index_order=index_order)
        return tot_state


    def crank_nicolson(self, dt, time: float = None, err_tol: float = np.sqrt(CUTOFF), inplace=False, max_iter:int = 100,
                       compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
        """
        | 1/dt * I, -c/2 curl | |E^(n+1)| = | 1/dt * I, c/2 curl | |E^(n)|  - |J^(n+1/2)|
        |-c/2 curl,  1/dt * I | |B^(n+1)| = | c/2 curl, 1/dt * I | |B^(n)|  + |    0    |
        where s is a sign and coefficient as defined by Maxwell's eq
        """
        new_sys = self if inplace else self.copy()
        use_dmrg = True

        ## current, theoretically at time n + 1/2
        j = self.current_density.copy()
        if self.matl_params.is_cgs:
            j.scalar_multiply(-4 * np.pi, inplace=True)
        else:
            j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)


        ### dmrg method
        if use_dmrg:

            all_fields_vec = self.get_combined_state()
            all_fields_time_deriv = self.get_combined_derivative_mpo()

            jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
                                               index_order=list(range(6)))
            jF = jF.scalar_multiply(dt, inplace=False)

            ## explicit contribution
            dFdt1 = all_fields_vec.apply(all_fields_time_deriv, compress=compress_level)    ## nans in compress

            # plt.figure()
            # for i in range(0, 3):
            #     dBdt0 = self.grid_X.select_indexed_gtn(dFdt1, i)
            #     if dBdt0 is not None and dBdt0.data is not None:
            #         plt.plot(dBdt0.get_data(), label=f'dE/dt {i}')
            # plt.show()
            #
            # plt.figure()
            # for i in range(3,6):
            #     dBdt0 = self.grid_X.select_indexed_gtn(dFdt1, i)
            #     if dBdt0 is not None and dBdt0.data is not None:
            #         plt.plot(dBdt0.get_data(), label=f'dB/dt {i}')
            # plt.show()

            dFdt1 = dFdt1.scalar_multiply(0.5 * dt, inplace=False)
            F1 = all_fields_vec.add(dFdt1, inplace=False)
            F1 = F1.add(jF, inplace=False, compress=compress_level)                          ## nans in compress
            # F1 = all_fields_vec.copy()

            ## implicit contribution operator
            dFdt_gtn = all_fields_time_deriv.scalar_multiply(-0.5 * dt, inplace=False)
            iden_gtn = dFdt_gtn.get_like_iden()
            imp_gtn = dFdt_gtn.add(iden_gtn)

            ## dmrg solve
            compress_opts = new_sys.E.compress_config.get_compress_opts(1)
            print('implicity dmrg solve')
            new_F1 = F1.solve(imp_gtn, compress_type=CompressType.DMRG, is_H=False, compress_opts=compress_opts)

            # reset fields
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]

                ref_EC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
                EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
                EC.ax_deriv_configs = ax_deriv_configs
                if not(EC is None and compID not in new_sys.E.componentIDs):
                    new_sys.E.components[compID] = EC

                ref_BC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
                BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
                BC.ax_deriv_configs = ax_deriv_configs
                if not(BC is None and compID not in new_sys.E.componentIDs):
                    new_sys.B.components[compID] = BC

            # ## exact solve
            # tensor: qtn.Tensor = F1.data.contract()  # contract all tensors
            # tensor.modify(apply = lambda x: x * 10**F1.data.exponent)
            # out_inds = [F1.data.site_ind_id.format(i) for i in range(F1.data.num_tensors)]
            # tensor.transpose(*out_inds, inplace=True)
            # tensor_data = tensor.data.reshape(-1)
            # npts = len(tensor_data)
            #
            # imp_tensor: qtn.Tensor = imp_gtn.data.contract()  # contract all tensors
            # imp_tensor.modify(apply = lambda x: x * 10**imp_gtn.data.exponent)
            # i_inds = [imp_gtn.data.lower_ind_id.format(i) for i in range(imp_gtn.data.num_tensors)]
            # o_inds = [imp_gtn.data.upper_ind_id.format(i) for i in range(imp_gtn.data.num_tensors)]
            # imp_tensor.transpose(*o_inds, *i_inds, inplace=True)
            # imp_tensor_data = imp_tensor.data.reshape(npts,npts)
            #
            # new_F1_data = np.linalg.solve(imp_tensor_data, tensor_data)
            # new_F1_data = new_F1_data.reshape(6, -1)
            #
            # # plt.figure()
            # # plt.plot(new_F1_data.T)
            # # plt.title('new F1')
            # # plt.show()
            #
            # ## reset fields
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #
            #     ref_EC = new_sys.E.components.get(compID, None)
            #     ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
            #     EC = new_sys.grid_X.map_state_to_mps(new_F1_data[i,:], ax_deriv_configs=ax_deriv_configs)
            #     if EC is not None: # and compID in new_sys.E.componentIDs:
            #         new_sys.E.components[compID] = EC
            #
            #     ref_BC = new_sys.E.components.get(compID, None)
            #     ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
            #     BC = new_sys.grid_X.map_state_to_mps(new_F1_data[3 + i, :], ax_deriv_configs=ax_deriv_configs)
            #     if BC is not None: # and compID in new_sys.B.componentIDs:
            #         new_sys.B.components[compID] = BC

        else:   ## solve iteratively until convergence

            ## only the curl(B), curl(E) components of dE/dt, dB/dt
            tmp_sys = self.copy()
            tmp_sys.current_density = None
            dEdt = tmp_sys._calculate_time_derivative_E(compress=compress_level + 3)
            dBdt = tmp_sys._calculate_time_derivative_B(compress=compress_level + 3)

            ## contributions from E, B fields at time step n + 1
            E1 = dEdt.scalar_multiply(0.5 * dt, inplace=False)      # 0.5 * curl(B)
            E1 = E1.add(tmp_sys.E, compress_level=0)
            j = j.scalar_multiply(dt, inplace=False)
            E1 = E1.add(j, inplace=True, compress_level=compress_level + 1)

            B1 = dBdt.scalar_multiply(0.5 * dt, inplace=False)
            B1 = B1.add(tmp_sys.B, compress_level=compress_level + 1)

            ## contributions from E, B fields at time step n + 1. initial guess is fields at time n
            E2 = dEdt.scalar_multiply(0.5 * dt)
            B2 = dBdt.scalar_multiply(0.5 * dt)

            it, err = 0, np.inf
            while err > err_tol and it < max_iter:
                old_sys = tmp_sys.copy()

                ## solve for fields at time step n + 1, given current guess of other fields
                for C in tmp_sys.coords_x.coords:
                    E_comp = tmp_sys.E.components.get(C, None)
                    E1_comp = E1.components.get(C, None)
                    E2_comp = E2.components.get(C, None)

                    if E1_comp is not None:
                        new_E_comp = E1_comp.add(E2_comp)
                    else:
                        new_E_comp = E2_comp.copy()

                    if E2_comp is not None:
                        E_comp.data = new_E_comp.data

                ## update guess for dBdt with new E fields at time n + 1
                dBdt = tmp_sys._calculate_time_derivative_B(compress=compress_level + 3)
                B2 = dBdt.scalar_multiply(0.5 * dt)

                for C in tmp_sys.coords_x.coords:
                    B_comp = tmp_sys.B.components.get(C, None)
                    B1_comp = B1.components.get(C, None)
                    B2_comp = B2.components.get(C, None)

                    if B1_comp is not None:
                        new_B_comp = B1_comp.add(B2_comp)
                    else:
                        new_B_comp = B2_comp.copy()

                    if B2_comp is not None:
                        B_comp.data = new_B_comp.data

                ## update guess for dEdt with new B fields at time n + 1
                dEdt = tmp_sys._calculate_time_derivative_E(compress=compress_level + 3)
                E2 = dEdt.scalar_multiply(0.5 * dt)

                err = tmp_sys.distances(old_sys, total=True)
                print('err', it, err)
                it += 1

                # errs = new_sys.distances(old_sys)
                # print('errs', errs)
                # err = np.sum([val for k, val in errs.items()])

            new_sys.E._components = tmp_sys.E.components
            new_sys.B._components = tmp_sys.B.components

        return new_sys