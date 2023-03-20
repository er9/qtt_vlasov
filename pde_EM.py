from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system

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

        self.coords_x = coords_x

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
                deriv0 = state0.create_like(dEdt, None, None, None)
            state0.euler(dt, deriv0=deriv0, inplace=True, compress_level=compress)
        elif method == 'rk4':
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
        elif method == 'rk4':
            print('rk4')
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
                # X,Y,Z = self.coords_x.coords
                # plt.figure()
                # for C in self.coords_x.coords:
                #     B_data = self.B.get_comp_data(C)
                #     if B_data is not None:
                #         plt.plot(np.real(B_data), label=f'{C} re')
                #         plt.plot(np.imag(B_data), '--', label=f'{C} im')
                # plt.xlabel('x')
                # plt.legend()
                # plt.title('B')
                # plt.show()

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
                #     plt.plot(curlB_z)
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

