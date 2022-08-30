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
                 is_electrostatic=False, normalize=True, upwind=False, zipup=False, clean=True,
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
            new_system = Maxwell(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                 coords_x=self.coords_x, matl_params=self.matl_params,
                                 background_E0=self.background_E0, background_B0=self.background_B0,
                                 normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                 clean=self.clean, is_electrostatic=self.is_electrostatic,
                                 te_order=self.te_order,
                                 compress_levels=self._comp_levels,
                                 )
        else:
            new_system = Maxwell(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                 coords_x=self.coords_x, matl_params=self.matl_params,
                                 background_E0=self.background_E0, background_B0=self.background_B0,
                                 normalize=False, upwind=self.upwind, zipup=self.zipup, clean=self.clean,
                                 is_electrostatic=self.is_electrostatic,
                                 te_order=self.te_order,
                                 compress_levels=self._comp_levels,
                                 )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_E = self.init_norm_E
            new_system.init_norm_B = self.init_norm_B

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

    # def copy(self):
    #     """ create a new system like this with new fields
    #     """
    #     fe = self.fe.copy() if self.fe is not None else None
    #     fi = self.fi.copy() if self.fi is not None else None
    #     E = self.E.copy() if self.E is not None else None
    #     B = self.B.copy() if self.B is not None else None
    #     phi = self.phi.copy() if self.phi is not None else None
    #     psi = self.psi.copy() if self.psi is not None else None
    #     new_system = self.create_like(fe,fi,E,B,phi=phi,psi=psi)
    #     return new_system

    def normalize(self, target_B=None, target_E=None):
        """ normalized density field
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


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method = None,
                   is_first_time_step=False, is_last_time_step=False,
                   compress_level=1, verbose_plot: bool = False) -> 'VlasovMaxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
        """
        ## at initialization, need to evolve EM_sys with dt/2
        ## maybe include time as a part of self?
        # self.upwind = True
        state0 = self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if is_first_time_step:
            dEdt = state0._calculate_time_derivative_E(compress=comp4, compress1=comp5, compress2=0)
            dFdt = state0.create_like(dEdt, None, None, None)
            state0 = state0.euler(dt/2, deriv0=dFdt, compress_level=comp2)

        dBdt = state0._calculate_time_derivative_B(compress=comp4, compress1=comp5, compress2=0)
        dFdt = state0.create_like(None, dBdt, None, None)
        state0 = state0.euler(dt, deriv0=dFdt, compress_level=comp2)
        # print('dBdt', dBdt.components)
        # print('dFdt B', dFdt.B.components)
        # print('state0 B', state0.B.components)

        if is_last_time_step:
            dt_ = dt/2
        else:
            dt_ = dt

        dEdt = state0._calculate_time_derivative_E(compress=comp4, compress1=comp5, compress2=0)
        dFdt = state0.create_like(dEdt, None, None, None)
        state0 = state0.euler(dt_, deriv0=dFdt, compress_level=comp2)

        if comp1 < comp2:
            state0.compress(inplace=True, compress_level=comp1)

        # print('state0 B', state0.B.components)

        state0.time = self.time + dt if self.time is not None else None

        return state0


    def calculate_time_derivative(self, time=None, # current_density=None, charge_density=None,
                                  compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'Maxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt_B = self._calculate_time_derivative_B(compress=compress_level, compress1=compress_level1) \
                      if self.E is not None else None
        dFdt_E = self._calculate_time_derivative_E(# current_density=self.current_density,
                                                   compress=compress_level, compress1=compress_level1)

        if self.clean:
            dFdt_psi = self._calculate_time_derivative_psi(compress=compress_level, compress1=compress_level1)
            dFdt_phi = self._calculate_time_derivative_phi(# charge_density=self.charge_density,
                                                           compress=compress_level, compress1=compress_level1)
        else:
            dFdt_phi = None
            dFdt_psi = None

        dFdt = self.create_like(dFdt_E, dFdt_B, dFdt_phi, dFdt_psi)
        return dFdt


    def _calculate_time_derivative_E(self, current_density=None, compress=1, compress1=0, compress2=0):
        """
        e0*mu0 dE/dt - curl(B) + chi*grad(phi) = -mu0 J
        J = sum_s qs ns vs
        """
        x_axes = self.coords_x.axes

        ## current
        # j = self.compute_current(compress=compress1, compress1=0)
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
            j.scalar_multiply(-1 * self.matl_params.mu0, inplace=True)

        dEdt = j
        # print('dEdt', dEdt[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)

        ## curl(B)
        if self.B is not None:
            # print('getting curl B')
            # print('B', self.B[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)
            curlB = self.B.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            # print('curlB',curlB,curlB.components.keys())
            # curlB.scalar_multiply(self.params['c']**2, inplace=True)

            # X,Y,Z = self.coords_x.coords
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

            if dEdt is not None:
                dEdt.add(curlB, inplace=True, compress_level=0)
            else:
                dEdt = curlB

        if self.background_B0 is not None:
            if self.curlB0 is None:
                curlB0 = self.background_B0.curl(self.coords_x, compress_level=compress1,
                                                 inner_compress_level=compress2)
                self.curlB0 = curlB0

            if dEdt is not None:
                dEdt.add(self.curlB0, inplace=True, compress_level=0)
            else:
                dEdt = self.curlB0.copy()

        ## correction
        if self.phi is not None:
            grad_phi = self.phi.gradient(compress_level=compress1)
            grad_phi.scalar_multiply(-1 * self.chi, inplace=True)
            dEdt.add(grad_phi, inplace=True, compress_level=0)

        if dEdt is not None:
            if compress:    dEdt.compress(inplace=True, compress_level=compress)
            dEdt.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            dEdt.name = self.E.name if self.E is not None else 'E'

        # print(dEdt.components)
        # for X in dEdt.components.keys():
        #     print(dEdt[X].data)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(dEdt.get_comp_data('Z'))
        # plt.title('dE/dt')
        # plt.show()

        return dEdt

    def _calculate_time_derivative_B(self, compress=1, compress1=0, compress2=0):
        """
        dB/dt + curl(E) + gamma*grad(psi) = 0
        """
        dBdt = None

        ## current
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

        if self.background_E0 is not None:
            if self.curlE0 is None:
                curlE0 = self.background_E0.curl(self.coords_x, compress_level=compress1,
                                                 inner_compress_level=compress2)
                self.curlE0 = curlE0

            if dBdt is not None:
                dBdt.add(self.curlE0, inplace=True, compress_level=0)
            else:
                dBdt = self.curlE0.copy()

        if self.psi is not None:
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
            dBdt.scalar_multiply(-1, inplace=True)
            dBdt.name = self.B.name if self.B is not None else 'B'

            # X,Y,Z = self.coords_x.coords
            # plt.figure()
            # plt.plot(np.real(dBdt.get_comp_data(Z)))
            # plt.plot(np.imag(dBdt.get_comp_data(Z)),'--')
            # plt.title('dBdt')
            # plt.show()

        # X, Y = dBdt.components.keys()
        # print(dBdt.components, dBdt.components[X].data, dBdt.components[Y].data)
        # exit()

        return dBdt

    def _calculate_time_derivative_phi(self, charge_density=None, compress=1, compress1=0, compress2=0):
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

        return dPhidt


    def _calculate_time_derivative_psi(self, compress=1, compress1=0, compress2=0):
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
            divB.scalar_multiply(-self.gamma * self.matl_params.c ** 2, inplace=True)
            divB.name = self.psi.name if self.psi is not None else 'psi'

        return divB

