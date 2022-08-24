from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis

class Maxwell(PDE_system):
    """ Maxwell's equations solver
    """
    def __init__(self,
                 E: Optional[Field], B: Optional[Field],
                 phi: Optional[ScalarField] = None, psi: Optional[ScalarField] = None,
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
            self.grid_X = None

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
            return self.fields[self.names['E']]
        except KeyError:
            return None

    @E.setter
    def E(self, new_field: 'Field'):
        self.fields[self.names['E']] = new_field

    @property
    def B(self) -> Optional['Field']:
        try:
            return self.fields[self.names['B']]
        except KeyError:
            return None

    @B.setter
    def B(self, new_field: 'Field'):
        self.fields[self.names['B']] = new_field

    @property
    def phi(self) -> Optional['ScalarField']:
        try:
            return self.fields[self.names['phi']]
        except KeyError:
            return None

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        self.fields[self.names['phi']] = new_field

    @property
    def psi(self) -> Optional['ScalarField']:
        try:
            return self.fields[self.names['psi']]
        except KeyError:
            return None

    @psi.setter
    def psi(self, new_field: 'ScalarField'):
        self.fields[self.names['psi']] = new_field


    def create_like(self, *new_fields, recalc=True):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 4:
            new_fields = new_fields + (None,) * (4 - len(new_fields))

        E, B, phi, psi = new_fields[:4]
        if recalc:
            new_system = Maxwell(E, B, phi=phi, psi=psi,
                                 coords_x=self.coords_x, matl_params=self.matl_params,
                                 background_E0=self.background_E0, background_B0=self.background_B0,
                                 normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                 clean=self.clean, is_electrostatic=self.is_electrostatic,
                                 te_order=self.te_order,
                                 compress_levels=self._comp_levels,
                                 )
        else:
            new_system = Maxwell(E, B, phi=phi, psi=psi,
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


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
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



class VlasovMaxwell(PDE_system):
    """ df/dt + v grad(f) + q/m (E + v x B) grad_v(f) = 0

        - EM method 1:    hyperbolic Maxwell's equations (Maxwell's eq with equation cleaning)
                          http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
                          http://ammar-hakim.org/maxwell-eigensystem.html

            dB/dt + curl(E) + gamma grad(psi) = 0
            eps0 mu0 dE/dt - curl(B) + chi grad(phi) = 0
            1/chi dphi/dt + div(E) = rho/eps0
            eps0 mu0 /gamma dpsi/dt + div(B) = 0

            choose chi = gamma = 1

        - EM method 2:   Yee cell
            to be implemented. 
    """
    def __init__(self, dist_e: Optional[ScalarField], dist_i: Optional[ScalarField],
                 EM_sys: Optional['Maxwell'],
                 # E: Optional[Field], B: Optional[Field],
                 # phi: Optional[ScalarField] = None, psi: Optional[ScalarField] = None,
                 coords_x : Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 # matl_params: Optional[UnitsConfiguration] = None,
                 f0_gradv_e: Optional['Field'] = None,
                 f0_gradv_i: Optional['Field'] = None,
                 background_fe0: Optional['ScalarField'] = None,
                 background_fi0: Optional['ScalarField'] = None,
                 # background_E0: Optional['Field'] = None,
                 # background_B0: Optional['Field'] = None,
                 # is_electrostatic=False,
                 evolve_ion=True, normalize=True,
                 upwind=False, zipup=False, # clean=True,
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
        self.names['fe']  = dist_e.name if dist_e is not None else 'fe'
        self.names['fi']  = dist_i.name if dist_i is not None else 'fi'
        self.names['E']   = EM_sys.E.name if EM_sys.E is not None else 'E'
        self.names['B']   = EM_sys.B.name if EM_sys.B is not None else 'B'
        self.names['phi'] = EM_sys.phi.name if EM_sys.phi is not None else 'phi'
        self.names['psi'] = EM_sys.psi.name if EM_sys.psi is not None else 'psi'

        self.EM_sys = Maxwell(None,None) if EM_sys is None else EM_sys

        field_names = [self.names[x] for x in ['fe','fi','E','B','phi','psi']]

        super().__init__(dist_e, dist_i, # E, B, phi, psi,
                         field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         # field_compress_config=field_compress_config,
                         )

        self.coords_x  = coords_x
        self.coords_ve = coords_ve
        self.coords_vi = coords_vi

        self.x_axes = self.coords_x.axes
        self.vi_axes_dict = {ax: self.coords_vi.get_axis(ax.coordinate.type) for ax in self.x_axes}
        self.ve_axes_dict = {ax: self.coords_ve.get_axis(ax.coordinate.type) for ax in self.x_axes}

        self.grid_X = EM_sys.grid_X

        if self.fe is not None:
            v_dict = {}
            for v_ax in self.coords_ve.axes:
                try:
                    # x_ax = self.coords_x.get_axis(v_ax.coordinate.type)
                    x_coord = self.coords_x.get_coord(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mpo()
                    v_dict[x_coord] = self.fe.grid.make_mpo_ndim({v_ax:v_mult})
                except (AttributeError, KeyError):
                    pass
            self.velocities_e = self.fe.create_like_vector(v_dict)
            self.velocities_e.name = 'vel'

        if self.fi is not None:
            v_dict = {}
            for v_ax in self.coords_vi.axes:
                try:
                    # x_ax = self.coords_x.get_axis(v_ax.coordinate.type)
                    x_coord = self.coords_x.get_coord(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mpo()
                    v_dict[x_coord] = self.fi.grid.make_mpo_ndim({v_ax:v_mult})
                except (AttributeError,KeyError):
                    pass
            self.velocities_i = self.fi.create_like_vector(v_dict)
            self.velocities_i.name = 'vel'

        ## background fields
        self.background_fe0 = background_fe0
        self.background_fi0 = background_fi0
        self.gradx_fe0 = None
        self.gradv_fe0 = None
        self.gradx_fi0 = None
        self.gradv_fi0 = None
        self.background_j0 = None   # current density
        self.background_p0 = None   # charge density

        # self.background_E0 = background_E0
        # self.divE0 = None
        # self.curlE0 = None
        #
        # self.background_B0 = background_B0
        # self.divB0 = None
        # self.curlB0 = None


        ## save initial normalization for normalization
        self.init_norm_i = None
        self.init_norm_e = None
        if normalize:
            self.init_norm_i = self.total_fi.norm() if self.fi is not None else None
            self.init_norm_e = self.total_fe.norm() if self.fe is not None else None

        ## define for linearized Vlasov equation
        self.df0dv_e = f0_gradv_e
        self.df0dv_i = f0_gradv_i

        ## species parameters
        if elc_params is None:
            self.elc_params = ElcConfiguration(units_config=self.EM_sys.matl_params)
        else:
            elc_params.units_config = self.EM_sys.matl_params
            self.elc_params = elc_params

        if ion_params is None:
            self.ion_params = IonConfiguration(units_config=self.EM_sys.matl_params)
        else:
            ion_params.units_config = self.EM_sys.matl_params
            self.ion_params = ion_params

        self.evolve_ion = evolve_ion
        self.upwind = upwind
        self.zipup = zipup

        self._evolve_EM = True
        self._x_advection = True
        self._v_advection = True

        self.compress_F = True
        self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}

        self.collision = CollisionConfiguration(None)
        self.dist0_e = None
        self.dist0_i = None


    @property
    def fe(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fe']]
        except KeyError:   return None

    @fe.setter
    def fe(self,new_field: 'ScalarField'):
        self.fields[self.names['fe']] = new_field

    @property
    def total_fe(self) -> Optional['ScalarField']:
        fe = self.fe
        fe = self.background_fe0 if fe is None else fe.add(self.background_fe0)
        return fe


    @property
    def fi(self) -> Optional['ScalarField']:
        try:   return self.fields[self.names['fi']]
        except KeyError:   return None

    @fi.setter
    def fi(self,new_field: 'ScalarField'):
        self.fields[self.names['fi']] = new_field

    @property
    def total_fi(self) -> Optional['ScalarField']:
        fi = self.fi
        fi = self.background_fi0 if fi is None else fi.add(self.background_fi0)
        return fi


    @property
    def E(self) -> Optional['Field']:
        return self.EM_sys.E

    @E.setter
    def E(self,new_field: 'Field'):
        self.EM_sys.E = new_field


    @property
    def B(self) -> Optional['Field']:
        return self.EM_sys.B

    @B.setter
    def B(self,new_field: 'Field'):
        self.EM_sys.B = new_field


    @property
    def phi(self) -> Optional['ScalarField']:
        return self.EM_sys.phi

    @phi.setter
    def phi(self,new_field: 'ScalarField'):
        self.EM_sys.phi = new_field


    @property
    def psi(self) -> Optional['ScalarField']:
        return self.EM_sys.psi

    @psi.setter
    def psi(self,new_field: 'ScalarField'):
        self.EM_sys.psi = new_field



    def get_field(self, field_name):
        """ return ith component of the field. for convenience
                """
        if field_name == self.names['E']:
            return self.E
        elif field_name == self.names['B']:
            return self.B
        elif field_name == self.names['phi']:
            return self.phi
        elif field_name == self.names['psi']:
            return self.psi
        else:
            return self.fields[field_name]

    def set_field(self, field_name, new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        assert (isinstance(new_field, Field) or new_field is None), 'new_field must be Field object or None'

        if field_name == self.names['E']:
            self.E = new_field
        elif field_name == self.names['B']:
            self.B = new_field
        elif field_name == self.names['phi']:
            self.phi = new_field
        elif field_name == self.names['psi']:
            self.psi = new_field
        else:
            self.fields[field_name] = new_field

    # def __getitem__(self,field_name):
    #     """ return ith component of the field. for convenience
    #     """
    #     if   field_name == self.names['E']:     return self.E
    #     elif field_name == self.names['B']:     return self.B
    #     elif field_name == self.names['phi']:   return self.phi
    #     elif field_name == self.names['psi']:   return self.psi
    #     else:
    #         return self.fields[field_name]


    def create_like(self, *new_fields, recalc=True):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 6:
            new_fields = new_fields + (None,) *(6-len(new_fields))

        dist_e, dist_i, E, B, phi, psi = new_fields[:6]
        EM_sys = self.EM_sys.create_like(E,B,phi,psi, recalc=recalc)
        if recalc:
            new_system = VlasovMaxwell(dist_e, dist_i, EM_sys,
                                       coords_x=self.coords_x, coords_ve=self.coords_ve, coords_vi=self.coords_vi,
                                       elc_params=self.elc_params, ion_params=self.ion_params,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i,
                                       background_fe0=self.background_fe0,
                                       background_fi0=self.background_fi0,
                                       # background_E0=self.background_E0, background_B0=self.background_B0,
                                       # is_electrostatic=self.is_electrostatic,
                                       evolve_ion=self.evolve_ion, normalize = self.do_normalization,
                                       upwind=self.upwind, zipup=self.zipup, # clean=self.clean,
                                       te_order=self.te_order,
                                       compress_levels=self._comp_levels,
                                       )
        else:
            new_system = VlasovMaxwell(dist_e, dist_i, EM_sys,
                                       coords_x=self.coords_x, coords_ve=self.coords_ve, coords_vi=self.coords_vi,
                                       elc_params=self.elc_params, ion_params=self.ion_params,
                                       f0_gradv_e=self.df0dv_e, f0_gradv_i=self.df0dv_i,
                                       background_fe0=self.background_fe0,
                                       background_fi0=self.background_fi0,
                                       # background_E0=self.background_E0, background_B0=self.background_B0,
                                       # is_electrostatic=self.is_electrostatic,
                                       evolve_ion=self.evolve_ion, normalize=False,
                                       upwind=self.upwind, zipup=self.zipup, # clean=self.clean,
                                       te_order=self.te_order,
                                       compress_levels=self._comp_levels,
                                       )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_e = self.init_norm_e
            new_system.init_norm_i = self.init_norm_i

        if dist_e is None:      new_system.names['fe'] = self.names['fe']
        if dist_i is None:      new_system.names['fi'] = self.names['fi']
        if E is None:           new_system.names['E'] = self.names['E']
        if B is None:           new_system.names['B'] = self.names['B']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if psi is None:         new_system.names['psi'] = self.names['psi']

        new_system.collision = self.collision

        new_system.compress_E = self.compress_F
        new_system.E_compress_opts = self.compress_F_opts

        new_system.dist0_i = self.dist0_i
        new_system.dist0_e = self.dist0_e

        new_system._evolve_EM = self._evolve_EM
        new_system._x_advection = self._x_advection
        new_system._v_advection = self._v_advection

        new_system.gradx_fe0 = self.gradx_fe0
        new_system.gradv_fe0 = self.gradv_fe0
        new_system.gradx_fi0 = self.gradx_fi0
        new_system.gradv_fi0 = self.gradv_fi0
        new_system.background_p0 = self.background_p0
        new_system.background_j0 = self.background_j0

        # new_system.divE0 = self.divE0
        # new_system.divB0 = self.divB0
        # new_system.curlE0 = self.curlE0
        # new_system.curlB0 = self.curlB0

        return new_system

    # def copy(self):
    #     """ create a new system like this with new fields
    #     """
    #     new_system = super().copy()
    #     new_system.EM_sys = self.EM_sys.copy()
    #     return new_system


    def normalize(self,target_e=None,target_i=None):
        """ normalized density field
        """
        fi = self.fi
        fe = self.fe

        if fi is not None:
            if target_i is None:
                target_i = self.init_norm_i if self.init_norm_i is not None else 1.0
            # norm_i = fi.norm()
            # scale_i = target_i/np.conj(norm_i)
            # if np.isnan(scale_i) and np.abs(norm_i) < 1.0e-12:
            #     scale_i = 0.0
            # self.fi.scalar_multiply( scale_i, inplace=True )
            self.fi.component.normalize(target_i, inplace=True)

        if fe is not None:
            if target_e is None:
                target_e = self.init_norm_e if self.init_norm_e is not None else 1.0
            # norm_e = fe.norm()     # integrate(0)
            # scale_e = target_e / np.conj(norm_e)
            # if np.isnan(scale_e) and np.abs(norm_e) < 1.0e-12:
            #     scale_e = 0.0
            # self.fe.scalar_multiply( scale_e, inplace=True)
            self.fe.component.normalize(target_e, inplace=True)
        return


    def get_time_derivative_operator(self,compress=1,is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

        x_dim = v_dim = self.grid.ndim // 2

        convective_term = None
        for i in range(x_dim):
            tmp = self.fe_grid.get_mpo_firstderivative(i).copy()
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i])
            tmp = helper.apply( self.velocities[i], tmp )
            convective_term = helper.add_MPO(convective_term,tmp,compress=max(0,compress-1))
        helper.scalar_multiply(grads,-1,inplace=True)

        if self.B is not None:  em_term = velocity.cross_product(self[self.B], compress_level=compress)
        else:                   em_term = None
        if self.E_func is not None:  em_term = self.E_func.add(em_term)
        else:                   em_term = None

        if is_ion:    em_term.scalar_multiply(self.params['e']/self.params['mass_i']*-1, inplace=True)
        else:         em_term.scalar_multiply(self.params['e']/self.params['mass_e']   , inplace=True)

        if em_term.ndim <= self.fe.grid.ndim:
            em_term = em_term.pad_to_new_grid(new_grid=self.fe.grid)

        lorentz_term = None
        for i in range(v_dim):
            tmp = self.fe.grid.get_mpo_secondderivative(i+x_dim,i+x_dim).copy()     # velocity derivative
            tmp = helper.scalar_multiply(tmp, 1./self.grid.dx[i+x_dim])
            try:               tmp = helper.apply( em_term[i], tmp )
            except KeyError:  continue
            lorentz_term = helper.add_MPO(lorentz_term,tmp,compress=max(0,compress-1))

        tot_op = helper.add_MPO(convective_term, lorentz_term, compress=compress)

        return tot_op  ## an MPO


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method=None,
                   is_first_time_step=False, is_last_time_step=False,
                   compress_level=1, verbose_plot: bool = False) -> 'VlasovMaxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
        """
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        self._evolve_EM = False
        state0 = self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if is_first_time_step:
            # state0 = self.copy()  # if not inplace else self
            state0.EM_sys = state0.EM_sys.split_step(dt / 2)
            state0.EM_sys.time = self.EM_sys.time + dt / 2 if self.EM_sys.time is not None else None

        ## evolve force advection dt/
        state0 = state0._get_force_advection(dt/2, inplace=True, method=None, compress=comp2, compress1=comp4,
                                             compress2=comp5, verbose_plot=verbose_plot)
        ## evolve velocity advection dt
        state0 = state0._get_vel_advection(dt, inplace=True, method=method, compress=comp2, compress1=comp4,
                                           compress2=comp5, verbose_plot=verbose_plot)
        ## evolve force advection dt/2
        state0 = state0._get_force_advection(dt / 2, inplace=True, method=None, compress=comp2, compress1=comp4,
                                             compress2=comp5, verbose_plot=verbose_plot)

        ## include collisions
        if self.collision.coll_type is not None:
            raise NotImplementedError
            deriv_coll_e = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=False)
            deriv_coll_i = self.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=True)
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt / 2, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None

        state0.EM_sys.current_density = state0.compute_current(compress=comp3)
        state0.EM_sys.charge_density  = state0.compute_current(compress=comp3)

        # print('current density', state0.EM_sys.current_density.components)

        if is_last_time_step:
            state0.EM_sys = state0.EM_sys.split_step(dt / 2)
            state0.EM_sys.time = self.EM_sys.time + dt / 2 if self.EM_sys.time is not None else None
        else:
            state0.EM_sys = state0.EM_sys.split_step(dt)
            state0.EM_sys.time = self.EM_sys.time + dt if self.EM_sys.time is not None else None

        return state0


    def _get_force_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                             verbose_plot=False):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        if method is None:
            deriv_vi0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
                                                                  is_ion=True, verbose_plot=verbose_plot)
            deriv_ve0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
                                                                  is_ion=False, verbose_plot=verbose_plot)
            dFdt_v = state0.create_like(deriv_ve0, deriv_vi0, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)
        else:
            raise NotImplementedError

        return state0


    def _get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                           verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        if method is None:
            deriv_xe0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                      is_ion=False, verbose_plot=verbose_plot)
            deriv_xi0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                      is_ion=True, verbose_plot=verbose_plot)
            dFdt_x = state0.create_like(deriv_xe0, deriv_xi0, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_x, inplace=True, compress_level=compress)

        elif method == 'SL':
            print('SL advection')
            assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
                'SL time evolution for spatial basis only'

            for x_ax in self.x_axes:
                ## ions
                fi_gtn = state0.fi.component
                v_ax = self.vi_axes_dict[x_ax]
                deriv_params = fi_gtn.ax_deriv_configs[x_ax]
                compress_opts = state0.fi.compress_config.get_compress_opts(compress)
                fi_mpo_gtn = fi_gtn.grid._get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params)
                fi_gtn.apply(fi_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

                ## elecs
                fe_gtn = state0.fe.component
                v_ax = self.ve_axes_dict[x_ax]
                deriv_params = fi_gtn.ax_deriv_configs[x_ax]
                compress_opts = state0.fe.compress_config.get_compress_opts(compress)
                fe_mpo_gtn = fe_gtn.grid._get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params)
                fe_gtn.apply(fe_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

        elif method == 'SL-pfc':
            print('SL-pfc advection')
            assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
                    'SL time evolution for spatial basis only'

            for x_ax in self.x_axes:
                ## ions
                fi_gtn = state0.fi.component
                v_ax = self.vi_axes_dict[x_ax]
                deriv_params = fi_gtn.ax_deriv_configs[x_ax]
                compress_opts = state0.fi.compress_config.get_compress_opts(compress)
                fi_mpo_gtn = fi_gtn.grid._get_mpo_advection_SL_pfc(x_ax, v_ax, dt, deriv_params)
                fi_gtn.apply(fi_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

                ## elecs
                fe_gtn = state0.fe.component
                v_ax = self.ve_axes_dict[x_ax]
                deriv_params = fi_gtn.ax_deriv_configs[x_ax]
                compress_opts = state0.fe.compress_config.get_compress_opts(compress)
                fe_mpo_gtn = fe_gtn.grid._get_mpo_advection_SL_pfc(x_ax, v_ax, dt, deriv_params)
                fe_gtn.apply(fe_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
        else:
            raise NotImplementedError

        return state0



    #############################

    # @profile
    def compute_force_term(self, compress_level=1, compress_level1=0, compress_level2=0, is_ion=False):
        """ compute EM force:  q/m (E + v x B)
        """
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        background_B0 = self.EM_sys.background_B0
        background_E0 = self.EM_sys.background_E0

        # print('compress', compress_level, compress_level1, compress_level2)

        if self.B is not None:
            B = self.B.add(background_B0, compress_level=compress_level2)
        else:
            B = background_B0
        em_term = velocities.cross_product(B, compress_level=compress_level1,
                                           inner_compress_level=compress_level2) if self.B is not None else None

        if self.E is not None:
            E = self.E.add(self.EM_sys.background_E0, compress_level=compress_level2)
        else:
            E = background_E0

        if em_term is not None:
            em_term = em_term.add(E, compress_level=compress_level2)
        else:
            em_term = E

        # plt.figure()
        # x_axes = self.coords_x.axes
        # plt.plot(em_term.get_comp_data(x_axes[0].axID),label='em')
        # plt.legend()
        # plt.show()

        if em_term is not None:
            em_term.scalar_multiply(matl_params.e * matl_params.Z / matl_params.mass, inplace=True)
            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

        # print('E + v x B term', em_term.components)

        return em_term


    def compute_charge_density(self, compress=1):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        charge_density = None

        ve_axes = self.coords_ve.axes if self.fe is not None else None
        vi_axes = self.coords_vi.axes if self.fi is not None else None

        if self.fe is not None:

            e_density = self.fe.integrate(axes=ve_axes, new_grid=self.grid_X,
                                          new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e / self.elc_params.eps0, inplace=True)
            charge_density = e_density

        if self.fi is not None:

            i_density = self.fi.integrate(axes=vi_axes, new_grid=self.grid_X,
                                          new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e / self.ion_params.eps0)
            if charge_density is not None:
                charge_density.add(i_density, inplace=True, compress_level=0)

        # plt.figure()
        # print(type(charge_density))
        # charge_data = charge_density.get_comp_data()
        # plt.plot(charge_data)
        # plt.plot('charge')

        # print('n(x) deriv config', charge_density.component.ax_deriv_configs)

        if self.background_fe0 is not None or self.background_fi0 is not None:
            if self.background_p0 is None:
                charge_density_0 = None
                if self.background_fe0 is not None:
                    fe0 = self.background_fe0
                    e_density_0 = fe0.integrate(axes=ve_axes, new_grid=self.grid_X,
                                                new_ax_deriv_configs=fe0.component.ax_deriv_configs)
                    e_density_0.scalar_multiply(self.elc_params.Z * self.elc_params.e / self.elc_params.eps0,
                                              inplace=True)
                    charge_density_0 = e_density_0

                if self.background_fi0 is not None:
                    vi_axes = self.coords_vi.axes
                    fi0 = self.background_fi0
                    i_density_0 = fi0.integrate(axes=vi_axes, new_grid=self.grid_X,
                                              new_ax_deriv_configs=fi0.component.ax_deriv_configs)
                    i_density_0.scalar_multiply(self.ion_params.Z * self.ion_params.e / self.ion_params.eps0)
                    if charge_density_0 is not None:
                        charge_density_0.add(i_density_0, inplace=True, compress_level=0)
                    else:
                        charge_density = i_density_0

                self.background_p0 = charge_density_0

            charge_density.add(self.background_p0, inplace=True)

        if charge_density is not None:
            charge_density.compress(inplace=True, compress_level=compress)

        return charge_density


    def compute_species_current(self, is_ion=False, compress=1, compress1=0) -> 'Field':

        dist = self.fi if is_ion else self.fe
        v_axes = self.coords_vi.axes if is_ion else self.coords_ve.axes
        Z = self.ion_params.Z if is_ion else self.elc_params.Z
        e = self.ion_params.e if is_ion else self.elc_params.e

        j = None
        if dist is not None:

            j = dist.xmultiply(v_axes, compress_level=compress1)

            j = j.integrate(axes=v_axes, new_grid=self.grid_X,
                            new_ax_deriv_configs=dist.component.ax_deriv_configs)
            j.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in v_axes})
            j.scalar_multiply(Z * e, inplace=True)

        return j

    # @profile
    def compute_current(self, compress=1, compress1=0) -> 'Field':
        """ compute j(x) = \integ dv \sum_s q_s v_s f_s
        """
        j = None

        if self.fe is not None:

            ve_axes = self.coords_ve.axes
            je = self.fe.xmultiply(ve_axes, compress_level=compress1)

            # plt.figure()
            # plt.imshow(self.fe.get_comp_data())

            # plt.figure()
            # plt.imshow(je.get_field_data(comps=0))

            # plt.figure()
            # VX, VY, VZ = self.coords_ve.coords
            # X, Y, Z = self.coords_x.coords
            # dv = np.prod( [ax.dx for ax in self.fe.grid.axes[1:]] )
            # je_z = je.get_comp_data(VZ)
            # fe_data = self.fe.get_comp_data()
            # if je_z is not None:
            #     # v_max = np.max(np.abs(ve_axes[-1].xpts))
            #     fe_data = fe_data.dot(np.diag(ve_axes[-1].xpts))
            #     # print('v_max', np.max(np.abs(ve_axes[-1].xpts)))
            #     # print('v sum', np.sum(np.abs(ve_axes[-1].xpts))*ve_axes[-1].dx)
            #     sum_fe = np.sum( np.real(fe_data), axis=(1,2,3)) * dv * -1
            #     out = np.sum( je_z, axis=(1,2,3)) * dv * -1
            #     plt.plot(np.real(sum_fe), label='fe')
            #     # print(sum_fe[0], sum_fe[-1])
            #     # print(out[0], out[-1])
            #     plt.plot( np.real(out), label='je np' )

            je = je.integrate(axes=ve_axes, new_grid=self.grid_X,
                              new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            je.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in ve_axes})
            je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)

            # je_z = je.get_comp_data(Z)
            # if je_z is not None:
            #     print(je_z[0], je_z[-1])
            #     plt.plot(np.real(je_z),'--', label='je tn')
            #     plt.legend()
            #     plt.title('Jz')
            #     plt.show()
            #     # print('je',je.get_field_data(comps=0)[0])

            j = je.copy()

        if self.fi is not None:

            vi_axes = self.coords_vi.axes
            # ji = self.fi.elemental_multiply(self.velocities_i[0],compress=compress)
            ji = self.fi.xmultiply(vi_axes, compress_level=compress1)

            # plt.figure()
            # plt.imshow(self.fi.get_field_data(comps=0))

            # plt.figure()
            # plt.imshow(ji.get_field_data(comps=0))

            # plt.figure()
            # ji_ = np.sum(ji.get_comp_data(x_axes[0].axID),axis=1)*vi_axes[0].dx
            # plt.plot(ji_)
            # print('ji',ji_[0])

            ji = ji.integrate(axes=vi_axes, new_grid=self.grid_X,
                              new_ax_deriv_configs=self.fi.component.ax_deriv_configs)
            ji.scalar_multiply(self.ion_params.Z * self.ion_params.e)
            ji.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in vi_axes})
            # print('ji',ji.get_field_data(comps=0)[0])

            # plt.figure()
            # plt.plot(ji.get_comp_data(Z),'--',label='ion')
            # plt.plot(je.get_comp_data(Z),'--',label='elec')
            # plt.title('current')

            if j is not None:
                j.add(ji, compress_level=0, inplace=True)
            else:
                j = ji

            # plt.plot(j.get_comp_data(Z), ':', label='total')
            # plt.legend()

            # print('j deriv config', j[x_axes[0].axID].ax_deriv_configs[x_axes[0]].bc)

        if self.background_fe0 is not None or self.background_fi0 is not None:
            if self.background_j0 is None:
                j0 = None
                if self.background_fe0 is not None:
                    ve_axes = self.coords_ve.axes
                    je0 = self.background_fe0.xmultiply(ve_axes, compress_level=compress1)
                    je0 = je0.integrate(axes=ve_axes, new_grid=self.grid_X,
                                        new_ax_deriv_configs=self.background_fe0.component.ax_deriv_configs)
                    je0.replace_compIDs(
                         {v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in ve_axes})
                    je0.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
                    j0 = je0

                if self.background_fi0 is not None:
                    vi_axes = self.coords_vi.axes
                    ji0 = self.background_fi0.xmultiply(vi_axes, compress_level=compress1)
                    ji0 = ji0.integrate(axes=vi_axes, new_grid=self.grid_X,
                                      new_ax_deriv_configs=self.fi.component.ax_deriv_configs)
                    ji0.scalar_multiply(self.ion_params.Z * self.ion_params.e)
                    ji0.replace_compIDs(
                        {v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in vi_axes})
                    if j0 is None:
                        j0 = ji0
                    else:
                        j0.add(ji0, inplace=True, compress_level=0)

                j0.compress(inplace=True, compress_level=compress)
                self.background_j0 = j0
            j.add(self.background_j0, inplace=True)

        if j is not None:
            j.compress(inplace=True, compress_level=compress)

        return j

    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'VlasovMaxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs
            
            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt_e = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1)
        dFdt_i = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1, is_ion=True) \
                      if self.fi is not None and self.evolve_ion else None

        if self._evolve_EM:
            self.EM_sys.charge_density = self.compute_charge_density(compress=compress_level1)
            self.EM_sys.current_density = self.compute_current(compress=compress_level1)
            EM_sys_deriv = self.EM_sys.calculate_time_derivative(time=time,
                                                                 compress_level=compress_level,
                                                                 compress_level1=compress_level1)
        else:
            EM_sys_deriv = None

        # dFdt = self.create_like(dFdt_e, dFdt_i) #, dFdt_E, dFdt_B, dFdt_phi, dFdt_psi)
        EM_sys_deriv_fields = [EM_sys_deriv.get_field(k) for k in EM_sys_deriv.field_names]
        dFdt = self.create_like(dFdt_e, dFdt_i, *EM_sys_deriv_fields, recalc=False)

        return dFdt

    # @profile
    def _calculate_time_derivative_f(self, compress:int = 1, compress1:int = 0, compress2: int = 0, is_ion=False,
                                     verbose_plot=False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.fi if is_ion else self.fe
        dist0 = self.background_fi0 if is_ion else self.background_fe0

        if dist is None:  return None

        x_coords = self.coords_x.coords
        v_coordsys = self.coords_vi if is_ion else self.coords_ve
        v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
        v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
        v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
        # print('v_axes', x_coords, v_coords, v_axes_dict, v_axes)

        df0dv = self.df0dv_i if is_ion else self.df0dv_e
        v_grads = dist.gradient(deriv_axes=v_axes, compress_level=0) if df0dv is None else df0dv
        # print('vgrad', v_grads.componentIDs)

        ## grad_v of background distribution
        v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
        if dist0 is not None and v_grads_0 is None:
            v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
            if is_ion:
                self.gradv_fi0 = v_grads_0
            else:
                self.gradv_fe0 = v_grads_0
        # print('vgrad0', v_grads_0.componentIDs)

        v_grads.add(v_grads_0, inplace=True, compress_level=compress1)



        ### advection term
        convective_term = self._calculate_time_derivative_f_advection(compress1 = compress1, compress2 = compress2,
                                                                      is_ion=is_ion, verbose_plot=verbose_plot)
        ### force term
        lorentz_term = self._calculate_time_derivative_f_force(v_grads=v_grads,
                                                               compress1=compress1, compress2=compress2,
                                                               is_ion=is_ion, verbose_plot=verbose_plot)
        # exit()

        dFdt_s = convective_term.add(lorentz_term, compress_level=0)
        dFdt_s.name = self.fi.name if is_ion else self.fe.name

        ### get collisions
        coll_field = self.get_collision_term(is_ion=is_ion, v_axes=v_axes, v_grads=v_grads,
                                             compress1=compress1, compress2=compress2)
        dFdt_s.add(coll_field, inplace=True, compress_level=0)

        if compress:
            dFdt_s.compress(compress_level=compress)

        return dFdt_s


    def _calculate_time_derivative_f_advection(self, compress1:int = 0, compress2: int = 0,
                                               is_ion=False, verbose_plot:bool=False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.fi if is_ion else self.fe
        dist0 = self.background_fi0 if is_ion else self.background_fe0

        if dist is None:  return None

        x_axes = self.coords_x.axes
        v_coordsys = self.coords_vi if is_ion else self.coords_ve
        v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
        v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        if not self.upwind:
            v_axes_dict = None

        x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)

        ## grad_v of background distribution
        x_grads_0 = self.gradx_fi0 if is_ion else self.gradx_fe0
        if dist0 is not None and x_grads_0 is None:
            x_grads_0 = dist0.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
            if is_ion:
                self.gradx_fi0 = x_grads_0
            else:
                self.gradx_fe0 = x_grads_0
        x_grads.add(x_grads_0, inplace=True)


        ## intermediate plots
        if verbose_plot:
            dist_data = dist.get_comp_data() #  - self.fi.get_comp_data()
            xg_data = x_grads.get_comp_data(x_axes[0].coordinate)

            plt.figure()
            plt.imshow(dist_data)
            plt.xlabel('v')
            plt.ylabel('x')
            plt.title(f'dist, is ion {is_ion}')
            plt.colorbar()

            plt.figure()
            plt.plot(np.sum(dist_data,axis=1)*self.grid_X.dx)
            plt.xlabel('x')

            plt.figure()
            plt.plot(dist_data[0,:])
            plt.ylabel('v')

            plt.show()

            plt.figure()
            plt.imshow(xg_data)
            plt.xlabel('v')
            plt.ylabel('x')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(xg_data[:,0],label='gradx')
            plt.xlabel('v')
            plt.legend()
            plt.show()
            exit()

        ## convective term
        convective_term = x_grads.xdot(v_axes, compress_level=compress1)
        convective_term.scalar_multiply(-1, inplace=True)

        if verbose_plot:
            plt.figure()
            plt.imshow(convective_term.get_comp_data())
            plt.colorbar()
            plt.title('convective term')

        return convective_term


    # @profile
    def _calculate_time_derivative_f_force(self, v_grads = None,
                                           compress1: int = 0, compress2: int = 0,
                                           is_ion=False, verbose_plot:bool=False) -> Optional['Field']:

        dist = self.fi if is_ion else self.fe
        dist0 = self.background_fi0 if is_ion else self.background_fe0

        if dist is None and dist0 is None:  return None

        # x_axes = self.coords_x.axes
        # v_coordsys = self.coords_vi if is_ion else self.coords_ve
        # v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
        # v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        x_axes = self.coords_x.axes
        x_coords = self.coords_x.coords
        v_coordsys = self.coords_vi if is_ion else self.coords_ve
        v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
        v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
        v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
        # print('x coords', x_coords, v_coords, v_axes)

        ## grad_v of main distribution
        if v_grads is None:
            df0dv = self.df0dv_i if is_ion else self.df0dv_e
            v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress2) if df0dv is None else df0dv

            ## grad_v of background distribution
            v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
            if dist0 is not None and v_grads_0 is None:
                v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
                if is_ion:
                    self.gradv_fi0 = v_grads_0
                else:
                    self.gradv_fe0 = v_grads_0
            v_grads.add(v_grads_0, inplace=True, compress_level=compress1)


        ## intermediate plots
        if verbose_plot:
            dist_data = dist.get_comp_data()  # - self.fi.get_comp_data()
            vg_data = v_grads.get_comp_data(x_axes[0].coordinate)

            plt.figure()
            plt.imshow(vg_data)
            plt.xlabel('v')
            plt.ylabel('x')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(vg_data,label='gradx')
            plt.xlabel('v')
            plt.legend()
            plt.show()
            # exit()

        ## (EM) force term
        em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0, is_ion=is_ion)
        if em_term is not None:
            em_term.scalar_multiply(-1, inplace=True)

        if em_term is not None:
            em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
            print('em term', em_term.max_bonds())
            print('v grads', v_grads.max_bonds())
            # lorentz_term = v_grads.dot(em_term, comps=[ax.coordinate for ax in v_axes],
            #                            other_comps=[ax.coordinate for ax in x_axes], zipup=self.zipup,
            #                            compress_level=compress1, inner_compress_level=compress2)    # 5, 0
            lorentz_term = v_grads.dot(em_term, comps=v_coords,
                                       other_comps=x_coords, zipup=self.zipup,
                                       compress_level=compress1, inner_compress_level=compress2)  # 5, 0
        else:
            lorentz_term = None

        if verbose_plot:
            plt.figure()
            plt.imshow(lorentz_term.get_comp_data())
            plt.title('lorentz')
            plt.colorbar()
            plt.show()

        return lorentz_term


    def get_collision_term(self, is_ion=False, v_axes:Sequence['Axis']=None, v_grads: 'Field'=None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        dist = self.fi if is_ion else self.fe
        dist0 = self.background_fi0 if is_ion else self.background_fe0

        if v_axes is None:
            x_axes = self.coords_x.axes
            v_coordsys = self.coords_vi if is_ion else self.coords_ve
            v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
            v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]

        if self.collision.coll_type is None:
            return None

        elif self.collision.coll_type == CollisionType.LB:

            if dist.is_sqrt:   raise NotImplementedError('LB operator not implemented for sqrt field')

            if v_grads is None:
                df0dv = self.df0dv_i if is_ion else self.df0dv_e  # for linearized Vlasov
                v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                    if df0dv is None else df0dv

            ## grad_v of background distribution
            v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
            if dist0 is not None and v_grads_0 is None:
                v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
            v_grads.add(v_grads_0, inplace=True)

            ############  get maxwellian ######
            if is_ion:
                vth2 = self.ion_params.vth ** 2
                coll_rate = self.collision.coll_rate_i
            else:
                vth2 = self.elc_params.vth ** 2
                coll_rate = self.collision.coll_rate_e

            ## alternate method of computing collisions
            # v_drift = self.v_from_v0_i if is_ion else self.v_from_v0_e
            # coll = v_drift.elemental_multiply(dist[0], compress_level=0)  ## components are self.v_axes
            v0 = self.collision.v0_i if is_ion else self.collision.v0_e
            try:
                v_offsets = {v_ax: -v0[v_ax.coordinate] for v_ax in v_axes}
            except (IndexError, TypeError):
                v_offsets = {v_ax: -v0 for v_ax in v_axes}
            coll = dist.xmultiply(v_axes, offsets=v_offsets, )
            # exit()

            coll_1 = v_grads.scalar_multiply(vth2)  ## components are self.v_axes
            coll.add(coll_1, inplace=True, compress_level=0)
            # print('coll deriv params', coll[v_axes[0].coordinate].ax_deriv_configs)
            # coll.update_comp_deriv_params(axes=v_axes)

            coords_v = self.coords_vi if is_ion else self.coords_ve
            coll_field = coll.divergence(coords_v, compress_level=compress1, inner_compress_level=compress2)

            coll_field.scalar_multiply(coll_rate, inplace=True)

            return coll_field



