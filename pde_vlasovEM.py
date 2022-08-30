from setup.configs import *
import matplotlib.pyplot as plt
from field import Field, ScalarField
from pde_system import PDE_system
from pde_EM import Maxwell
from pde_vlasov import Vlasov
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis


class VlasovMaxwell(Vlasov):
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
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField','Boltzmann']],
                 sys_fi: Optional[Union['ScalarField','Boltzmann']],
                 EM_sys: Optional['Maxwell'] = None,
                 grid_X: Optional['Grid'] = None,
                 coords_x : Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional['SpeciesConfiguration'] = None,
                 ion_params: Optional['SpeciesConfiguration'] = None,
                 f0_gradv_e: Optional['Field'] = None,
                 f0_gradv_i: Optional['Field'] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 evolve_background=False,
                 evolve_ion=True, evolve_EM=True, normalize=True,
                 upwind=False, zipup=False,
                 te_order=2, compress_levels=None,
                 **kwargs
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov. 
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """

        grid_X = grid_X if EM_sys is None else EM_sys.grid_X

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         evolve_background=evolve_background,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         te_order=te_order, compress_levels=compress_levels)

        self.EM_sys = EM_sys
        self.evolve_EM = evolve_EM

        # self._x_advection = True
        # self._v_advection = True

        # self.compress_F = True
        # self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}
        #
        # self.collision = CollisionConfiguration(None)
        # self.dist0_e = None
        # self.dist0_i = None

    @property
    def EM_sys(self) -> Optional['Maxwell']:
        return self._EM_sys

    @EM_sys.setter
    def EM_sys(self, new_EM_sys: Optional['Maxwell']):
        self._EM_sys = Maxwell(None, None, grid_X=self.grid_X) if new_EM_sys is None else new_EM_sys

        if new_EM_sys is not None:
            assert(self._EM_sys.grid_X == self.grid_X), 'Maxwell grid_X does not match self'

            self.names['E'] = self._EM_sys.names['E'] if self._EM_sys.names['E'] is not None else 'E'
            self.names['B'] = self._EM_sys.names['B'] if self._EM_sys.names['B'] is not None else 'B'
            self.names['phi'] = self._EM_sys.names['phi'] if self._EM_sys.names['phi'] is not None else 'phi'
            self.names['psi'] = self._EM_sys.names['psi'] if self._EM_sys.names['psi'] is not None else 'psi'
            self.field_names = self.field_names[:2] + [self.names['E'], self.names['B'],
                                                       self.names['phi'], self.names['psi']]

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
            return super().get_field(field_name)

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
            self._fields[field_name] = new_field


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 6:
            new_fields = new_fields + (None,) *(6-len(new_fields))

        dist_e, dist_i, E, B, phi, psi = new_fields[:6]

        new_system = super().create_like(dist_e, dist_i, recalc=recalc, deep=deep)

        # if isinstance(dist_e, ScalarField):
        #     sys_fe = self.sys_fe.create_like(dist_e, recalc=recalc, deep=deep)
        # elif isinstance(dist_e, Boltzmann) or dist_e is None:
        #     sys_fe = dist_e
        # else:
        #     raise TypeError('dist e should be ScalarField or Boltzmann object')
        #
        # if isinstance(dist_i, ScalarField):
        #     sys_fi = self.sys_fe.create_like(dist_i, recalc=recalc, deep=deep)
        # elif isinstance(dist_i, Boltzmann) or dist_i is None:
        #     sys_fi = dist_i
        # else:
        #     raise TypeError('dist e should be ScalarField or Boltzmann object')


        EM_sys = self.EM_sys.create_like(E, B, phi, psi, recalc=recalc)
        new_system.EM_sys = EM_sys
        new_system.evolve_EM = self.evolve_EM

        # new_system = VlasovMaxwell(sys_fe, sys_fi, EM_sys,
        #                            evolve_ion=self.evolve_ion, normalize=self.do_normalization,
        #                            upwind=self.upwind, zipup=self.zipup, evolve_EM=self.evolve_EM,
        #                            te_order=self.te_order, compress_levels=self._comp_levels,
        #                            )
        #
        # if dist_e is None:      new_system.names['fe'] = self.names['fe']
        # if dist_i is None:      new_system.names['fi'] = self.names['fi']
        if E is None:           new_system.names['E'] = self.names['E']
        if B is None:           new_system.names['B'] = self.names['B']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if psi is None:         new_system.names['psi'] = self.names['psi']

        # new_system.collision = self.collision
        #
        # new_system.compress_F = self.compress_F
        # new_system.compress_F_opts = self.compress_F_opts
        #
        # new_system.background_p0 = self.background_p0
        # new_system.background_j0 = self.background_j0

        # new_system._x_advection = self._x_advection
        # new_system._v_advection = self._v_advection

        return new_system


    def copy(self):
        new_system = super().copy()
        ## update EM saved fields?
        return new_system



    def get_time_derivative_operator(self,compress=1,is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, force_term: Optional['Field'] = None,
                   method=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'VlasovMaxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
        """
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        self.evolve_EM = False
        state0 = self if inplace else self.copy()

        comp3 = 0 if compress_level == 0 else compress_level + 3
        # if compress_level == 0:
        #     comp1 = comp2 = comp3 = comp4 = comp5 = 0
        # else:
        #     comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if is_first_time_step:
            state0.EM_sys = state0.EM_sys.split_step(dt / 2)
            state0.EM_sys.time = self.EM_sys.time + dt / 2 if self.EM_sys.time is not None else None

        em_term = self.compute_force_term(compress_level1=comp3, compress_level2=0)

        # dt/2, dt, dt/2 structure
        super(VlasovMaxwell, state0).split_step(dt, deriv0, method=method, force_term=em_term,
                                                inplace=True, is_first_time_step=True, is_last_time_step=True,
                                                compress_level=compress_level, verbose_plot=verbose_plot)

        # ## evolve force advection dt/2
        # state0.fe_sys = state0.fe_sys.get_force_advection(dt / 2, inplace=True, method=None, compress=comp2, compress1=comp4,
        #                                                   compress2=comp5, verbose_plot=verbose_plot)
        # ## evolve velocity advection dt
        # state0 = state0.get_vel_advection(dt, inplace=True, method=method, compress=comp2, compress1=comp4,
        #                                   compress2=comp5, verbose_plot=verbose_plot)
        # ## evolve force advection dt/2
        # state0 = state0.get_force_advection(dt / 2, inplace=True, method=None, compress=comp2, compress1=comp4,
        #                                     compress2=comp5, verbose_plot=verbose_plot)

        # ## include collisions
        # if self.collision.coll_type is not None:
        #     raise NotImplementedError
        #     deriv_coll_e = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=False)
        #     deriv_coll_i = self.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=True)
        #     dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
        #     state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        # state0.time = self.time + dt if self.time is not None else None

        state0.EM_sys.current_density = state0.compute_current(compress=comp3)
        state0.EM_sys.charge_density  = state0.compute_current(compress=comp3)

        # print('current density', state0.EM_sys.current_density.components)

        dt_ = dt/2 if is_last_time_step else dt
        state0.EM_sys = state0.EM_sys.split_step(dt_)
        state0.EM_sys.time = self.EM_sys.time + dt_ if self.EM_sys.time is not None else None

        return state0


    # def get_force_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
    #                          verbose_plot=False):
    #     """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
    #     """
    #     state0 = self if inplace else self.copy()
    #
    #     if method is None:
    #         deriv_vi0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                               is_ion=True, verbose_plot=verbose_plot)
    #         deriv_ve0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                               is_ion=False, verbose_plot=verbose_plot)
    #         dFdt_v = state0.create_like(deriv_ve0, deriv_vi0, recalc=False)
    #         state0 = state0.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)
    #     else:
    #         raise NotImplementedError
    #
    #     return state0
    #
    #
    # def get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
    #                        verbose_plot=False):
    #     """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
    #     """
    #     state0 = self if inplace else self.copy()
    #
    #     if method is None:
    #         deriv_xe0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
    #                                                                   is_ion=False, verbose_plot=verbose_plot)
    #         deriv_xi0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
    #                                                                   is_ion=True, verbose_plot=verbose_plot)
    #         dFdt_x = state0.create_like(deriv_xe0, deriv_xi0, recalc=False)
    #         state0 = state0.euler(dt, deriv0=dFdt_x, inplace=True, compress_level=compress)
    #
    #     elif method == 'SL':
    #         print('SL advection')
    #         assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
    #             'SL time evolution for spatial basis only'
    #
    #         for x_ax in self.x_axes:
    #             ## ions
    #             fi_gtn = state0.fi.component
    #             v_ax = self.vi_axes_dict[x_ax]
    #             deriv_params = fi_gtn.ax_deriv_configs[x_ax]
    #             compress_opts = state0.fi.compress_config.get_compress_opts(compress)
    #             fi_mpo_gtn = fi_gtn.grid._get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params)
    #             fi_gtn.apply(fi_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #
    #             ## elecs
    #             fe_gtn = state0.fe.component
    #             v_ax = self.ve_axes_dict[x_ax]
    #             deriv_params = fi_gtn.ax_deriv_configs[x_ax]
    #             compress_opts = state0.fe.compress_config.get_compress_opts(compress)
    #             fe_mpo_gtn = fe_gtn.grid._get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params)
    #             fe_gtn.apply(fe_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #
    #     elif method == 'SL-pfc':
    #         print('SL-pfc advection')
    #         assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
    #                 'SL time evolution for spatial basis only'
    #
    #         for x_ax in self.x_axes:
    #             ## ions
    #             fi_gtn = state0.fi.component
    #             v_ax = self.vi_axes_dict[x_ax]
    #             deriv_params = fi_gtn.ax_deriv_configs[x_ax]
    #             compress_opts = state0.fi.compress_config.get_compress_opts(compress)
    #             fi_mpo_gtn = fi_gtn.grid._get_mpo_advection_SL_pfc(x_ax, v_ax, dt, deriv_params)
    #             fi_gtn.apply(fi_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #
    #             ## elecs
    #             fe_gtn = state0.fe.component
    #             v_ax = self.ve_axes_dict[x_ax]
    #             deriv_params = fi_gtn.ax_deriv_configs[x_ax]
    #             compress_opts = state0.fe.compress_config.get_compress_opts(compress)
    #             fe_mpo_gtn = fe_gtn.grid._get_mpo_advection_SL_pfc(x_ax, v_ax, dt, deriv_params)
    #             fe_gtn.apply(fe_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #     else:
    #         raise NotImplementedError
    #
    #     return state0


    #############################

    # @profile
    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False):
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

        if em_term is not None:
            em_term.scalar_multiply(matl_params.e * matl_params.Z / matl_params.mass, inplace=True)
            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

        if verbose_plot:
            print('get force term plot')
            x_coords = self.coords_x.coords
            ax_x, ax_y, ax_vx, ax_vy, ax_vz = em_term[em_term.componentIDs[0]].grid.axes
            for x_c in x_coords:
                plt.figure()
                em_term_data = em_term.get_comp_data(x_c, ax_select={ax_vx: 0,
                                                                     ax_vy: 0,
                                                                     ax_vz: 0})
                if em_term_data is None:
                    print('em term data is None', x_c)
                    continue
                plt.imshow(np.real(em_term_data))
                plt.title(f'em {x_c}')
                plt.colorbar()
            plt.show()

        # print('E + v x B term', em_term.components)

        return em_term


    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'VlasovMaxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs
            
            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt = super().calculate_time_derivative(time=time, compress_level=compress_level,
                                                 compress_level1=compress_level1, compress_level2=compress_level2,
                                                 verbose_plot=verbose_plot)

        if self.evolve_EM:
            self.EM_sys.charge_density  = self.compute_charge_density(compress=compress_level1)
            self.EM_sys.current_density = self.compute_current(compress=compress_level1)
            EM_sys_deriv = self.EM_sys.calculate_time_derivative(time=time,
                                                                 compress_level=compress_level,
                                                                 compress_level1=compress_level1)

            dFdt.EM_sys = EM_sys_deriv

            # EM_sys_deriv_fields = [EM_sys_deriv.get_field(k) for k in EM_sys_deriv.field_names]
        # else:
        #     EM_sys_deriv_fields = []

        return dFdt


    # # @profile
    # def _calculate_time_derivative_f(self, compress:int = 1, compress1:int = 0, compress2: int = 0, is_ion=False,
    #                                  verbose_plot=False) -> Optional['Field']:
    #     """ df/dt = ...
    #         dB/dt + curl(E) = 0
    #         e0*mu0 dE/dt - curl(B) = -mu0 J
    #         J = sum_s qs ns vs
    #
    #         df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #     print(dist.component)
    #     # if dist is None:  return None
    #
    #     x_coords = self.coords_x.coords
    #     X, Y, Z = x_coords
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
    #     VX, VY, VZ = v_coords
    #     v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
    #     v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
    #     # print('v_axes', x_coords, v_coords, v_axes_dict, v_axes)
    #
    #     df0dv = self.df0dv_i if is_ion else self.df0dv_e
    #     v_grads = dist.gradient(deriv_axes=v_axes, compress_level=0) if df0dv is None else df0dv
    #     if verbose_plot:
    #         plt.figure()
    #         x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist.component.grid.axes
    #         v_grads_y = v_grads.get_comp_data(VY, ax_select={x_ax: x_ax.npts//2, y_ax:y_ax.npts//2,
    #                                                                vx_ax: vx_ax.npts//2})
    #         if v_grads_y is not None:
    #             plt.imshow(np.real(v_grads_y))
    #             plt.colorbar()
    #             plt.title('v grad y')
    #             plt.show()
    #
    #     ## grad_v of background distribution
    #     v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #     if dist0 is not None and v_grads_0 is None:
    #         v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #         if is_ion:
    #             self.gradv_fi0 = v_grads_0
    #         else:
    #             self.gradv_fe0 = v_grads_0
    #
    #         if verbose_plot:
    #             plt.figure()
    #             x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist0.component.grid.axes
    #             v_grads_y = v_grads_0.get_comp_data(VY, ax_select={x_ax: x_ax.npts//2, y_ax:y_ax.npts//2,
    #                                                                vx_ax: vx_ax.npts//2})
    #             plt.imshow(np.real(v_grads_y))
    #             plt.colorbar()
    #             plt.title('v grad 0 y')
    #             plt.show()
    #
    #     # print('vgrad0', v_grads_0.componentIDs)
    #     v_grads.add(v_grads_0, inplace=True, compress_level=compress1)
    #
    #
    #     ### advection term
    #     convective_term = self._calculate_time_derivative_f_advection(compress1 = compress1, compress2 = compress2,
    #                                                                   is_ion=is_ion, verbose_plot=verbose_plot)
    #     ### force term
    #     lorentz_term = self._calculate_time_derivative_f_force(v_grads=v_grads,
    #                                                            compress1=compress1, compress2=compress2,
    #                                                            is_ion=is_ion, verbose_plot=verbose_plot)
    #     # exit()
    #
    #     dFdt_s = convective_term.add(lorentz_term, compress_level=0)
    #     dFdt_s.name = self.fi.name if is_ion else self.fe.name
    #
    #     ### get collisions
    #     coll_field = self.get_collision_term(is_ion=is_ion, v_axes=v_axes, v_grads=v_grads,
    #                                          compress1=compress1, compress2=compress2)
    #     dFdt_s.add(coll_field, inplace=True, compress_level=0)
    #
    #     if compress:
    #         dFdt_s.compress(compress_level=compress)
    #
    #     return dFdt_s
    #
    #
    # def _calculate_time_derivative_f_advection(self, compress1:int = 0, compress2: int = 0,
    #                                            is_ion=False, verbose_plot:bool=False) -> Optional['Field']:
    #     """ df/dt = ...
    #         dB/dt + curl(E) = 0
    #         e0*mu0 dE/dt - curl(B) = -mu0 J
    #         J = sum_s qs ns vs
    #
    #         df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     # if dist is None:  return None
    #
    #     x_axes = self.coords_x.axes
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #     v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     if not self.upwind:
    #         v_axes_dict = None
    #
    #     x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
    #
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #         xg_data = x_grads.get_comp_data(ax_y.coordinate,
    #                                         ax_select={ax_vx: ax_vx.npts//2, ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(dist_data)
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title(f'dist, is ion {is_ion}')
    #         plt.colorbar()
    #
    #         plt.figure()
    #         plt.plot(dist_data[0,:])
    #         plt.title(f'dist, is ion {is_ion}')
    #
    #         plt.show()
    #
    #         plt.figure()
    #         plt.imshow(xg_data)
    #         plt.xlabel('x')
    #         plt.ylabel('y')
    #         plt.title('xgrad')
    #         plt.colorbar()
    #         # plt.show()
    #
    #         plt.figure()
    #         plt.plot(xg_data[0,:],label='gradx')
    #         plt.xlabel('x')
    #         plt.legend()
    #         plt.show()
    #
    #     exit()
    #
    #     ## grad_v of background distribution
    #     x_grads_0 = self.gradx_fi0 if is_ion else self.gradx_fe0
    #     if dist0 is not None and x_grads_0 is None:
    #         x_grads_0 = dist0.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
    #         if is_ion:
    #             self.gradx_fi0 = x_grads_0
    #         else:
    #             self.gradx_fe0 = x_grads_0
    #     x_grads.add(x_grads_0, inplace=True)
    #
    #
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist0.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #         xg_data = x_grads_0.get_comp_data(ax_y.coordinate,
    #                                         ax_select={ax_vx: ax_vx.npts//2, ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(dist_data)
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title(f'dist 0, is ion {is_ion}')
    #         plt.colorbar()
    #
    #         # plt.figure()
    #         # plt.plot(np.sum(dist_data,axis=1)*self.grid_X.dx)
    #         # plt.xlabel('x')
    #         #
    #         # plt.figure()
    #         # plt.plot(dist_data[0,:])
    #         # plt.ylabel('v')
    #
    #         plt.show()
    #
    #         plt.figure()
    #         plt.imshow(xg_data)
    #         plt.xlabel('x')
    #         plt.ylabel('y')
    #         plt.title('xgrad 0')
    #         plt.colorbar()
    #         # plt.show()
    #
    #         plt.figure()
    #         plt.plot(xg_data[0,:],label='gradx')
    #         plt.xlabel('x')
    #         plt.legend()
    #         plt.show()
    #
    #     ## convective term
    #     print('x_grads', x_grads.componentIDs)
    #     print(v_axes, self.coords_x.coords)
    #     convective_term = x_grads.xdot(v_axes, compIDs=self.coords_x.coords, compress_level=compress1)
    #     convective_term.scalar_multiply(-1, inplace=True)
    #
    #     if verbose_plot:
    #         plt.figure()
    #         plt.imshow(convective_term.get_comp_data(ax_select={ax_vx: ax_vx.npts//2,
    #                                                             ax_vy:ax_vy.npts//2,
    #                                                             ax_vz: ax_vz.npts//2}))
    #         plt.colorbar()
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title('convective term')
    #         plt.show()
    #         # exit()
    #
    #     return convective_term
    #
    #
    # # @profile
    # def _calculate_time_derivative_f_force(self, v_grads = None,
    #                                        compress1: int = 0, compress2: int = 0,
    #                                        is_ion=False, verbose_plot:bool=False) -> Optional['Field']:
    #
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     if dist is None and dist0 is None:  return None
    #
    #     # x_axes = self.coords_x.axes
    #     # v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     # v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #     # v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     x_axes = self.coords_x.axes
    #     x_coords = self.coords_x.coords
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
    #     v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
    #     v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
    #     # print('x coords', x_coords, v_coords, v_axes)
    #
    #     ## grad_v of main distribution
    #     if v_grads is None:
    #         df0dv = self.df0dv_i if is_ion else self.df0dv_e
    #         v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress2) if df0dv is None else df0dv
    #
    #         ## grad_v of background distribution
    #         v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #         if dist0 is not None and v_grads_0 is None:
    #             v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #             if is_ion:
    #                 self.gradv_fi0 = v_grads_0
    #             else:
    #                 self.gradv_fe0 = v_grads_0
    #         v_grads.add(v_grads_0, inplace=True, compress_level=compress1)
    #
    #
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                   ax_vx: ax_vx.npts//2,
    #                                                   ax_vy:ax_vy.npts//2,
    #                                                   ax_vz: ax_vz.npts//2})
    #         vg_data = v_grads.get_comp_data(ax_vx.coordinate,
    #                                         ax_select={ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    # ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(np.real(vg_data))
    #         plt.xlabel('vy')
    #         plt.ylabel('y')
    #         plt.colorbar()
    #         plt.title('grad v')
    #         plt.show()
    #
    #         plt.figure()
    #         plt.plot(np.real(vg_data[ax_y.npts//2,:]),label='gradv')
    #         plt.xlabel('vy')
    #         plt.legend()
    #         plt.show()
    #         # exit()
    #
    #     ## (EM) force term
    #     em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0, is_ion=is_ion,
    #                                       verbose_plot=verbose_plot)
    #     if em_term is not None:
    #         em_term.scalar_multiply(-1, inplace=True)
    #
    #     if em_term is not None:
    #         em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
    #         print('em term', em_term.max_bonds())
    #         print('v grads', v_grads.max_bonds())
    #         # lorentz_term = v_grads.dot(em_term, comps=[ax.coordinate for ax in v_axes],
    #         #                            other_comps=[ax.coordinate for ax in x_axes], zipup=self.zipup,
    #         #                            compress_level=compress1, inner_compress_level=compress2)    # 5, 0
    #         lorentz_term = v_grads.dot(em_term, comps=v_coords,
    #                                    other_comps=x_coords, zipup=self.zipup,
    #                                    compress_level=compress1, inner_compress_level=compress2)  # 5, 0
    #     else:
    #         lorentz_term = None
    #
    #     if verbose_plot:
    #         plt.figure()
    #         lorentz_data = lorentz_term.get_comp_data(ax_select={ax_vx: ax_vx.npts//2,
    #                                                          ax_vy: ax_vy.npts//2,
    #                                                          ax_vz: ax_vz.npts//2})
    #         plt.imshow(np.real(lorentz_data))
    #         plt.title('lorentz')
    #         plt.colorbar()
    #         plt.show()
    #
    #         exit()
    #
    #     return lorentz_term
    #
    #
    # def get_collision_term(self, is_ion=False, v_axes:Sequence['Axis']=None, v_grads: 'Field'=None,
    #                        compress1=0, compress2=0):
    #     """ obtain collision term
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     if v_axes is None:
    #         x_axes = self.coords_x.axes
    #         v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #         v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #         v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     if self.collision.coll_type is None:
    #         return None
    #
    #     elif self.collision.coll_type == CollisionType.LB:
    #
    #         if dist.is_sqrt:   raise NotImplementedError('LB operator not implemented for sqrt field')
    #
    #         if v_grads is None:
    #             df0dv = self.df0dv_i if is_ion else self.df0dv_e  # for linearized Vlasov
    #             v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
    #                 if df0dv is None else df0dv
    #
    #         ## grad_v of background distribution
    #         v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #         if dist0 is not None and v_grads_0 is None:
    #             v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #         v_grads.add(v_grads_0, inplace=True)
    #
    #         ############  get maxwellian ######
    #         if is_ion:
    #             vth2 = self.ion_params.vth ** 2
    #             coll_rate = self.collision.coll_rate_i
    #         else:
    #             vth2 = self.elc_params.vth ** 2
    #             coll_rate = self.collision.coll_rate_e
    #
    #         ## alternate method of computing collisions
    #         # v_drift = self.v_from_v0_i if is_ion else self.v_from_v0_e
    #         # coll = v_drift.elemental_multiply(dist[0], compress_level=0)  ## components are self.v_axes
    #         v0 = self.collision.v0_i if is_ion else self.collision.v0_e
    #         try:
    #             v_offsets = {v_ax: -v0[v_ax.coordinate] for v_ax in v_axes}
    #         except (IndexError, TypeError):
    #             v_offsets = {v_ax: -v0 for v_ax in v_axes}
    #         coll = dist.xmultiply(v_axes, offsets=v_offsets, )
    #         # exit()
    #
    #         coll_1 = v_grads.scalar_multiply(vth2)  ## components are self.v_axes
    #         coll.add(coll_1, inplace=True, compress_level=0)
    #         # print('coll deriv params', coll[v_axes[0].coordinate].ax_deriv_configs)
    #         # coll.update_comp_deriv_params(axes=v_axes)
    #
    #         coords_v = self.coords_vi if is_ion else self.coords_ve
    #         coll_field = coll.divergence(coords_v, compress_level=compress1, inner_compress_level=compress2)
    #
    #         coll_field.scalar_multiply(coll_rate, inplace=True)
    #
    #         return coll_field
    #


