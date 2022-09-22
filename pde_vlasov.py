from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid

class Vlasov(PDE_system):
    """ df/dt + v grad(f) + F grad_v(f) = 0
    """
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField','Boltzmann']],
                 sys_fi: Optional[Union['ScalarField','Boltzmann']],
                 # sys_fe: Optional['ScalarField'],
                 # sys_fi: Optional['ScalarField'],
                 grid_X: Optional['Grid']=None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional['SpeciesConfiguration'] = None,
                 ion_params: Optional['SpeciesConfiguration'] = None,
                 f0_gradv_e: Optional['Field'] = None,
                 f0_gradv_i: Optional['Field'] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_force_neg_e: Optional['Field'] = None,
                 background_force_neg_i: Optional['Field'] = None,
                 normalize=True, upwind=False, zipup=False, evolve_ion=True,
                 te_order=4, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov.
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        if isinstance(sys_fe, ScalarField):
            fe = sys_fe
            sys_fe = Boltzmann(fe, grid_X=grid_X, coords_x=coords_x, coords_v=coords_ve, matl_params=elc_params,
                               background_f=background_fe0, background_force_neg=background_force_neg_e,
                               f0_gradv=f0_gradv_e, normalize=normalize, upwind=upwind, zipup=zipup,
                               te_order=te_order, compress_levels=compress_levels)
        elif isinstance(sys_fe, Boltzmann):
            fe = sys_fe.f
            if background_fe0 is not None:
                sys_fe.background_f = background_fe0
        elif sys_fe is None:
            fe = None
        else:
            raise TypeError('sys_fe should be ScalarField or Boltzmann object')

        if isinstance(sys_fi, ScalarField):
            fi = sys_fi
            sys_fi = Boltzmann(fi, grid_X=sys_fe.grid_X, coords_x=coords_x, coords_v=coords_vi, matl_params=ion_params,
                               background_f=background_fi0, background_force_neg=background_force_neg_i,
                               f0_gradv=f0_gradv_i, normalize=normalize, upwind=upwind, zipup=zipup,
                               te_order=te_order, compress_levels=compress_levels)
        elif isinstance(sys_fi, Boltzmann):
            fi = sys_fi.f
            if background_fi0 is not None:
                sys_fi.background_f = background_fi0
        elif sys_fi is None:
            fi = None
        else:
            raise TypeError('sys_fe should be ScalarField or Boltzmann object')


        self.names = {}
        if fe is not None and fi is not None and fe.name == fi.name:
            fe.name += 'e'
            fi.name += 'i'
        self.names['fe'] = fe.name if fe is not None else 'fe'
        self.names['fi'] = fi.name if fi is not None else 'fi'

        field_names = [self.names[x] for x in ['fe', 'fi']]

        super().__init__(# fe, fi,
                         field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         # field_compress_config=field_compress_config,
                         )

        self.sys_fe = sys_fe
        self.sys_fi = sys_fi

        self.coords_x  = sys_fe.coords_x if sys_fe is not None else sys_fi.coords_x
        self.coords_ve = sys_fe.coords_v if sys_fe is not None else coords_ve
        self.coords_vi = sys_fi.coords_v if sys_fi is not None else coords_vi
        self.grid_X = sys_fe.grid_X

        self.x_axes = sys_fe.x_axes
        self.ve_axes_dict = self.sys_fe.v_axes_dict
        self.vi_axes_dict = self.sys_fi.v_axes_dict

        self.elc_params = self.sys_fe.matl_params
        self.ion_params = self.sys_fi.matl_params

        self.velocities_e, self.velocities_i = None, None
        if self.sys_fe is not None:
            self.velocities_e = self.sys_fe.velocities
            self.velocities_e.name = 'vel_e'
        if self.sys_fi is not None:
            self.velocities_i = self.sys_fi.velocities
            self.velocities_i.name = 'vel_i'

        self.background_p0 = None
        self.background_j0 = None

        # self.evolve_background = sys_fe.evolve_background or sys_fi.evolve_background
        self.evolve_ion = evolve_ion
        self.upwind = upwind
        self.zipup = zipup

        self.compress_F = True
        self.compress_F_opts = {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}

        self.collision = CollisionConfiguration(None)
        self.coll_dist0 = None
        # self.v_from_v0_e = None   # data_ve - vdrift_e
        # self.v_from_v0_i = None   # data_vi - vdrift_i


    @property
    def fe(self) -> Optional['ScalarField']:
        return self.sys_fe.f

    @fe.setter
    def fe(self, new_field: 'ScalarField'):
        # self.fields[self.names['fe']] = new_field
        self.sys_fe.f = new_field

    @property
    def fi(self) -> Optional['ScalarField']:
        return self.sys_fi.f

    @fi.setter
    def fi(self, new_field: 'ScalarField'):
        # self.fields[self.names['fi']] = new_field
        self.sys_fi.f = new_field

    # @property
    # def fields:
    #     return {self.names['fe']: self.fe, self.names['fi']: self.fi}

    def get_field(self, field_name):
        """ return ith component of the field. for convenience
                """
        if field_name == self.names['fe']:
            return self.fe
        elif field_name == self.names['fi']:
            return self.fi
        else:
            return self._fields[field_name]

    @property
    def background_fe(self) -> Optional['Boltzmann']:
        return self.sys_fe.background_f

    @background_fe.setter
    def background_fe(self, f0: Optional[Union['ScalarField', 'Boltzmann']]):
        self.sys_fe.background_f = f0
        self.background_j0 = None
        self.background_p0 = None

    @property
    def background_fi(self) -> Optional['Boltzmann']:
        return self.sys_fi.background_f

    @background_fi.setter
    def background_fi(self, f0: Optional[Union['ScalarField', 'Boltzmann']]):
        self.sys_fi.background_f = f0
        self.background_j0 = None
        self.background_p0 = None

    @property
    def total_fe(self) -> Optional['ScalarField']:
        f = self.fe
        f = self.background_fe if f is None else f.add(self.background_fe)
        return f

    @property
    def total_fi(self) -> Optional['ScalarField']:
        f = self.fi
        f = self.background_fi if f is None else f.add(self.background_fi)
        return f


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 2:
            new_fields = new_fields + (None,) * (2 - len(new_fields))

        dist_e, dist_i = new_fields[:2]

        if isinstance(dist_e, ScalarField):
            sys_fe = self.sys_fe.create_like(dist_e, recalc=recalc, deep=deep)
        elif isinstance(dist_e, Boltzmann) or dist_e is None:
            sys_fe = dist_e
        else:
            raise TypeError('dist e should be ScalarField or Boltzmann object')

        if isinstance(dist_i, ScalarField):
            sys_fi = self.sys_fi.create_like(dist_i, recalc=recalc, deep=deep)
        elif isinstance(dist_i, Boltzmann) or dist_i is None:
            sys_fi = dist_i
        else:
            raise TypeError('dist e should be ScalarField or Boltzmann object')

        new_system = self.__class__(sys_fe, sys_fi, grid_X=self.grid_X,
                                    evolve_ion=self.evolve_ion, normalize=self.do_normalization,
                                    upwind=self.upwind, zipup=self.zipup,
                                    te_order=self.te_order, compress_levels=self._comp_levels,
                                    )

        if dist_e is None:      new_system.names['fe'] = self.names['fe']
        if dist_i is None:      new_system.names['fi'] = self.names['fi']

        new_system.collision = self.collision
        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

        new_system.background_p0 = self.background_p0
        new_system.background_j0 = self.background_j0

        # new_system.v_from_v0_i = self.v_from_v0_i
        # new_system.v_from_v0_e = self.v_from_v0_e

        return new_system


    def copy(self):
        """ create a new system like this with new fields
        """
        new_system = super().copy()

        new_system.sys_fe.gradx_f = self.sys_fe.gradx_f
        new_system.sys_fe.gradv_f = self.sys_fe.gradv_f
        new_system.sys_fe.density = self.sys_fe.density
        new_system.sys_fe.current = self.sys_fe.flow

        new_system.sys_fi.gradx_f = self.sys_fi.gradx_f
        new_system.sys_fi.gradv_f = self.sys_fi.gradv_f
        new_system.sys_fi.density = self.sys_fi.density
        new_system.sys_fi.current = self.sys_fi.flow

        return new_system


    def normalize(self, target_e=None, target_i=None):
        """ normalized density field
            in place operation
        """
        if self.sys_fi is not None:
            self.sys_fi.normalize(target_i)

        if self.sys_fe is not None:
            self.sys_fe.normalize(target_e)
        return


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                   method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False,
                   inplace=False, compress_level=1, verbose_plot: bool = False) -> 'Vlasov':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)

            Steps: (init force advection); vel advection; EM update; force advection
        """
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)


        if is_first_time_step:
            ## evolve force advection dt/2
            state0.get_force_advection(dt/2, inplace=True, method=method_f, verbose_plot=verbose_plot,
                                       compress=comp1, compress1=comp4, compress2=comp5,)

        ## evolve velocity advection dt
        state0.get_vel_advection(dt, inplace=True, method=method_v, verbose_plot=verbose_plot,
                                 compress=comp1, compress1=comp4, compress2=comp5,)

        ## update V or EM sys
        state0.update_EM_sys(dt, inplace=True, compress=comp1)

        if verbose_plot:
            from pde_vlasovES import VlasovPoisson
            if isinstance(self, VlasovPoisson):
                V_data = state0.V.get_comp_data()
                plt.figure()
                plt.plot(V_data)
                plt.title('V')
                plt.show()

        dt_ = dt/2 if is_last_time_step else dt
        state0.get_force_advection(dt_, inplace=True, method=method_f, verbose_plot=verbose_plot,
                                   compress=comp1, compress1=comp4, compress2=comp5,)

        ## include collisions (cross ion/elc collisions)
        if self.collision.coll_type is not None:
            raise NotImplementedError
            deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None

        return state0


    def update_EM_sys(self, dt, inplace=False, compress=1, compress1=2, compress2=4, **kwargs):
        """ update E, B (called by split step)
        """
        return self if inplace else self.copy()


    # def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
    #                method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False,
    #                inplace=False, compress_level=1, compress_level1=4, verbose_plot: bool = False) -> 'Vlasov':
    #     """ take split step for EM system and ion,electron advection terms
    #         each step is just an Euler update
    #         TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
    #
    #         change to a more general:
    #         (init force advection); vel advection; EM update; force advection
    #     """
    #     ## at initialization, need to evolve EM_sys with dt/2
    #     # self.upwind = True
    #     state0 = self if inplace else self.copy()
    #
    #     state0.sys_fe.force_term = state0.compute_force_term(is_ion=False, compress_level=compress_level,
    #                                                        compress_level1=compress_level1, compress_level2=0)
    #     state0e = state0.sys_fe.split_step(dt, method_v=method_v, method_f=method_f, inplace=True,
    #                                        is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
    #                                        compress_level=compress_level, verbose_plot=verbose_plot)
    #
    #     state0.sys_fi.force_term = state0.compute_force_term(is_ion=True, compress_level=compress_level,
    #                                                          compress_level1=compress_level1, compress_level2=0)
    #     state0i = state0.sys_fi.split_step(dt, method_v=method_v, method_f=method_f, inplace=True,
    #                                        is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
    #                                        compress_level=compress_level, verbose_plot=verbose_plot)
    #
    #     # state0.sys_fe = state0e
    #     # state0.sys_fi = state0i
    #
    #     ## include collisions (cross ion/elc collisions)
    #     if self.collision.coll_type is not None:
    #         raise NotImplementedError
    #         deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
    #         dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
    #         state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)
    #
    #     state0.time = self.time + dt if self.time is not None else None
    #
    #     return state0


    def get_force_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                            verbose_plot=False):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=compress1, compress_level2=0)
        state0.sys_fe.get_force_advection(dt, inplace=True, method=method,
                                          compress=compress, compress1=compress1, compress2=compress2,
                                          verbose_plot=verbose_plot)

        state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=compress1, compress_level2=0)
        state0.sys_fi.get_force_advection(dt, inplace=True, method=method,
                                          compress=compress, compress1=compress1, compress2=compress2,
                                          verbose_plot=verbose_plot)
        return state0


    def get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                          verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        state0.sys_fe = state0.sys_fe.get_vel_advection(dt, inplace=inplace, method=method, compress=compress,
                                                        compress1=compress1, compress2=compress2,
                                                        verbose_plot=verbose_plot)
        state0.sys_fi = state0.sys_fi.get_vel_advection(dt, inplace=inplace, method=method, compress=compress,
                                                        compress1=compress1, compress2=compress2,
                                                        verbose_plot=verbose_plot)
        return state0

    #############################

    # @profile
    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_only=False) -> 'Field':
        """ compute EM force:  q/m (E + v x B)
        """
        raise NotImplementedError


    # @profile
    def compute_charge_density(self, compress=1):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        # print('vlasov computing charge density')
        charge_density = None

        if self.fe is not None:
            e_density = self.sys_fe.compute_density(compress=0)
            e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e / self.elc_params.eps0, inplace=True)
            charge_density = e_density

        if self.fi is not None:
            i_density = self.sys_fi.compute_density(compress=0)
            i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e / self.ion_params.eps0, inplace=True)

            if charge_density is not None:
                charge_density.add(i_density, inplace=True, compress_level=0)
            else:
                charge_density = i_density

        if charge_density is not None and compress:
            charge_density.compress(inplace=True, compress_level=compress)

        # plt.figure()
        # print(type(charge_density))
        # charge_data = charge_density.get_comp_data()
        # plt.plot(charge_data)
        # plt.plot('charge')

        # print('n(x) deriv config', charge_density.component.ax_deriv_configs)

        return charge_density


    # @profile
    def compute_current(self, compress=1, compress1=0) -> 'Field':
        """ compute j(x) = \integ dv \sum_s q_s v_s f_s
        """
        # print('vlasov computing current density')
        j = None

        if self.fe is not None:
            je = self.sys_fe.compute_current(compress=0, compress1=compress1)
            je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
            j = je

            # ve_axes = self.coords_ve.axes
            # je = self.fe.xmultiply(ve_axes, compress_level=compress1)

            # # plt.figure()
            # # plt.imshow(self.fe.get_comp_data())
            #
            # # plt.figure()
            # # plt.imshow(je.get_field_data(comps=0))
            #
            # # plt.figure()
            # # VX, VY, VZ = self.coords_ve.coords
            # # X, Y, Z = self.coords_x.coords
            # # dv = np.prod( [ax.dx for ax in self.fe.grid.axes[1:]] )
            # # je_z = je.get_comp_data(VZ)
            # # fe_data = self.fe.get_comp_data()
            # # if je_z is not None:
            # #     # v_max = np.max(np.abs(ve_axes[-1].xpts))
            # #     fe_data = fe_data.dot(np.diag(ve_axes[-1].xpts))
            # #     # print('v_max', np.max(np.abs(ve_axes[-1].xpts)))
            # #     # print('v sum', np.sum(np.abs(ve_axes[-1].xpts))*ve_axes[-1].dx)
            # #     sum_fe = np.sum( np.real(fe_data), axis=(1,2,3)) * dv * -1
            # #     out = np.sum( je_z, axis=(1,2,3)) * dv * -1
            # #     plt.plot(np.real(sum_fe), label='fe')
            # #     # print(sum_fe[0], sum_fe[-1])
            # #     # print(out[0], out[-1])
            # #     plt.plot( np.real(out), label='je np' )
            #
            # je = je.integrate(axes=ve_axes, new_grid=self.grid_X,
            #                   new_ax_deriv_configs=self.fe.component.ax_deriv_configs)
            # je.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in ve_axes})
            # je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
            #
            # # je_z = je.get_comp_data(Z)
            # # if je_z is not None:
            # #     print(je_z[0], je_z[-1])
            # #     plt.plot(np.real(je_z),'--', label='je tn')
            # #     plt.legend()
            # #     plt.title('Jz')
            # #     plt.show()
            # #     # print('je',je.get_field_data(comps=0)[0])
            #
            # j = je.copy()

        if self.fi is not None:
            ji = self.sys_fi.compute_current(compress=0, compress1=compress1)
            ji.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)
            if j is not None:
                j.add(ji, compress_level=0, inplace=True)
            else:
                j = ji

            # vi_axes = self.coords_vi.axes
            # # ji = self.fi.elemental_multiply(self.velocities_i[0],compress=compress)
            # ji = self.fi.xmultiply(vi_axes, compress_level=compress1)
            #
            # # plt.figure()
            # # plt.imshow(self.fi.get_field_data(comps=0))
            #
            # # plt.figure()
            # # plt.imshow(ji.get_field_data(comps=0))
            #
            # # plt.figure()
            # # ji_ = np.sum(ji.get_comp_data(x_axes[0].axID),axis=1)*vi_axes[0].dx
            # # plt.plot(ji_)
            # # print('ji',ji_[0])
            #
            # ji = ji.integrate(axes=vi_axes, new_grid=self.grid_X,
            #                   new_ax_deriv_configs=self.fi.component.ax_deriv_configs)
            # ji.scalar_multiply(self.ion_params.Z * self.ion_params.e)
            # ji.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in vi_axes})
            # # print('ji',ji.get_field_data(comps=0)[0])
            #
            # # plt.figure()
            # # plt.plot(ji.get_comp_data(Z),'--',label='ion')
            # # plt.plot(je.get_comp_data(Z),'--',label='elec')
            # # plt.title('current')
            #
            # if j is not None:
            #     j.add(ji, compress_level=0, inplace=True)
            # else:
            #     j = ji
            #
            # # plt.plot(j.get_comp_data(Z), ':', label='total')
            # # plt.legend()
            #
            # # print('j deriv config', j[x_axes[0].axID].ax_deriv_configs[x_axes[0]].bc)

        # if self.background_fe0 is not None or self.background_fi0 is not None:
        #     if self.background_j0 is None:
        #         j0 = None
        #         if self.background_fe0 is not None:
        #             ve_axes = self.coords_ve.axes
        #             je0 = self.background_fe0.xmultiply(ve_axes, compress_level=compress1)
        #             je0 = je0.integrate(axes=ve_axes, new_grid=self.grid_X,
        #                                 new_ax_deriv_configs=self.background_fe0.component.ax_deriv_configs)
        #             je0.replace_compIDs(
        #                 {v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in ve_axes})
        #             je0.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
        #             j0 = je0
        #
        #         if self.background_fi0 is not None:
        #             vi_axes = self.coords_vi.axes
        #             ji0 = self.background_fi0.xmultiply(vi_axes, compress_level=compress1)
        #             ji0 = ji0.integrate(axes=vi_axes, new_grid=self.grid_X,
        #                                 new_ax_deriv_configs=self.fi.component.ax_deriv_configs)
        #             ji0.scalar_multiply(self.ion_params.Z * self.ion_params.e)
        #             ji0.replace_compIDs(
        #                 {v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in vi_axes})
        #             if j0 is None:
        #                 j0 = ji0
        #             else:
        #                 j0.add(ji0, inplace=True, compress_level=0)
        #
        #         j0.compress(inplace=True, compress_level=compress)
        #         self.background_j0 = j0
        #     j.add(self.background_j0, inplace=True)

        # if True:
        #     for C in self.coords_x.coords:
        #         plt.figure()
        #         j_data = j.get_comp_data(C)
        #         plt.plot(np.real(j_data), label=f'{C} re')
        #         plt.plot(np.imag(j_data), '--', label=f'{C} im')
        #         plt.legend()
        #         plt.show()

        if j is not None and compress:
            j.compress(inplace=True, compress_level=compress)

        return j


    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['Vlasov'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False) \
            -> 'Vlasov':

        state1 = self if inplace else self.copy()

        # print('vlasov deriv0 is None?', deriv0 is None)
        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=compress_level,
                                                    compress_level1=compress_level1, compress_level2=compress_level2,
                                                    verbose_plot=verbose_plot)

        # state1.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level=compress_level,
        #                                                    compress_level1=compress_level1, compress_level2=0)
        state1.sys_fe.euler(dt, deriv0=deriv0.sys_fe, inplace=True, compress_level=compress_level,
                            compress_level1=compress_level1, compress_level2=compress_level2,
                            verbose_plot=verbose_plot)

        # state1.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level=compress_level,
        #                                                    compress_level1=compress_level1, compress_level2=0)
        state1.sys_fi.euler(dt, deriv0=deriv0.sys_fi, inplace=True, compress_level=compress_level,
                            compress_level1=compress_level1, compress_level2=compress_level2,
                            verbose_plot=verbose_plot)

        return state1


    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'Vlasov':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        force_term = self.compute_force_term(is_ion=False, compress_level=compress_level,
                                             compress_level1=compress_level1, compress_level2=0,
                                             verbose_plot=verbose_plot)
        # print('force compress', compress_level, compress_level1, compress_level2)
        self.sys_fe.force_term = force_term
        dFdt_e = self.sys_fe.calculate_time_derivative(time=time, compress_level=compress_level,
                                                       compress_level1=compress_level1, compress_level2=compress_level2,
                                                       verbose_plot=verbose_plot) \
                    if self.sys_fe is not None else None

        force_term = self.compute_force_term(is_ion=True, compress_level=compress_level,
                                             compress_level1=compress_level1, compress_level2=0,
                                             verbose_plot=verbose_plot)
        self.sys_fi.force_term = force_term
        dFdt_i = self.sys_fi.calculate_time_derivative(time=time, compress_level=compress_level,
                                                       compress_level1=compress_level1, compress_level2=compress_level2,
                                                       verbose_plot=verbose_plot) \
                    if self.sys_fi is not None and self.evolve_ion else None

        dFdt = self.create_like(dFdt_e, dFdt_i, recalc=False)
        return dFdt

    # @profile
    # def _calculate_time_derivative_f(self, compress: int = 1, compress1: int = 0, compress2: int = 0, is_ion=False,
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
    #         v_grads_y = v_grads.get_comp_data(VY, ax_select={x_ax: x_ax.npts // 2, y_ax: y_ax.npts // 2,
    #                                                          vx_ax: vx_ax.npts // 2})
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
    #             v_grads_y = v_grads_0.get_comp_data(VY, ax_select={x_ax: x_ax.npts // 2, y_ax: y_ax.npts // 2,
    #                                                                vx_ax: vx_ax.npts // 2})
    #             plt.imshow(np.real(v_grads_y))
    #             plt.colorbar()
    #             plt.title('v grad 0 y')
    #             plt.show()
    #
    #     # print('vgrad0', v_grads_0.componentIDs)
    #     v_grads.add(v_grads_0, inplace=True, compress_level=compress1)
    #
    #     ### advection term
    #     convective_term = self._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
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
    # def _calculate_time_derivative_f_advection(self, compress1: int = 0, compress2: int = 0,
    #                                            is_ion=False, verbose_plot: bool = False) -> Optional['Field']:
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
    #         dist_data = dist.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
    #             ax_vx: ax_vx.npts // 2,
    #             ax_vy: ax_vy.npts // 2,
    #             ax_vz: ax_vz.npts // 2})
    #         xg_data = x_grads.get_comp_data(ax_y.coordinate,
    #                                         ax_select={ax_vx: ax_vx.npts // 2, ax_vy: ax_vy.npts // 2,
    #                                                    ax_vz: ax_vz.npts // 2})
    #
    #         plt.figure()
    #         plt.imshow(dist_data)
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title(f'dist, is ion {is_ion}')
    #         plt.colorbar()
    #
    #         plt.figure()
    #         plt.plot(dist_data[0, :])
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
    #         plt.plot(xg_data[0, :], label='gradx')
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
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist0.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
    #             ax_vx: ax_vx.npts // 2,
    #             ax_vy: ax_vy.npts // 2,
    #             ax_vz: ax_vz.npts // 2})
    #         xg_data = x_grads_0.get_comp_data(ax_y.coordinate,
    #                                           ax_select={ax_vx: ax_vx.npts // 2, ax_vy: ax_vy.npts // 2,
    #                                                      ax_vz: ax_vz.npts // 2})
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
    #         plt.plot(xg_data[0, :], label='gradx')
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
    #         plt.imshow(convective_term.get_comp_data(ax_select={ax_vx: ax_vx.npts // 2,
    #                                                             ax_vy: ax_vy.npts // 2,
    #                                                             ax_vz: ax_vz.npts // 2}))
    #         plt.colorbar()
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title('convective term')
    #         plt.show()
    #         # exit()
    #
    #     return convective_term
    #
    # # @profile
    # def _calculate_time_derivative_f_force(self, v_grads=None,
    #                                        compress1: int = 0, compress2: int = 0,
    #                                        is_ion=False, verbose_plot: bool = False) -> Optional['Field']:
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
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
    #             ax_vx: ax_vx.npts // 2,
    #             ax_vy: ax_vy.npts // 2,
    #             ax_vz: ax_vz.npts // 2})
    #         vg_data = v_grads.get_comp_data(ax_vx.coordinate,
    #                                         ax_select={ax_x: ax_x.npts // 2,
    #                                                    ax_vx: ax_vx.npts // 2,
    #                                                    # ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts // 2})
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
    #         plt.plot(np.real(vg_data[ax_y.npts // 2, :]), label='gradv')
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
    #         lorentz_data = lorentz_term.get_comp_data(ax_select={ax_vx: ax_vx.npts // 2,
    #                                                              ax_vy: ax_vy.npts // 2,
    #                                                              ax_vz: ax_vz.npts // 2})
    #         plt.imshow(np.real(lorentz_data))
    #         plt.title('lorentz')
    #         plt.colorbar()
    #         plt.show()
    #
    #         exit()
    #
    #     return lorentz_term

    def get_collision_term(self, v_axes: Sequence['Axis'] = None, v_grads: 'Field' = None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        coll_e = self.sys_fe.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                                compress1=compress1, compress2=compress2)
        coll_i = self.sys_fi.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                                compress1=compress1, compress2=compress2)

        ### TODO: implement ion/elec collisions

        return coll_e, coll_i



