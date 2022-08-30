from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid


class Boltzmann(PDE_system):
    """ df/dt + v grad(f) + F grad_v(f) = 0
    """
    def __init__(self, dist: Optional[ScalarField],
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 coords_v: Optional['CoordinateSystem'] = None,
                 f0_gradv: Optional['Field'] = None,
                 matl_params:  Optional['SpeciesConfiguration'] = None,
                 background_f: Optional[Union['ScalarField','Boltzmann']] = None,
                 evolve_background=False,
                 normalize=True, upwind=False, zipup=False,
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

        self.names = {'f': dist.name if dist is not None else 'f'}
        field_names = [self.names[x] for x in ['f']]

        super().__init__(dist,
                         field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         # background_pde=background_f, evolve_background=evolve_background, ## set later
                         # field_compress_config=field_compress_config,
                         )

        if matl_params is None:
            self.matl_params = ElcConfiguration()
        else:
            self.matl_params = matl_params

        self.coords_x = coords_x
        self.coords_v = coords_v
        self.x_axes = self.coords_x.axes
        self.v_axes_dict = {coord: self.coords_v.get_axis(coord.type) for coord in coords_x.coords}
        self.v_axes = [self.v_axes_dict[C] for C in coords_x.coords]


        if grid_X is None:
            self.grid_X = dist.grid.create_like(new_axes=self.coords_x.axes, new_gridID='GX')
        else:
            assert(grid_X.axes == self.x_axes)
            self.grid_X = grid_X

        if self.f is not None:
            v_dict = {}
            for v_ax in self.coords_v.axes:
                try:
                    x_coord = self.coords_x.get_coord(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mpo()
                    v_dict[x_coord] = self.f.grid.make_mpo_ndim({v_ax: v_mult})
                except (AttributeError, KeyError):
                    pass
            self.velocities = self.f.create_like_vector(v_dict)
            self.velocities.name = 'vel'

        ## save important values
        self.gradx_f = None
        self.gradv_f = None
        self.flow = None     # flow density (-> current)
        self.density = None  # density
        self.force_term = None

        self.upwind = upwind
        self.zipup = zipup

        ## background fields
        self.background_f = background_f
        self.evolve_background = evolve_background if background_f is not None else False

        ## save initial normalization for normalization
        self.init_norm = None
        if normalize:
            self.init_norm = self.total_f.norm() if self.f is not None else None

        ## define for linearized equation
        self.df0dv = f0_gradv

        ## force/em-term compression
        self.compress_F = True
        self.compress_F_opts = {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}

        ## collisions
        self.collision = CollisionConfiguration(None)
        self.coll_dist0 = None



    @property
    def f(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['f']]
        except KeyError:
            return None

    @f.setter
    def f(self, new_field: 'ScalarField'):
        self._fields[self.names['f']] = new_field
        self.gradx_f = None
        self.gradv_f = None
        self.flow = None
        self.density = None

    @property
    def total_f(self) -> Optional['ScalarField']:
        f = self.f
        if self.background_f is not None:
            f = self.background_f.f if f is None else f.add(self.background_f.f)
        return f

    @property
    def background_f(self) -> Optional['Boltzmann']:
        return self.background_pde

    @background_f.setter
    def background_f(self, f0: Optional[Union['ScalarField','Boltzmann']]):
        if isinstance(f0, ScalarField):
            f0 = Boltzmann(f0, grid_X=self.grid_X, coords_x=self.coords_x, coords_v=self.coords_v,
                           matl_params=self.matl_params, upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                           compress_levels=self._comp_levels)
        elif isinstance(f0, Boltzmann) or f0 is None:
            pass    ## PDE system
        else:
            raise TypeError
        self.background_pde = f0


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 1:
            new_fields = new_fields + (None,) * (1 - len(new_fields))

        dist, = new_fields[:1]

        f0_copy = self.background_f if not deep else \
            self.background_f.copy() if self.background_f is not None else None

        if recalc:
            new_system = Boltzmann(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x, coords_v=self.coords_v,
                                   f0_gradv=self.df0dv, background_f=f0_copy, matl_params=self.matl_params,
                                   normalize=self.do_normalization, evolve_background=self.evolve_background,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )
        else:
            new_system = Boltzmann(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x, coords_v=self.coords_v,
                                   f0_gradv=self.df0dv, background_f=f0_copy, matl_params=self.matl_params,
                                   normalize=False, evolve_background=self.evolve_background,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )

            new_system.do_normalization = self.do_normalization
            new_system.init_norm = self.init_norm

        if dist is None:      new_system.names['f'] = self.names['f']

        new_system.collision = self.collision
        new_system.coll_dist0 = self.coll_dist0

        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

        return new_system


    def copy(self) -> 'Boltzmann':
        new_system = super().copy()
        new_system.gradx_f = self.gradx_f
        new_system.gradv_f = self.gradv_f
        new_system.density = self.density
        new_system.current = self.flow
        new_system.force_term = self.force_term
        return new_system


    def normalize(self, target_val=None):
        """ normalized density field
            in place operation
        """
        if self.f is not None and self.f.component is not None:
            if target_val is None:
                target_val = self.init_norm if self.init_norm is not None else 1.0
            # norm_i = fi.norm()
            # scale_i = target_i/np.conj(norm_i)
            # if np.isnan(scale_i) and np.abs(norm_i) < 1.0e-12:
            #     scale_i = 0.0
            # self.fi.scalar_multiply( scale_i, inplace=True )
            self.f.component.normalize(target_val, inplace=True)
        return


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
              inplace: bool = False, compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False) -> 'Boltzmann':

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time,
                                                    compress_level=compress_level, compress_level1=compress_level1,
                                                    compress_level2=compress_level2, verbose_plot=verbose_plot)

        state1 = self.add(deriv0 * dt, compress_level=0, inplace=inplace)

        ## acts iff bc is absorbing
        x_axes = self.coords_x.axes
        if state1.f is not None:
            for x_ax in x_axes:
                v_ax = state1.v_axes_dict[x_ax.coordinate]
                state1.f.apply_absorbing_bc(x_ax, v_ax, inplace=True, compress_level=compress_level1)

        if compress_level:
            state1.compress(compress_level=compress_level, verbose=False)

        if state1.do_normalization:
            state1.normalize()

        return state1


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                   method=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'Boltzmann':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
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
            state0 = state0.get_force_advection(dt / 2, inplace=True, method=method,
                                                compress=comp2, compress1=comp4, compress2=comp5,
                                                verbose_plot=verbose_plot)
        ## evolve velocity advection dt
        state0 = state0.get_vel_advection(dt, inplace=True, method=method,
                                          compress=comp2, compress1=comp4, compress2=comp5, verbose_plot=verbose_plot)

        ## evolve force advection dt/2 if last time step or dt
        dt_ = dt/2 if is_last_time_step else dt
        state0 = state0.get_force_advection(dt_, inplace=True, method=method,
                                            compress=comp2, compress1=comp4, compress2=comp5,
                                            verbose_plot=verbose_plot)

        ## include collisions
        if self.collision.coll_type is not None:
            raise NotImplementedError
            deriv_coll = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
            dFdt_coll = self.create_like(deriv_coll, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None
        return state0


    def get_force_advection(self, dt, inplace=False, method=None,
                            compress=1, compress1=4, compress2=5, verbose_plot=False):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        if method is None:
            deriv_v0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
                                                                 verbose_plot=verbose_plot)
            dFdt_v = state0.create_like(deriv_v0, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)

        elif method[:2] == 'SL':
            assert all([v_ax.basis.type == BasisType.SPATIAL for v_ax in self.v_axes]), \
                'SL time evolution for spatial basis only'
            assert (len(self.x_axes) == 1), 'SL method only valid for 1DnV'

            f_gtn = state0.f.component
            if method == 'SL':
                print('SL F advection')
                f_mpo_func = f_gtn.grid._get_mpo_advection_SL
            elif method == 'SL-pfc':
                print('SL-pfc F advection')
                f_mpo_func = f_gtn.grid._get_mpo_advection_SL_pfc
            else:
                raise ValueError('not valid SL method')


            for x_ax in self.x_axes:
                C = self.coords_x.get_coord(x_ax.coordinate.type)

                if self.force_term is None:
                    force_term = self.compute_force_term(compress_level1=compress1, compress_level2=compress2)

                if force_term is None:
                    print('force is None')
                    continue

                force_data = force_term.get_comp_data(C)
                if force_data is None:  # no field
                    print('force is None')
                    continue


                if verbose_plot:
                    plt.figure()
                    plt.plot(force_data)
                    plt.title('elc lorentz force')
                    plt.show()

                ## ions
                v_ax = self.v_axes_dict[x_ax]
                deriv_params = f_gtn.ax_deriv_configs[v_ax]
                compress_opts = state0.f.compress_config.get_compress_opts(compress)
                f_mpo_gtn = f_mpo_func(v_ax, x_ax, dt, deriv_params, advec_coeffs=force_data)
                f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

        else:
            raise NotImplementedError

        return state0


    def get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                          verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        if method is None:
            deriv_x0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                     verbose_plot=verbose_plot)
            dFdt_x = state0.create_like(deriv_x0, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_x, inplace=True, compress_level=compress)

        elif method[:2] == 'SL':
            assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
                'SL time evolution for spatial basis only'

            for x_ax in self.x_axes:
                f_gtn = state0.f.component
                v_ax = self.v_axes_dict[x_ax.coordinate]
                deriv_params = f_gtn.ax_deriv_configs[x_ax]
                compress_opts = state0.f.compress_config.get_compress_opts(compress)
                f_mpo_gtn = f_gtn.grid.get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params, method=method)
                f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
        else:
            raise NotImplementedError

        return state0

    #############################

    # @profile
    def compute_force_term(self, compress_level=1, compress_level1=0, compress_level2=0, verbose_plot=False) -> 'Field':
        """ compute EM force:  q/m (E + v x B)
        """
        raise NotImplementedError


    # @profile
    def compute_density(self, compress=1, store=False):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        density = None
        v_axes = self.coords_v.axes if self.f is not None else None

        if self.f is not None:
            density = self.f.integrate(axes=v_axes, new_grid=self.grid_X,
                                          new_ax_deriv_configs=self.f.component.ax_deriv_configs)

        if self.background_f is not None:
            if self.background_f.density is not None:
                self.background_f.compute_density(compress=5, store=True)
            density.add(self.background_f.density, inplace=True)

        if density is not None and compress:
            density.compress(inplace=True, compress_level=compress)

        if store:
            self.density = density

        return density


    def compute_current(self, compress=1, compress1=5, store=False) -> 'Field':

        dist = self.f
        v_axes = self.coords_v.axes

        j = None
        if dist is not None:
            j = dist.xmultiply(v_axes, compress_level=compress1)
            j = j.integrate(axes=v_axes, new_grid=self.grid_X,
                            new_ax_deriv_configs=dist.component.ax_deriv_configs)
            j.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in v_axes})

        if self.background_f is not None:
            if self.background_f.flow is None:
                self.background_f.compute_current(compress=5, compress1=0, store=True)

            j.add(self.background_f.flow, inplace=True)

        if j is not None:
            if compress:
                j.compress(inplace=True, compress_level=compress)

        if store:
            self.flow = j

        return j


    # @profile
    def calculate_time_derivative(self, time=None,
                                  compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'Boltzmann':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt = self._calculate_time_derivative_f(compress=compress_level, compress1=compress_level1,
                                                 verbose_plot=verbose_plot)
        dFdt = self.create_like(dFdt, recalc=False)

        # ## grad_v of background distribution
        # if self.background_f is not None and self.evolve_background:
        #     dF0dt = self.background_f.calculate_time_derivative(time=time,
        #                                                         compress_level=compress_level,
        #                                                         compress_level1=compress_level1,
        #                                                         compress_level2=compress_level2,
        #                                                         verbose_plot=verbose_plot)
        #     dFdt.add(dF0dt, inplace=True, compress_level=compress_level)

        return dFdt


    # @profile
    def _calculate_time_derivative_f(self, compress: int = 1, compress1: int = 0, compress2: int = 0,
                                     verbose_plot=False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.f

        if dist is None:  return None

        # X, Y, Z = self.coords_x.coords
        # VX, VY, VZ = self.coords_v.coords
        v_axes = [self.v_axes_dict.get(coord, None) for coord in self.coords_x.coords]

        df0dv = self.df0dv
        # print('vgrads', compress1, dist.max_bond())
        v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) if df0dv is None else df0dv

        if verbose_plot:
            x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist.component.grid.axes
            VX, VY, VZ = self.coords_v.coords
            v_grads_y = v_grads.get_comp_data(VY, ax_select={x_ax: x_ax.npts // 2, y_ax: y_ax.npts // 2,
                                                             vx_ax: vx_ax.npts // 2})
            # VX, = self.coords_v.coords[:1]
            # v_grads_y = v_grads.get_comp_data(VX)
            if v_grads_y is not None:
                plt.figure()
                plt.imshow(np.real(v_grads_y))
                plt.colorbar()
                plt.title('v grad y')
                plt.show()

        ### advection term
        # print('adv compress,', compress1, compress2)
        convective_term = self._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                      verbose_plot=verbose_plot)
        ### force term
        # print('lor compress,', compress1, compress2)
        lorentz_term = self._calculate_time_derivative_f_force(v_grads=v_grads,
                                                               compress1=compress1, compress2=compress2,
                                                               verbose_plot=verbose_plot)
        # exit()

        dFdt_s = convective_term.add(lorentz_term, compress_level=0)
        dFdt_s.name = self.f.name

        ### get collisions
        coll_field = self.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                             compress1=compress1, compress2=compress2)
        dFdt_s.add(coll_field, inplace=True, compress_level=0)

        # print('dFdt compress', compress)
        if compress:
            dFdt_s.compress(compress_level=compress)

        return dFdt_s

    # @profile
    def _calculate_time_derivative_f_advection(self, compress1: int = 0, compress2: int = 0,
                                               verbose_plot: bool = False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.f
        # dist0 = self.background_f

        if dist is None:  return None

        x_axes = self.coords_x.axes
        v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]

        if not self.upwind:
            v_axes_dict = None
        else:
            v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}

        x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)

        if verbose_plot:
            ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
                ax_vx: ax_vx.npts // 2,
                ax_vy: ax_vy.npts // 2,
                ax_vz: ax_vz.npts // 2})
            xg_data = x_grads.get_comp_data(ax_y.coordinate,
                                            ax_select={ax_vx: ax_vx.npts // 2, ax_vy: ax_vy.npts // 2,
                                                       ax_vz: ax_vz.npts // 2})

            if dist_data is not None:
                print('dist data is None')
                plt.figure()
                plt.imshow(np.real(dist_data))
                plt.xlabel('y')
                plt.ylabel('x')
                plt.title(f'dist')
                plt.colorbar()

                plt.figure()
                plt.plot(np.real(dist_data[0, :]))
                plt.title(f'dist')

                plt.show()

            if xg_data is not None:
                print('xg data is None')
                plt.figure()
                plt.imshow(np.real(xg_data))
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('xgrad')
                plt.colorbar()
                # plt.show()

                plt.figure()
                plt.plot(xg_data[0, :], label='gradx')
                plt.xlabel('x')
                plt.legend()
                plt.show()


        ## convective term
        # print('x_grads', x_grads.max_bonds())
        # print(v_axes, self.coords_x.coords)
        convective_term = x_grads.xdot(v_axes, compIDs=self.coords_x.coords, compress_level=compress1,
                                       inner_compress_level=0)
        convective_term.scalar_multiply(-1, inplace=True)

        if verbose_plot:
            conv_data = convective_term.get_comp_data(ax_select={ax_vx: ax_vx.npts // 2,
                                                                 ax_vy: ax_vy.npts // 2,
                                                                 ax_vz: ax_vz.npts // 2})
            if conv_data is not None:
                plt.figure()
                plt.imshow(conv_data)
                plt.colorbar()
                plt.xlabel('y')
                plt.ylabel('x')
                plt.title('convective term')
                plt.show()
                # exit()

        return convective_term

    # @profile
    def _calculate_time_derivative_f_force(self, v_grads=None, compress1: int = 0, compress2: int = 0,
                                           verbose_plot: bool = False) -> Optional['Field']:

        dist = self.f
        # dist0 = self.background_f

        if dist is None:  return None

        x_axes = self.coords_x.axes
        v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]

        ## grad_v of main distribution
        if v_grads is None:
            df0dv = self.df0dv
            v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) if df0dv is None else df0dv

        ## intermediate plots
        if verbose_plot:
            ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
                                                      ax_vx: ax_vx.npts // 2,
                                                      ax_vy: ax_vy.npts // 2,
                                                      ax_vz: ax_vz.npts // 2})
            vg_data = v_grads.get_comp_data(ax_vx.coordinate,
                                            ax_select={ax_x: ax_x.npts // 2,
                                                       ax_vx: ax_vx.npts // 2,
                                                       # ax_vy:ax_vy.npts//2,
                                                       ax_vz: ax_vz.npts // 2})

            if vg_data is not None:
                plt.figure()
                plt.imshow(np.real(vg_data))
                plt.xlabel('vy')
                plt.ylabel('y')
                plt.colorbar()
                plt.title('grad v')
                plt.show()

                plt.figure()
                plt.plot(np.real(vg_data[ax_y.npts // 2, :]), label='gradv')
                plt.xlabel('vy')
                plt.legend()
                plt.show()
                # exit()

        ## (EM) force term
        if self.force_term is None:
            em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0,
                                              verbose_plot=verbose_plot)
        else:
            em_term = self.force_term

        if em_term is not None:
            em_term = em_term.scalar_multiply(-1)

            for compID, comp in em_term.components.items():
                em_term.components[compID] = comp.apply_elemental_multiply_op(compress=True)

            em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
            # print('em term', em_term.max_bonds())
            # print('v grads', v_grads.max_bonds())
            # print('self.coords', self.coords_v.coords, self.coords_x.coords)
            # lorentz_term = v_grads.dot(em_term, comps=[ax.coordinate for ax in v_axes],
            #                            other_comps=[ax.coordinate for ax in x_axes], zipup=self.zipup,
            #                            compress_level=compress1, inner_compress_level=compress2)    # 5, 0
            lorentz_term = v_grads.dot(em_term, comps=self.coords_v.coords, other_comps=self.coords_x.coords,
                                       zipup=self.zipup, compress_level=compress1, inner_compress_level=0)  # 5, 0
        else:
            lorentz_term = None

        if verbose_plot:
            plt.figure()
            # lorentz_data = lorentz_term.get_comp_data()
            lorentz_data = lorentz_term.get_comp_data(ax_select={ax_vx: ax_vx.npts // 2,
                                                                 ax_vy: ax_vy.npts // 2,
                                                                 ax_vz: ax_vz.npts // 2})
            if lorentz_data is not None:
                plt.imshow(np.real(lorentz_data))
                plt.title('lorentz')
                plt.colorbar()
                plt.show()

        return lorentz_term


    def get_collision_term(self, v_axes: Sequence['Axis'] = None, v_grads: 'Field' = None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        dist = self.f

        if v_axes is None:
            x_axes = self.coords_x.axes
            v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]

        if self.collision.coll_type is None:
            return None

        elif self.collision.coll_type == CollisionType.LB:
            if dist.is_sqrt:
                raise NotImplementedError('LB operator not implemented for sqrt field')

            if v_grads is None:
                df0dv = self.df0dv  # for linearized Vlasov
                v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                    if df0dv is None else df0dv

            ############  get maxwellian ######
            vth2 = self.matl_params.vth ** 2
            coll_rate = self.collision.coll_rate_e

            ## alternate method of computing collisions
            v0 = self.collision.v0
            try:
                v_offsets = {v_ax: -v0[v_ax.coordinate] for v_ax in v_axes}
            except (IndexError, TypeError):
                v_offsets = {v_ax: -v0 for v_ax in v_axes}
            coll = dist.xmultiply(v_axes, offsets=v_offsets, )

            coll_1 = v_grads.scalar_multiply(vth2)  ## components are self.v_axes
            coll.add(coll_1, inplace=True, compress_level=0)

            coords_v = self.coords_v
            coll_field = coll.divergence(coords_v, compress_level=compress1, inner_compress_level=compress2)

            coll_field.scalar_multiply(coll_rate, inplace=True)

            return coll_field



