from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
import helper_sl as helper_sl
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridTN


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
                 background_force_neg: Optional['Field'] = None,
                 # evolve_background=False,
                 normalize=True, upwind=False, zipup=False,
                 te_order=4, compress_levels=None,
                 compress_F=True, compress_F_opts=None,
                 collision: Optional['CollisionConfiguration'] = None,
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
        self.x_axes: Sequence['Axis'] = self.coords_x.axes

        self.v_axes_dict = {}
        self.v_axes = []
        for coord in coords_x.coords:
            v_ax = self.coords_v.get_axis(coord.type)
            if v_ax is not None:
                self.v_axes_dict[coord] = v_ax
                self.v_axes += [v_ax]

        if grid_X is None:
            # self.grid_X = dist.grid.create_like(new_axes=self.coords_x.axes, new_gridID='GX')
            self.grid_X = dist.grid.get_subgrid(select_axes=self.coords_x.axes, new_gridID='GX')
        else:
            assert(grid_X.axes == self.x_axes)
            self.grid_X = grid_X

        if self.f is not None:
            v_dict = {}
            for v_ax in self.coords_v.axes:
                try:
                    x_coord = self.coords_x.get_coord(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mps()
                    v_gtn = self.f.grid.make_mps_ndim({v_ax: v_mult})
                    v_gtn.constant_axes = list(self.coords_x.axes) + [v for v in self.coords_v.axes if v != v_ax]
                    v_dict[x_coord] = v_gtn
                    # v_mult = v_ax.get_xmultiply_mpo()
                    # v_dict[x_coord] = self.f.grid.make_mpo_ndim({v_ax: v_mult})
                except (AttributeError, KeyError):
                    pass
            self.velocities = self.f.create_like_vector(v_dict)
            self.velocities.name = 'vel'

        ## save important values
        self.gradx_f: Optional['Field'] = None
        self.gradv_f: Optional['Field'] = None
        self.flow: Optional['Field'] = None     # flow density (-> current)
        self.density: Optional['ScalarField'] = None  # density

        self.force_term: Optional['Field'] = None
        self.force_term_nobg: Optional['Field'] = None
        self.force_term_bg: Optional['Field'] = None
        ## background force is separated by number of cells traveled (for SL advection)
        self.force_term_bg_SL: dict['Coordinate', 'helper_sl.SL_data'] = {}

        self.upwind = upwind
        self.zipup = zipup

        ## background fields
        self.background_f = background_f
        self.background_force_neg = background_force_neg       # negative of background force
        self.evolve_background = False # evolve_background if background_f is not None else False

        ## save initial normalization for normalization
        self.init_norm = None
        if normalize:
            self.init_norm = self.total_f.norm() if self.f is not None else None

        ## define for linearized equation
        self.df0dv = f0_gradv

        ## force/em-term compression
        self.compress_F = compress_F   # True
        self.compress_F_opts = {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}\
                                    if compress_F_opts is None else compress_F_opts

        ## collisions
        self.collision = SpeciesCollisionConfiguration(None) if collision is None else collision
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


    def set_force_term(self, force, background_force=True, internal_force=True):
        if background_force and internal_force:
            self.force_term = force
        elif background_force:
            self.force_term_bg = force
        elif internal_force:
            self.force_term_nobg = force


    def get_force_term(self,  background_force=True, internal_force=True):
        if background_force and internal_force:
            return self.force_term
        elif background_force:
            # print('GET BACKGROUND FORCE', self.force_term_bg.components)
            return self.force_term_bg
        elif internal_force:
            return self.force_term_nobg


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
                                   f0_gradv=self.df0dv, matl_params=self.matl_params,
                                   background_f=f0_copy, background_force_neg=self.background_force_neg,
                                   normalize=self.do_normalization,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )
        else:
            new_system = Boltzmann(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x, coords_v=self.coords_v,
                                   f0_gradv=self.df0dv, matl_params=self.matl_params,
                                   background_f=f0_copy, background_force_neg=self.background_force_neg,
                                   normalize=False,
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

        new_system.force_term_bg_SL = self.force_term_bg_SL
        new_system.force_term_bg = self.force_term_bg

        return new_system


    def copy(self) -> 'Boltzmann':
        new_system = super().copy()
        new_system.gradx_f = self.gradx_f
        new_system.gradv_f = self.gradv_f
        new_system.density = self.density
        new_system.current = self.flow
        new_system.force_term = self.force_term
        new_system.force_term_nobg = self.force_term_nobg
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
            self.f.component.normalize(target_val, inplace=True, is_sqrt=self.f.is_sqrt)
        return


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
              inplace: bool = False, compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False, do_x_advection=True, do_v_advection=True) -> 'Boltzmann':

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=compress_level,
                                                    compress_level1=compress_level1, compress_level2=compress_level2,
                                                    background_force=do_x_advection, internal_force=do_v_advection,
                                                    verbose_plot=verbose_plot)

        state1 = self.add(deriv0 * dt, compress_level=0, inplace=inplace)
        if verbose_plot:
            ax_x, ax_y, ax_vx, ax_vy, ax_vz = state1.f.grid.axes
            dist_data = state1.f.get_comp_data(ax_select={ax_x: 0, ax_vx: ax_vx.npts//2, ax_vz: ax_vz.npts//2})
            plt.figure()
            plt.imshow(dist_data)
            plt.colorbar()
            plt.xlabel('vy'), plt.ylabel('y')
            plt.title('euler')
            plt.show()

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


    # @profile
    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                   method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'Boltzmann':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
        """
        # print('boltzmann split step')
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        assert all([v_ax.basis.type == BasisType.SPATIAL for v_ax in self.v_axes]), \
            'SL time evolution for spatial basis only'


        if is_first_time_step:
            ## evolve force advection dt/2
            state0 = state0.get_force_advection(dt / 2, inplace=True, method=method_f, compress=comp1, compress1=comp2,
                                                compress2=comp5, verbose_plot=verbose_plot)

        # print('compress', comp2, state0.f.max_bond())

        ## evolve velocity advection dt
        state0 = state0.get_vel_advection(dt, inplace=True, method=method_v,
                                          compress=comp1, compress1=comp2, compress2=comp5, verbose_plot=verbose_plot)

        # print('vel compress', comp2, state0.f.max_bond())

        ## evolve force advection dt/2 if last time step or dt
        dt_ = dt/2 if is_last_time_step else dt
        state0 = state0.get_force_advection(dt_, inplace=True, method=method_f, compress=comp1, compress1=comp2,
                                            compress2=comp5, verbose_plot=verbose_plot)

        # print('f compress', comp2, state0.f.max_bond())

        ## include collisions
        if self.collision.coll_type is not None:
            # raise NotImplementedError
            deriv_coll = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
            dFdt_coll = self.create_like(deriv_coll, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None
        return state0


    def _get_split_step_func(self, method):
        if method is None:
            sl_func, kwargs = None, {}
        elif method[:2] == 'SL':
            # assert (len(self.x_axes) == 1), 'SL method only valid for 1DnV'
            sl_func = self.semilagrangian
            kwargs = {'method': method}
        elif method[:3] == 'mac':
            sl_func = self.maccormack
            kwargs = {'coeff_is_constant': True}
        elif method[:3] == 'lax':
            sl_func = self.lax_wendroff
            kwargs = {'coeff_is_constant': True}
        else:
            raise NotImplementedError(f'{method} not yet implemented')
        return sl_func, kwargs


    def _get_two_step_func(self, method):
        if method == 'two' or method[:8] == 'two-leap':
            func = self.two_step_leapfrog
        elif method[:10] == 'two-adams3':
            func = self.two_step_adamsbashforth3
        elif method[:9] == 'two-adams':
            func = self.two_step_adamsbashforth
        elif method[:7] == 'two-mag':
            func = self.two_step_magazenkov
        else:
            raise NotImplementedError(f'{method} not yet implemented')
        return func


    def _get_rk_func(self, method):
        if method == 'rk2':
            func = self.rk2
        elif method[:10] == 'rk3':
            func = self.rk3
        elif method[:9] == 'rk4':
            func = self.rk4
        else:
            raise NotImplementedError(f'{method} not yet implemented')
        return func


    def _get_implicit_func(self, method):
        if method == 'imp-mid':
            func = self.implicit_midpoint
        elif method == 'imp-cn':
            func = self.crank_nicolson
        else:
            raise NotImplementedError(f'{method} not yet implemented')
        return func


    def semilagrangian(self, ax: 'Axis', dt: float, method=None, advec_coeffs: 'GridTN' = None,
                       inplace=False, verbose_plot=False, compress_level=1,
                       # f_perturbation: 'GridTN' = None,
                       # background_sl=False, save_background_sl=False,
                       **kwargs):

        state0 = self if inplace else self.copy()
        f_gtn = state0.f.component

        if method[:2] == 'SL':
            print('SL advection', ax)
            # f_mpo_func = f_gtn.grid._get_mpo_advection_SL
            f_mpo_func = state0._get_mpo_advection_SL
            try:
                sl_order = int(method[2])
            except IndexError:
                sl_order = 3
        elif method == 'SL-pfc':
            print('SL-pfc advection', ax)
            raise NotImplementedError
        else:
            raise ValueError('not valid SL method')

        if advec_coeffs is None:
            coeff_gr = None # state0.coords_v.get_axis(ax.coordinate.type)  # corresponding v_ax for coeffs
            # advec_coeffs = coeff_ax.xpts
        else:
            # assert(advec_coeffs.grid.ndim == 1), 'only defined for 1-D coeffs'
            # coeff_ax = advec_coeffs.grid.axes[0]
            # coeff_gr = advec_coeffs.grid
            print('advec coeffs constant axes', ax, advec_coeffs.constant_axes)
            advec_coeffs, coeff_gr = advec_coeffs.get_squeezed_data()
            # print('advec_coeffs', advec_coeffs.shape)
            # print('advec coeffs', advec_coeffs)
            # print('ax', ax, ax.dx, dt)
            if advec_coeffs is None:  # no field
                print('force is None')

        if verbose_plot:
            if advec_coeffs is not None:
                plt.figure()
                plt.plot(advec_coeffs)
                plt.title('advection coeffs')
                plt.show()
            else:
                print('SL advec coeffs is None')

        if advec_coeffs is not None: # or background_sl:
            deriv_params = f_gtn.ax_deriv_configs[ax]
            f_mpo_gtn = f_mpo_func(ax, coeff_gr, dt, deriv_params, advec_coeffs=advec_coeffs, sl_order=sl_order,
                                   # background_sl=background_sl, save_background_sl=save_background_sl
                                   )
            # print('SL f mpo gtn', f_mpo_gtn)
            # print('SL f gtn', f_gtn)
            # print('zipup ?', self.zipup)
            f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup) #, compress=compress_level, compress_opts=compress_opts)


        if compress_level:
            compress_opts = state0.f.compress_config.get_compress_opts(compress_level)
            f_gtn.compress(inplace=True, compress_opts=compress_opts)

        # f_field = state0.f.create_like_scalar(f_gtn)
        # state0.f = f_field
        state0.time = self.time + dt if self.time is not None else None
        return state0


    def _get_mpo_advection_SL(self, advec_ax: 'Axis', coeff_gr: 'Grid', dt: float,
                              deriv_params: 'DerivativeConfiguration',
                              advec_coeffs: np.ndarray = None,
                              sl_order=3,
                              sl_adv_obj: 'helper_sl.SL_data' = None,
                              # background_sl = False, save_background_sl = False
                              ) \
        -> 'GTN_TYPE':
        """ derivative in x for SemiLagrangian t
            S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative
                BCType or Tuple of BCTypes (left, right boundary)
            order:  order of finite difference method

            see Kormann 2011 equation after 4.2 for f_(n+1)
            note: returns f_(n+1), NOT f_(n+1) - f_(n)
        """
        # sl_order = 1
        print('SL order', sl_order, coeff_gr, coeff_gr.axes)
        # print('advec ax', advec_ax, coeff_gr.axes)

        # order = deriv_params.order
        #
        # if order != 1:
        #     raise NotImplementedError

        if sl_adv_obj is None:

            if advec_coeffs is None:  ###  return None
                print('advec coeffs is None', advec_ax)
                return None

            else:
                # plt.figure()
                # print(advec_coeffs.shape)
                # plt.imshow(advec_coeffs)
                # plt.show()

                # print('get cell data ADVEC COEFFS', dt, advec_ax, advec_ax.dx)
                if False: # background_sl:   # and advec_ax in self.coords_v.axes:
                    # print('is background_sl')
                    sl_adv_obj = self.force_term_bg_SL.get(advec_ax.coordinate, None)
                    # print('sl adv obj', sl_adv_obj is None, advec_ax.coordinate)
                    if sl_adv_obj is None: #  and advec_ax in self.coords_x.axes:
                        # print('get bg cell data advec coeffs', advec_ax, advec_coeffs.shape)
                        dx = advec_ax.dx
                        sl_adv_obj = helper_sl.get_cell_data(dt, dx, advec_coeffs, sl_order=sl_order,
                                                             coeff_grid=coeff_gr)
                        self.force_term_bg_SL[advec_ax.coordinate] = sl_adv_obj
                    # print('got bg sl force', advec_ax)
                else:
                    print('get cell data ADVEC COEFFS', dt, advec_ax, advec_ax.dx)
                    dx = advec_ax.dx
                    sl_adv_obj = helper_sl.get_cell_data(dt, dx, advec_coeffs, sl_order=sl_order, coeff_grid=coeff_gr)
                    # if save_background_sl:
                    #     self.force_term_bg_SL[advec_ax.coordinate] = sl_adv_obj
                    #     print('saved new sl adv obj', coeff_gr)

        # print('advection_SL sl adv obj', sl_adv_obj.vals)
        # print('getting weights')

        # sl_weights = helper_sl.get_multidimensional_weights(sl_adv_obj, sl_order=sl_order)
        # sl_weights = {k[0]: val for k,val in sl_weights.items()}
        sl_weights = sl_adv_obj.get_weights(sl_order)
        if sl_adv_obj.coeff_grid is not None:
            coeff_gr = sl_adv_obj.coeff_grid
            # print('using new coeff gr', coeff_gr, coeff_gr.axes)

        gtn_mpo = None
        for sh in sl_weights.keys():
            # print('SL mpo shift', sh)
            # print('advec ax', advec_ax)
            shift_mpo_gtn = advec_ax.get_shift_mpo(sh, boundary_conditions=deriv_params)
            shift_mpo_gtn = self.f.grid.make_mpo_ndim({advec_ax: shift_mpo_gtn})

            # print('sl weights', sl_weights[sh])
            # print('sl weights', sl_weights[sh].shape, coeff_gr)
            if (not isinstance(sl_weights[sh], np.ndarray)) or sl_weights[sh].size == 1:
                weights_mpo = coeff_gr.get_iden_mpo().scalar_multiply(sl_weights[sh], inplace=True)
            else:
                weights_mps = coeff_gr.map_state_to_mps(sl_weights[sh])
                weights_mpo = weights_mps.apply_elemental_multiply_op()

            # print('shift mpo', shift_mpo_gtn)
            # print('weights mpo', weights_mpo)

            gtn_mpo_2 = shift_mpo_gtn.apply(weights_mpo, compress=False)
            # print('gtn mpo 2', gtn_mpo_2)
            if gtn_mpo is None:
                gtn_mpo = gtn_mpo_2
            else:
                gtn_mpo.add(gtn_mpo_2, compress=True, inplace=True)

        gtn_mpo.compress(inplace=True)
        # print('SL mpo', gtn_mpo)

        return gtn_mpo


    def _get_mpo_advection_SL_multidim(self, sl_adv_objs: dict['Axis', 'helper_sl.SL_data'],
                                       ax_deriv_configs: dict['Axis','DerivativeConfiguration'],
                                       sl_order=3,
                                       ) \
        -> 'GTN_TYPE':
        """ derivative in x for SemiLagrangian t
            S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative
                BCType or Tuple of BCTypes (left, right boundary)
            order:  order of finite difference method

            see Kormann 2011 equation after 4.2 for f_(n+1)
            note: returns f_(n+1), NOT f_(n+1) - f_(n)
        """
        # sl_order = 1
        print('SL order', sl_order)

        coeff_gr = sl_adv_objs[next(iter(sl_adv_objs))].coeff_grid
        advec_axes, sl_adv_objs_list = [], []
        for ax in sl_adv_objs.keys():
            sl_adv_objs_list += [sl_adv_objs[ax]]
            advec_axes += [ax]
        multidim_weights = helper_sl.get_multidimensional_weights(*sl_adv_objs_list)
        # print('multidim weights', multidim_weights)

        gtn_mpo = None
        for shifts, sl_weights in multidim_weights.items():
            # print('shifts', shifts)

            shift_mpo_gtn_dict = {}
            for i in range(len(shifts)):
                sh = shifts[i]
                advec_ax = advec_axes[i]
                deriv_params = ax_deriv_configs[advec_ax]
                # print('SL mpo shift', sh)
                # print('advec ax', advec_ax)
                shift_mpo_gtn_dict[advec_ax] = advec_ax.get_shift_mpo(sh, boundary_conditions=deriv_params)
            shift_mpo_gtn = self.f.grid.make_mpo_ndim(shift_mpo_gtn_dict)

            # print('sl weights', sl_weights[sh])
            # print('sl weights', sl_weights[sh].shape, coeff_gr)
            if (not isinstance(sl_weights, np.ndarray)) or sl_weights.size == 1:
                weights_mpo = coeff_gr.get_iden_mpo().scalar_multiply(sl_weights[sh], inplace=True)
            else:
                weights_mps = coeff_gr.map_state_to_mps(sl_weights)
                weights_mpo = weights_mps.apply_elemental_multiply_op()

            # print('shift mpo', shift_mpo_gtn)
            # print('weights mpo', weights_mpo)

            gtn_mpo_2 = shift_mpo_gtn.apply(weights_mpo, compress=False)
            # print('gtn mpo 2', gtn_mpo_2)
            if gtn_mpo is None:
                gtn_mpo = gtn_mpo_2
            else:
                gtn_mpo.add(gtn_mpo_2, compress=True, inplace=True)

        gtn_mpo.compress(inplace=True)

        return gtn_mpo


    # @profile
    def maccormack(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeff_is_constant=True,
                   inplace=False, get_df=False, compress_level=1, **kwargs):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        print('mac', ax, compress_level)

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)


        ## gradient of state along x
        deriv_config = self.f.component.ax_deriv_configs[ax].copy()
        deriv_config.update(order=0)    # 1st order accurate. kind of jank implementation

        state = self if inplace else self.copy()

        # if advec_coeffs is None:
        #     # v_ax = self.coords_v.get_axis(ax.coordinate.type)
        #     # advec_coeffs = v_ax.xpts
        #     print('self velocities', self.velocities.componentIDs, ax)
        #     advec_coeffs = self.velocities[ax.coordinate]
        #     # print('advec coeffs', advec_coeffs)

        if advec_coeffs is None:    ## no advection occurs
            print('NO ADVECTION', ax)
            return state

        # print('advec coeff', advec_coeffs)

        # advec_data = advec_coeffs.get_data()
        # if advec_data is not None:
        #     plt.figure()
        #     plt.imshow(advec_data[0,:,:])
        #     plt.colorbar()
        #     plt.title('advec coeff mac')
        #     plt.show()
        # else:
        #     print('ADVEC DATA NONE')

        f_active = state.f.copy()
        for it in range(2):
            if coeff_is_constant:
                if it == 0:
                    deriv_config.update(fd_type=FDType.BACKWARD)
                elif it == 1:
                    deriv_config.update(fd_type=FDType.FORWARD)
                else:
                    raise ValueError

                grad_f = f_active.gradient(deriv_axes=[ax], deriv_config=deriv_config, compress_level=0)
                grad_f.scalar_multiply(-dt, inplace=True)  # dx already included in gradient operation
                # print('advec coeffs', advec_coeffs)
                # print('advec coeffs data type', advec_coeffs.data_type)
                # print('grad f', grad_f.max_bonds())
                # print('advec coeffs', advec_coeffs.max_bond(), advec_coeffs.grid)
                # print('zipup', self.zipup)
                grad_f_term = grad_f.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                # print('advec coeffs', advec_coeffs.max_bond())

                # plt.figure()
                # plt.imshow(grad_f.get_comp_data(compID=ax.coordinate)[0,:,:])
                # plt.title('grad f * - dt'), plt.colorbar()
                # plt.show()
                #
                # plt.figure()
                # if grad_f_term[ax.coordinate].data is not None:
                #     plt.imshow(grad_f_term.get_comp_data(compID=ax.coordinate)[0,:,:])
                #     plt.title(f'grad f * advec_coeffs {it}'), plt.colorbar()
                # plt.show()
            else:
                term1 = f_active.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                shift = -1 if it == 0 else 1
                shift_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config.left_bc)  ## intended: j -> j-1, and then j -> j+1
                print('WARNING: lax wendroff need to check shift direction')
                # term2 = state.f.component.apply(shift_mpo, inplace=False,)
                term2 = f_active.component.apply(shift_mpo, inplace=False, )
                term2 = term2.elemental_multiply(advec_coeffs, compress_level=comp4, inplace=True)
                term2 = term2.scalar_multiply(-1, inplace=True)
                grad_f_term = term1.add(term2)
                grad_f_term.scalar_multiply(-dt, inplace=True)

            grad_f_term = grad_f_term.create_like_scalar(grad_f_term[ax.coordinate])
            f_active = f_active.add( grad_f_term, inplace=True, compress_level=comp3)

        state2 = self.create_like( f_active, recalc=False )
        if get_df:
            print('get df')
            neg_state = state.scalar_multiply(-1, inplace=True)
            new_state = neg_state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1./2, inplace=True, )

            # plt.figure()
            # plt.imshow(new_state.f.get_comp_data()[0,:,:])
            # plt.colorbar()
            # plt.show()

            # state.add(new_state, inplace=True)
        else:
            new_state = state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1./2, inplace=True,)
        return new_state


    def lax_wendroff(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None,
                     inplace=False, get_df=False, compress_level=1, **kwargs):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        state = self if inplace else self.copy()

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        dist = state.f.copy()
        deriv_config = state.f.component.ax_deriv_configs[ax].copy()
        shift_p = dist.grid.get_shift_mpo({ax: 1}, ax_boundary_conditions=dist.component.ax_deriv_configs)
        shift_m = dist.grid.get_shift_mpo({ax:-1}, ax_boundary_conditions=dist.component.ax_deriv_configs)

        if advec_coeffs is None:
            advec_coeffs = self.velocities[ax.coordinate]

        ## forward difference
        deriv_config.update(order=0, fd_type=FDType.FORWARD)
        f1 = dist.gradient(deriv_axes=[ax], deriv_config=deriv_config)
        f1 = dist.create_like_scalar(f1[ax.coordinate])
        f1 = f1.elemental_multiply(advec_coeffs, inplace=True, zipup=self.zipup, compress_level=comp4)
        f1.scalar_multiply(-dt/2, inplace=True)

        # plt.figure()
        # if f1.component.data is not None:
        #     plt.imshow(f1.get_comp_data())
        #     plt.title('f1 * advec_coeffs'), plt.colorbar()
        # plt.show()

        fp = dist.component.apply(shift_p, inplace=False)
        fp = dist.create_like_scalar(fp)
        f1_avg = dist.add(fp, inplace=False, compress_level=0)
        f1_avg.scalar_multiply(1./2, inplace=True)

        # if f1_avg.component.data is not None:
        #     plt.imshow(f1_avg.get_comp_data())
        #     plt.colorbar()
        # plt.title('f1')
        # plt.show()

        # print('F1 + Favg')
        f1 = f1.add(f1_avg, inplace=True, compress_level=comp3)

        # plt.figure()
        # if f1.component.data is not None:
        #     plt.imshow(f1.get_comp_data())
        #     plt.colorbar()
        # plt.title('f1')
        # plt.show()

        ## backward difference
        deriv_config.update(order=0, fd_type=FDType.BACKWARD)
        f2 = dist.gradient(deriv_axes=[ax], deriv_config=deriv_config)
        f2 = dist.create_like_scalar(f2[ax.coordinate])
        f2 = f2.elemental_multiply(advec_coeffs, inplace=True, zipup=self.zipup, compress_level=comp4)
        f2.scalar_multiply(-dt / 2, inplace=True)

        # plt.figure()
        # if f2.component.data is not None:
        #     plt.imshow(f2.get_comp_data())
        #     plt.title('f2 * advec_coeffs'), plt.colorbar()
        # plt.show()

        # print('advec coeffs', advec_coeffs.grid)

        # plt.figure()
        # if advec_coeffs.data is not None:
        #     data = advec_coeffs.get_data()
        #     if data.ndim == 2:
        #         plt.imshow(data)
        #         plt.colorbar()
        #     else:
        #         plt.plot(data)
        # plt.show()

        fm = dist.component.apply(shift_m, inplace=False)
        fm = dist.create_like_scalar(fm)
        f2_avg = dist.add(fm, inplace=False, compress_level=0)
        f2_avg.scalar_multiply(1. / 2, inplace=True)

        f2 = f2.add(f2_avg, inplace=True, compress_level=comp3)

        # plt.figure()
        # if f2.component.data is not None:
        #     plt.imshow(f2.get_comp_data())
        #     plt.colorbar()
        # plt.title('f2')
        # plt.show()

        ## calculate next time step
        f2.scalar_multiply(-1, inplace=True)
        deriv = f1.add(f2, inplace=True, compress_level=comp4)
        deriv.scalar_multiply(-dt/ax.dx, inplace=True)

        # plt.figure()
        # if deriv.component.data is not None:
        #     plt.imshow(deriv.get_comp_data())
        #     plt.colorbar()
        # plt.title('deriv')
        # plt.show()

        deriv.elemental_multiply(advec_coeffs, inplace=True, zipup=self.zipup, compress_level=comp4)

        # plt.figure()
        # if deriv.component.data is not None:
        #     plt.imshow(deriv.get_comp_data())
        #     plt.colorbar()
        # plt.title('deriv * c')
        # plt.show()

        deriv_state = state.create_like(deriv, recalc=False)
        if get_df:
            return deriv_state
        else:
            new_state = state.add(deriv_state, inplace=True, compress_level=comp1)
            return new_state


    def warming_beam(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeffs_is_constant=True,
                     inplace=False, compress_level=1, compress_level_1=5):
        """
        advec_coeffs (c)
        """
        mu = advec_coeffs.scalar_multiply( dt / ax.dx )      # c * dt/dx

        dist = self.f
        deriv_config = dist.component.ax_deriv_configs[ax].copy()

        raise NotImplementedError('Warming-Beam requires elemental analysis of advection coeff (ie. >0, <0)')


    @staticmethod
    def _get_splitstep_axes_list(axes: Sequence['Axis'], order=2):
        """ determine ordering of axes in split step method
        """
        print('split order', order)
        if len(axes) > 1 and order == 2:
            ax_list = [ax for ax in axes if ax is not None] + \
                      [ax for ax in axes[-2::-1] if ax is not None]
            scales = [1. / 2, ] * len(ax_list)
            scales[len(axes) - 1] = 1.
        elif len(axes) > 1 and order == 3:
            ax_list = [ax for ax in axes if ax is not None] + \
                      [ax for ax in axes[::-1] if ax is not None]
            scales = [1. / 2, ] * len(ax_list)
        else:
            ax_list = axes
            scales = [1.]
        return ax_list, scales


    def get_force_advection(self, dt, inplace=False, method=None, background_force=True, internal_force=True,
                            split_order=2,
                            compress=1, compress1=2, compress2=5, verbose_plot=False, **method_kwargs):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        # print('boltzmann force advection', compress, compress1, compress2)
        new_state = self if inplace else self.copy()

        if method is None or method == '':
            deriv_v0 = new_state._calculate_time_derivative_f_force(background_force=background_force,
                                                                    internal_force=internal_force,
                                                                    compress1=compress1, compress2=compress2,
                                                                    verbose_plot=verbose_plot)
            dFdt_v = new_state.create_like(deriv_v0, recalc=False)
            new_state = new_state.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)

        elif method[:2] == 'rk':
            func = new_state._get_rk_func(method)
            state1 = func(dt, compress_level=compress, do_x_advection=False,
                          background_force=background_force, internal_force=internal_force)
            new_state.f = state1.f
            new_state.time = self.time + dt if self.time is not None else None

            # deriv_v0 = new_state._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
            #                                                         verbose_plot=verbose_plot)
            # dFdt_v0 = new_state.create_like(deriv_v0, recalc=False)
            # state1 = new_state.euler(dt, deriv0=dFdt_v0, inplace=False, compress_level=compress)
            #
            # deriv_v1 = state1._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
            #                                                      verbose_plot=verbose_plot)
            # dFdt_v1 = state1.create_like(deriv_v1, recalc=False)
            #
            # dFdt_sum = dFdt_v0.add(dFdt_v1, inplace=True, compress_level=0)
            # new_state = new_state.euler(0.5 * dt, deriv0=dFdt_sum, inplace=True, compress_level=compress)
            # new_state.time = self.time + dt if self.time is not None else None

        elif method[:3] == 'two':
            new_state = new_state.two_step(dt, inplace=True, method=method, compress_level=compress,
                                           do_x_advection=False)
        elif ',' in method:  ## SL{x},{split_step str}

            ### SL + other split step
            method0, method1 = method.split(',')

            if True: # method1[:2] == 'rk':
                pert_state = new_state.get_force_advection(dt / 2, inplace=True, method=method1, background_force=False,
                                                           internal_force=True, compress=compress,
                                                           compress1=compress1, compress2=compress2)

                new_state = pert_state.get_force_advection(dt, inplace=True, method=method0, background_force=True,
                                                           internal_force=False, split_order=split_order,
                                                           compress=compress, compress1=compress1, compress2=compress2)

                new_state = new_state.get_force_advection(dt / 2, inplace=True, method=method1, background_force=False,
                                                          internal_force=True, compress=compress, compress1=compress1,
                                                          compress2=compress2)

            else:
                ### old version, slightly faster but right now not compatible with rk methods (see above)

                ### SL + other split step
                method0, method1 = method.split(',')

                v_axes = new_state.v_axes
                ax_list, scales = Boltzmann._get_splitstep_axes_list(v_axes, order=split_order)

                force_term_pt: 'Field' = new_state.get_force_term(background_force=False, internal_force=True)
                force_term_bg: 'Field' = new_state.get_force_term(background_force=True, internal_force=False)

                for v_ax, scale in zip(ax_list, scales):
                    C = new_state.coords_x.get_coord(v_ax.coordinate.type)

                    try:
                        force_term_pt_C = force_term_pt[C]
                        print('got force term nobg', C)
                    except (KeyError, TypeError):
                        print('force nobg is None', C)
                        force_term_pt_C = None

                    try:
                        force_term_bg_C = force_term_bg[C]
                        print('got force term bg', C)
                        # print('bg force data type', force_term_bg_C.data_type)
                    except (KeyError, TypeError):
                        print('force bg is None', C)
                        force_term_bg_C = None

                    func1, kwargs1 = new_state._get_split_step_func(method1)
                    pert_state = func1(v_ax, dt * scale, advec_coeffs=force_term_pt_C, inplace=True,
                                      compress_level=compress, get_df=False, **kwargs1)
                    ## use pert_state as initial state for SL advection of background forces

                    func0, kwargs0 = pert_state._get_split_step_func(method0)
                    new_state = func0(v_ax, dt * scale, advec_coeffs=force_term_bg_C, inplace=True,
                                      compress_level=compress, **kwargs0)

                    ## using background_sl doesn't work because it doesn't take into account the scaling of dt

                    ## for some reason, inplace operation leads to wrong results... (at least for func1 calc above)
                    ## so set inplace = False

        elif method[:3] == 'SLd':   ## d-dimensional semilagrangian projection

            sl_order = int(method[-1])

            # print('getting force term', background_force, internal_force)
            force_term: 'Field' = new_state.get_force_term(background_force=background_force,
                                                           internal_force=internal_force)

            sl_adv_obj_dict = {}
            for C in force_term.componentIDs:
                force_C = force_term[C]
                # print('C', C)
                if force_C is not None:
                    advec_coeffs, coeff_gr = force_C.get_squeezed_data()

                    advec_ax = self.coords_v.get_axis(C.type)
                    if advec_ax is None:
                        continue
                    # print('get cell data ADVEC COEFFS', dt, advec_ax, advec_ax.dx)
                    dx = advec_ax.dx
                    sl_adv_obj = helper_sl.get_cell_data(dt, dx, advec_coeffs, sl_order=sl_order, coeff_grid=coeff_gr)
                    sl_adv_obj_dict[advec_ax] = sl_adv_obj

            if new_state.background_f is not None:
                raise NotImplementedError('split step with background f0 not implemented')
                ## need to include f0 * dF term


            f_gtn = new_state.f.component
            gtn_mpo = new_state._get_mpo_advection_SL_multidim(sl_adv_obj_dict, sl_order=sl_order,
                                                               ax_deriv_configs=f_gtn.ax_deriv_configs)

            compress_opts = new_state.f.compress_config.get_compress_opts(compress)
            new_state.f.component.apply(gtn_mpo, inplace=True, zipup=self.zipup,
                                        compress=compress, compress_opts=compress_opts)

        else:
            func, kwargs = new_state._get_split_step_func(method)

            v_axes = new_state.v_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(v_axes, order=split_order)

            # print('getting force term', background_force, internal_force)
            force_term: 'Field' = new_state.get_force_term(background_force=background_force,
                                                           internal_force=internal_force)

            for v_ax, scale in zip(ax_list, scales):
                C = new_state.coords_x.get_coord(v_ax.coordinate.type)
                # print('advec v_ax', v_ax, C)

                if verbose_plot:
                    force_data = force_term.get_comp_data(C, ax_select={v_ax: 0})
                    # print('force data', C, v_ax, force_data is None)
                    if force_data is not None:
                        # print('force data', force_data)
                        plt.figure()
                        # plt.plot(force_data)
                        plt.imshow(force_data)
                        plt.colorbar()
                        plt.title('elc lorentz force')
                        plt.show()
                    else:
                        print('force term is None')

                if new_state.background_f is not None:
                    raise NotImplementedError('split step with background f0 not implemented')
                    ## need to include f0 * dF term

                # print('force term', v_ax, C, force_term[C])

                try:
                    force_term_C = force_term[C]
                    print('got force term', C)
                except (KeyError, TypeError):
                    print('force is None', C)
                    force_term_C = None
                    # continue

                new_state = func(v_ax, dt * scale, advec_coeffs=force_term_C, inplace=True,
                                 compress_level=compress, **kwargs, **method_kwargs)

        return new_state


    def get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
                          verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        # print('boltzmann vel advection', compress, compress1, compress2)
        state0 = self if inplace else self.copy()

        if method is None:
            deriv_x0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                     verbose_plot=verbose_plot)
            dFdt_x = state0.create_like(deriv_x0, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_x, inplace=True, compress_level=compress)

        elif method[:2] == 'rk':
            func = state0._get_rk_func(method)
            state1 = func(dt, compress_level=compress, do_v_advection=False)
            state0.f = state1.f
            state0.time = self.time + dt if self.time is not None else None

            # deriv_v0 = state0._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
            #                                                         verbose_plot=verbose_plot)
            # dFdt_v0 = state0.create_like(deriv_v0, recalc=False)
            # state1 = state0.euler(dt, deriv0=dFdt_v0, inplace=False, compress_level=compress)
            #
            # deriv_v1 = state1._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
            #                                                          verbose_plot=verbose_plot)
            # dFdt_v1 = state1.create_like(deriv_v1, recalc=False)
            #
            # dFdt_sum = dFdt_v0.add(dFdt_v1, inplace=True, compress_level=0)
            # state0 = state0.euler(0.5 * dt, deriv0=dFdt_sum, inplace=True, compress_level=compress)
            # state0.time = self.time + dt if self.time is not None else None

        elif method[:3] == 'imp':
            func = state0._get_implicit_func(method)
            state1 = func(dt, compress_level=compress, do_v_advection=False)
            state0.f = state1.f
            state0.time = self.time + dt if self.time is not None else None

        elif method[:3] == 'two':
            state0 = state0.two_step(dt, inplace=True, method=method, compress_level=compress, do_v_advection=False)

        else:
            func, kwargs = state0._get_split_step_func(method)

            # assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
            #     'SL time evolution for spatial basis only'

            x_axes = state0.x_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(x_axes)

            # print('vel axes', ax_list, scales)

            for x_ax, scale in zip(ax_list, scales):
                assert(x_ax.basis.type==BasisType.SPATIAL), 'axes must be in spatial basis'

                # print('split step', x_ax, self.velocities[x_ax.coordinate])
                state0 = func(x_ax, dt * scale, advec_coeffs = self.velocities[x_ax.coordinate],
                              inplace=True, compress_level=compress, **kwargs)

                # f_gtn = state0.f.component
                # v_ax = self.v_axes_dict[x_ax.coordinate]
                # deriv_params = f_gtn.ax_deriv_configs[x_ax]
                # compress_opts = state0.f.compress_config.get_compress_opts(compress)
                # f_mpo_gtn = f_gtn.grid.get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params, method=method)
                # f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

        return state0

    #############################

    # @profile
    def compute_force_term(self, compress_level=1, compress_level1=0, compress_level2=0, verbose_plot=False,
                           background_force=True, internal_force=True, **kwargs) -> 'Field':
        """ compute EM force:  q/m (E + v x B)
        """
        raise NotImplementedError


    # @profile
    def compute_density(self, compress=1, store=False):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        density: Optional['ScalarField'] = None
        v_axes = self.coords_v.axes if self.f is not None else None

        if self.f is not None:
            new_grid = self.grid_X if not isinstance(self.f.grid, GridTN1DComb) else None
            density = self.f.integrate(axes=v_axes, new_grid=new_grid, # self.grid_X,
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


    def compute_flow(self, compress=1, compress1=5, store=False, verbose_plot=False) -> 'Field':

        dist = self.f
        v_axes = self.coords_v.axes

        j: Optional['Field'] = None
        if dist is not None:
            new_grid = self.grid_X if not isinstance(self.f.grid, GridTN1DComb) else None
            v_mpos = { self.coords_x.coords[i]: dist.grid.get_xmultiply_mpo([v_axes[i]]) for i in range(len(v_axes)) }
            j = dist.meas_expec(v_mpos, integ_axes=v_axes, new_grid=new_grid, # self.grid_X,
                                new_ax_deriv_configs=dist.component.ax_deriv_configs)

        if self.background_f is not None:
            if self.background_f.flow is None:
                self.background_f.compute_flow(compress=5, compress1=0, store=True)

            j.add(self.background_f.flow, inplace=True)

        if verbose_plot:
            for C in self.coords_x.coords:
                plt.figure()
                j_data = j.get_comp_data(C)
                plt.plot(np.real(j_data), label=f'{C} re')
                plt.plot(np.imag(j_data), '--', label=f'{C} im')
                plt.legend()
                plt.title('flow (boltzmann)')


            ax_x, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={ax_vy: ax_vy.npts//2, ax_vz: ax_vz.npts//2 })
            plt.figure()
            plt.imshow(np.real(dist_data))
            plt.title('dist (boltzmann) re')
            plt.xlabel('vx'), plt.ylabel('x')
            plt.colorbar()
            plt.figure()
            plt.imshow(np.imag(dist_data))
            plt.title('dist (boltzmann) im')
            plt.xlabel('vx'), plt.ylabel('x')
            plt.colorbar()
            plt.show()


        if j is not None:
            if compress:
                j.compress(inplace=True, compress_level=compress)

        if store:
            self.flow = j

        return j


    def compute_moment(self, axes: Sequence['Axis'], powers: Sequence['int']=None, integ_axes=None, compress=1):
        """ compute the moment e.g. integ dist * x^p dV
            where x is the grid values along the specified axes, raise to the p^th power
            default is to integrate over velocity space
        """

        if powers is None:
            powers = [0] * len(axes)

        grid_X = None
        if integ_axes is None:
            integ_axes = self.v_axes
            grid_X = self.grid_X

        ## compute integrand
        # moment_dist = None
        # for ax, power in zip(axes, powers):
        #     ax_dist = self.f.xmultiply([ax], x_powers=[power], compress_level=0)
        #
        #     if moment_dist is None:
        #         moment_dist = ax_dist
        #     else:
        #         moment_dist.add(ax_dist, inplace=True, compress_level=0)
        #
        # ## take integral
        # moment = moment_dist.integrate(axes=integ_axes, new_grid=grid_X)

        moment_mpo_dict = {}
        for ax, power in zip(axes, powers):
            if power > 0:
                moment_mpo_dict[ax] = ax.get_xmultiply_mpo(x_power=power)

        moment_mpo = self.f.grid.make_mpo_ndim(moment_mpo_dict)

        ## take integral
        moment = self.f.meas_expec(moment_mpo, integ_axes=integ_axes, new_grid=grid_X)


        if compress:
            moment.compress(inplace=True, compress_level=compress)

        return moment


    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  verbose_plot=False, **kwargs) -> 'Boltzmann':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt = self._calculate_time_derivative_f(do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                                                 background_force=background_force, internal_force=internal_force,
                                                 compress=compress_level, compress1=compress_level1,
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


    def get_time_derivative_mpo(self, time=None,
                                compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                                do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                verbose_plot: bool =False, **kwargs) -> 'PDE_system':
        raise NotImplementedError

    # @profile
    def _calculate_time_derivative_f(self, do_x_advection=True, do_v_advection=True,
                                     background_force=True, internal_force=True,
                                     compress: int = 1, compress1: int = 0, compress2: int = 0,
                                     verbose_plot=False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.f

        # if dist is None:  return None

        # X, Y, Z = self.coords_x.coords
        # VX, VY, VZ = self.coords_v.coords
        v_axes = [self.v_axes_dict.get(coord, None) for coord in self.coords_x.coords]

        if self.df0dv is None:
            v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) if do_v_advection else None
        else:
            v_grads = self.df0dv

        if verbose_plot:
            # x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist.component.grid.axes
            x_ax, vx_ax, vy_ax, vz_ax = dist.component.grid.axes
            VX, VY, VZ = self.coords_v.coords

            dist_data = dist.get_comp_data(ax_select={# x_ax: x_ax.npts // 2,
                                                      # y_ax: y_ax.npts // 2,
                                                      vx_ax: vx_ax.npts // 2, vy_ax: 0,})
            if dist_data is not None:
                plt.figure()
                plt.imshow(np.real(dist_data))
                plt.colorbar()
                plt.xlabel('y')
                plt.ylabel('vz')
                plt.title('dist force')
                plt.show()

            v_grads_y = v_grads.get_comp_data(VY, ax_select={# x_ax: x_ax.npts // 2,
                                                             # y_ax: y_ax.npts // 2,
                                                             # vy_ax: 0,
                                                             vx_ax: vx_ax.npts // 2,
                                                             vz_ax: vx_ax.npts // 2,
                                                             })
            # VX, = self.coords_v.coords[:1]
            # v_grads_y = v_grads.get_comp_data(VX)
            if v_grads_y is not None:
                plt.figure()
                plt.imshow(np.real(v_grads_y))
                plt.colorbar()
                plt.xlabel('vy')
                plt.ylabel('y')
                plt.title('v grad [Y]')

                plt.figure()
                plt.plot(np.real(v_grads_y[:,vy_ax.npts//2]))
                plt.plot(np.imag(v_grads_y[:, vy_ax.npts // 2]))
                # plt.colorbar()
                plt.xlabel('vy')
                plt.ylabel('y')
                plt.title('v grad [Y]')
                plt.show()

        ### advection term
        if do_x_advection:
            convective_term = self._calculate_time_derivative_f_advection(compress1=compress1, compress2=compress2,
                                                                          verbose_plot=verbose_plot)
        else:
            convective_term = self.f.create_like(None)
            print('no x advection')

        ### force term
        if do_v_advection:
            lorentz_term = self._calculate_time_derivative_f_force(v_grads=v_grads,
                                                                   background_force=background_force,
                                                                   internal_force=internal_force,
                                                                   compress1=compress1, compress2=compress2,
                                                                   verbose_plot=verbose_plot)
        else:
            lorentz_term = self.f.create_like(None)
            print('no v advection')
        # exit()

        dFdt_s = convective_term.add(lorentz_term, compress_level=0)
        dFdt_s.name = self.f.name

        # if verbose_plot:
        #     ax_x, ax_y, ax_vx, ax_vy, ax_vz = dFdt_s.grid.axes
        #     plt.figure()
        #     dfdt_data = dFdt_s.get_comp_data(ax_select={ax_vx:ax_vx.npts//2, ax_vy:ax_vy.npts//2, ax_vz:ax_vz.npts//2})
        #     plt.imshow(dfdt_data)
        #     plt.colorbar()
        #     plt.xlabel('y'), plt.ylabel('x')
        #
        #     plt.figure()
        #     dfdt_data = dFdt_s.get_comp_data(
        #         ax_select={ax_x: 0, ax_vx: ax_vx.npts // 2, ax_vz: ax_vz.npts // 2})
        #     plt.imshow(dfdt_data)
        #     plt.xlabel('vy'), plt.ylabel('y')
        #     plt.colorbar()
        #     plt.show()

        ### get collisions
        coll_field = self.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                             compress1=compress1, compress2=compress2)
        dFdt_s.add(coll_field, inplace=True, compress_level=0)

        if compress:
            dFdt_s.compress(compress_level=compress)

        return dFdt_s


    # @profile
    def _calculate_time_derivative_f_advection(self, x_axes=None, compress1: int = 0, compress2: int = 0,
                                               verbose_plot: bool = False) -> Optional['Field']:
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
        """
        dist = self.f

        if dist is None:
            return None

        x_axes = self.coords_x.axes if x_axes is None else x_axes
        # v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]
        v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]

        if not self.upwind:
            v_axes_dict = None
        else:
            v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}

        x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)

        if verbose_plot:
            # ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            # dist_data = dist.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
            #     ax_vx: ax_vx.npts // 2,
            #     ax_vy: ax_vy.npts // 2,
            #     ax_vz: ax_vz.npts // 2})
            # xg_data = x_grads.get_comp_data(ax_y.coordinate,
            #                                 ax_select={ax_vx: ax_vx.npts // 2, ax_vy: ax_vy.npts // 2,
            #                                            ax_vz: ax_vz.npts // 2})
            ax_x, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={  # ax_x: ax_x.npts//2,
                                                        ax_vx: ax_vx.npts // 2,
                                                        # ax_vy: ax_vy.npts // 2,
                                                        ax_vz: ax_vz.npts // 2})
            xg_data = x_grads.get_comp_data(ax_x.coordinate, ax_select={ax_vx: ax_vx.npts // 2,
                                                                        # ax_vy: ax_vy.npts // 2,
                                                                        ax_vz: ax_vz.npts // 2})

            if dist_data is not None:
                plt.figure()
                plt.imshow(np.real(dist_data))
                plt.xlabel('y')
                plt.ylabel('x')
                plt.title(f'dist adv')
                plt.colorbar()

                plt.figure()
                # plt.plot(np.real(dist_data[0, :]))
                plt.plot(np.real(dist_data[:, ax_vy.npts // 2]))
                plt.plot(np.imag(dist_data[:, ax_vy.npts // 2]))
                plt.title(f'dist')

                # plt.show()

            if xg_data is not None:
                plt.figure()
                plt.imshow(np.real(xg_data))
                plt.xlabel('vy')
                plt.ylabel('x')
                plt.title('xgrad')
                plt.colorbar()
                # plt.show()

                plt.figure()
                plt.plot(np.real(xg_data[:, ax_vy.npts // 2]), label='gradx')
                plt.plot(np.imag(xg_data[:, ax_vy.npts // 2]), label='gradx')
                plt.xlabel('x')
                plt.legend()
                plt.show()

                print('xgdata is nan', np.any(np.isnan(xg_data)))


        ## convective term
        # print('x_grads', x_grads.max_bonds())
        # print(v_axes, self.coords_x.coords)
        convective_term = x_grads.xdot(v_axes, compIDs=self.coords_x.coords, compress_level=compress1,
                                       inner_compress_level=0)
        convective_term.scalar_multiply(-1, inplace=True)

        if verbose_plot:
            conv_data = convective_term.get_comp_data(ax_select={ax_vx: ax_vx.npts // 2,
                                                                 # ax_vy: ax_vy.npts // 2,
                                                                 ax_vz: ax_vz.npts // 2})
            if conv_data is not None:
                plt.figure()
                plt.imshow(np.real(conv_data))
                plt.colorbar()
                plt.xlabel('y')
                plt.ylabel('x')
                plt.title('convective term')
                plt.show()
                # exit()

        return convective_term


    # @profile
    def _calculate_time_derivative_f_force(self, v_axes=None, v_grads=None, background_force=True, internal_force=True,
                                           compress1: int = 0, compress2: int = 0,
                                           verbose_plot: bool = False) -> Optional['Field']:

        dist = self.f
        dist0 = self.background_f

        # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]
        v_axes = self.v_axes if v_axes is None else v_axes

        ## compute for background distribution (w/ deviation in force)
        force_term_0 = None
        if dist0 is not None:
            dist0_force = self.get_force_term(background_force=False)
            # # dist0_force = self.force_term.add(self.background_force_neg, compress_level=0)
            # if self.compress_F:
            #     dist0_force.compress(inplace=True, compress_opts=self.compress_F_opts)
            # dist0.force_term = dist0_force
            # force_term_0 = dist0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
            #                                                         verbose_plot=False)

            dist0.set_force_term(dist0_force, background_force=False)
            force_term_0 = dist0._calculate_time_derivative_f_force(background_force=False,
                                                                    compress1=compress1, compress2=compress2,
                                                                    verbose_plot=False)

        ## compute for original distribution
        if dist is None:
            return force_term_0

        ## grad_v of main distribution
        if v_grads is None:
            df0dv = self.df0dv
            v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) if df0dv is None else df0dv

        # plt.figure()
        # ax_vx, ax_vy, ax_vz = v_axes
        # vgrad_data = v_grads.get_comp_data(ax_vz.coordinate, ax_select={ax_vx: ax_vx.npts // 2,
        #                                                                 ax_vy: ax_vy.npts // 2,
        #                                                                 ax_vz: ax_vz.npts // 2})
        # if vgrad_data is not None:
        #     plt.imshow(np.real(vgrad_data))
        #     plt.title('vgrad [VZ]')
        #     plt.colorbar()
        #     plt.show()

        ## intermediate plots
        if verbose_plot:
            # ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            ax_x, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
                                                      # ax_y: ax_y.npts//2,
                                                      # ax_vx: ax_vx.npts // 2,
                                                      ax_vy: 0,
                                                      ax_vz: ax_vz.npts // 2})
            vg_data = v_grads.get_comp_data(ax_vx.coordinate,
                                            ax_select={# ax_x: ax_x.npts // 2,
                                                       ax_vx: ax_vx.npts // 2,
                                                       ax_vy: 0,
                                                       # ax_vz: ax_vz.npts // 2
                                                       })

            if vg_data is not None:
                plt.figure()
                plt.imshow(np.real(dist_data))
                plt.xlabel('vx')
                plt.ylabel('y')
                plt.colorbar()
                plt.title('dist (vy[0])')
                plt.show()

                plt.figure()
                plt.imshow(np.real(vg_data))
                plt.xlabel('vz')
                plt.ylabel('y')
                plt.colorbar()
                plt.title('grad v [X]')
                plt.show()

                plt.figure()
                plt.plot(np.real(vg_data[ax_x.npts // 2, :]), label='gradv [X]')
                plt.xlabel('vz')
                plt.legend()
                plt.show()
                # exit()

        ## (EM) force term
        # if self.force_term is None:
        #     em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0,
        #                                       verbose_plot=verbose_plot)
        #     self.force_term = em_term.copy()
        # else:

        # em_term = self.force_term.copy() if self.force_term is not None else None        # is zero if is None
        em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)
        em_term = em_term.copy() if em_term is not None else None

        if verbose_plot:
            if em_term is not None:
                for C in self.coords_x.coords:
                    force_data = em_term.get_comp_data(C, ax_select={# ax_x: 0,
                                                                ax_vx: ax_vx.npts // 2,
                                                                # ax_vy: ax_vy.npts // 2,
                                                                ax_vz: ax_vz.npts // 2})
                    if force_data is not None:
                        plt.figure()
                        plt.imshow(np.real(force_data))
                        plt.title(f'force {C} re')
                        plt.xlabel('vy'), plt.ylabel('x')
                        plt.colorbar()

                        plt.figure()
                        plt.imshow(np.imag(force_data))
                        plt.title(f'force {C} im')
                        plt.xlabel('vy'), plt.ylabel('x')
                        plt.colorbar()
                        plt.show()


        if em_term is not None:
            em_term = em_term.scalar_multiply(-1, inplace=True)

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
                                       zipup=self.zipup, compress_level=compress1, inner_compress_level=0,   # 5, 0
                                       verbose_plot=False)
        else:
            lorentz_term = None

        if force_term_0 is not None:
            lorentz_term.add(force_term_0, inplace=True, compress_level=0)

        if verbose_plot:
            # lorentz_data = lorentz_term.get_comp_data()
            lorentz_data = lorentz_term.get_comp_data(ax_select={# ax_x: 0,
                                                                 ax_vx: ax_vx.npts // 2,
                                                                 # ax_vy: ax_vy.npts // 2,
                                                                 ax_vz: ax_vz.npts // 2
                                                                 })
            if lorentz_data is not None:
                plt.figure()
                plt.imshow(np.real(lorentz_data))
                plt.title('lorentz re')
                plt.xlabel('vy'), plt.ylabel('x')
                plt.colorbar()

                plt.figure()
                plt.imshow(np.imag(lorentz_data))
                plt.title('lorentz im')
                plt.xlabel('vy'), plt.ylabel('x')
                plt.colorbar()
                plt.show()

        return lorentz_term


    def get_collision_term(self, v_axes: Sequence['Axis'] = None, v_grads: 'Field' = None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        dist = self.f
        if dist is None:
            return None

        # print('get collision', self.collision)
        # print('get collision', self.collision.coll_type, v_grads is None)

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
            coll_rate = self.collision.coll_rate

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



