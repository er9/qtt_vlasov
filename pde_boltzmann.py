from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system

from typing import TYPE_CHECKING
if TYPE_CHECKING:
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
            self.grid_X = dist.grid.create_like(new_axes=self.coords_x.axes, new_gridID='GX')
        else:
            assert(grid_X.axes == self.x_axes)
            self.grid_X = grid_X

        if self.f is not None:
            v_dict = {}
            for v_ax in self.coords_v.axes:
                try:
                    x_coord = self.coords_x.get_coord(v_ax.coordinate.type)
                    v_mult = v_ax.get_xmultiply_mps()
                    v_dict[x_coord] = self.f.grid.make_mps_ndim({v_ax: v_mult})
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
        self.collision = CollisionConfiguration(None) if collision is None else collision
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
              verbose_plot=False, do_x_advection=True, do_v_adection=True) -> 'Boltzmann':

        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time,
                                                    do_x_advection=do_x_advection, do_v_adection=do_v_adection,
                                                    compress_level=compress_level, compress_level1=compress_level1,
                                                    compress_level2=compress_level2, verbose_plot=verbose_plot)

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
            TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
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
            state0 = state0.get_force_advection(dt / 2, inplace=True, method=method_f,
                                                compress=comp1, compress1=comp2, compress2=comp5,
                                                verbose_plot=verbose_plot)

        # print('compress', comp2, state0.f.max_bond())

        ## evolve velocity advection dt
        state0 = state0.get_vel_advection(dt, inplace=True, method=method_v,
                                          compress=comp1, compress1=comp2, compress2=comp5, verbose_plot=verbose_plot)

        # print('vel compress', comp2, state0.f.max_bond())

        ## evolve force advection dt/2 if last time step or dt
        dt_ = dt/2 if is_last_time_step else dt
        state0 = state0.get_force_advection(dt_, inplace=True, method=method_f,
                                            compress=comp1, compress1=comp2, compress2=comp5,
                                            verbose_plot=verbose_plot)

        # print('f compress', comp2, state0.f.max_bond())

        ## include collisions
        if self.collision.coll_type is not None:
            raise NotImplementedError
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


    def semilagrangian(self, ax: 'Axis', dt: float, method=None, advec_coeffs: 'GridTN' = None, coeffs_is_constant=True,
                       inplace=False, verbose_plot=False, compress_level=1):

        state0 = self if inplace else self.copy()
        f_gtn = state0.f.component

        if method == 'SL':
            print('SL advection', ax)
            # f_mpo_func = f_gtn.grid._get_mpo_advection_SL
            f_mpo_func = state0._get_mpo_advection_SL
        elif method == 'SL-pfc':
            print('SL-pfc advection', ax)
            f_mpo_func = f_gtn.grid._get_mpo_advection_SL_pfc
        else:
            raise ValueError('not valid SL method')

        if advec_coeffs is None:
            coeff_ax = state0.coords_v.get_axis(ax.coordinate.type)  # corresponding v_ax for coeffs
            # advec_coeffs = coeff_ax.xpts
        else:
            assert(advec_coeffs.grid.ndim == 1), 'only defined for 1-D coeffs'
            coeff_ax = advec_coeffs.grid.axes[0]
            advec_coeffs = advec_coeffs.get_data()
            if advec_coeffs is None:  # no field
                print('force is None')
                return

        if verbose_plot:
            if advec_coeffs is not None:
                plt.figure()
                plt.plot(advec_coeffs)
                plt.title('advection coeffs')
                plt.show()
            else:
                print('SL advec coeffs is None')

        deriv_params = f_gtn.ax_deriv_configs[ax]
        # print('SL deriv params', deriv_params, ax, coeff_ax)
        compress_opts = state0.f.compress_config.get_compress_opts(compress_level)
        f_mpo_gtn = f_mpo_func(ax, coeff_ax, dt, deriv_params, advec_coeffs=advec_coeffs)
        f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress_level, compress_opts=compress_opts)

        # f_field = state0.f.create_like_scalar(f_gtn)
        # state0.f = f_field
        state0.time = self.time + dt if self.time is not None else None
        return state0


    def _get_mpo_advection_SL(self, x_ax: 'Axis', v_ax: 'Axis', dt: float, deriv_params: 'DerivativeConfiguration',
                              advec_coeffs: np.ndarray = None, sl_order=1,
                              pos_coeff: 'MPSType' = None, neg_coeff: 'MPSType' = None,) -> 'GTN_TYPE':
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

            Note: neg_coeff = abs(neg_coeff) > 0
        """
        sl_order = 2
        print('here', sl_order)
        order = deriv_params.order

        if order != 1:
            raise NotImplementedError

        if pos_coeff is None or neg_coeff is None:

            if advec_coeffs is None:  # use velocities
                # print('coeffs none')
                advec_coeffs = v_ax.map_state_to_mps(v_ax.xpts)
                # advec_coeffs = self.velocities[x_ax.coordinate]   ## defined on full grid, not just v_ax
                v_pos_mpo = v_ax.get_heaviside_mpo(v_ax.zero_ind, get='xright')
                v_neg_mpo = v_ax.get_heaviside_mpo(v_ax.zero_ind, get='xleft')
                pos_coeff = helper.apply(v_pos_mpo, advec_coeffs)
                neg_coeff = helper.apply(v_neg_mpo, advec_coeffs)
                helper.scalar_multiply(pos_coeff, dt / x_ax.dx, inplace=True)
                helper.scalar_multiply(neg_coeff, -1 * dt / x_ax.dx, inplace=True)

            else:
                dx = x_ax.dx
                v_pos = np.where(advec_coeffs <= 0, 0., advec_coeffs) * dt / dx
                v_neg = np.where(advec_coeffs >= 0, 0., advec_coeffs) * dt / dx * -1
                # v_abs = 1 - np.abs(advec_coeffs) * dt / dx

                pos_coeff = v_ax.map_state_to_mps(v_pos)
                neg_coeff = v_ax.map_state_to_mps(v_neg)
                # abs_coeff = v_ax.map_state_to_mps(v_abs)

        # plt.figure()
        # plt.plot(v_ax.map_mps_to_state(pos_coeff), label='pos')
        # plt.plot(v_ax.map_mps_to_state(neg_coeff), label='neg')
        # plt.legend()
        # plt.show()

        v_pos_mpo = v_ax.apply_elemental_multiply_op(pos_coeff)
        v_neg_mpo = v_ax.apply_elemental_multiply_op(neg_coeff)

        if sl_order == 1:       ## Kormann 2015 Section 4.1, Durran 7.5

            shift_0 = x_ax.get_iden_mpo()
            shift_p = x_ax.get_shift_mpo( 1, boundary_conditions=deriv_params)
            # print('shift p', x_ax.map_mpo_to_operator(shift_p))
            shift_m = x_ax.get_shift_mpo(-1, boundary_conditions=deriv_params)
            # print('shift m', x_ax.map_mpo_to_operator(shift_m))

            abs_coeff = helper.add_MPS(pos_coeff, neg_coeff, inplace=False)
            helper.scalar_multiply(abs_coeff, -1, inplace=True)
            helper.add_MPS(abs_coeff, v_ax.get_iden_mps(), inplace=True)

            v_abs_mpo = v_ax.apply_elemental_multiply_op(abs_coeff)

            # plt.figure()
            # plt.plot(v_ax.map_mps_to_state(abs_coeff), label='abs')
            # plt.legend()
            # plt.show()

            ## forward mpo
            # print('SL mpo', x_ax, v_ax)
            mpo_list = [{x_ax: shift_0, v_ax: v_abs_mpo},
                        {x_ax: shift_p, v_ax: v_neg_mpo},
                        {x_ax: shift_m, v_ax: v_pos_mpo}]

        elif sl_order == 2 or sl_order == 3:

            # print('deriv_params', deriv_params)
            shift_0 = x_ax.get_iden_mpo()
            shift_p1 = x_ax.get_shift_mpo(1, boundary_conditions=deriv_params)
            shift_p2 = x_ax.get_shift_mpo(2, boundary_conditions=deriv_params)
            shift_m1 = x_ax.get_shift_mpo(-1, boundary_conditions=deriv_params)
            shift_m2 = x_ax.get_shift_mpo(-2, boundary_conditions=deriv_params)

            ## positive coeffs
            if not 'v_pos' in locals():
                v_pos = v_ax.map_mps_to_state(pos_coeff)

            ## negative coeffs
            if not 'v_neg' in locals():
                v_neg = v_ax.map_mps_to_state(neg_coeff)

            # plt.figure()
            # plt.plot(v_pos)
            # plt.plot(v_neg)
            # plt.show()

            if sl_order == 2:     ## cubic (Rahn-Allman Eq. 27, 28)

                ## left boundary:  target i = -1
                ## v_pos, origin j = -1, target i = -1  (flux to left boundary)
                pos_m2 = 1 * -v_pos * (1 - v_pos) * (1 + v_pos) / 6
                pos_m1 = 1 * v_pos * (1 - (1 - v_pos) * (2 - v_pos) / 6 + (1 - v_pos) * (1 + v_pos) / 6)
                pos_00 = 1 * v_pos * (1 - v_pos) * (2 - v_pos) / 6
                pos_p1 = 0
                pos_p2 = 0

                ## v_neg, origin j = 0
                neg_m2 = 0
                neg_m1 = +1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
                neg_00 = +1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
                neg_p1 = +1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6
                neg_p2 = 0

                ## right boundary:  target i = 0
                ## v_pos, origin j = 0, target i = 0  (flux to right boundary)
                pos_m2 += 0
                pos_m1 += -1 * -v_pos * (1 - v_pos) * (1 + v_pos) / 6
                pos_00 += -1 * v_pos * (1 - (1 - v_pos) * (2 - v_pos) / 6 + (1 - v_pos) * (1 + v_pos) / 6)
                pos_p1 += -1 * v_pos * (1 - v_pos) * (2 - v_pos) / 6
                pos_p2 += 0

                ## v_neg, origin j = 1
                neg_m2 += 0
                neg_m1 += 0
                neg_00 += -1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
                neg_p1 += -1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
                neg_p2 += -1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6


                # plt.figure()
                # # plt.plot(pos_00, label='pos')
                # plt.plot(pos_m2 + pos_m1 + pos_00 + pos_p1 + pos_p2, label='pos')
                # plt.plot(neg_m2 + neg_m1 + neg_00 + neg_p1 + neg_p2, label='neg')
                # plt.show()

            elif sl_order == 3:     ## cubic (Durran 7.12)

                pos_m2 = -v_pos * (1 - v_pos**2) / 6
                pos_m1 = v_pos * ( 1 + v_pos ) * (2 - v_pos ) / 2
                pos_00 = (1-v_pos**2) * (2-v_pos) / 2 - 1
                pos_p1 = -v_pos * (1-v_pos) * (2-v_pos) / 6
                pos_p2 = 0

                neg_p2 = -v_neg * (1 - v_neg ** 2) / 6
                neg_p1 = v_neg * (1 + v_neg) * (2 - v_neg) / 2
                neg_00 = (1 - v_neg ** 2) * (2 - v_neg) / 2 - 1
                neg_m1 = -v_neg * (1 - v_neg) * (2 - v_neg) / 6
                neg_m2 = 0

            else:
                raise ValueError('check sl_order value')

            pos_heaviside = np.where(v_pos > 0, 1, 0)
            neg_heaviside = 1 - pos_heaviside

            pos_m2 = pos_m2 * pos_heaviside     # only keep vals for coeffs > 0
            pos_m1 = pos_m1 * pos_heaviside
            pos_00 = pos_00 * pos_heaviside
            pos_p1 = pos_p1 * pos_heaviside
            pos_p2 = pos_p2 * pos_heaviside

            neg_p2 = neg_p2 * neg_heaviside     # only keep vals for coeffs < 0
            neg_p1 = neg_p1 * neg_heaviside
            neg_00 = neg_00 * neg_heaviside
            neg_m1 = neg_m1 * neg_heaviside
            neg_m2 = neg_m2 * neg_heaviside

            # plt.figure()
            # plt.plot(v_pos, label='pos')
            # plt.plot(v_neg, label='neg')
            # plt.legend()
            # plt.show()

            ## assert(v_max * dt / x_ax.dx < 1), ie. assume p=0
            coeff_m2 = pos_m2 + neg_m2       ## can be zero
            coeff_m1 = pos_m1 + neg_m1
            coeff_00 = 1 + pos_00 + neg_00
            coeff_p1 = pos_p1 + neg_p1
            coeff_p2 = pos_p2 + neg_p2

            # plt.figure()
            # plt.plot(coeff_m2, label='m2')
            # plt.plot(coeff_m1, label='m1')
            # plt.plot(coeff_00, label='00')
            # plt.plot(coeff_p1, label='p1')
            # if coeff_p2 is not None:
            #     plt.plot(coeff_p2, label='p2')
            # plt.legend()
            # plt.show()

            coeff_00_mps = v_ax.map_state_to_mps(coeff_00)
            coeff_m1_mps = v_ax.map_state_to_mps(coeff_m1)
            coeff_m2_mps = v_ax.map_state_to_mps(coeff_m2) if coeff_m2 is not None else None
            coeff_p1_mps = v_ax.map_state_to_mps(coeff_p1)
            coeff_p2_mps = v_ax.map_state_to_mps(coeff_p2) if coeff_p2 is not None else None

            coeff_00_mpo = v_ax.apply_elemental_multiply_op(coeff_00_mps)
            coeff_m1_mpo = v_ax.apply_elemental_multiply_op(coeff_m1_mps)
            coeff_m2_mpo = v_ax.apply_elemental_multiply_op(coeff_m2_mps) if coeff_m2 is not None else None
            coeff_p1_mpo = v_ax.apply_elemental_multiply_op(coeff_p1_mps)
            coeff_p2_mpo = v_ax.apply_elemental_multiply_op(coeff_p2_mps) if coeff_p2 is not None else None

            mpo_list = [{x_ax: shift_0, v_ax: coeff_00_mpo},
                        {x_ax: shift_p1, v_ax: coeff_p1_mpo},
                        {x_ax: shift_m1, v_ax: coeff_m1_mpo}]
            if coeff_m2_mpo is not None:
                mpo_list += [{x_ax: shift_m2, v_ax: coeff_m2_mpo}]
            if coeff_p2_mpo is not None:
                mpo_list += [{x_ax: shift_p2, v_ax: coeff_p2_mpo}]
        else:
            raise ValueError('check sl_order')


        gtn_mpo = self.f.grid.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO, compress=True)
        return gtn_mpo


    # @profile
    def maccormack(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeff_is_constant=True,
                   inplace=False, compress_level=1):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        print('mac', ax, compress_level)

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if advec_coeffs is None:
            # v_ax = self.coords_v.get_axis(ax.coordinate.type)
            # advec_coeffs = v_ax.xpts
            advec_coeffs = self.velocities[ax.coordinate]
            # print('advec coeffs', advec_coeffs)

        ## gradient of state along x
        deriv_config = self.f.component.ax_deriv_configs[ax].copy()
        deriv_config.update(order=0)    # 1st order accurate. kind of jank implementation

        state = self if inplace else self.copy()

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
                grad_f_term = grad_f.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                # print('advec coeffs', advec_coeffs.max_bond())

                # plt.figure()
                # plt.imshow(grad_f.get_comp_data(compID=ax.coordinate))
                # plt.title('grad f * - dt'), plt.colorbar()
                # plt.show()
                #
                # plt.figure()
                # if grad_f_term[ax.coordinate].data is not None:
                #     plt.imshow(grad_f_term.get_comp_data(compID=ax.coordinate))
                #     plt.title(f'grad f * advec_coeffs {it}'), plt.colorbar()
                # plt.show()
            else:
                term1 = f_active.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                shift = -1 if it == 0 else 1
                shift_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config.left_bc)  ## intended: j -> j-1, and then j -> j+1
                print('WARNING: lax wendroff need to check shift direction')
                term2 = state.f.component.apply(shift_mpo, inplace=False,)
                term2 = term2.elemental_multiply(advec_coeffs, compress_level=comp4, inplace=True)
                term2 = term2.scalar_multiply(-1, inplace=True)
                grad_f_term = term1.add(term2)
                grad_f_term.scalar_multiply(-dt, inplace=True)

            grad_f_term = grad_f_term.create_like_scalar(grad_f_term[ax.coordinate])
            f_active = f_active.add( grad_f_term, inplace=False, compress_level=comp3)

        state2 = self.create_like( f_active, recalc=False )
        new_state = state.add(state2, inplace=True, compress_level=comp1)
        new_state.scalar_multiply(1./2, inplace=True,)
        return new_state


    def lax_wendroff(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeff_is_constant=True,
                     inplace=False, compress_level=1):
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


    def get_force_advection(self, dt, inplace=False, method=None,
                            compress=1, compress1=2, compress2=5, verbose_plot=False):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        # print('boltzmann force advection', compress, compress1, compress2)
        new_state = self if inplace else self.copy()

        if method is None:
            deriv_v0 = new_state._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
                                                                    verbose_plot=verbose_plot)
            dFdt_v = new_state.create_like(deriv_v0, recalc=False)
            new_state = new_state.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)

        elif method[:2] == 'rk':
            func = new_state._get_rk_func(method)
            state1 = func(dt, compress_level=compress, do_x_advection=False)
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
        else:
            func, kwargs = new_state._get_split_step_func(method)

            v_axes = new_state.v_axes
            if len(v_axes) > 1:
                ax_list = [ax for ax in v_axes if ax is not None] + \
                          [ax for ax in v_axes[-2::-1] if ax is not None]
                scales = [1. / 2, ] * len(ax_list)
                scales[len(v_axes) - 1] = 1.
            else:
                ax_list = v_axes
                scales = [1.]

            if new_state.force_term is None:
                force_term: 'Field' = new_state.compute_force_term(compress_level1=compress1, compress_level2=compress2)
            else:
                force_term: 'Field' = new_state.force_term

            # print('force term', force_term.components)
            # exit()

            # print('force axes', ax_list, scales)

            for v_ax, scale in zip(ax_list, scales):
                C = new_state.coords_x.get_coord(v_ax.coordinate.type)

                if verbose_plot:
                    force_data = force_term.get_comp_data(C)
                    if force_data is not None:
                        plt.figure()
                        plt.plot(force_data)
                        plt.title('elc lorentz force')
                        plt.show()
                    else:
                        print('force term is None')

                if new_state.background_f is not None:
                    raise NotImplementedError('split step with background f not implemented')
                    ## need to include f0 * dF term

                # print('force term', v_ax, C, force_term[C])

                try:
                    new_state = func(v_ax, dt * scale, advec_coeffs=force_term[C], inplace=True,
                                     compress_level=compress, **kwargs)
                except KeyError:
                    print('force is None', C)
                    continue

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
            if len(x_axes) > 1:
                ax_list = [ax for ax in x_axes if ax is not None] + \
                          [ax for ax in x_axes[-2::-1] if ax is not None]
                scales = [1. / 2, ] * len(ax_list)
                scales[len(x_axes) - 1] = 1.
            else:
                ax_list = x_axes
                scales = [1.]

            # print('vel axes', ax_list, scales)

            for x_ax, scale in zip(ax_list, scales):
                assert(x_ax.basis.type==BasisType.SPATIAL), 'axes must be in spatial basis'

                state0 = func(x_ax, dt * scale, inplace=True,
                              compress_level=compress, **kwargs)

                # f_gtn = state0.f.component
                # v_ax = self.v_axes_dict[x_ax.coordinate]
                # deriv_params = f_gtn.ax_deriv_configs[x_ax]
                # compress_opts = state0.f.compress_config.get_compress_opts(compress)
                # f_mpo_gtn = f_gtn.grid.get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params, method=method)
                # f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)

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
        density: Optional['ScalarField'] = None
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


    def compute_flow(self, compress=1, compress1=5, store=False, verbose_plot=False) -> 'Field':

        dist = self.f
        v_axes = self.coords_v.axes

        j: Optional['Field'] = None
        if dist is not None:
            j = dist.xmultiply(v_axes, compress_level=compress1)
            j = j.integrate(axes=v_axes, new_grid=self.grid_X,
                                     new_ax_deriv_configs=dist.component.ax_deriv_configs)
            j.replace_compIDs({v_ax.coordinate: self.coords_x.get_coord(v_ax.coordinate.type) for v_ax in v_axes})

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
            powers = [1] * len(axes)

        grid_X = None
        if integ_axes is None:
            integ_axes = self.v_axes
            grid_X = self.grid_X

        ## compute integrand
        moment_dist = None
        for ax, power in zip(axes, powers):
            ax_dist = self.f.xmultiply([ax], x_powers=[power], compress_level=0)

            if moment_dist is None:
                moment_dist = ax_dist
            else:
                moment_dist.add(ax_dist, inplace=True, compress_level=0)

        ## take integral
        moment = moment_dist.integrate(axes=integ_axes, new_grid=grid_X)

        if compress:
            moment.compress(inplace=True, compress_level=compress)

        return moment


    def calculate_time_derivative(self, time=None, do_x_advection=True, do_v_advection=True,
                                  compress_level=0, compress_level1=0, compress_level2=0,
                                  verbose_plot=False) -> 'Boltzmann':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        dFdt = self._calculate_time_derivative_f(do_x_advection=do_x_advection, do_v_advection=do_v_advection,
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


    # @profile
    def _calculate_time_derivative_f(self, do_x_advection=True, do_v_advection=True,
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
    def _calculate_time_derivative_f_force(self, v_axes=None, v_grads=None, compress1: int = 0, compress2: int = 0,
                                           verbose_plot: bool = False) -> Optional['Field']:

        dist = self.f
        dist0 = self.background_f

        # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]
        v_axes = self.v_axes if v_axes is None else v_axes

        ## compute for background distribution (w/ deviation in force)
        force_term_0 = None
        if dist0 is not None:
            dist0_force = self.force_term.add(self.background_force_neg, compress_level=0)
            if self.compress_F:
                dist0_force.compress(inplace=True, compress_opts=self.compress_F_opts)
            dist0.force_term = dist0_force
            force_term_0 = dist0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
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
        if self.force_term is None:
            em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0,
                                              verbose_plot=verbose_plot)
            self.force_term = em_term.copy()
        else:
            em_term = self.force_term.copy()


        if verbose_plot:
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



