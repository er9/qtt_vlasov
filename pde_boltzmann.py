from setup_.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
import helper_sl as helper_sl
import helper_dlr
from helper_tdvp import TDVPSolver
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system

import multiprocessing as mp
# import mpi4py.futures

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
                 matl_params: Optional['SpeciesConfiguration'] = None,
                 background_f: Optional[Union['ScalarField', 'Boltzmann']] = None,
                 background_force_neg: Optional['Field'] = None,
                 # evolve_background=False,
                 normalize=True, upwind=False, zipup=False, conservative=True,
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
                         conservative=conservative,
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
        self.v_axes: list['Axis'] = []
        for coord in coords_x.coords:
            v_ax = self.coords_v.get_axis(coord.type)
            if v_ax is not None:
                self.v_axes_dict[coord] = v_ax
                self.v_axes += [v_ax]

        if grid_X is None:
            # self.grid_X = dist.grid.create_like(new_axes=self.coords_x.axes, new_gridID='GX')
            self.grid_X = dist.grid.get_subgrid(select_axes=self.coords_x.axes, new_gridID='GX')
        else:
            assert (grid_X.axes == self.x_axes)
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

                    # grid_X = self.f.grid.__class__('x', [v_ax])
                    # v_gtn_ = grid_X.make_mps_ndim({v_ax: v_mult})
                    #
                    # check1 = v_ax.get_xmultiply_mpo()
                    # check2 = v_gtn_.apply_elemental_multiply_op()
                    #
                    # plt.figure()
                    # plt.imshow(np.real(v_ax.map_mpo_to_operator(check1)))
                    # plt.colorbar()
                    # plt.title('re check1')
                    # plt.figure()
                    # plt.imshow(np.imag(v_ax.map_mpo_to_operator(check1)))
                    # plt.colorbar()
                    # plt.title('im check1')
                    #
                    # plt.figure()
                    # plt.imshow(np.real(check2.get_data()))
                    # plt.colorbar()
                    # plt.title('re check2')
                    # plt.figure()
                    # plt.imshow(np.imag(check2.get_data()))
                    # plt.colorbar()
                    # plt.title('im check2')
                    #
                    # plt.show()
                    # exit()

                    # v_mult = v_ax.get_xmultiply_mpo()
                    # v_dict[x_coord] = self.f.grid.make_mpo_ndim({v_ax: v_mult})
                except (AttributeError, KeyError):
                    pass
            self.velocities = self.f.create_like_vector(v_dict)
            self.velocities.name = 'vel'

        ## save important values
        self.gradx_f: Optional['Field'] = None
        self.gradv_f: Optional['Field'] = None
        self.flow: Optional['Field'] = None  # flow density (-> current)
        self.density: Optional['ScalarField'] = None  # density

        self.force_term: Optional['Field'] = None
        self.force_term_nobg: Optional['Field'] = None
        self.force_term_bg: Optional['Field'] = None
        ## background force is separated by number of cells traveled (for SL advection)
        self.force_term_bg_SL: dict['Coordinate', 'helper_sl.SL_data'] = {}
        self.saved_SL_mpos: dict['Axis', dict[Numeric]] = {}  ## axis, dt nested dict

        self.upwind = upwind
        self.zipup = zipup

        ## background fields
        self.background_f = background_f
        self.background_force_neg = background_force_neg  # negative of background force
        self.evolve_background = False  # evolve_background if background_f is not None else False

        ## save initial normalization for normalization
        self.init_norm = None
        if normalize:
            self.init_norm = self.total_f.norm() if self.f is not None else None

        ## define for linearized equation
        self.df0dv = f0_gradv

        ## force/em-term compression
        self.compress_F = compress_F  # True
        self.compress_F_opts = {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE} \
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
    def background_f(self, f0: Optional[Union['ScalarField', 'Boltzmann']]):
        if isinstance(f0, ScalarField):
            f0 = Boltzmann(f0, grid_X=self.grid_X, coords_x=self.coords_x, coords_v=self.coords_v,
                           matl_params=self.matl_params, upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                           compress_levels=self._comp_levels)
        elif isinstance(f0, Boltzmann) or f0 is None:
            pass  ## PDE system
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

    def get_force_term(self, background_force=True, internal_force=True):
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
                                   conservative=self.conservative,
                                   )
        else:
            new_system = Boltzmann(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x, coords_v=self.coords_v,
                                   f0_gradv=self.df0dv, matl_params=self.matl_params,
                                   background_f=f0_copy, background_force_neg=self.background_force_neg,
                                   normalize=False,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   conservative=self.conservative,
                                   )

            new_system.do_normalization = self.do_normalization
            new_system.init_norm = self.init_norm

        if dist is None:      new_system.names['f'] = self.names['f']

        new_system.collision = self.collision
        new_system.coll_dist0 = self.coll_dist0

        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

        new_system.force_term_bg_SL = self.force_term_bg_SL
        new_system.force_term_bg = self.force_term_bg.copy() if self.force_term_bg is not None else None
        new_system.saved_SL_mpos = {k: val.copy() for k, val in self.saved_SL_mpos.items()}

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

    ######

    ######

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
              inplace: bool = False, compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False, do_x_advection=True, do_v_advection=True, **deriv_kwargs) -> 'Boltzmann':

        state1 = super().euler(dt, deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                               verbose_plot=verbose_plot, **deriv_kwargs)

        if verbose_plot:
            ax_x, ax_y, ax_vx, ax_vy, ax_vz = state1.f.grid.axes
            dist_data = state1.f.get_comp_data(ax_select={ax_x: 0, ax_vx: ax_vx.npts // 2, ax_vz: ax_vz.npts // 2})
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

        ## technically would have to renormalize after applying absorbing boundaries
        # if state1.do_normalization:
        #     state1.normalize()

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
        dt_ = dt / 2 if is_last_time_step else dt
        state0 = state0.get_force_advection(dt_, inplace=True, method=method_f, compress=comp1, compress1=comp2,
                                            compress2=comp5, verbose_plot=verbose_plot)

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
            if isinstance(self.f.component, GridTN1D):
                sl_func = self.maccormack # _v2  ## rdm
            elif isinstance(self.f.component, GridTN1DComb):
                sl_func = self.maccormack # _v2  ## zipup
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
        elif method == 'rk1':
            func = self.euler
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

    # @profile
    def semilagrangian(self, ax: 'Axis', dt: float, method=None, advec_coeffs: 'GridTN' = None,
                       inplace=False, verbose_plot=False, compress_level=1,
                       # f_perturbation: 'GridTN' = None,
                       # background_sl=False, save_background_sl=False,
                       store_sl_mpo=True,
                       **kwargs):

        state0 = self if inplace else self.copy()
        f_gtn = state0.f.component

        if method[:2] == 'SL':
            print('SL advection', ax)
            # f_mpo_func = f_gtn.grid._get_mpo_advection_SL
            if ax.is_k():
                f_mpo_func = state0._get_mpo_advection_SL_k
            else:
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
            coeff_gr = None  # state0.coords_v.get_axis(ax.coordinate.type)  # corresponding v_ax for coeffs
            # advec_coeffs = coeff_ax.xpts
        else:
            # assert(advec_coeffs.grid.ndim == 1), 'only defined for 1-D coeffs'
            # coeff_ax = advec_coeffs.grid.axes[0]
            # coeff_gr = advec_coeffs.grid
            # print('advec coeffs constant axes', ax, advec_coeffs.constant_axes)
            # print('advec ceoffs', advec_coeffs)
            advec_coeffs, coeff_gr = advec_coeffs.get_squeezed_data()
            # print('advec_coeffs', advec_coeffs.shape, coeff_gr)
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

        compress_opts = state0.f.compress_config.get_compress_opts(compress_level) if compress_level else {}

        if advec_coeffs is not None:  # or background_sl:
            deriv_params = f_gtn.ax_deriv_configs[ax]
            f_mpo_gtn = f_mpo_func(ax, coeff_gr, dt, deriv_params, advec_coeffs=advec_coeffs, sl_order=sl_order,
                                   store_sl_mpo=store_sl_mpo,
                                   # background_sl=background_sl, save_background_sl=save_background_sl
                                   )

            if f_mpo_gtn is not None:
                print('SL f mpo gtn', f_mpo_gtn.max_bond())
                # print('SL f gtn', f_gtn.max_bond())
                # print('zipup ?', self.zipup)
                # f_gtn.apply(f_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress_level, compress_opts=compress_opts)
                # print('compress opts', compress_level, compress_opts)
                print('SL rdm comp')
                f_gtn.apply_rdm(f_mpo_gtn, inplace=True, compress_opts=compress_opts)
                print('f gtn max bond', f_gtn.max_bond())

        # if compress_level:
        #     compress_opts = state0.f.compress_config.get_compress_opts(compress_level)
        #     f_gtn.compress(inplace=True, compress_opts=compress_opts)

        # f_field = state0.f.create_like_scalar(f_gtn)
        # state0.f = f_field
        state0.time = self.time + dt if self.time is not None else None
        return state0

    def _get_mpo_advection_SL_k(self, advec_ax: 'Axis', coeff_gr: 'Grid', dt: float,
                                deriv_params: 'DerivativeConfiguration',
                                advec_coeffs: np.ndarray = None,
                                sl_order=3,
                                sl_adv_obj: 'helper_sl.SL_data' = None,
                                store_sl_mpo=True
                                # background_sl = False, save_background_sl = False
                                ) \
            -> 'GTN_TYPE':
        """ derivative in x for exact TE, assuming advection axis is in Fourier basis
            f/dx = i k * v |fk>  --> fk = exp(ikv * dt) |fk>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt here
        """
        # sl_order = 1
        print('SL (k)', coeff_gr, coeff_gr.axes)
        # print('advec ax', advec_ax, coeff_gr.axes)

        try:
            if not store_sl_mpo:
                raise KeyError
            sl_adv_mpo = self.saved_SL_mpos[advec_ax][dt].copy()
            print('use saved sl obj', advec_ax, dt)
        except KeyError:
            sl_adv_mpo = None

        if sl_adv_mpo is None:
            if advec_coeffs is None:  ###  return None
                print('advec coeffs is None', advec_ax)
                return None

            else:
                # plt.figure()
                # print(advec_coeffs.shape)
                # plt.imshow(advec_coeffs)
                # plt.show()

                print('get cell data ADVEC COEFFS', dt, advec_ax, advec_ax.dx)
                ks = advec_ax.xpts
                exp_grid_axes = [advec_ax] + list(coeff_gr.axes)
                ordered_exp_grid_axes = [ax for ax in self.f.grid.axes if ax in exp_grid_axes]
                transpose_axes = [ordered_exp_grid_axes.index(ax) for ax in exp_grid_axes]
                print('ordered exp grid axes', ordered_exp_grid_axes, transpose_axes)
                exp_grid = self.f.grid.create_like_from_axes(ordered_exp_grid_axes, 'exp')

                exp_vals = np.tensordot(ks, advec_coeffs, axes=0)
                exp_vals = exp_vals.transpose(transpose_axes)
                exp_vals = np.exp(-1.j * exp_vals * dt)

                exp_vals_mps = exp_grid.map_state_to_mps(exp_vals)
                sl_adv_mpo = exp_vals_mps.mps_to_diag_mpo()

            # print('compress gtn mpo', gtn_mpo.L)
            sl_adv_mpo.compress(inplace=True)  # , compress_opts={'cutoff': 1e-14})

            if store_sl_mpo:
                self.saved_SL_mpos[advec_ax] = {dt: sl_adv_mpo}
                print('saved new sl adv obj', coeff_gr)

        return sl_adv_mpo

    def _get_mpo_advection_SL(self, advec_ax: 'Axis', coeff_gr: 'Grid', dt: float,
                              deriv_params: 'DerivativeConfiguration',
                              advec_coeffs: np.ndarray = None,
                              sl_order=3,
                              sl_adv_obj: 'helper_sl.SL_data' = None,
                              store_sl_mpo=True
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
        print('advec ax', advec_ax, coeff_gr.axes)

        # order = deriv_params.order
        #
        # if order != 1:
        #     raise NotImplementedError

        try:
            if not store_sl_mpo:
                raise KeyError
            sl_adv_mpo = self.saved_SL_mpos[advec_ax][dt].copy()
            print('use saved sl obj', advec_ax, dt)
        except KeyError:
            sl_adv_mpo = None

        if sl_adv_mpo is None:
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
                    if False:  # background_sl:   # and advec_ax in self.coords_v.axes:
                        # print('is background_sl')
                        sl_adv_obj = self.force_term_bg_SL.get(advec_ax.coordinate, None)
                        # print('sl adv obj', sl_adv_obj is None, advec_ax.coordinate)
                        if sl_adv_obj is None:  # and advec_ax in self.coords_x.axes:
                            # print('get bg cell data advec coeffs', advec_ax, advec_coeffs.shape)
                            dx = advec_ax.dx
                            sl_adv_obj = helper_sl.get_cell_data(dt, dx, advec_coeffs, sl_order=sl_order,
                                                                 coeff_grid=coeff_gr)
                            self.force_term_bg_SL[advec_ax.coordinate] = sl_adv_obj
                        # print('got bg sl force', advec_ax)
                    else:
                        # print('get cell data ADVEC COEFFS', dt, advec_ax, advec_ax.dx)
                        dx = advec_ax.dx
                        sl_adv_obj = helper_sl.get_cell_data(dt, dx, advec_coeffs, sl_order=sl_order,
                                                             coeff_grid=coeff_gr)
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
                shift_mpo_gtn = advec_ax.get_shift_mpo(sh, boundary_conditions=deriv_params)
                shift_mpo_gtn = self.f.grid.make_mpo_ndim({advec_ax: shift_mpo_gtn})

                if (not isinstance(sl_weights[sh], np.ndarray)) or sl_weights[sh].size == 1:
                    weights_mpo = coeff_gr.get_iden_mpo().scalar_multiply(sl_weights[sh], inplace=True)
                else:
                    weights_mps = coeff_gr.map_state_to_mps(sl_weights[sh])
                    if weights_mps.data is None:
                        print('no coeffs', advec_ax, sh)
                        continue
                    weights_mpo = weights_mps.apply_elemental_multiply_op()

                # print('shift mpo', shift_mpo_gtn)
                # print('weights mpo', weights_mpo)

                gtn_mpo_2 = shift_mpo_gtn.apply(weights_mpo, compress=False)

                if gtn_mpo is None:
                    gtn_mpo = gtn_mpo_2
                else:
                    gtn_mpo.add(gtn_mpo_2, compress=True, inplace=True)

            # print('compress gtn mpo', gtn_mpo.max_bond())
            gtn_mpo.compress(inplace=True)  # , compress_opts={'cutoff': 1e-8})
            sl_adv_mpo = gtn_mpo

            if store_sl_mpo:
                self.saved_SL_mpos[advec_ax] = {dt: sl_adv_mpo}
                print('saved new sl adv obj', coeff_gr)

        return sl_adv_mpo

    def _get_mpo_advection_SL_multidim(self, sl_adv_objs: dict['Axis', 'helper_sl.SL_data'],
                                       ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'],
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

    def maccormack_v2(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeff_is_constant=True,
                      inplace=False, get_df=False, compress_level=1, **kwargs):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        print('mac2', ax, dt, compress_level)
        assert (ax.basis.type == BasisType.SPATIAL), 'axes must be in spatial basis for mac'

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        ## gradient of state along x
        deriv_config = self.f.component.ax_deriv_configs[ax].copy()
        deriv_config.update(order=0)  # 1st order accurate. kind of jank implementation

        state = self if inplace else self.copy()

        if advec_coeffs is None:  ## no advection occurs
            print('NO ADVECTION', ax)
            return state

        f_active = state.f.component.copy()
        for it in range(2):
            if coeff_is_constant:
                if it == 0:
                    deriv_config.update(fd_type=FDType.BACKWARD)
                elif it == 1:
                    deriv_config.update(fd_type=FDType.FORWARD)
                else:
                    raise ValueError

                grad_mpo = f_active.grid.get_firstderivative_mpo(ax, deriv_config=deriv_config)
                grad_mpo = grad_mpo.scalar_multiply(-dt, inplace=False)  # dx already included in gradient operation
                advec_mpo = advec_coeffs.apply_elemental_multiply_op()  # advec_coeffs is the force
                advec_coeff_mpo = grad_mpo.apply(advec_mpo, compress=False)

                iden_mpo = f_active.grid.get_iden_mpo()

                advec_mpo = iden_mpo.add(advec_coeff_mpo, inplace=False)
            else:
                raise NotImplementedError
                # term1 = f_active.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                # shift = -1 if it == 0 else 1
                # shift_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config.left_bc)  ## intended: j -> j-1, and then j -> j+1
                # print('WARNING: lax wendroff need to check shift direction')
                # # term2 = state.f.component.apply(shift_mpo, inplace=False,)
                # term2 = f_active.component.apply(shift_mpo, inplace=False, )
                # term2 = term2.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4, inplace=True)
                # term2 = term2.scalar_multiply(-1, inplace=True)
                # grad_f_term = term1.add(term2)
                # grad_f_term.scalar_multiply(-dt, inplace=True)

            compress_opts = state.f.compress_config.get_compress_opts(comp3)
            # print('advec mpo', advec_mpo)
            # print('compress opts', compress_opts)
            f_active = f_active.apply_rdm(advec_mpo, compress_opts=compress_opts, verbose=False)
            # f_active = f_active.apply(advec_mpo, zipup=self.zipup, compress_opts=compress_opts)

        state2 = self.create_like(state.f.create_like({SCALAR_COORD: f_active}), recalc=False)
        # state2 = self.create_like( f_active, recalc=False )
        if get_df:
            print('get df')
            neg_state = state.scalar_multiply(-1, inplace=True)
            new_state = neg_state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1. / 2, inplace=True, )
        else:
            new_state = state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1. / 2, inplace=True, )
        return new_state

    # @profile
    def maccormack(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None, coeff_is_constant=True,
                   inplace=False, get_df=False, compress_level=1, **kwargs):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        print('mac', ax, compress_level, dt)
        assert (ax.basis.type == BasisType.SPATIAL), 'axes must be in spatial basis for mac'

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        ## gradient of state along x
        deriv_config = self.f.component.ax_deriv_configs[ax].copy()
        deriv_config.update(order=0)  # 1st order accurate. kind of jank implementation

        state = self if inplace else self.copy()

        # if advec_coeffs is None:
        #     # v_ax = self.coords_v.get_axis(ax.coordinate.type)
        #     # advec_coeffs = v_ax.xpts
        #     print('self velocities', self.velocities.componentIDs, ax)
        #     advec_coeffs = self.velocities[ax.coordinate]
        #     # print('advec coeffs', advec_coeffs)

        if advec_coeffs is None:  ## no advection occurs
            print('NO ADVECTION', ax)
            return state

        f_active = state.f.copy()
        for it in range(2):
            if coeff_is_constant:
                if it == 0:
                    deriv_config.update(fd_type=FDType.BACKWARD)
                elif it == 1:
                    deriv_config.update(fd_type=FDType.FORWARD)
                else:
                    raise ValueError

                grad_f = f_active.gradient(deriv_axes=[ax], compress_level=comp5)
                grad_f = grad_f.scalar_multiply(-dt, inplace=False)  # dx already included in gradient operation
                print('mac zipup', self.zipup)
                # print('advec coeff', advec_coeffs.max_bond())
                # print('grad f compress opts', grad_f.compress_config.get_compress_opts(comp4))
                grad_f_term = grad_f.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4)
                # ## advec_coeffs is the force
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
                shift_mpo = ax.get_shift_mpo(shift,
                                             boundary_conditions=deriv_config.left_bc)  ## intended: j -> j-1, and then j -> j+1
                print('WARNING: lax wendroff need to check shift direction')
                # term2 = state.f.component.apply(shift_mpo, inplace=False,)
                term2 = f_active.component.apply(shift_mpo, inplace=False, )
                term2 = term2.elemental_multiply(advec_coeffs, zipup=self.zipup, compress_level=comp4, inplace=True)
                term2 = term2.scalar_multiply(-1, inplace=True)
                grad_f_term = term1.add(term2)
                grad_f_term.scalar_multiply(-dt, inplace=True)

            grad_f_term = grad_f_term.create_like_scalar(grad_f_term[ax.coordinate])
            f_active = f_active.add(grad_f_term, inplace=False, compress_level=comp3)

        state2 = self.create_like(f_active, recalc=False)
        if get_df:
            print('get df')
            neg_state = state.scalar_multiply(-1, inplace=True)
            new_state = neg_state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1. / 2, inplace=True, )

            # plt.figure()
            # plt.imshow(new_state.f.get_comp_data()[0,:,:])
            # plt.colorbar()
            # plt.show()

            # state.add(new_state, inplace=True)
        else:
            new_state = state.add(state2, inplace=True, compress_level=comp1)
            new_state.scalar_multiply(1. / 2, inplace=True, )
        return new_state

    def lax_wendroff(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None,
                     inplace=False, get_df=False, compress_level=1, **kwargs):
        """
        ax:  axis along which advection occurs (d/dx axis)
        note: assume advec_coeffs is constant along ax
        """
        assert (ax.basis.type == BasisType.SPATIAL), 'axes must be in spatial basis for lax'
        state = self if inplace else self.copy()

        comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        dist = state.f.copy()
        deriv_config = state.f.component.ax_deriv_configs[ax].copy()
        shift_p = dist.grid.get_shift_mpo({ax: 1}, ax_boundary_conditions=dist.component.ax_deriv_configs)
        shift_m = dist.grid.get_shift_mpo({ax: -1}, ax_boundary_conditions=dist.component.ax_deriv_configs)

        if advec_coeffs is None:
            advec_coeffs = self.velocities[ax.coordinate]

        ## forward difference
        deriv_config.update(order=0, fd_type=FDType.FORWARD)
        f1 = dist.gradient(deriv_axes=[ax])
        f1 = dist.create_like_scalar(f1[ax.coordinate])
        f1 = f1.elemental_multiply(advec_coeffs, inplace=True, zipup=self.zipup, compress_level=comp4)
        f1.scalar_multiply(-dt / 2, inplace=True)

        # plt.figure()
        # if f1.component.data is not None:
        #     plt.imshow(f1.get_comp_data())
        #     plt.title('f1 * advec_coeffs'), plt.colorbar()
        # plt.show()

        fp = dist.component.apply(shift_p, inplace=False)
        fp = dist.create_like_scalar(fp)
        f1_avg = dist.add(fp, inplace=False, compress_level=0)
        f1_avg.scalar_multiply(1. / 2, inplace=True)

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
        f2 = dist.gradient(deriv_axes=[ax])
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
        deriv.scalar_multiply(-dt / ax.dx, inplace=True)

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
        mu = advec_coeffs.scalar_multiply(dt / ax.dx)  # c * dt/dx

        dist = self.f
        deriv_config = dist.component.ax_deriv_configs[ax].copy()

        raise NotImplementedError('Warming-Beam requires elemental analysis of advection coeff (ie. >0, <0)')

    @staticmethod
    def _get_splitstep_axes_list(axes: Sequence['Axis'], order=2):
        """ determine ordering of axes in split step method
        """
        # print('split order', order)
        if len(axes) > 1 and order == 2:
            ax_list = [ax for ax in axes if ax is not None] + \
                      [ax for ax in axes[-2::-1] if ax is not None]
            scales = [1. / 2, ] * len(ax_list)
            scales[len(axes) - 1] = 1.
        elif len(axes) > 1 and order == 3:
            ax_list = [ax for ax in axes if ax is not None] + \
                      [ax for ax in axes[::-1] if ax is not None]
            scales = [1. / 2, ] * len(ax_list)
        elif order == -1:
            ax_list = axes[::-1]
            scales = [1.] * len(axes)
        else:  # elif order == 1:
            ax_list = axes
            scales = [1.] * len(axes)

        return ax_list, scales

    # @profile
    def get_force_advection(self, dt, inplace=False, method=None, background_force=True, internal_force=True,
                            split_order=2, compress=1, compress1=2, compress2=5, verbose_plot=False, **method_kwargs):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        print('boltzmann force advection', compress, compress1, compress2, dt)
        new_state = self if inplace else self.copy()
        if new_state.f is None or new_state.f.component is None or new_state.f.component.data is None:
            return new_state

        if method is None or method == '':
            deriv_v0 = new_state._calculate_time_derivative_f_force(background_force=background_force,
                                                                    internal_force=internal_force,
                                                                    compress1=compress1, compress2=compress2,
                                                                    verbose_plot=verbose_plot)
            dFdt_v = new_state.create_like(deriv_v0, recalc=False)
            new_state = new_state.euler(dt, deriv0=dFdt_v, inplace=True, compress_level=compress)

        elif method[:2] == 'rk' and ',' not in method:
            func = new_state._get_rk_func(method)
            state1 = func(dt, compress_level=compress, do_x_advection=False,
                          background_force=background_force, internal_force=internal_force)
            new_state.f = state1.f
            new_state.time = self.time + dt if self.time is not None else None

        elif method[:4] == 'tdvp':
            te_order = 0 if len(method) == 4 else int(method[-1])
            print('TDVP advec', new_state.v_axes, te_order)

            ## expand basis with euler
            # euler_state = new_state.get_force_advection(dt, inplace=False, method=None,
            #                                        background_force=True, internal_force=True,split_order=split_order,
            #                                        compress=compress, compress1=compress1, compress2=compress2,
            #                                        verbose_plot=False, **method_kwargs)

            new_state = new_state.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=False, # True,
                                                                       inplace=True, compress_level=compress,
                                                                       advec_axes=new_state.v_axes,
                                                                       background_force=background_force,
                                                                       internal_force=internal_force,
                                                                       # expand_basis=[euler_state.f.component],
                                                                       )
        elif method[:10] == 'split-tdvp':
            te_order = 0 if len(method) == 10 else int(method[-1])
            v_axes = new_state.v_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(v_axes, order=split_order)

            for ax, scale in zip(ax_list, scales):
                print('SPLIT TDVP', ax, scale, split_order)
                new_state = new_state.time_dependent_variational_principle(dt * scale, te_order=te_order, do_adapt=True,
                                                                           inplace=True, compress_level=compress,
                                                                           advec_axes=[ax],
                                                                           background_force=background_force,
                                                                           internal_force=internal_force)

        elif method[:9] == 'tdmrg_new':
            te_order = 0 if len(method) == 5 else int(method[-1])
            print('tdmrg advec', new_state.v_axes, te_order)
            new_state = new_state.time_dmrg_new(dt, te_order=te_order, do_adapt=True,
                                                inplace=True, compress_level=compress, advec_axes=new_state.v_axes,
                                                background_force=background_force, internal_force=internal_force)


        elif method[:5] == 'tdmrg':
            te_order = 0 if len(method) == 5 else int(method[-1])
            print('tdmrg advec', new_state.v_axes, te_order)
            new_state = new_state.time_dmrg(dt, te_order=te_order, do_adapt=True,
                                            inplace=True, compress_level=compress, advec_axes=new_state.v_axes,
                                            background_force=background_force, internal_force=internal_force)

        elif method[:11] == 'split-tdmrg':
            te_order = 0 if len(method) == 11 else int(method[-1])
            v_axes = new_state.v_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(v_axes, order=split_order)

            for ax, scale in zip(ax_list, scales):
                print('split-tdmrg force', ax)
                new_state = new_state.time_dmrg(dt * scale, te_order=te_order, do_adapt=True,
                                                inplace=True, compress_level=compress, advec_axes=[ax],
                                                background_force=background_force, internal_force=internal_force)


        elif method[:3] == 'two':
            new_state = new_state.two_step(dt, inplace=True, method=method, compress_level=compress,
                                           do_x_advection=False)
        elif ',' in method:  ## SL{x},{split_step str}

            ### SL + other split step
            method0, method1 = method.split(',')

            if True:  # method1[:2] == 'rk':
                new_state = new_state.get_force_advection(dt / 2, inplace=True, method=method0, background_force=True,
                                                          internal_force=False, split_order=split_order,
                                                          compress=compress, compress1=compress1, compress2=compress2)

                new_state = new_state.get_force_advection(dt, inplace=True, method=method1, background_force=False,
                                                          internal_force=True, compress=compress,
                                                          compress1=compress1, compress2=compress2)

                new_state = new_state.get_force_advection(dt / 2, inplace=True, method=method0, background_force=True,
                                                          internal_force=False, split_order=split_order,
                                                          compress=compress, compress1=compress1, compress2=compress2)

                # new_state = new_state.get_force_advection(dt / 2, inplace=True, method=method1, background_force=False,
                #                                           internal_force=True, compress=compress, compress1=compress1,
                #                                           compress2=compress2)

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

        elif method[:3] == 'SLd':  ## d-dimensional semilagrangian projection

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

                if force_term_C is None or force_term_C.data is None:
                    continue

                store_sl_mpo = background_force and not internal_force  ## internal force will vary with time

                new_state = func(v_ax, dt * scale, advec_coeffs=force_term_C, inplace=True,
                                 compress_level=compress, store_sl_mpo=store_sl_mpo, **kwargs, **method_kwargs)

        return new_state

    def get_vel_advection(self, dt, inplace=False, method=None, split_order=2, compress=1, compress1=4, compress2=5,
                          verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        # print('boltzmann vel advection', compress, compress1, compress2)
        state0 = self if inplace else self.copy()
        if state0.f is None or state0.f.component is None or state0.f.component.data is None:
            return state0

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

        elif method[:4] == 'tdvp':
            te_order = 0 if len(method) == 4 else int(method[-1])
            state0 = state0.time_dependent_variational_principle(dt, te_prder=te_order, do_adapt=True,
                                                                 inplace=True, compress_level=compress,
                                                                 advec_axes=state0.x_axes)
        elif method[:10] == 'split-tdvp':
            te_order = 0 if len(method) == 4 else int(method[-1])
            x_axes = state0.x_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(x_axes, order=split_order)

            for ax, scale in zip(ax_list, scales):
                state0 = state0.time_dependent_variational_principle(dt * scale, te_prder=te_order, do_adapt=True,
                                                                     inplace=True, compress_level=compress,
                                                                     advec_axes=[ax])

        elif method[:5] == 'tdmrg':
            te_order = 0 if len(method) == 5 else int(method[-1])
            state0 = state0.time_dmrg(dt, te_prder=te_order, do_adapt=True, inplace=True,
                                      compress_level=compress, advec_axes=state0.x_axes)

        elif method[:11] == 'split-tdmrg':
            te_order = 0 if len(method) == 11 else int(method[-1])
            x_axes = state0.x_axes
            ax_list, scales = Boltzmann._get_splitstep_axes_list(x_axes, order=split_order)

            for ax, scale in zip(ax_list, scales):
                print('split-tdmrg vel', ax)
                state0 = state0.time_dmrg(dt * scale, te_prder=te_order, do_adapt=True,
                                          inplace=True, compress_level=compress, advec_axes=[ax])

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
            ax_list, scales = Boltzmann._get_splitstep_axes_list(x_axes, order=split_order)

            # print('x axes', ax_list, scales)

            for x_ax, scale in zip(ax_list, scales):
                # assert(x_ax.basis.type==BasisType.SPATIAL), 'axes must be in spatial basis'

                # print('split step', x_ax, self.velocities[x_ax.coordinate])
                state0 = func(x_ax, dt * scale, advec_coeffs=self.velocities[x_ax.coordinate], inplace=True,
                              store_sl_mpo=True, compress_level=compress, **kwargs)

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
            # print('compute density', self.f.max_bond(), compress)
            # print('self.f', self.f.component)
            new_grid = self.grid_X if not isinstance(self.f.grid, GridTN1DComb) else None
            density = self.f.integrate(axes=v_axes, new_grid=new_grid,  # self.grid_X,
                                       new_ax_deriv_configs=self.f.component.ax_deriv_configs)

        if self.background_f is not None:
            if self.background_f.density is not None:
                self.background_f.compute_density(compress=5, store=True)
            density.add(self.background_f.density, inplace=True)

        # print('density', density.component)
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
            v_mpos = {self.coords_x.coords[i]: dist.grid.get_xmultiply_mpo([v_axes[i]]) for i in range(len(v_axes))}

            # print('compute flow')
            # for k, v in v_mpos.items():
            #     print('current',k,v)

            j = dist.meas_expec(v_mpos, integ_axes=v_axes, new_grid=new_grid,  # self.grid_X,
                                new_ax_deriv_configs=dist.component.ax_deriv_configs)

            # print('norm', dist.norm())
            #
            # plt.figure()
            # axes = dist.grid.axes
            # nx = dist.grid.axes[1].npts
            # ny = dist.grid.axes[2].npts
            # nz1, nz2 = dist.grid.axes[3].xpts[0], dist.grid.axes[3].xpts[-1]
            # plt.imshow(np.real(dist.get_comp_data(ax_select={axes[1]:nx//2, axes[2]: ny//2})),
            #            extent=(nz1, nz2, 0, 1), aspect='auto')
            # plt.show()
            #
            # for C in self.coords_x.coords:
            #     j_data = j.get_comp_data(C)
            #     plt.figure()
            #     if j_data is not None:
            #         plt.plot(np.real(j_data))
            #         plt.plot(np.imag(j_data), '--')
            #         plt.title(f'compute flow {C}')
            #     plt.show()

        if self.background_f is not None:
            if self.background_f.flow is None:
                self.background_f.compute_flow(compress=5, compress1=0, store=True)

            j.add(self.background_f.flow, inplace=True)

        verbose_plot = False
        # for C, jC in j.components.items():
        #     # if C != j.componentIDs[0]:  continue
        #     if jC is not None and jC.data is not None:
        #         plt.figure()
        #         jC_data = jC.get_data()
        #         ax_x, ax_y = j.grid.axes
        #         # jC_data = ax_x.basis.get_realspace_1D(jC_data, 0)
        #         # jC_data = ax_y.basis.get_realspace_1D(jC_data, 1)
        #         plt.imshow(np.imag(jC_data))
        #         plt.title(f'j{C}, grid {self.f.grid}')
        #         plt.colorbar()
        # plt.show()

        if verbose_plot:
            for C in self.coords_x.coords:
                plt.figure()
                j_data = j.get_comp_data(C)
                # plt.imshow(np.real(j_data))
                # plt.colorbar()
                plt.plot(np.real(j_data), label=f'{C} re')
                plt.plot(np.imag(j_data), '--', label=f'{C} im')
                plt.legend()
                plt.title(f'flow {C} (boltzmann)')

            ax_x, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
            dist_data = dist.get_comp_data(ax_select={ax_vy: ax_vy.npts // 2, ax_vz: ax_vz.npts // 2,
                                                      })  # ax_y: ax_y.npts//2})
            plt.figure()
            plt.imshow(np.real(dist_data))
            plt.title('dist (boltzmann) re')
            plt.xlabel('vz'), plt.ylabel('x')
            plt.colorbar()
            # plt.figure()
            # plt.imshow(np.imag(dist_data))
            # plt.title('dist (boltzmann) im')
            # plt.xlabel('vz'), plt.ylabel('x')
            # plt.colorbar()
            plt.show()

        if j is not None:
            if compress:
                j.compress(inplace=True, compress_level=compress)

        if store:
            self.flow = j

        return j

    def compute_moment(self, axes: Sequence['Axis'], powers: Sequence['int'] = None, integ_axes=None, compress=1):
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

    def get_time_derivative_op(self, time=None,
                               compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                               do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                               verbose_plot: bool = False, **kwargs) -> 'PDE_system':
        """ get G where G[f] = df/dt
            as a pde_system object
        """
        mpo = self.get_time_derivative_f_mpo(time=time, compress_level=compress_level, compress_level1=compress_level1,
                                             compress_level2=compress_level2, do_x_advectiom=do_x_advection,
                                             do_v_advection=do_v_advection, background_force=background_force,
                                             internal_force=internal_force, verbose_plot=verbose_plot, **kwargs)

        dfdt_op = self.f.create_like(new_components={SCALAR_COORD: mpo})
        dfdt_sys = self.create_like(dfdt_op, recalc=False)
        return dfdt_sys

    def get_time_derivative_f_mpo(self, time=None,
                                  compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  verbose_plot: bool = False, **kwargs) -> 'GridTN':
        """ get G where G[f] = df/dt
        """
        axes = []
        if do_x_advection:
            axes += self.grid_X.axes
        if do_v_advection:
            axes += [ax for ax in self.f.grid.axes if ax not in self.grid_X.axes]

        mpos = self._get_time_evolution_mpos(advec_axes=axes, background_force=background_force,
                                             internal_force=internal_force, **kwargs)

        dfdt_op = mpos[0].copy()
        for i in range(1, len(mpos)):
            tmp_comp_val = 0 if i == len(mpos) - 1 else compress_level1
            compress_opts = self.f.compress_config.get_compress_opts(tmp_comp_val)
            dfdt_op = dfdt_op.add(mpos[i], inplace=True, compress=tmp_comp_val, compress_opts=compress_opts)

        if compress_level:
            compress_opts = self.f.compress_config.get_compress_opts(compress_level)
            dfdt_op = dfdt_op.compress(inplace=True, compress_opts=compress_opts)

        return dfdt_op

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

            dist_data = dist.get_comp_data(ax_select={  # x_ax: x_ax.npts // 2,
                # y_ax: y_ax.npts // 2,
                vx_ax: vx_ax.npts // 2, vy_ax: 0, })
            if dist_data is not None:
                plt.figure()
                plt.imshow(np.real(dist_data))
                plt.colorbar()
                plt.xlabel('y')
                plt.ylabel('vz')
                plt.title('dist force')
                plt.show()

            v_grads_y = v_grads.get_comp_data(VY, ax_select={  # x_ax: x_ax.npts // 2,
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
                plt.plot(np.real(v_grads_y[:, vy_ax.npts // 2]))
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
        # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]

        mpo_list = self._get_time_evolution_mpos(advec_axes=x_axes)

        convective_term: Optional['GridTN'] = None
        for mpo in mpo_list:
            compress_opts = dist.compress_config.get_compress_opts(compress2)
            term_comp = dist.component.apply(mpo, compress=compress2, compress_opts=compress_opts)
            if convective_term is None:
                convective_term = term_comp.copy()
            else:
                convective_term.add(term_comp, compress=0, inplace=True)

        if convective_term is not None:
            if compress1:
                compress_opts = dist.compress_config.get(compress1)
                convective_term.compress(inplace=True, compress_opts=compress_opts)
            # convective_term.scalar_multiply(-1, inplace=True)

        return dist.create_like_scalar(convective_term)

    # @profile
    def _calculate_time_derivative_f_force(self, v_axes=None, v_grads=None, background_force=True, internal_force=True,
                                           compress1: int = 0, compress2: int = 0,
                                           verbose_plot: bool = False) -> Optional['Field']:

        dist = self.f

        if dist is None:
            return None

        v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords] if v_axes is None else v_axes

        mpo_list = self._get_time_evolution_mpos(advec_axes=v_axes, background_force=background_force,
                                                 internal_force=internal_force)
        # print('len mpos list', len(mpo_list))

        use_dmrg = False
        if use_dmrg:
            # print('calc time deriv force use dmrg')
            # compress_opts = dist.compress_config.get_compress_opts(compress2)
            # lorentz_terms = [dist.component.apply(mpo, zipup=self.zipup, compress=False,  # compress2,
            #                                       compress_opts=compress_opts) for mpo in mpo_list]
            # compress_opts = dist.compress_config.get(1)
            # print('dist compress config',dist.compress_config)
            # print('compress_opts', compress_opts, compress1)
            # lorentz_term = lorentz_terms[0].add_dmrg(*lorentz_terms[1:], compress_opts=compress_opts)

            print('calc time deriv force use dmrg sum_apply')
            compress_opts = dist.compress_config.get(1)
            lorentz_term = dist.component.sum_apply_dmrg(mpo_list, compress_opts=compress_opts)

        else:
            lorentz_term: Optional['GridTN'] = None
            for mpo in mpo_list:
                compress_opts = dist.compress_config.get_compress_opts(1) # compress2)
                # print('self.zipup f v', self.zipup)
                term_comp = dist.component.apply(mpo, zipup=self.zipup, compress=compress2, compress_opts=compress_opts)
                if lorentz_term is None:
                    lorentz_term = term_comp.copy()
                else:
                    lorentz_term.add(term_comp, compress=0, inplace=True)

            # if lorentz_term is not None:
            #     if compress1:
            #         compress_opts = dist.compress_config.get(compress1)
            #         lorentz_term.compress(inplace=True, compress_opts=compress_opts)
            #     # lorentz_term.scalar_multiply(-1, inplace=True)

        return dist.create_like_scalar(lorentz_term)

    # def _get_time_evolution_mpos(self, advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
    #                              get_collisions=False, **kwargs) -> list['GridTN']:
    #     """ get list of mpos needed to compute df/dt = -v grad(f) - F grad_v(f)
    #     """
    #     x_axes = self.coords_x.axes
    #     # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]
    #     v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}
    #     # if not self.upwind:
    #     #     v_axes_dict = None
    #     # else:
    #     #     v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}
    #
    #     dist = self.f
    #
    #     ## advection terms
    #     mpo_list = []
    #     for x_ax in x_axes:
    #         if x_ax is None:  continue
    #         if advec_axes is not None and x_ax not in advec_axes:  continue
    #         v_ax = v_axes_dict[x_ax]
    #         upwind_ax = v_ax if self.upwind else None
    #         print('x advection upwind ax', upwind_ax)
    #
    #         deriv_config = dist.component.ax_deriv_configs[x_ax]
    #         ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config, upwind_ax=upwind_ax)
    #         # vel = self.velocities[x_ax.coordinate]
    #         vel_mpo = v_ax.get_xmultiply_mpo()
    #         x_advec_mpo = dist.grid.build_mpo_from_subgtns([ddx_gtn, ([v_ax], vel_mpo)])
    #         x_advec_mpo.scalar_multiply(-1, inplace=True)
    #
    #         if x_advec_mpo is not None:
    #             mpo_list += [x_advec_mpo]
    #
    #     ## force terms
    #     em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)
    #     if em_term is not None:
    #         for x_coord in self.coords_x.coords:
    #             v_ax = self.v_axes_dict.get(x_coord)
    #             if v_ax is None:  continue
    #             if advec_axes is not None and v_ax not in advec_axes:  continue
    #
    #             if False:  # self.upwind:
    #                 print('upwind', self.upwind)
    #                 # print('WARNING: does not seem to work well?')
    #                 force_component = em_term[x_coord]
    #                 force_norm = force_component.frobenius_norm() / np.sqrt(force_component.grid.npts)
    #                 threshold =  0.1 * force_norm
    #                 print('force norm', force_norm, 'threshold', threshold)
    #
    #                 mask_pos = force_component.evaluate_func(lambda x: (x >  threshold), inplace=False, max_bond=32)
    #                 mask_neg = force_component.evaluate_func(lambda x: (x < -threshold), inplace=False, max_bond=32)
    #                 mask_cen = force_component.grid.get_ones_mps()
    #                 mask_cen = mask_cen.add( mask_pos.scalar_multiply(-1) ).add( mask_neg.scalar_multiply(-1) )
    #                 # print('bond dims', mask_pos.max_bond(), mask_neg.max_bond(), mask_neg.L)
    #                 # mask_neg = force_component.grid.get_ones_mps().add( mask_pos.scalar_multiply(-1) )
    #
    #                 # force_data = force_component.get_data()
    #                 #
    #                 # plt.figure()
    #                 # plt.plot(mask_neg.get_data()[0, :], label='neg cross')
    #                 # plt.plot(mask_pos.get_data()[0, :], label='pos cross')
    #                 # plt.plot(force_data[0, :], 'x--', label='data')
    #                 # plt.plot(force_data[0, :] > 0.5, '--', label='data > 0.5')
    #                 # plt.plot(force_data[0, :] < -0.5, '--', label='data < -0.5')
    #                 # plt.legend()
    #                 # plt.show()
    #
    #                 # plt.figure()
    #                 # plt.imshow(mask_neg.get_data())
    #                 # plt.colorbar()
    #                 # plt.title('maks neg 1')
    #                 # plt.figure()
    #                 # plt.imshow(mask_pos.get_data())
    #                 # plt.colorbar()
    #                 # plt.title('maks pos 1')
    #
    #                 # plt.figure()
    #                 # plt.imshow(force_data * (force_data > 0))
    #                 # plt.title('data')
    #                 # plt.colorbar()
    #
    #                 # print('diff', np.linalg.norm(mask_pos.get_data() - force_data > 0))
    #                 # print('diff', np.linalg.norm(mask_neg.get_data() - force_data <= 0))
    #
    #                 # force_data = force_component.get_data()
    #                 # mask_pos_data = force_data > 0
    #                 # mask_neg_data = force_data <= 0
    #                 # mask_pos = force_component.create_like(mask_pos_data)
    #                 # mask_neg = force_component.create_like(mask_neg_data)
    #
    #                 # plt.figure()
    #                 # plt.imshow(mask_neg.get_data())
    #                 # plt.colorbar()
    #                 # plt.title('maks neg')
    #                 # plt.figure()
    #                 # plt.imshow(mask_pos.get_data())
    #                 # plt.colorbar()
    #                 # plt.title('maks pos')
    #                 # plt.show()
    #
    #                 mask_pos = mask_pos.apply_elemental_multiply_op()
    #                 mask_neg = mask_neg.apply_elemental_multiply_op()
    #                 mask_cen = mask_cen.apply_elemental_multiply_op()
    #
    #                 op = force_component.apply_elemental_multiply_op()
    #                 op_pos = op.apply(mask_pos, zipup=self.zipup)
    #                 op_neg = op.apply(mask_neg, zipup=self.zipup)
    #                 op_cen = op.apply(mask_cen, zipup=self.zipup)
    #
    #                 deriv_config = dist.component.ax_deriv_configs[v_ax]
    #                 deriv_config.fd_type = FDType.BACKWARD  # BACKWARD
    #                 ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
    #                 v_advec_mpo = dist.grid.build_mpo_from_subgtns([([v_ax], ddv_mpo), op_pos])
    #                 v_advec_mpo.scalar_multiply(-1, inplace=True)
    #
    #                 if v_advec_mpo is not None:
    #                     mpo_list += [v_advec_mpo]
    #
    #                 deriv_config.fd_type = FDType.FORWARD  # FORWARD
    #                 ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
    #                 v_advec_mpo = dist.grid.build_mpo_from_subgtns([([v_ax], ddv_mpo), op_neg])
    #                 v_advec_mpo.scalar_multiply(-1, inplace=True)
    #
    #                 if v_advec_mpo is not None:
    #                     mpo_list += [v_advec_mpo]
    #
    #                 deriv_config.fd_type = FDType.CENTER  # CENTER
    #                 ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
    #                 v_advec_mpo = dist.grid.build_mpo_from_subgtns([([v_ax], ddv_mpo), op_cen])
    #                 v_advec_mpo.scalar_multiply(-1, inplace=True)
    #
    #                 if v_advec_mpo is not None:
    #                     mpo_list += [v_advec_mpo]
    #
    #             else:
    #                 force_component = em_term[x_coord]
    #                 # print('force_component', x_coord, force_component.data)
    #                 if force_component is None or force_component.data is None:
    #                     print('force is None', x_coord)
    #                     continue
    #
    #                 ## for k-space
    #                 # force_data = force_component.get_data()
    #                 # print('is H?', np.linalg.norm(force_data[1:] - force_data[:0:-1].conj()))
    #
    #                 force_component = force_component.apply_elemental_multiply_op()
    #
    #                 # force_mpo_data = force_component.get_data()
    #                 # print('is H?', np.linalg.norm(force_mpo_data - force_mpo_data.T.conj()))
    #
    #                 deriv_config = dist.component.ax_deriv_configs[v_ax]
    #                 ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
    #                 v_advec_mpo = dist.grid.build_mpo_from_subgtns([force_component, ([v_ax], ddv_mpo)])
    #                 v_advec_mpo.scalar_multiply(-1, inplace=True)
    #                 # print('v_ax', v_ax, v_advec_mpo)
    #                 if v_advec_mpo is not None:
    #                     mpo_list += [v_advec_mpo]
    #
    #
    #     ## collision terms
    #     if get_collisions:
    #         print('mpo list get collisions')
    #
    #         # x_axes = self.coords_x.axes
    #         # v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]
    #         v_axes = self.v_axes
    #
    #         if self.collision.coll_type is None:
    #             pass
    #
    #         elif self.collision.coll_type == CollisionType.LB:
    #
    #             ############  get maxwellian ######
    #             if dist.is_sqrt:
    #                 print('dist is sqrt')
    #                 vth2 = 2 * self.matl_params.vth ** 2
    #             else:
    #                 vth2 = self.matl_params.vth ** 2
    #             coll_rate = self.collision.coll_rate
    #
    #             ## beta * d/dv \cdot ((v-u_sr) f_s + v_th^2 df/dv)
    #             for v_ax in v_axes:
    #                 deriv_config = dist.component.ax_deriv_configs[v_ax]
    #
    #                 ddv = dist.grid.get_firstderivative_mpo(v_ax, deriv_config=deriv_config)
    #                 v_with_off = dist.grid.get_xmultiply_mpo([v_ax], offsets=-self.collision.v0)
    #                 ddv_v = v_with_off.apply(ddv)
    #                 ddv_v = ddv_v.scalar_multiply(coll_rate)
    #
    #                 d2dv2 = dist.grid.get_secondderivative_mpo(v_ax, v_ax, deriv_config1=deriv_config)
    #                 d2dv2 = d2dv2.scalar_multiply(vth2 * coll_rate)
    #
    #                 print('add collisions to mpo_list')
    #                 mpo_list += [ddv_v, d2dv2]
    #
    #     return mpo_list

    def _get_time_evolution_mpos(self, advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                                 get_collisions=False, ax_deriv_configs=None, **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = -v grad(f) - F grad_v(f)
        """
        x_axes = self.coords_x.axes
        # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]
        v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}
        # if not self.upwind:
        #     v_axes_dict = None
        # else:
        #     v_axes_dict = {ax: self.v_axes_dict[ax.coordinate] for ax in x_axes}

        dist = self.f
        ax_deriv_configs = dist.component.ax_deriv_configs if ax_deriv_configs is None else ax_deriv_configs

        ## advection terms
        mpo_list = []
        for x_ax in x_axes:
            if x_ax is None:  continue
            if advec_axes is not None and x_ax not in advec_axes:  continue
            v_ax = v_axes_dict[x_ax]
            upwind_ax = v_ax if self.upwind else None
            print('x advection upwind ax', upwind_ax)

            deriv_config = ax_deriv_configs[x_ax]
            ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config, upwind_ax=upwind_ax)
            # vel = self.velocities[x_ax.coordinate]
            vel_mpo = v_ax.get_xmultiply_mpo()
            x_advec_mpo = dist.grid.build_mpo_from_subgtns([ddx_gtn, ([v_ax], vel_mpo)])
            x_advec_mpo.scalar_multiply(-1, inplace=True)

            if x_advec_mpo is not None:
                mpo_list += [x_advec_mpo]

        ## force terms
        em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)
        if em_term is not None:
            for x_coord in self.coords_x.coords:
                v_ax = self.v_axes_dict.get(x_coord)
                if v_ax is None:  continue
                if advec_axes is not None and v_ax not in advec_axes:  continue

                force_component = em_term[x_coord]
                # print('force_component', x_coord, force_component.data)
                if force_component is None or force_component.data is None:
                    print('force is None', x_coord)
                    continue

                ## for k-space
                # force_data = force_component.get_data()
                # print('is H?', np.linalg.norm(force_data[1:] - force_data[:0:-1].conj()))

                force_component = force_component.apply_elemental_multiply_op()

                # force_mpo_data = force_component.get_data()
                # print('is H?', np.linalg.norm(force_mpo_data - force_mpo_data.T.conj()))

                deriv_config = dist.component.ax_deriv_configs[v_ax]
                ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
                v_advec_mpo = dist.grid.build_mpo_from_subgtns([force_component, ([v_ax], ddv_mpo)])
                v_advec_mpo.scalar_multiply(-1, inplace=True)
                # print('v_ax', v_ax, v_advec_mpo)
                if v_advec_mpo is not None:
                    mpo_list += [v_advec_mpo]


        ## collision terms
        if get_collisions:
            print('mpo list get collisions')

            # x_axes = self.coords_x.axes
            # v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]
            v_axes = self.v_axes

            if self.collision.coll_type is None:
                pass

            elif self.collision.coll_type == CollisionType.LB:

                ############  get maxwellian ######
                if dist.is_sqrt:
                    print('dist is sqrt')
                    vth2 = 2 * self.matl_params.vth ** 2
                else:
                    vth2 = self.matl_params.vth ** 2
                coll_rate = self.collision.coll_rate

                ## beta * d/dv \cdot ((v-u_sr) f_s + v_th^2 df/dv)
                for v_ax in v_axes:
                    deriv_config = dist.component.ax_deriv_configs[v_ax]

                    ddv = dist.grid.get_firstderivative_mpo(v_ax, deriv_config=deriv_config)
                    v_with_off = dist.grid.get_xmultiply_mpo([v_ax], offsets=-self.collision.v0)
                    ddv_v = v_with_off.apply(ddv)
                    ddv_v = ddv_v.scalar_multiply(coll_rate)

                    d2dv2 = dist.grid.get_secondderivative_mpo(v_ax, v_ax, deriv_config1=deriv_config)
                    d2dv2 = d2dv2.scalar_multiply(vth2 * coll_rate)

                    print('add collisions to mpo_list')
                    mpo_list += [ddv_v, d2dv2]

        for m in mpo_list:
            m.data.mangle_inner_()

        return mpo_list


    def dynamical_low_rank(self, dt: Numeric, te_order=4, do_adapt: bool = True,
                           inplace: bool = False, compress_level: int = 1, compress_level_2: int = 4,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
        """
        new_state = self if inplace else self.copy()

        dist_gtn = new_state.f.component
        compress_config = new_state.f.compress_config
        max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force)

        ## project by subspaces defined by each axis
        bond_ind = -1
        for ax in dist_gtn.grid.axes:  ## requires sequential geometry
            if bond_ind > 0:
                print('BACKPROP bond ind', bond_ind, dist_gtn.norm(is_sqrt=True))
                dist_gtn = dist_gtn.convert_to_USVT(canon_site=bond_ind, inplace=True)
                # print('canon?')
                # helper.check_orthog(dist_gtn.data)
                helper_dlr.dlr_bond_time_evolution(dist_gtn, -dt / 2, mpo_list, bond_ind, te_order=1,
                                                   inplace=True, canonize=False, )
                dist_gtn = dist_gtn.convert_from_USVT(inplace=True)
                # print('done backprop', dist_gtn.norm(is_sqrt=True))
                # exit()

            if ax != dist_gtn.grid.axes[-1]:
                bond_ind += ax.L

            dist_gtn = dist_gtn.canonize_axes([ax], inplace=True, scale=False)
            # print('canon?', ax)
            # helper.check_orthog(dist_gtn.data)
            # print('bond ind size', bond_ind, dist_gtn.data.bond_size(bond_ind, bond_ind+1))
            adapt = do_adapt and dist_gtn.data.bond_size(bond_ind, bond_ind + 1) < max_bond
            print('FORWARD PROP ax', ax, ax.L, adapt)
            helper_dlr.dlr_subspace_time_evolution(dist_gtn, dt / 2, mpo_list, [ax], te_order=te_order,
                                                   adapt=adapt, canonize=False, inplace=True,
                                                   compress_level=compress_level, compress_level_2=compress_level_2,
                                                   compress_opts_dict=compress_config)
            print('done forward prop')
            # print('distgtn', dist_gtn)

        # exit()

        ## reverse sweep
        bond_ind = dist_gtn.grid.L - 1
        for ax in dist_gtn.grid.axes[::-1]:  ## requires sequential geometry

            if bond_ind < dist_gtn.grid.L - 1:
                print('BACKPROP (-1) bond ind', bond_ind)
                dist_gtn.convert_to_USVT(canon_site=bond_ind, inplace=True)
                helper_dlr.dlr_bond_time_evolution(dist_gtn, -dt / 2, mpo_list, bond_ind, te_order=1,
                                                   inplace=True, canonize=False, )
                dist_gtn.convert_from_USVT(inplace=True)
                print('done back prop')

            if ax != dist_gtn.grid.axes[0]:
                bond_ind -= ax.L
            # print('bond ind', bond_ind)

            dist_gtn = dist_gtn.canonize_axes([ax], inplace=True, scale=False)
            adapt = do_adapt and dist_gtn.data.bond_size(bond_ind, bond_ind + 1) < max_bond
            print('FORWARD PROP (-1) ax', ax, ax.L, adapt)
            helper_dlr.dlr_subspace_time_evolution(dist_gtn, dt / 2, mpo_list, [ax], te_order=te_order,
                                                   adapt=adapt, canonize=False, inplace=True,
                                                   compress_level=compress_level, compress_level_2=compress_level_2,
                                                   compress_opts_dict=compress_config)
            print('done forward prop', dist_gtn.max_bond())

        return new_state

    # @profile
    def time_dependent_variational_principle(self, dt: Numeric, te_order=4, do_adapt: bool = True,
                                             inplace: bool = False,
                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                             advec_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True,
                                             expand_basis: Sequence['GridTN'] = None,
                                             **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force,
                                                 get_collisions=True)

        # print('mpo bond dim', [m.max_bond() for m in mpo_list])
        # dist_gtn_copy = dist_gtn.copy()
        dist_gtn.evolve_tdvp(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                             compress_config=compress_config, expand_basis=expand_basis)
        # print('distance tdvp', dist_gtn.distance(dist_gtn_copy), dist_gtn_copy.norm(is_sqrt=new_state.f.is_sqrt),
        #       dist_gtn.norm(is_sqrt=new_state.f.is_sqrt))
        # print('distance tdvp', new_state.f.component.distance(dist_gtn_copy))

        print('done boltzmann tdvp')
        return new_state

    def time_dmrg(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                  solver_type=LocalSolverType.TDDMRG, **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        print('boltzmann time dmrg', solver_type)

        if solver_type == LocalSolverType.TDCross:
            return self.time_dmrg_cross(dt, te_order, do_adapt=do_adapt, inplace=inplace,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force)

        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force,
                                                 get_collisions=True)

        dist_gtn.evolve_tdmrg(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                              compress_config=compress_config)
        # print('distance tdvp', dist_gtn.distance(dist_gtn_copy), dist_gtn_copy.norm(is_sqrt=new_state.f.is_sqrt),
        #       dist_gtn.norm(is_sqrt=new_state.f.is_sqrt))
        # print('distance tdvp', new_state.f.component.distance(dist_gtn_copy))

        print('done boltzmann tdmrg')
        return new_state


    def time_dmrg_cross(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                        # compress_level: int = 1, compress_level_2: int = 4, direction=1,
                        advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                        solver_type=LocalSolverType.TDCross, **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        dist_gtn.data.distribute_exponent()
        dist_gtn_copy = dist_gtn.copy()
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        print('td cross', advec_axes)
        advec_axes = self.f.grid.axes if advec_axes is None else advec_axes
        # mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
        #                                          internal_force=internal_force,
        #                                          get_collisions=True)
        # out_gtn = dist_gtn.evolve_tdmrg_new(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
        #                                     compress_config=compress_config, solver_type=LocalSolverType.TDCross,
        #                                     verbose_plot=self.verbose_plot, )

        ax_deriv_configs = self.f.component.ax_deriv_configs
        f_ax_deriv_configs = {k: v.copy() for k, v in ax_deriv_configs.items()}
        b_ax_deriv_configs = {k: v.copy() for k, v in ax_deriv_configs.items()}
        for k, v in f_ax_deriv_configs.items():
            v.update(fd_type=FDType.FORWARD, order=0)

        for k, v in b_ax_deriv_configs.items():
            v.update(fd_type=FDType.BACKWARD, order=0)

        grid = self.f.grid
        fd_mpo_dict = {('FD', ax): [grid.get_firstderivative_mpo(ax, deriv_config=f_ax_deriv_configs[ax]).data]
                       for ax in advec_axes}
        bd_mpo_dict = {('BD', ax): [grid.get_firstderivative_mpo(ax, deriv_config=b_ax_deriv_configs[ax]).data]
                       for ax in advec_axes}

        em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)
        coeff_dict = {}
        for x_coord in self.coords_x.coords:
            v_ax = self.v_axes_dict.get(x_coord)
            force_comp = em_term[x_coord]
            if v_ax is None or force_comp is None:
                continue
            coeff_dict[('fcoeff', v_ax)] = [force_comp.data]

        def upwind_func(dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                        upwind_terms_dict: dict[Any, 'qtn.Tensor'], ref_deriv:'qtn.Tensor'=None,
                        left_site_pos=None, nsites=None, select_inds=None):
            return self.euler_upwind_v4(dt, ket, submat, selectors, upwind_terms_dict, ref_deriv=ref_deriv,
                                        left_site_pos=left_site_pos, order=te_order,
                                        internal_force=internal_force, background_force=background_force )

        out_gtn = dist_gtn.evolve_tdmrg_new(dt, [], te_order=te_order, do_adapt=do_adapt, inplace=True,
                                            compress_config=compress_config, solver_type=LocalSolverType.TDCross,
                                            upwind_mpo_list={**fd_mpo_dict, **bd_mpo_dict, **coeff_dict},
                                            upwind_func=upwind_func, verbose_plot=self.verbose_plot,)

        print('distance tdcross', dist_gtn.distance(dist_gtn_copy), dist_gtn_copy.norm(is_sqrt=new_state.f.is_sqrt),
              dist_gtn.norm(is_sqrt=new_state.f.is_sqrt))
        print('distance tdcross', new_state.f.component.distance(dist_gtn_copy))

        new_state.f.component.data = out_gtn.data
        print('done boltzmann td cross (upwind)')

        return new_state


    def tdvp_new(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                 compress_level: int = 1, compress_level_2: int = 4, direction=1,
                 advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                 **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force,
                                                 get_collisions=True)

        # print('mpo bond dim', [m.max_bond() for m in mpo_list])
        print('boltzmann tdvp new', te_order)
        dist_gtn.evolve_tdvp_new(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                 compress_config=compress_config, time=self.time)

        print('done boltzmann tdvp new')
        return new_state


    def time_dmrg_new(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                      compress_level: int = 1, compress_level_2: int = 4, direction=1,
                      advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                      solver_type=LocalSolverType.TDDMRG, **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        if solver_type == LocalSolverType.TDCross:
            return self.time_dmrg_cross(dt, te_order, do_adapt=do_adapt, inplace=inplace,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force)

        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force,
                                                 get_collisions=True)

        # print('mpo bond dim', [m.max_bond() for m in mpo_list])
        print('boltzmann time dmrg', te_order)
        dist_gtn.evolve_tdmrg_new(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                  compress_config=compress_config, time=self.time, solver_type=solver_type,
                                  **kwargs)

        print('done boltzmann tdmrg')
        return new_state


    def time_local_global(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                          compress_level: int = 1, compress_level_2: int = 4, direction=1,
                          advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                          **kwargs) -> 'Boltzmann':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force,
                                                 get_collisions=True)

        # print('mpo bond dim', [m.max_bond() for m in mpo_list])
        print('boltzmann time dmrg', te_order)
        dist_gtn.evolve_time_local_global(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                          compress_config=compress_config, time=self.time)

        print('done boltzmann tdmrg')
        return new_state

    def euler_upwind(self, dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                     upwind_terms: dict[Any, 'qtn.Tensor'], ref_deriv: 'qtn.Tensor'=None, left_site_pos: int=None,
                     order=1, do_x_advection=True, do_v_advection=True, internal_force=True, background_force=True,
                     **kwargs) -> 'qtn.Tensor':
        """ perform a single time-step update at the specified indices
            upwinding for advection: df/dt + c d/dx f = 0
            if c_i > 0:
                backward difference:  u_i(t+dt) = u_i(t) - c_i (u_{i}(t) - u_{i-1}(t)) / dx * dt
            else c_i < 0:
                forward difference:   u_i(t+dt) = u_i(t) - c_i (u_{i+1}(t) - u_{i}(t)) / dx * dt
        """

        if True:
            return self.euler_upwind_v3(dt, ket, submat, selectors, upwind_terms, ref_deriv=ref_deriv,
                                        left_site_pos=left_site_pos, order=order,
                                        do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                                        internal_force=internal_force, background_force=background_force, **kwargs)

        grid = self.f.grid
        # axes = self.f.grid.axes
        ax_deriv_configs = self.f.component.ax_deriv_configs
        charge, mass = self.matl_params.charge, self.matl_params.mass

        submat_data = submat.data.reshape(-1)
        out_data = np.zeros(submat_data.shape)

        order = 2

        def _upwind_deriv(ax: 'Axis', coeff: Numeric, inds: Sequence[int]):
            """ inds: binary string
            """
            idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
            val1 = grid.meas_elem_bin(ket, idx1) * sign1
            idx1_, sign1_ = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
            val1_ = grid.meas_elem_bin(ket, idx1_) * sign1_
            # deriv = - (val1 - val1_) / 2 / ax.dx
            if ax.coordinate == self.coords_v.coords[0]:
                deriv = - (val1 - val1_) / 2 / ax.dx
            elif ax.coordinate == self.coords_v.coords[1]:
                deriv = - (val1 - val1_) / 2 / ax.dx
            else:
                raise ValueError
            return deriv


        # def _upwind_deriv(ax: 'Axis', coeff: Numeric, inds: Sequence[int]):
        #     """ inds: binary string
        #     """
        #     if coeff < 0:
        #         if order == 1:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             deriv = (val0 - val1) / ax.dx
        #         elif order == 2:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
        #             idx2, sign2 = grid.get_shifted_index_and_sign(inds, {ax: -2}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             val2 = grid.meas_elem_bin(ket, idx2) * sign2
        #             deriv = (3 * val0 - 4 * val1 + val2) / 2 / ax.dx
        #         else:
        #             raise NotImplementedError
        #     else:
        #         if order == 1:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             deriv = (val1 - val0) / ax.dx
        #         elif order == 2:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
        #             idx2, sign2 = grid.get_shifted_index_and_sign(inds, {ax: 2}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             val2 = grid.meas_elem_bin(ket, idx2) * sign2
        #             deriv = (-val2 + 4 * val1 - 3 * val0) / 2 / ax.dx
        #         else:
        #             raise NotImplementedError
        #
        #     # print('ind0', inds)
        #     # print('idx1', idx1)
        #     #
        #     # print(grid.get_axis_positions(inds))
        #     # print(grid.get_axis_positions(idx1))
        #     #
        #     # print(self.f.component.meas_elem(inds), grid.meas_elem_bin(ket, inds))
        #     # print(self.f.component.meas_elem(idx1), grid.meas_elem_bin(ket, idx1))
        #
        #     # exit()
        #
        #     return deriv


        for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):

            # print('ind0', ix, ind, val0)

            sel_inds = [int(x) for x in bin(ind)[2:]]
            sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds
            ax_index = grid.get_axis_positions(sel_inds)
            # print('ax index', ax_index)
            tot_deriv_0 = 0

            if do_x_advection:

                for x_ax in self.x_axes:
                    v_ax = self.v_axes_dict[x_ax.coordinate]
                    v_coeff = v_ax.x0 + v_ax.dx * ax_index[v_ax]
                    deriv = _upwind_deriv(x_ax, v_coeff, sel_inds)

                    tot_deriv_0 += deriv * v_coeff

            if do_v_advection:

                em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)

                for x_coord in self.coords_x.coords:
                    v_ax = self.v_axes_dict.get(x_coord)
                    if v_ax is None:  continue
                    if em_term is None:  continue

                    force_component = em_term[x_coord]
                    # print('force_component', x_coord, force_component.data)
                    if force_component is None or force_component.data is None:
                        print('force is None', x_coord)
                        continue

                    f_coeff = force_component.meas_elem(sel_inds) * charge / mass
                    deriv = _upwind_deriv(v_ax, f_coeff, sel_inds)

                    tot_deriv_0 += deriv * f_coeff

            out_data[ix] = val0 - tot_deriv_0 * dt

        # print('submat data', submat_data)
        # print('boltzmann upwind', out_data)
        # print('upwind mat diff', np.linalg.norm(out_data - submat_data))
        # exit()

        out_tens = submat.copy()
        out_tens.modify(data=out_data.reshape(submat.shape))
        return out_tens


    def euler_upwind_v2(self, dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                        upwind_terms : dict[Any, 'qtn.Tensor'] = None, order=1, do_x_advection=True,
                        do_v_advection=True, internal_force=True, background_force=True, **kwargs) -> 'qtn.Tensor':
        """ perform a single time-step update at the specified indices
            upwinding for advection: df/dt + c d/dx f = 0
            if c_i > 0:
                backward difference:  u_i(t+dt) = u_i(t) - c_i (u_{i}(t) - u_{i-1}(t)) / dx * dt
            else c_i < 0:
                forward difference:   u_i(t+dt) = u_i(t) - c_i (u_{i+1}(t) - u_{i}(t)) / dx * dt

            explicitly compute D[x] * f, and measure points from that.
        """
        grid = self.f.grid
        # axes = self.f.grid.axes
        ax_deriv_configs = self.f.component.ax_deriv_configs
        charge, mass = self.matl_params.charge, self.matl_params.mass

        submat_data = submat.data.reshape(-1)
        out_data = np.zeros(submat_data.shape)

        # order = 2

        f_ax_deriv_configs = {k: v.copy() for k, v in ax_deriv_configs.items()}
        b_ax_deriv_configs = {k: v.copy() for k, v in ax_deriv_configs.items()}
        # for k, v in f_ax_deriv_configs.items():
        #     v.update(fd_type=FDType.FORWARD)

        # for k, v in b_ax_deriv_configs.items():
        #     v.update(fd_type=FDType.BACKWARD)

        ket_gtn = self.f.component.create_like(new_data=ket)
        fd_deriv_ax = {ax: ket_gtn.take_firstderivative(ax, ax_deriv_config=f_ax_deriv_configs) for ax in grid.axes}
        bd_deriv_ax = {ax: ket_gtn.take_firstderivative(ax, ax_deriv_config=b_ax_deriv_configs) for ax in grid.axes}

        def _upwind_deriv(ax: 'Axis', coeff: Numeric, inds: Sequence[int]):
            """ inds: binary string
            """
            fd_ket = fd_deriv_ax[ax]
            bd_ket = fd_deriv_ax[ax]

            if coeff < 0:
                deriv = fd_ket.meas_elem(inds)  # grid.meas_elem_bin(fd_ket, inds)
            else:
                deriv = bd_ket.meas_elem(inds)  # grid.meas_elem_bin(bd_ket, inds)

            return deriv


        for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):

            sel_inds = [int(x) for x in bin(ind)[2:]]
            sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds
            ax_index = grid.get_axis_positions(sel_inds)
            # print('ax index', ax_index)
            tot_deriv_0 = 0

            if do_x_advection:

                for x_ax in self.x_axes:
                    v_ax = self.v_axes_dict[x_ax.coordinate]
                    v_coeff = v_ax.x0 + v_ax.dx * ax_index[v_ax]
                    deriv = _upwind_deriv(x_ax, v_coeff, sel_inds)

                    tot_deriv_0 += deriv * v_coeff

            if do_v_advection:

                em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)

                for x_coord in self.coords_x.coords:
                    v_ax = self.v_axes_dict.get(x_coord)
                    if v_ax is None:  continue
                    if em_term is None:  continue

                    force_component = em_term[x_coord]
                    # print('force_component', x_coord, force_component.data)
                    if force_component is None or force_component.data is None:
                        print('force is None', x_coord)
                        continue

                    f_coeff = force_component.meas_elem(sel_inds) * charge / mass
                    deriv = _upwind_deriv(v_ax, f_coeff, sel_inds)

                    tot_deriv_0 += deriv * f_coeff

            out_data[ix] = val0 - tot_deriv_0 * dt

        # print('submat data', submat_data)
        # print('boltzmann upwind', out_data)
        # print('upwind mat diff', np.linalg.norm(out_data - submat_data))
        # exit()

        out_tens = submat.copy()
        out_tens.modify(data=out_data.reshape(submat.shape))
        return out_tens


    def euler_upwind_v3(self, dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                        upwind_terms: dict[Any, 'qtn.Tensor'], ref_deriv: 'qtn.Tensor'=None,
                        left_site_pos:int = None, order=1, do_x_advection=True,
                        do_v_advection=True, internal_force=True, background_force=True, **kwargs) -> 'qtn.Tensor':
        """ perform a single time-step update at the specified indices
            upwinding for advection: df/dt + c d/dx f = 0
            if c_i > 0:
                backward difference:  u_i(t+dt) = u_i(t) - c_i (u_{i}(t) - u_{i-1}(t)) / dx * dt
            else c_i < 0:
                forward difference:   u_i(t+dt) = u_i(t) - c_i (u_{i+1}(t) - u_{i}(t)) / dx * dt

            explicitly compute D[x] * f, and measure points from that.
        """
        grid = self.f.grid
        # axes = self.f.grid.axes
        # ax_deriv_configs = self.f.component.ax_deriv_configs
        charge, mass = self.matl_params.charge, self.matl_params.mass

        submat_data = submat.data.reshape(-1)
        npts = len(submat_data)

        tot_deriv_0 = np.zeros((npts,))
        if do_x_advection:

            for x_ax in self.x_axes:
                fd_deriv_key = ('FD', x_ax)
                bd_deriv_key = ('BD', x_ax)

                ## already transposed ot match submat
                fd_deriv_submat = upwind_terms[fd_deriv_key].data.reshape(-1)
                bd_deriv_submat = upwind_terms[bd_deriv_key].data.reshape(-1)

                v_coeffs = np.zeros((npts,))
                v_ax = self.v_axes_dict[x_ax.coordinate]
                for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):
                    sel_inds = [int(x) for x in bin(ind)[2:]]
                    sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds

                    ax_index = grid.get_axis_positions(sel_inds)
                    v_coeff = v_ax.x0 + v_ax.dx * ax_index[v_ax]
                    v_coeffs[ix] = v_coeff

                deriv = np.where(v_coeffs > 0, bd_deriv_submat, fd_deriv_submat)
                # deriv = fd_deriv_submat
                tot_deriv_0 += deriv * v_coeffs

        if do_v_advection:

            em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)

            for x_coord in self.coords_x.coords:
                v_ax = self.v_axes_dict.get(x_coord)
                if v_ax is None:  continue
                if em_term is None:  continue

                fd_deriv_key = ('FD', v_ax)
                bd_deriv_key = ('BD', v_ax)

                ## already transposed ot match submat
                fd_deriv_submat = upwind_terms[fd_deriv_key].data.reshape(-1)
                bd_deriv_submat = upwind_terms[bd_deriv_key].data.reshape(-1)

                force_component = em_term[x_coord]
                # print('force_component', x_coord, force_component.data)
                if force_component is None or force_component.data is None:
                    print('force is None', x_coord)
                    continue

                num_proc = 6
                num_proc = min(num_proc, npts)
                parts = [m * npts // num_proc for m in range(num_proc)] + [npts]
                f_coeffs = np.zeros((npts,))

                # f_coeffs = np.zeros((npts,))
                # for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):
                #     sel_inds = [int(x) for x in bin(ind)[2:]]
                #     sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds
                #
                #     f_coeff = force_component.meas_elem(sel_inds) * charge / mass
                #     f_coeffs[ix] = f_coeff

                def f_eval_func(out_data: 'mp.Array'):
                    ix0 = int(out_data[0])
                    for ix in out_data:
                        ix = int(ix)
                        ind = selectors[ix]

                        sel_inds = [int(x) for x in bin(ind)[2:]]
                        sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds

                        f_coeff = force_component.meas_elem(sel_inds) * charge / mass
                        out_data[ix - ix0] = f_coeff
                    return ix0, out_data

                processes = []
                outputs = []
                for nn in range(num_proc):
                    print('nn', nn)
                    out_array_xx = mp.Array('d', range(parts[nn], parts[nn + 1]))

                    p = mp.Process(target=f_eval_func, args=(out_array_xx,))  # , submat_array, sel_array))
                    p.start()
                    processes += [p]
                    outputs += [out_array_xx]

                for p in processes:
                    p.join()

                for nn, out_array_xx in enumerate(outputs):
                    f_coeffs[parts[nn]: parts[nn + 1]] = np.array(out_array_xx)

                f_tens = submat.copy()
                f_tens.modify(data = f_coeffs.reshape(submat.shape))
                # from local_solvers import helper_cross
                # force_comp = force_component.scalar_multiply(charge / mass).data
                # helper_cross.plot_submat(ket, left_site_pos, 2, f_tens, ref_kets=[force_comp], plt_title='force')
                #
                # ket_deriv = helper.apply(self.f.grid.get_firstderivative_mpo(v_ax).data, ket)
                # helper_cross.plot_submat(ket, left_site_pos, 2, upwind_terms[fd_deriv_key],
                #                          ref_kets=[ket_deriv],
                #                          # ref_kets=[self.f.component.take_firstderivative(v_ax).data],
                #                          plt_title='deriv')


                deriv = np.where(f_coeffs > 0, bd_deriv_submat, fd_deriv_submat)
                # deriv = fd_deriv_submat
                tot_deriv_0 += deriv * f_coeffs

        if ref_deriv is not None:
            plt.figure()
            # plt.plot(ref_deriv.data.reshape(-1) * -1 - tot_deriv_0)
            plt.plot(ref_deriv.data.reshape(-1) * -1)
            plt.plot(tot_deriv_0)
            plt.show()
        print('submat', submat)
        print('upwind term', upwind_terms[next(iter(upwind_terms))])



        # tot_deriv_0 = upwind_terms[next(iter(upwind_terms))].data.reshape(-1) * -1
        out_data = submat_data - tot_deriv_0 * dt

        # print('submat data', submat_data)
        # print('boltzmann upwind', out_data)
        # print('upwind mat diff', np.linalg.norm(out_data - submat_data))
        # exit()

        out_tens = submat.copy()
        out_tens.modify(data=out_data.reshape(submat.shape))
        return out_tens


    def euler_upwind_v4(self, dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                        upwind_terms: dict[Any, 'qtn.Tensor'], ref_deriv: 'qtn.Tensor'=None,
                        left_site_pos:int = None, order=1, do_x_advection=True,
                        do_v_advection=True, internal_force=True, background_force=True, **kwargs) -> 'qtn.Tensor':
        """ perform a single time-step update at the specified indices
            upwinding for advection: df/dt + c d/dx f = 0
            if c_i > 0:
                backward difference:  u_i(t+dt) = u_i(t) - c_i (u_{i}(t) - u_{i-1}(t)) / dx * dt
            else c_i < 0:
                forward difference:   u_i(t+dt) = u_i(t) - c_i (u_{i+1}(t) - u_{i}(t)) / dx * dt

            explicitly compute D[x] * f, and measure points from that.
        """
        print('euler upwind v4')
        grid = self.f.grid
        charge, mass = self.matl_params.charge, self.matl_params.mass

        submat_data = submat.data.reshape(-1)
        npts = len(submat_data)

        tot_deriv_0 = np.zeros((npts,))
        if do_x_advection:

            for x_ax in self.x_axes:
                fd_deriv_key = ('FD', x_ax)
                bd_deriv_key = ('BD', x_ax)

                ## already transposed ot match submat
                fd_deriv_submat = upwind_terms[fd_deriv_key].data.reshape(-1)
                bd_deriv_submat = upwind_terms[bd_deriv_key].data.reshape(-1)

                v_coeffs = np.zeros((npts,))
                v_ax = self.v_axes_dict[x_ax.coordinate]
                for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):
                    sel_inds = [int(x) for x in bin(ind)[2:]]
                    sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds

                    ax_index = grid.get_axis_positions(sel_inds)
                    v_coeff = v_ax.x0 + v_ax.dx * ax_index[v_ax]
                    v_coeffs[ix] = v_coeff

                deriv = np.where(v_coeffs > 0, bd_deriv_submat, fd_deriv_submat)
                # deriv = fd_deriv_submat
                tot_deriv_0 += deriv * v_coeffs

        if do_v_advection:

            em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)

            for x_coord in self.coords_x.coords:
                v_ax = self.v_axes_dict.get(x_coord)
                if v_ax is None:  continue
                if em_term is None:  continue

                fd_deriv_key = ('FD', v_ax)
                bd_deriv_key = ('BD', v_ax)

                ## already transposed ot match submat
                fd_deriv_submat = upwind_terms[fd_deriv_key].data.reshape(-1)
                bd_deriv_submat = upwind_terms[bd_deriv_key].data.reshape(-1)

                f_coeffs = upwind_terms[('fcoeff', v_ax)].data.reshape(-1)

                deriv = np.where(f_coeffs > 0, bd_deriv_submat, fd_deriv_submat)
                # deriv = fd_deriv_submat
                tot_deriv_0 += deriv * f_coeffs

        if ref_deriv is not None:
            plt.figure()
            # plt.plot(ref_deriv.data.reshape(-1) * -1 - tot_deriv_0)
            plt.plot(ref_deriv.data.reshape(-1) * -1)
            plt.plot(tot_deriv_0)
            plt.show()
        print('submat', submat)
        print('upwind term', upwind_terms[next(iter(upwind_terms))])

        out_data = submat_data - tot_deriv_0 * dt

        out_tens = submat.copy()
        out_tens.modify(data=out_data.reshape(submat.shape))
        return out_tens


    def euler_upwind_mp(self, dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                        upwind_terms=None, order=1, do_x_advection=True, do_v_advection=True,
                        internal_force=True, background_force=True, **kwargs) -> 'qtn.Tensor':
        """ perform a single time-step update at the specified indices
            upwinding for advection: df/dt + c d/dx f = 0
            if c_i > 0:
                backward difference:  u_i(t+dt) = u_i(t) - c_i (u_{i}(t) - u_{i-1}(t)) / dx * dt
            else c_i < 0:
                forward difference:   u_i(t+dt) = u_i(t) - c_i (u_{i+1}(t) - u_{i}(t)) / dx * dt
        """
        grid = self.f.grid
        # axes = self.f.grid.axes
        ax_deriv_configs = self.f.component.ax_deriv_configs
        charge, mass = self.matl_params.charge, self.matl_params.mass

        submat_data = submat.data.reshape(-1)
        npts = len(submat_data)
        # out_data = np.zeros(submat_data.shape)

        submat_array = mp.Array('d', submat_data)
        out_array = mp.Array('d', range(npts))
        sel_array = mp.Array('i', selectors)

        order = 2

        def _upwind_deriv(ax: 'Axis', coeff: Numeric, inds: Sequence[int]):
            """ inds: binary string
            """
            idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
            val1 = grid.meas_elem_bin(ket, idx1) * sign1
            idx1_, sign1_ = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
            val1_ = grid.meas_elem_bin(ket, idx1_) * sign1_
            # deriv = - (val1 - val1_) / 2 / ax.dx
            if ax.coordinate == self.coords_v.coords[0]:
                deriv = - (val1 - val1_) / 2 / ax.dx
            elif ax.coordinate == self.coords_v.coords[1]:
                deriv = - (val1 - val1_) / 2 / ax.dx
            else:
                raise ValueError
            return deriv


        # def _upwind_deriv(ax: 'Axis', coeff: Numeric, inds: Sequence[int]):
        #     """ inds: binary string
        #     """
        #     if coeff < 0:
        #         if order == 1:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             deriv = (val0 - val1) / ax.dx
        #         elif order == 2:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: -1}, ax_deriv_configs)
        #             idx2, sign2 = grid.get_shifted_index_and_sign(inds, {ax: -2}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             val2 = grid.meas_elem_bin(ket, idx2) * sign2
        #             deriv = (3 * val0 - 4 * val1 + val2) / 2 / ax.dx
        #         else:
        #             raise NotImplementedError
        #     else:
        #         if order == 1:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             deriv = (val1 - val0) / ax.dx
        #         elif order == 2:
        #             idx1, sign1 = grid.get_shifted_index_and_sign(inds, {ax: 1}, ax_deriv_configs)
        #             idx2, sign2 = grid.get_shifted_index_and_sign(inds, {ax: 2}, ax_deriv_configs)
        #             val1 = grid.meas_elem_bin(ket, idx1) * sign1
        #             val2 = grid.meas_elem_bin(ket, idx2) * sign2
        #             deriv = (-val2 + 4 * val1 - 3 * val0) / 2 / ax.dx
        #         else:
        #             raise NotImplementedError
        #
        #     # print('ind0', inds)
        #     # print('idx1', idx1)
        #     #
        #     # print(grid.get_axis_positions(inds))
        #     # print(grid.get_axis_positions(idx1))
        #     #
        #     # print(self.f.component.meas_elem(inds), grid.meas_elem_bin(ket, inds))
        #     # print(self.f.component.meas_elem(idx1), grid.meas_elem_bin(ket, idx1))
        #
        #     # exit()
        #
        #     return deriv


        # for ix, (ind, val0) in enumerate(zip(selectors, submat_data)):
        def eval_func(out_data: 'mp.Array'):  # , val0_data: 'mp.Array', select_inds: 'mp.Array'):

            ix0 = int(out_data[0])
            for ix in out_data:
                ix = int(ix)
                val0 =  submat_data[ix]
                ind = selectors[ix]
                # val0 = val0_data[ix]
                # ind = select_inds[ix]
                # print('ind0', ix, ind, val0)

                sel_inds = [int(x) for x in bin(ind)[2:]]
                sel_inds = [0] * (grid.num_tensors - len(sel_inds)) + sel_inds
                ax_index = grid.get_axis_positions(sel_inds)
                # print('ax index', ax_index)
                tot_deriv_0 = 0

                if do_x_advection:

                    for x_ax in self.x_axes:
                        v_ax = self.v_axes_dict[x_ax.coordinate]
                        v_coeff = v_ax.x0 + v_ax.dx * ax_index[v_ax]
                        deriv = _upwind_deriv(x_ax, v_coeff, sel_inds)

                        tot_deriv_0 += deriv * v_coeff

                if do_v_advection:

                    em_term = self.get_force_term(background_force=background_force, internal_force=internal_force)

                    for x_coord in self.coords_x.coords:
                        v_ax = self.v_axes_dict.get(x_coord)
                        if v_ax is None:  continue
                        if em_term is None:  continue

                        force_component = em_term[x_coord]
                        # print('force_component', x_coord, force_component.data)
                        if force_component is None or force_component.data is None:
                            print('force is None', x_coord)
                            continue

                        f_coeff = force_component.meas_elem(sel_inds) * charge / mass
                        deriv = _upwind_deriv(v_ax, f_coeff, sel_inds)

                        tot_deriv_0 += deriv * f_coeff

                out_data[ix - ix0] = val0 - tot_deriv_0 * dt
            return ix0, out_data


        num_proc = 6
        num_proc = min(num_proc, npts)
        parts = [m * npts // num_proc for m in range(num_proc)] + [npts]
        out_data = np.zeros(npts)
        print('seps', parts, npts)

        # executor = mpi4py.futures.MPIPoolExecutor(num_proc)
        # futures = []

        # for nn in range(num_proc):
        #     print('nn', nn)
        #     out_array_xx = mp.Array('d', range(parts[nn], parts[nn + 1]))
        #
        #     futures += [executor.submit(eval_func, *(out_array_xx,))]
        #
        #     # p = mp.Process(target=eval_func, args=(out_array_xx,)) #, submat_array, sel_array))
        #     # p.start()
        #     # p.join()
        #     # print('out array', out_array)
        #
        #     # out_data[parts[nn]: parts[nn + 1]] = np.array(out_array_xx)
        #
        # mpi4py.futures.wait(futures)
        # for x in futures:
        #     idx0, out_array_xx = x.result()
        #     print('futures idx0', idx0)
        #     nn = parts.index(idx0)
        #     out_data[parts[nn]: parts[nn + 1]] = np.array(out_array_xx)
        # exit()

        processes = []
        outputs = []
        for nn in range(num_proc):
            print('nn', nn)
            out_array_xx = mp.Array('d', range(parts[nn], parts[nn + 1]))

            p = mp.Process(target=eval_func, args=(out_array_xx,)) #, submat_array, sel_array))
            p.start()
            processes += [p]
            outputs += [out_array_xx]

        for p in processes:
            p.join()

        for nn, out_array_xx in enumerate(outputs):
            out_data[parts[nn]: parts[nn + 1]] = np.array(out_array_xx)
            # p.close()

        # print('out array', out_data)


        # pool_obj = mp.Pool()
        # ans = pool_obj.map(eval_func, range(len(submat_data)))
        # out_data = np.array(ans)

        # print('submat data', submat_data)
        # print('boltzmann upwind', out_data)
        print('upwind mat diff', np.linalg.norm(out_data - submat_data))
        # exit()

        out_tens = submat.copy()
        out_tens.modify(data=out_data.reshape(submat.shape))
        return out_tens


    #####################

    def get_collision_term(self, v_axes: Sequence['Axis'] = None, v_grads: 'Field' = None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        dist = self.f
        if dist is None:
            return None

        print('get collision', self.collision)
        # print('get collision', self.collision.coll_type, v_grads is None)

        if v_axes is None:
            x_axes = self.coords_x.axes
            v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]

        if self.collision.coll_type is None:
            return None

        elif self.collision.coll_type == CollisionType.LB:
            # if dist.is_sqrt:
            #     raise NotImplementedError('LB operator not implemented for sqrt field')

            if v_grads is None:
                df0dv = self.df0dv  # for linearized Vlasov
                v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                    if df0dv is None else df0dv

            ############  get maxwellian ######
            if dist.is_sqrt:
                print('dist is sqrt')
                vth2 = 2 * self.matl_params.vth ** 2
            else:
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
