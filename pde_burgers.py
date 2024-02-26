from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
import helper_sl as helper_sl
import helper_dlr
from helper_tdvp import TDVPSolver
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridTN


class Burgers(PDE_system):
    """ flux form:  df/dt + d/dx (f^m) / m = eps * d^2 f / dx^2
        note: advection form:  df/dt + f^(m-1) d/dx f = eps * d^2 f / dx^2
    """

    def __init__(self, dist: Optional[ScalarField],
                 dissip_coeff: float = 0.0, power: int = 2, flux_coeff=1.0,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 normalize=True, upwind=False, zipup=False,
                 te_order=4, compress_levels=None,
                 ):
        """ dist:  function f
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in f
        """

        self.names = {'f': dist.name if dist is not None else 'f'}
        field_names = [self.names[x] for x in ['f']]

        super().__init__(dist,
                         field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         )

        self.grid_X = grid_X
        self.coords_x = coords_x if coords_x is not None else dist.grid.axes[0].coord_sys
        self.x_axes: Sequence['Axis'] = self.coords_x.axes

        ## equation parameters
        self.dissip_coeff = dissip_coeff
        self.power = power
        self.flux_coeff = flux_coeff

        ## save important values

        ## calculation flags
        self.upwind = upwind
        self.zipup = zipup

        ## save initial normalization for normalization
        self.init_norm = None
        if normalize:
            self.init_norm = self.total_f.norm() if self.f is not None else None

        ## could set a max_bond for the nonlinear term


    @property
    def f(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['f']]
        except KeyError:
            return None

    @f.setter
    def f(self, new_field: 'ScalarField'):
        self._fields[self.names['f']] = new_field

    @property
    def total_f(self) -> Optional['ScalarField']:
        f = self.f
        # if self.background_f is not None:
        #     f = self.background_f.f if f is None else f.add(self.background_f.f)
        return f

    # @property
    # def background_f(self) -> Optional['Burgers']:
    #     return self.background_pde
    #
    # @background_f.setter
    # def background_f(self, f0: Optional[Union['ScalarField', 'Burgers']]):
    #     if isinstance(f0, ScalarField):
    #         f0 = Burgers(f0, grid_X=self.grid_X, coords_x=self.coords_x,
    #                        upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
    #                        compress_levels=self._comp_levels)
    #     elif isinstance(f0, ScalarField) or f0 is None:
    #         pass  ## PDE system
    #     else:
    #         raise TypeError
    #     self.background_pde = f0

    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 1:
            new_fields = new_fields + (None,) * (1 - len(new_fields))

        dist, = new_fields[:1]

        # f0_copy = self.background_f if not deep else \
        #     self.background_f.copy() if self.background_f is not None else None

        if recalc:
            new_system = Burgers(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x,
                                   normalize=self.do_normalization,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )
        else:
            new_system = Burgers(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x,
                                   normalize=False,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )

            new_system.do_normalization = self.do_normalization
            new_system.init_norm = self.init_norm

        if dist is None:      new_system.names['f'] = self.names['f']

        new_system.dissip_coeff = self.dissip_coeff
        new_system.power = self.power
        new_system.flux_coeff = self.flux_coeff

        return new_system

    def copy(self) -> 'Burgers':
        new_system = super().copy()
        return new_system

    def normalize(self, target_val=None):
        """ normalized density field
            in place operation
        """
        if self.f is not None and self.f.component is not None:
            if target_val is None:
                target_val = self.init_norm if self.init_norm is not None else 1.0
            self.f.component.normalize(target_val, inplace=True, is_sqrt=self.f.is_sqrt)
        return

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
              inplace: bool = False, compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0,
              verbose_plot=False, do_x_advection=True, do_v_advection=True) -> 'Burgers':

        state1 = super().euler(dt, deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               background_force=do_x_advection, internal_force=do_v_advection,
                               verbose_plot=verbose_plot, )

        if verbose_plot:
            pass

        return state1

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


    def evolve_dissipation(self, dt, inplace=False, method='rk4', compress=1, compress1=4, compress2=5,
                           verbose_plot=False):
        raise NotImplementedError


    def evolve_flux(self, dt, inplace=False, method='rk4', compress=1, compress1=4, compress2=5,
                    verbose_plot=False):
        raise NotImplementedError

    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_v=None, method_f=None,
                   is_first_time_step = False, is_last_time_step = False, inplace=False,
                   compress_level: int = 1, verbose_plot=False, ) -> 'Burgers':
        """ separates diffusion (method_f) and nonlinear term (method_v)
        """
        # print('boltzmann split step')
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if is_first_time_step:
            ## evolve force advection dt/2
            state0 = state0.evolve_dissipation(dt / 2, inplace=True, method=method_f, compress=comp1, compress1=comp2,
                                            compress2=comp5, verbose_plot=verbose_plot)

        ## evolve velocity advection dt
        state0 = state0.evolve_flux(dt, inplace=True, method=method_v,
                                    compress=comp1, compress1=comp2, compress2=comp5, verbose_plot=verbose_plot)

        ## evolve force advection dt/2 if last time step or dt
        dt_ = dt / 2 if is_last_time_step else dt
        state0 = state0.evolve_dissipation(dt_, inplace=True, method=method_f, compress=comp1, compress1=comp2,
                                           compress2=comp5, verbose_plot=verbose_plot)

        state0.time = self.time + dt if self.time is not None else None
        return state0


    def _get_split_step_func(self, method):
        if method is None:
            sl_func, kwargs = None, {}
        elif method[:2] == 'SL':
            raise NotImplementedError
        elif method[:3] == 'mac':
            sl_func = self.maccormack  ## rdm
            kwargs = {'coeff_is_constant': True}
        elif method[:3] == 'lax':
            sl_func = self.lax_wendroff
            kwargs = {'coeff_is_constant': True}
        else:
            raise NotImplementedError(f'{method} not yet implemented')
        return sl_func, kwargs


    def maccormack(self, ax: 'Axis', dt: float, advec_coeffs: 'GridTN' = None,
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

        ## forward difference
        deriv_config.update(order=0, fd_type=FDType.FORWARD)

        ## backward difference
        deriv_config.update(order=0, fd_type=FDType.BACKWARD)

        raise NotImplementedError

        ## calculate next time step
        f2.scalar_multiply(-1, inplace=True)
        deriv = f1.add(f2, inplace=True, compress_level=comp4)
        deriv.scalar_multiply(-dt / ax.dx, inplace=True)

        deriv_state = state.create_like(deriv, recalc=False)
        if get_df:
            return deriv_state
        else:
            new_state = state.add(deriv_state, inplace=True, compress_level=comp1)
            return new_state

    #############################

    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  verbose_plot=False, **kwargs) -> 'Burgers':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """

        ### advection term
        flux_term = self._calculate_time_derivative_f_flux(compress1=compress_level1, compress2=compress_level2)

        ### force term
        diff_term = self._calculate_time_derivative_f_dissipation(compress1=compress_level1, compress2=compress_level2)

        dFdt_s = flux_term.add(diff_term, compress_level=0)
        dFdt_s.name = self.f.name

        if compress_level:
            dFdt_s.compress(compress_level=compress_level)

        # dFdt = self._calculate_time_derivative_f(do_x_advection=do_x_advection, do_v_advection=do_v_advection,
        #                                          background_force=background_force, internal_force=internal_force,
        #                                          compress=compress_level, compress1=compress_level1,
        #                                          verbose_plot=verbose_plot)
        dFdt = self.create_like(dFdt_s, recalc=False)

        return dFdt

    # def get_time_derivative_op(self, time=None,
    #                            compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
    #                            do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
    #                            verbose_plot: bool = False, **kwargs) -> 'PDE_system':
    #     """ get G where G[f] = df/dt
    #         as a pde_system object
    #     """
    #     mpo = self.get_time_derivative_f_mpo(time=time, compress_level=compress_level, compress_level1=compress_level1,
    #                                          compress_level2=compress_level2, do_x_advectiom=do_x_advection,
    #                                          do_v_advection=do_v_advection, background_force=background_force,
    #                                          internal_force=internal_force, verbose_plot=verbose_plot, **kwargs)
    #
    #     dfdt_op = self.f.create_like(new_components={SCALAR_COORD: mpo})
    #     dfdt_sys = self.create_like(dfdt_op, recalc=False)
    #     return dfdt_sys

    # def get_time_derivative_f_mpo(self, time=None,
    #                               compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
    #                               do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
    #                               verbose_plot: bool = False, **kwargs) -> 'GridTN':
    #     """ get G where G[f] = df/dt
    #     """
    #     axes = []
    #     if do_x_advection:
    #         axes += self.grid_X.axes
    #     if do_v_advection:
    #         axes += [ax for ax in self.f.grid.axes if ax not in self.grid_X.axes]
    #
    #     mpos = self._get_time_evolution_mpos(advec_axes=axes, background_force=background_force,
    #                                          internal_force=internal_force, **kwargs)
    #
    #     dfdt_op = mpos[0].copy()
    #     for i in range(1, len(mpos)):
    #         tmp_comp_val = 0 if i == len(mpos) - 1 else compress_level1
    #         compress_opts = self.f.compress_config.get_compress_opts(tmp_comp_val)
    #         dfdt_op = dfdt_op.add(mpos[i], inplace=True, compress=tmp_comp_val, compress_opts=compress_opts)
    #
    #     if compress_level:
    #         compress_opts = self.f.compress_config.get_compress_opts(compress_level)
    #         dfdt_op = dfdt_op.compress(inplace=True, compress_opts=compress_opts)
    #
    #     return dfdt_op

    # # @profile
    # def _calculate_time_derivative_f(self, do_x_advection=True, do_v_advection=True,
    #                                  background_force=True, internal_force=True,
    #                                  compress: int = 1, compress1: int = 0, compress2: int = 0,
    #                                  verbose_plot=False) -> Optional['Field']:
    #     """ df/dt = -a * d/dx f^m  + b * d^2 f / dt^2
    #     """
    #     dist = self.f
    #
    #     # X, Y, Z = self.coords_x.coords
    #
    #
    #     ### advection term
    #     flux_term = self._calculate_time_derivative_f_flux(compress1=compress1, compress2=compress2)
    #
    #     ### force term
    #     diff_term = self._calculate_time_derivative_f_dissipation(compress1=compress1, compress2=compress2)
    #
    #     dFdt_s = flux_term.add(diff_term, compress_level=0)
    #     dFdt_s.name = self.f.name
    #
    #     if compress:
    #         dFdt_s.compress(compress_level=compress)
    #
    #     return dFdt_s

    # @profile
    def _calculate_time_derivative_f_flux(self, x_axes=None, compress1: int = 0, compress2: int = 0,
                                          verbose_plot: bool = False) -> Optional['Field']:
        """ df/dt = -a * d/dx f^m  + b * d^2 f / dt^2
        """
        dist = self.f

        if dist is None or self.flux_coeff == 0.0:
            return None

        x_axes = self.coords_x.axes if x_axes is None else x_axes

        mpo_list = self._get_time_evolution_mpos_flux()

        flux_term: Optional['GridTN'] = None
        for mpo in mpo_list:
            compress_opts = dist.compress_config.get_compress_opts(compress2)
            term_comp = dist.component.apply(mpo, compress=compress2, compress_opts=compress_opts)
            if flux_term is None:
                flux_term = term_comp.copy()
            else:
                flux_term.add(term_comp, compress=0, inplace=True)

        if flux_term is not None:
            if compress1:
                compress_opts = dist.compress_config.get(compress1)
                flux_term.compress(inplace=True, compress_opts=compress_opts)
            # convective_term.scalar_multiply(-1, inplace=True)

            # op_data = mpo_list[0].get_data()
            # plt.figure()
            # plt.plot(-2 * np.diag(op_data),label='op diag')
            # plt.plot(-4 * np.diag(op_data) ** 2, label='op diag**2')
            # plt.plot(self.f.get_field_data(), label='self')
            # plt.plot(flux_term.get_data(), label='flux term')
            # plt.legend()
            # plt.show()

        return dist.create_like_scalar(flux_term)

    # @profile
    def _calculate_time_derivative_f_dissipation(self, compress1: int = 0, compress2: int = 0,
                                                 verbose_plot: bool = False) -> Optional['Field']:

        dist = self.f

        if dist is None:
            return None


        mpo_list = self._get_time_evolution_mpos_dissipation()

        dissip_term: Optional['GridTN'] = None
        for mpo in mpo_list:
            compress_opts = dist.compress_config.get_compress_opts(compress2)
            # print('self.zipup f v', self.zipup)
            term_comp = dist.component.apply(mpo, zipup=self.zipup, compress=compress2, compress_opts=compress_opts)
            if dissip_term is None:
                dissip_term = term_comp.copy()
            else:
                dissip_term.add(term_comp, compress=0, inplace=True)

        if dissip_term is not None:
            if compress1:
                compress_opts = dist.compress_config.get(compress1)
                dissip_term.compress(inplace=True, compress_opts=compress_opts)
            # lorentz_term.scalar_multiply(-1, inplace=True)

        return dist.create_like_scalar(dissip_term)


    # def _get_time_evolution_mpos(self, advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
    #                              **kwargs) -> list['GridTN']:
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
    #             force_component = em_term[x_coord]
    #             # print('force_component', x_coord, force_component.data)
    #             if force_component is None or force_component.data is None:
    #                 print('force is None', x_coord)
    #                 continue
    #
    #             ## for k-space
    #             # force_data = force_component.get_data()
    #             # print('is H?', np.linalg.norm(force_data[1:] - force_data[:0:-1].conj()))
    #
    #             force_component = force_component.apply_elemental_multiply_op()
    #
    #             # force_mpo_data = force_component.get_data()
    #             # print('is H?', np.linalg.norm(force_mpo_data - force_mpo_data.T.conj()))
    #
    #             deriv_config = dist.component.ax_deriv_configs[v_ax]
    #             ddv_mpo = v_ax.build_firstderivative_mpo(deriv_config=deriv_config)
    #             v_advec_mpo = dist.grid.build_mpo_from_subgtns([force_component, ([v_ax], ddv_mpo)])
    #             v_advec_mpo.scalar_multiply(-1, inplace=True)
    #             # print('v_ax', v_ax, v_advec_mpo)
    #             if v_advec_mpo is not None:
    #                 mpo_list += [v_advec_mpo]
    #
    #     return mpo_list


    def _get_time_evolution_mpos_flux(self, deriv_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                                      **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = - d/dx f^m / m
        """
        deriv_axes = self.coords_x.axes if deriv_axes is None else deriv_axes

        dist = self.f

        ## flux = f^power
        # f_diag_mpo = self.f.component.apply_elemental_multiply_op()

        # if self.upwind:
        #     ## avg velocity at interface at grid point (i + 1/2) [S_i]:  1/2 (u_i + u_(i+1))
        #     ## for upwinding, let u_(i+1/2) = u_i if S_i > 0 else u_(i+1)
        #
        #     avg_u = dist.integrate().component
        #     print('avg flux', avg_u)
        #
        #     mask_pos_mps = dist.component.evaluate_func(func=lambda x: (x >= avg_u) * 1)
        #     mask_pos = dist.component.create_like(mask_pos_mps)
        #     mask_pos.gaussian_smooth_data(inplace=True)
        #     ### use TT-rounding???
        #     mask_neg_mps = dist.grid.get_ones_mps().add(mask_pos_mps.scalar_multiply(-1))
        #     mask_neg = dist.component.create_like(mask_neg_mps)
        #
        #     plt.figure()
        #     # plt.plot(dist.get_field_data(), label='data')
        #     plt.semilogy(mask_pos.get_data() - dist.get_field_data(), label='pos')
        #     # plt.plot(mask_neg.get_data(), label='neg')
        #     plt.legend()
        #     plt.show()
        #
        #     mask_pos = mask_pos.apply_elemental_multiply_op()
        #     mask_neg = mask_neg.apply_elemental_multiply_op()

        # op_data = f_diag_mpo.get_data()
        # plt.figure()
        # plt.plot(np.diag(op_data))
        # plt.plot(np.diag(op_data)**2)
        # plt.title('diag op')
        # plt.show()

        ## d/dx flux
        mpo_list = []
        if self.flux_coeff != 0.0:
            for x_ax in deriv_axes:

                deriv_config = dist.component.ax_deriv_configs[x_ax]


                if self.upwind:

                    shock_speed = dist.component.integrate()

                    ## shock_speed at i+1/2:  (u_{i} + u_{i+1})/2
                    ## if less than 0 (all velocities u < 0) then shock_speed < 0
                    ## if shock < 0: shift points left (1) (use u_{i+1})
                    ## if shock > 0: shift points right (-1) (use u_{i})

                    ## whether shock or rarefaction
                    ## ul - ur at i+1/2:  dx * d/dx u = (u_{i} - u_{i+1})
                    ## if ul > ur:  shock (regardless of sign)
                    ## if ul < ur:  rarefaction
                    ## use ul if ul > 0; o.w. ur if ur < 0; o.w. 0

                    ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                    deriv_u = dist.component.apply(ddx_gtn)
                    ul_vs_ur = deriv_u.integrate()
                    print('ul > ur?', ul_vs_ur)

                    ## finite volume:  f_{i} --> f_{i+1/2} = 1/2 (f_{i} + f_{i+1})
                    # shift = dist.grid.get_shift_mpo({x_ax: -1}, ax_boundary_conditions=dist.component.ax_deriv_configs)
                    # avg_u = dist.component.integrate()
                    if ul_vs_ur <= 0 and shock_speed > 0:
                        shift = dist.grid.get_shift_mpo({x_ax: -1},
                                                        ax_boundary_conditions=dist.component.ax_deriv_configs)
                    else:
                        print('here')
                        shift = dist.grid.get_shift_mpo({x_ax: 1},
                                                        ax_boundary_conditions=dist.component.ax_deriv_configs)

                    f_shift = dist.component.apply(shift)
                    avg_f = dist.component.add(f_shift, compress=True, inplace=False)
                    avg_f.scalar_multiply(0.5, inplace=True)

                    # plt.figure()
                    # plt.plot(dist.component.get_data(), label='orig')
                    # plt.plot(avg_f.get_data(), label='avg')
                    # plt.legend()
                    # plt.show()

                    # ## no traveling of shock if used with upwind
                    # avg_f = dist.component.copy()

                    f_diag_mpo = avg_f.apply_elemental_multiply_op()
                    op = f_diag_mpo
                    for m in range(self.power - 2):
                        op = f_diag_mpo.apply(op, zipup=True)
                    op = op.scalar_multiply(-self.flux_coeff / self.power)

                    mask_pos_mps = avg_f.evaluate_func(func=lambda x: (x >= 0) * 1)
                    mask_pos = dist.component.create_like(mask_pos_mps)
                    mask_pos.gaussian_smooth_data(inplace=True)

                    mask_neg_mps = dist.grid.get_ones_mps().add(mask_pos_mps.scalar_multiply(-1))
                    mask_neg = dist.component.create_like(mask_neg_mps)

                    # plt.figure()
                    # plt.plot(dist.get_field_data(), label='data')
                    # plt.semilogy(mask_pos.get_data() - dist.get_field_data(), label='pos')
                    # plt.plot(mask_pos.get_data(), label='pos')
                    # plt.plot(mask_neg.get_data(), label='neg')
                    # plt.legend()
                    # plt.show()

                    mask_pos = mask_pos.apply_elemental_multiply_op()
                    mask_neg = mask_neg.apply_elemental_multiply_op()

                    # deriv_config.update(fd_type=FDType.FORWARD)
                    deriv_config.update(fd_type=FDType.BACKWARD)
                    ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                    ddx_gtn_pos = ddx_gtn.apply(mask_pos)
                    # plt.figure()
                    # plt.imshow(ddx_gtn_pos.get_data())
                    # plt.title('pos')

                    # deriv_config.update(fd_type=FDType.BACKWARD)
                    deriv_config.update(fd_type=FDType.FORWARD)
                    ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                    ddx_gtn_neg = ddx_gtn.apply(mask_neg)
                    # plt.figure()
                    # plt.imshow(ddx_gtn_neg.get_data())
                    # plt.title('neg')
                    # plt.show()

                    op_pos = ddx_gtn_pos.apply(op)
                    op_neg = ddx_gtn_neg.apply(op)

                    mpo_list += [op_pos, op_neg]

                    # out_pos = dist.component.apply(op_pos)
                    # out_neg = dist.component.apply(op_neg)
                    #
                    # plt.figure()
                    # plt.plot(out_pos.get_data(),label='pos')
                    # plt.plot(out_neg.get_data(),label='neg')
                    # plt.legend()
                    # plt.show()

                else:
                    avg_f = dist.component.copy()
                    f_diag_mpo = avg_f.apply_elemental_multiply_op()
                    op = f_diag_mpo
                    for m in range(self.power - 2):
                        op = f_diag_mpo.apply(op, zipup=True)
                    op = op.scalar_multiply(-self.flux_coeff / self.power)
                    
                    deriv_config = dist.component.ax_deriv_configs[x_ax]
                    ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                    op = op.apply(ddx_gtn)

                    mpo_list += [op]

        return mpo_list

    def _get_time_evolution_mpos_dissipation(self, deriv_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True, **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = + d^2/dx^2 f
        """
        deriv_axes = self.coords_x.axes if deriv_axes is None else deriv_axes
        dist = self.f

        ## d/dx flux
        mpo_list = []
        if self.dissip_coeff != 0.0:
            for x_ax in deriv_axes:
                deriv_config = dist.component.ax_deriv_configs[x_ax]
                ddx_gtn = dist.grid.get_secondderivative_mpo(x_ax, None, deriv_config1=deriv_config)

                op = ddx_gtn.scalar_multiply(self.dissip_coeff / self.power, inplace=False)
                mpo_list += [op]

        return mpo_list


    def _get_nonlinear_terms(self, solver_type: LocalSolverType, deriv_axes: Sequence['Axis'] = None,
                             **kwargs) -> list['Term']:
        """ get list of mpos needed to compute df/dt = - d/dx f^m / m
        """
        deriv_axes = self.coords_x.axes if deriv_axes is None else deriv_axes

        dist = self.f

        # ## flux = f^power
        # f_diag_mpo = self.f.component.apply_elemental_multiply_op()
        # op = f_diag_mpo
        # for m in range(self.power - 1):
        #     op = f_diag_mpo.apply(op, zipup=True)

        # solver_type = LocalSolverType.TDDMRG  # TDCross  # TDDMRG

        ## d/dx flux * 1/m
        mpo_list = []
        if self.flux_coeff != 0.0:
            for x_ax in deriv_axes:

                deriv_config = dist.component.ax_deriv_configs[x_ax]
                ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                # ddx_gtn = dist.grid.get_iden_mpo()
                op = ddx_gtn.copy()

                op = op.scalar_multiply(-self.flux_coeff / self.power, inplace=False)
                mpo_list += [op.data]

            if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:
                term = Term_DMRG(self.f.component.data, operators=mpo_list, mps_power=self.power,
                                 num_tiers=2)
                # from local_solvers.local_dmrg_eval import local_dmrg_evaluator
                # out = local_dmrg_evaluator([term])
            else:
                term = Term_Cross(self.f.component.data, operators=mpo_list, mps_power=self.power)
                # from local_solvers.local_cross_eval import local_cross_evaluator
                # out = local_cross_evaluator([term])

            terms = [term]

        else:
            terms = []

        return terms


    def _get_nonlinear_value(self, solver_type: LocalSolverType, deriv_axes: Sequence['Axis'] = None,
                             **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = - d/dx f^m / m
        """
        terms = self._get_nonlinear_terms(solver_type, deriv_axes=deriv_axes, **kwargs)

        eval_terms = []
        for term in terms:
            if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:
                from local_solvers.local_dmrg_eval import local_dmrg_evaluator
                out = local_dmrg_evaluator([term])
            else:
                from local_solvers.local_cross_eval import local_cross_evaluator
                out = local_cross_evaluator([term])

            eval_terms += [self.f.grid.make_gridTN(out)]

            # gtn = self.f.grid.make_gridTN(out)
            # plt.figure()
            # plt.plot(gtn.get_data(), label='term squared')
            # gtn = self.f.grid.make_gridTN(term.ket)
            # plt.plot(gtn.get_data() ** 2 / 2, label='orig ket')
            # plt.legend()
            # plt.title('term squared')
            # plt.show()

        return eval_terms


    def dynamical_low_rank(self, dt: Numeric, te_order=4, do_adapt: bool = True,
                           inplace: bool = False, compress_level: int = 1, compress_level_2: int = 4,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           **kwargs) -> 'Burgers':
        """ perform dynamical low rank TE with axes defining subspace
        """
        raise NotImplementedError

    # @profile
    def time_dependent_variational_principle(self, dt: Numeric, te_order=4, do_adapt: bool = True,
                                             inplace: bool = False,
                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                             advec_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True,
                                             expand_basis: Sequence['GridTN'] = None,
                                             **kwargs) -> 'Burgers':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config
        max_bond = compress_config.get_compress_opts()['max_bond'] if compress_config is not None else None

        dist_gtn = new_state.f.component
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force)



        from local_solvers.terms_3 import Term_DMRG as LocalTerm
        from local_solvers.local_dmrg_eval import local_dmrg_evaluator as LocalEval
        # from local_solvers.terms_3 import Term_Cross as LocalTerm
        # from local_solvers.local_cross_eval import local_cross_evaluator as LocalEval
        from local_solvers.time_integrator import TDVP_DMRG as TimeInteg

        lin_term = LocalTerm(dist_gtn.data, operators=[op.data for op in mpo_list], max_bond=max_bond)

        nonlin_terms = []
        if self.flux_coeff != 0:
            for ax in self.grid_X.axes:
                deriv_config = new_state.f.component.ax_deriv_configs[ax].copy()
                deriv_op = self.grid_X.get_firstderivative_mpo(ax, deriv_config)
                nonlin_terms += [LocalTerm(dist_gtn.data, [deriv_op.data], mps_coeff=self.flux_coeff, mps_power=3)]

        out_mps = TimeInteg(dist_gtn.data, [lin_term, *nonlin_terms], max_bond=max_bond)
        dist_gtn.data = out_mps

        # dist_gtn.evolve_tdvp(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
        #                      compress_config=compress_config, expand_basis=expand_basis)

        print('done boltzmann tdvp')
        return new_state


    def time_dmrg(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  **kwargs) -> 'Burgers':
        """ perform dynamical low rank TE with axes defining subspace
            do 2site if do_adapt is True (and bond dimension can still be expanded)
            direction is +1, -1
        """
        new_state = self if inplace else self.copy()
        compress_config = new_state.f.compress_config

        dist_gtn = new_state.f.component
        # max_bond = compress_config.max_bonds.get(compress_level, np.inf)

        solver_type = LocalSolverType.TDCross  # LocalSolverType.TDCross

        # lin_ops = [op.data for op in self._get_time_evolution_mpos_dissipation()]
        lin_ops = self._get_time_evolution_mpos_dissipation()
        nonlin_terms = self._get_nonlinear_terms(solver_type=solver_type)
        # sources = self._get_nonlinear_value(solver_type=solver_type)

        # print('lin ops', lin_ops)
        # print('non lin terms', nonlin_terms)
        # for s in sources:
        #     print('s.norm()', helper.norm(s.data), s.exponent)
        #     # s.scalar_multiply(10, inplace=True)
        #     # print('s.norm()', helper.norm(s.data), s.exponent)
        # print('sources', sources)
        dist_gtn.evolve_tdmrg(dt, lin_ops, te_order=self.te_order, inplace=True,
                              nonlinear_terms=nonlin_terms,
                              # sources=sources,
                              solver_type=solver_type,
                              compress_config=compress_config)
        # print(nonlin_terms[0].vec_block.power)
        # plt.figure()
        # plt.plot(dist_gtn.get_data(), label='te result')
        # plt.plot(dist_gtn.get_data(), label='te result')
        # # tmp = dist_gtn.create_like(nonlin_terms[0].vec_block.bra)
        # # plt.plot(tmp.get_data(), label='vecblock bra')
        # # tmp = dist_gtn.create_like(nonlin_terms[0].ket)
        # # plt.plot(tmp.get_data(), label='nonlin ket')
        # plt.title('nonlin kets')
        # plt.legend()
        # plt.show()


        print('done burgers tdmrg')
        return new_state



