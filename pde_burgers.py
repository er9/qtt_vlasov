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
                 background_f = None,
                 normalize=True, upwind=False, zipup=False,
                 te_order=4, compress_levels=None,
                 compress_F=True, compress_F_opts=None,
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
        self.coords_x = coords_x
        self.x_axes: Sequence['Axis'] = self.coords_x.axes

        ## equation parameters
        self.dissip_coeff = dissip_coeff
        self.power = power
        self.flux_coeff = flux_coeff

        ## save important values

        ## calculation flags
        self.upwind = upwind
        self.zipup = zipup

        ## background fields
        self.background_f = background_f

        ## save initial normalization for normalization
        self.init_norm = None
        if normalize:
            self.init_norm = self.total_f.norm() if self.f is not None else None

        ## force/em-term compression
        self.compress_F = compress_F  # True
        self.compress_F_opts = {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE} \
            if compress_F_opts is None else compress_F_opts


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
        if self.background_f is not None:
            f = self.background_f.f if f is None else f.add(self.background_f.f)
        return f

    @property
    def background_f(self) -> Optional['Burgers']:
        return self.background_pde

    @background_f.setter
    def background_f(self, f0: Optional[Union['ScalarField', 'Burgers']]):
        if isinstance(f0, ScalarField):
            f0 = Burgers(f0, grid_X=self.grid_X, coords_x=self.coords_x,
                           upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                           compress_levels=self._comp_levels)
        elif isinstance(f0, ScalarField) or f0 is None:
            pass  ## PDE system
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
            new_system = Burgers(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x,
                                   background_f=f0_copy,
                                   normalize=self.do_normalization,
                                   upwind=self.upwind, zipup=self.zipup, te_order=self.te_order,
                                   compress_levels=self._comp_levels,
                                   )
        else:
            new_system = Burgers(dist, grid_X=self.grid_X,
                                   coords_x=self.coords_x,
                                   background_f=f0_copy,
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

        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

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
              verbose_plot=False, do_x_advection=True, do_v_advection=True) -> 'Boltzmann':

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
        """ df/dt = -a * d/dx f^m  + b * d^2 f / dt^2
        """
        dist = self.f

        # X, Y, Z = self.coords_x.coords


        ### advection term
        flux_term = self._calculate_time_derivative_f_flux(compress1=compress1, compress2=compress2)

        ### force term
        diff_term = self._calculate_time_derivative_f_dissipation(compress1=compress1, compress2=compress2)

        dFdt_s = flux_term.add(diff_term, compress_level=0)
        dFdt_s.name = self.f.name

        if compress:
            dFdt_s.compress(compress_level=compress)

        return dFdt_s

    # @profile
    def _calculate_time_derivative_f_flux(self, x_axes=None, compress1: int = 0, compress2: int = 0,
                                          verbose_plot: bool = False) -> Optional['Field']:
        """ df/dt = -a * d/dx f^m  + b * d^2 f / dt^2
        """
        dist = self.f

        if dist is None or self.flux_coeff == 0.0:
            return None

        x_axes = self.coords_x.axes if x_axes is None else x_axes
        # v_axes = [self.v_axes_dict.get(ax.coordinate, None) for ax in x_axes]
        # v_axes = [self.v_axes_dict.get(c, None) for c in self.coords_x.coords]

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


    def _get_time_evolution_mpos(self, advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                                 **kwargs) -> list['GridTN']:
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

        ## advection terms
        mpo_list = []
        for x_ax in x_axes:
            if x_ax is None:  continue
            if advec_axes is not None and x_ax not in advec_axes:  continue
            v_ax = v_axes_dict[x_ax]
            upwind_ax = v_ax if self.upwind else None

            deriv_config = dist.component.ax_deriv_configs[x_ax]
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

        return mpo_list


    def _get_time_evolution_mpos_flux(self, deriv_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                                      **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = - d/dx f^m / m
        """
        deriv_axes = self.coords_x.axes if deriv_axes is None else deriv_axes

        dist = self.f

        ## flux = f^power
        f_diag_mpo = self.f.component.apply_elemental_multiply_op()
        op = f_diag_mpo
        for m in range(self.power - 1):
            op = f_diag_mpo.apply(op, zipup=True)

        ## d/dx flux
        mpo_list = []
        if self.flux_coeff != 0.0:
            for x_ax in deriv_axes:

                deriv_config = dist.component.ax_deriv_configs[x_ax]
                ddx_gtn = dist.grid.get_firstderivative_mpo(x_ax, deriv_config=deriv_config)
                op = op.apply(ddx_gtn)

                op.scalar_multiply(-self.flux_coeff / self.power, inplace=True)
                mpo_list += [op]

        return mpo_list

    def _get_time_evolution_mpos_dissipation(self, deriv_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True, **kwargs) -> list['GridTN']:
        """ get list of mpos needed to compute df/dt = - d/dx f^m / m + d^2/dx^2 f
        """
        deriv_axes = self.coords_x.axes if deriv_axes is None else deriv_axes
        dist = self.f

        ## d/dx flux
        mpo_list = []
        for x_ax in deriv_axes:
            deriv_config = dist.component.ax_deriv_configs[x_ax]
            ddx_gtn = dist.grid.get_secondderivative_mpo(x_ax, None, deriv_config1=deriv_config)

            op = ddx_gtn.scalar_multiply(self.dissip_coeff / self.power, inplace=False)
            mpo_list += [op]

        return mpo_list


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

        dist_gtn = new_state.f.component
        mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
                                                 internal_force=internal_force)


        def nonlinear_func(ket, t=None):
            


        ## create a local solver (base TT function) and pass it to a gtn solver
        ## treat nonlinear term as time-dependent sources  (best we can do is imex anyways)
        ## either a gridTN or a function of gridTNs taking t, ket, other args as inputs
        ## introduce time-dependent sources (to update nonlinearity)
        solver = get_local_solver(type='tdvp', ket=self.f, operators=self._get_time_evolution_mpos_dissipation(),
                                  targets=nonlinear_func)

        ## also have get_block_local_solver



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
                                                 internal_force=internal_force)

        # print('mpo bond dim', [m.max_bond() for m in mpo_list])
        # dist_gtn_copy = dist_gtn.copy()
        dist_gtn.evolve_tdmrg(dt, mpo_list, te_order=te_order, do_adapt=do_adapt, inplace=True,
                              compress_config=compress_config)
        # print('distance tdvp', dist_gtn.distance(dist_gtn_copy), dist_gtn_copy.norm(is_sqrt=new_state.f.is_sqrt),
        #       dist_gtn.norm(is_sqrt=new_state.f.is_sqrt))
        # print('distance tdvp', new_state.f.component.distance(dist_gtn_copy))

        print('done boltzmann tdmrg')
        return new_state

    # def time_dependent_variational_principle_SL(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
    #                                             advec_axes=None, bg_method = 'SL', bg_split_order = 1,
    #                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
    #                                             verbose_plot: bool = False, **kwargs):
    #     """ computes bg / pert / bg dynamics via split step.
    #     """
    #
    #     new_state = self if inplace else self.copy()
    #
    #     new_state = new_state.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
    #                                               internal_force=False, split_order=bg_split_order,
    #                                               ompress=compress_level, compress1=compress_level_2)
    #
    #     new_state.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
    #                                                    compress_level=compress_level, compress_level_2=compress_level_2,
    #                                                    direction=direction, advec_axes=advec_axes, background_force=False,
    #                                                    internal_force=True)
    #
    #     bg_split_order = -1 if bg_split_order == 1 else bg_split_order
    #     new_state = new_state.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
    #                                               internal_force=False, split_order=bg_split_order,
    #                                               ompress=compress_level, compress1=compress_level_2)
    #
    #     return new_state

    # def time_dependent_variational_principle(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace: bool = False,
    #                                          compress_level: int = 1, compress_level_2: int = 4, direction=1,
    #                                          advec_axes: Sequence['Axis']=None, background_force=True, internal_force=True,
    #                                          **kwargs) -> 'Boltzmann':
    #     """ perform dynamical low rank TE with axes defining subspace
    #         do 2site if do_adapt is True (and bond dimension can still be expanded)
    #         direction is +1, -1
    #     """
    #     new_state = self if inplace else self.copy()
    #     compress_config = new_state.f.compress_config
    #
    #     dist_gtn = new_state.f.component
    #     max_bond = compress_config.max_bonds.get(compress_level, np.inf)
    #     mpo_list = self._get_time_evolution_mpos(advec_axes=advec_axes, background_force=background_force,
    #                                              internal_force=internal_force)
    #
    #     ## project by subspaces defined by each axis
    #     nsites = 0
    #     direction = 1
    #     site_ind = 0
    #     cur_orthog = None
    #     while site_ind < dist_gtn.grid.L:
    #
    #         if site_ind == dist_gtn.grid.L - 1 and nsites == 2:
    #             ## previously updated L-2, L-1
    #             print('continue', site_ind)
    #             site_ind += 1
    #             continue
    #
    #         ## num sites for update
    #         if site_ind == dist_gtn.grid.L - 1:
    #             adapt = False
    #         else:
    #             adapt = do_adapt and dist_gtn.data.bond_size(site_ind, site_ind + 1) < max_bond
    #
    #         if not adapt:   # one site update of site_ind
    #             if nsites == 2:  ## previously updated this site
    #                 nsites = 1
    #                 site_ind += 1
    #                 continue
    #
    #         ## backward propagation (skipped for first iteration)
    #         if 0 < site_ind < dist_gtn.grid.L:
    #             if nsites == 1:
    #                 bond_ind = site_ind - 1
    #                 print('back prop (+1) bond ind', bond_ind)
    #                 dist_gtn = dist_gtn.convert_to_USVT(canon_site=bond_ind, inplace=True, cur_orthog=cur_orthog)
    #                 helper_dlr.dlr_bond_time_evolution(dist_gtn, -dt / 2, mpo_list, bond_ind, te_order=te_order,
    #                                                    inplace=True, canonize=False, )
    #                 dist_gtn = dist_gtn.convert_from_USVT(inplace=True, canon_site=cur_orthog)
    #                 # site_ind += 1
    #             else:
    #                 print('back prop (+1) site ind', site_ind)
    #                 # dist_gtn = dist_gtn.canonize(i=site_ind + (nsites-1) * direction, inplace=True, scale=False)
    #                 helper_dlr.dlr_site_time_evolution(dist_gtn, -dt / 2, mpo_list, site_ind,
    #                                                    nsites=1, te_order=te_order,
    #                                                    canonize=False, inplace=True, compress_direction=direction,
    #                                                    compress_level=compress_level,
    #                                                    compress_opts_dict=compress_config)
    #
    #         nsites = 2 if adapt else 1
    #
    #         ## forward propagation
    #         print('forward prop sites (+1)', site_ind, nsites)
    #         dist_gtn = dist_gtn.canonize(i=site_ind, inplace=True, scale=False, cur_orthog=cur_orthog)
    #         helper_dlr.dlr_site_time_evolution(dist_gtn, dt / 2, mpo_list, site_ind, nsites=nsites, te_order=te_order,
    #                                                canonize=False, inplace=True, compress_direction=direction,
    #                                                compress_level=compress_level, compress_opts_dict=compress_config)
    #         cur_orthog = site_ind + nsites - 1
    #         site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites
    #         # print('next site ind', site_ind)
    #         # print('bond ind', bond_ind)
    #
    #     nsites = 0
    #     direction = -1
    #     site_ind = dist_gtn.grid.L - 1
    #     while site_ind >= 0:
    #
    #         if site_ind == 0 and nsites == 2:
    #             ## previously updated 0, 1
    #             site_ind -= 1
    #             continue
    #
    #         adapt = do_adapt and dist_gtn.data.bond_size(site_ind, site_ind - 1) < max_bond
    #         if not adapt:   # one site update of site_ind
    #             if nsites == 2:  ## previously updated this site
    #                 nsites = 1
    #                 site_ind -= 1
    #                 continue
    #
    #         if 0 <= site_ind < dist_gtn.grid.L - 1:
    #             if nsites == 1:
    #                 bond_ind = site_ind # - 1
    #                 print('back prop (-1) bond ind', bond_ind)
    #                 dist_gtn = dist_gtn.convert_to_USVT(canon_site=bond_ind, inplace=True, cur_orthog=cur_orthog)
    #                 helper_dlr.dlr_bond_time_evolution(dist_gtn, -dt / 2, mpo_list, bond_ind, te_order=te_order,
    #                                                    inplace=True, canonize=False, )
    #                 dist_gtn = dist_gtn.convert_from_USVT(inplace=True, canon_site=cur_orthog)
    #                 # site_ind = bond_ind - 1
    #             else:
    #                 print('back prop (-1) site ind', site_ind)
    #                 # dist_gtn = dist_gtn.canonize(i=site_ind + (nsites-1) * direction, inplace=True, scale=False)
    #                 helper_dlr.dlr_site_time_evolution(dist_gtn, -dt / 2, mpo_list, site_ind,
    #                                                    nsites=1, te_order=te_order,
    #                                                    canonize=False, inplace=True, compress_direction=direction,
    #                                                    compress_level=compress_level, compress_opts_dict=compress_config)
    #         nsites = 2 if adapt else 1
    #         site_ind = site_ind - nsites + 1
    #         print('forward prop sites (-1)', site_ind, nsites)
    #         dist_gtn = dist_gtn.canonize(i=site_ind, inplace=True, scale=False, cur_orthog=cur_orthog)
    #         helper_dlr.dlr_site_time_evolution(dist_gtn, dt / 2, mpo_list, site_ind, nsites=nsites, te_order=te_order,
    #                                            canonize=False, inplace=True, compress_direction=direction,
    #                                            compress_level=compress_level, compress_opts_dict=compress_config)
    #         cur_orthog = site_ind
    #         if nsites <= 1:
    #             site_ind -= nsites
    #
    #         # print('next site', site_ind)
    #         # print('next cur orthog', cur_orthog)
    #
    #     return new_state

    #####################

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
            # if dist.is_sqrt:
            #     raise NotImplementedError('LB operator not implemented for sqrt field')

            if v_grads is None:
                df0dv = self.df0dv  # for linearized Vlasov
                v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
                    if df0dv is None else df0dv

            ############  get maxwellian ######
            if dist.is_sqrt:
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
