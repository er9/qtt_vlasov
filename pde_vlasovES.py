from setup.configs import *
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from field import Field, ScalarField
from pde_system import PDE_system
from pde_vlasov import Vlasov
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid


class VlasovPoisson(Vlasov):
    """ df/dt + v grad(f) + q/m (E) grad_v(f) = 0
    """
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField', 'Boltzmann']],
                 sys_fi: Optional[Union['ScalarField', 'Boltzmann']],
                 potential: Optional['ScalarField'] = None,
                 grid_X: Optional['Grid']=None,
                 coords_x:  Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 f0_gradv_e=None, f0_gradv_i=None,
                 evolve_ion=True, normalize=True, upwind=False, zipup=False,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_E0: Optional['Field'] = None,
                 potential_boundary_conditions=(None,None),
                 te_order=2,
                 compress_levels=None,
                 # init_compress_opts=None, te_compress_opts=None
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            boundary_conditions: boundary conditions on V
               tuple of (bc_vectors [m vectors or mps], bc_results [m float]) such that
               bc_vectors (dot) V_data = bc_results
        """
        grid_X = grid_X if potential is None else potential.grid

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         te_order=te_order, compress_levels=compress_levels)

        self.names['V']  = potential.name if potential is not None else 'V'
        self.field_names += [self.names['V']]
        if potential is not None:
            self._fields[self.names['V']] = potential

        self.evolve_V = True
        self.inv_laplacian_mpo = None
        self.potential_boundary_conditions = potential_boundary_conditions

        ## background fields
        self.background_E0 = background_E0      # a field (no time evolution equations)
        if background_E0 is not None:
            sys_fe.background_force_neg = self.compute_force_term(background_only=True)
            sys_fi.background_force_neg = self.compute_force_term(is_ion=True, background_only=True)
            sys_fe.background_force_neg.scalar_multiply(-1, inplace=True)
            sys_fi.background_force_neg.scalar_multiply(-1, inplace=True)
        self.divE0 = None
        self._total_E = None

        # self.compress_F = True
        # self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}
        #
        # self.collision = CollisionConfiguration(None)
        #
        # self.dist0_e = None
        # self.dist0_i = None
        #
        # self.v_from_v0_e = None   # data_ve - vdrift_e
        # self.v_from_v0_i = None   # data_vi - vdrift_i


    # @property
    # def fe(self) -> Optional['ScalarField']:
    #     try:   return self.fields[self.names['fe']]
    #     except KeyError:   return None
    #
    # @fe.setter
    # def fe(self,new_field):
    #     self.fields[self.names['fe']] = new_field
    #
    #
    # @property
    # def fi(self) -> Optional['ScalarField']:
    #     try:   return self.fields[self.names['fi']]
    #     except KeyError:   return None
    #
    # @fi.setter
    # def fi(self,new_field):
    #     self.fields[self.names['fi']] = new_field


    @property
    def V(self) -> Optional['ScalarField']:
        try:   return self._fields[self.names['V']]
        except KeyError:   return None

    @V.setter
    def V(self,new_field):
        self._fields[self.names['V']] = new_field
        self._total_E = None

    @property
    def total_E(self):
        return self._total_E


    @property
    def matl_params(self):
        if self.sys_fe is not None:
            return self.sys_fe.matl_params
        elif self.sys_fi is not None:
            return self.sys_fi.matl_params
        return None


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
            deep:  copy background PDE objects too
        """
        if len(new_fields) < 3:
            new_fields = new_fields + (None,) *(3-len(new_fields))

        dist_e, dist_i, potential = new_fields[:3]
        new_system = super().create_like(dist_e, dist_i, recalc=recalc, deep=deep)
        new_system.V = potential
        # if potential is None:   new_system.names['V' ] = self.names['V' ]     ## not needed anymore

        new_system.inv_laplacian_mpo = self.inv_laplacian_mpo
        new_system.background_E0 = self.background_E0
        new_system.sys_fe.background_force_neg = self.sys_fe.background_force_neg
        new_system.sys_fi.background_force_neg = self.sys_fi.background_force_neg
        new_system.divE0 = self.divE0

        # new_system.collision = self.collision
        #
        # new_system.compress_F = self.compress_F
        # new_system.compress_F_opts = self.compress_F_opts
        #
        # new_system.dist0_i = self.dist0_i
        # new_system.dist0_e = self.dist0_e

        # new_system.v_from_v0_i = self.v_from_v0_i
        # new_system.v_from_v0_e = self.v_from_v0_e

        return new_system


    def copy(self):
        """ create a new system like this with new fields
        """
        # fe = self.fe.copy() if self.fe is not None else None
        # fi = self.fi.copy() if self.fi is not None else None
        # V = self.V.copy() if self.V is not None else None
        new_system = super().copy()
        new_system._total_E = self._total_E
        return new_system


    # def normalize(self,target_e=None,target_i=None):
    #     """ normalized density field
    #     """
    #     if self.fi is not None:
    #         if target_i is None:
    #             target_i = self.init_norm_i if self.init_norm_i is not None else 1.0
    #         norm_i = self.fi.norm()
    #         # print('norm i', target_i, norm_i)
    #         # if self.fi.is_sqrt:  norm_i = np.sqrt(norm_i)
    #         # self.fi.scalar_multiply( np.abs(target_i/norm_i), inplace=True )
    #         self.fi.scalar_multiply( target_i/np.conj(norm_i), inplace=True )
    #         # print('norm i', self.fi.norm())
    #
    #     if self.fe is not None:
    #         if target_e is None:
    #             target_e = self.init_norm_e if self.init_norm_e is not None else 1.0
    #         norm_e = self.fe.norm()
    #         # print('norm e', target_e, norm_e)
    #         # if self.fe.is_sqrt:  norm_e = np.sqrt(norm_e)
    #         # self.fe.scalar_multiply( np.abs(target_e/norm_e), inplace=True )
    #         self.fe.scalar_multiply( target_e/np.conj(norm_e), inplace=True )
    #         # print('norm e', self.fe.norm())
    #     return
    #
    #
    # def get_time_derivative_operator(self, compress=1):
    #     """ dF/dt = G*prob[t].  Returns G
    #         NOT DEBUGGED
    #     """
    #     raise NotImplementedError


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                   method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'VlasovPoisson':
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

        # ## dt (dt/2 if first), dt (dt/2 if last), V structure
        # state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
        # state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
        # super(VlasovPoisson, state0).split_step(dt, deriv0, method_h=method_h, method_v=method_v, inplace=True,
        #                                         is_first_time_step=is_first_time_step,
        #                                         is_last_time_step=is_last_time_step,
        #                                         compress_level=compress_level, verbose_plot=verbose_plot)

        def get_func(pde_boltz, method):
            if method[:2] == 'SL':
                # assert all([v_ax.basis.type == BasisType.SPATIAL for v_ax in self.sys_fe.v_axes]), \
                #     'SL time evolution for spatial basis only'
                # assert (len(self.sys_fe.x_axes) == 1), 'SL method only valid for 1DnV'

                sl_func = pde_boltz.semilagrangian
                kwargs = {'method': method}
            elif method[:3] == 'mac':
                sl_func = pde_boltz.maccormack
                kwargs = {'coeff_is_constant': True}
            elif method[:3] == 'lax':
                sl_func = pde_boltz.lax_wendroff
                kwargs = {'coeff_is_constant': True}
            else:
                raise NotImplementedError(f'{method} not yet implemented')
            return sl_func, kwargs

        adv_f_func_fe = get_func(state0.sys_fe, method_f)
        adv_f_func_fi = get_func(state0.sys_fi, method_f)
        adv_v_func_fe = get_func(state0.sys_fe, method_v)
        adv_v_func_fi = get_func(state0.sys_fi, method_v)

        if is_first_time_step:
            ## evolve force advection dt/2
            state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
            state0.sys_fe.get_force_advection(dt / 2, inplace=True, adv_func=adv_f_func_fe,
                                              compress=comp2, compress1=comp4, compress2=comp5,
                                              verbose_plot=verbose_plot)
            state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
            state0.sys_fi.get_force_advection(dt / 2, inplace=True, adv_func=adv_f_func_fi,
                                              compress=comp2, compress1=comp4, compress2=comp5,
                                              verbose_plot=verbose_plot)

        ## evolve velocity advection dt
        state0.sys_fe.get_vel_advection(dt, inplace=True, adv_func=adv_v_func_fe, compress=comp2, compress1=comp4,
                                        compress2=comp5, verbose_plot=verbose_plot)
        state0.sys_fi.get_vel_advection(dt, inplace=True, adv_func=adv_v_func_fi, compress=comp2, compress1=comp4,
                                        compress2=comp5, verbose_plot=verbose_plot)

        if np.abs(state0.matl_params.e) > 0:
            state0.update_V(inplace=True, compress=compress_level, compress1=comp2)

        if verbose_plot:
            V_data = state0.V.get_comp_data()
            plt.figure()
            plt.plot(V_data)
            plt.show()

        ## evolve force advection dt/2
        dt_ = dt /2 if is_last_time_step else dt
        state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
        state0.sys_fe.get_force_advection(dt_, inplace=True, adv_func=adv_f_func_fe,
                                          compress=comp2, compress1=comp4,
                                          compress2=comp5, verbose_plot=verbose_plot)
        state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
        state0.sys_fi.get_force_advection(dt_, inplace=True, adv_func=adv_f_func_fi,
                                          compress=comp2, compress1=comp4,
                                          compress2=comp5, verbose_plot=verbose_plot)

        ## include collisions
        if self.collision.coll_type is not None:
            raise NotImplementedError
            deriv_coll_e = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=False)
            deriv_coll_i = self.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=True)
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None

        if compress_level:
            state0.compress(compress_level=compress_level, verbose=False)

        if state0.do_normalization:
            state0.normalize()

        return state0


    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['VlasovPoisson'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False) \
            -> 'VlasovPoisson':
        """ perform explicit Euler time evolution
            compress:  final compression of final state, updateV
            compress1: compression of derivative
            compress2: compress1 of calculate derivative, updateV
        """
        state1 = super().euler(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               verbose_plot=verbose_plot)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            # old_V = state1.V.copy()
            # print('state bond V', compress_level, state1.V.max_bond())
            # state1.V = self.V
            state1.update_V(inplace=True, compress=compress_level, compress1=compress_level2)
            # print('V diff', helper.distance(old_V.component.data, state1.V.component.data))
            # print('state bond comp', compress_level, state1.V.max_bond())

        state1.time = self.time + dt if self.time is not None else None

        return state1


    # def _get_force_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
    #                          verbose_plot=False):
    #     """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
    #     """
    #     state0 = self if inplace else self.copy()
    #
    #     if method is None:
    #         # deriv_vi0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #         #                                                       is_ion=True, verbose_plot=verbose_plot)
    #         # deriv_ve0 = state0._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #         #                                                       is_ion=False, verbose_plot=verbose_plot)
    #         # dFdt_v = state0.create_like(deriv_ve0, deriv_vi0, recalc=False)
    #         # new_state = state0.euler(dt, deriv0=dFdt_v, inplace=True, do_update_V=False, compress_level=compress)
    #
    #         ## rk2
    #         # print('warning: time not included')
    #         deriv0_vi0 = self._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                              is_ion=True, verbose_plot=verbose_plot, )
    #         deriv0_ve0 = self._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                              is_ion=False, verbose_plot=verbose_plot, )
    #         deriv0 = state0.create_like(deriv0_ve0, deriv0_vi0, recalc=False)
    #         state1 = state0.euler(dt, deriv0=deriv0, do_update_V=False, compress_level=compress)
    #
    #         deriv1_vi0 = state1._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                                is_ion=True, verbose_plot=verbose_plot, )
    #         deriv1_ve0 = state1._calculate_time_derivative_f_force(compress1=compress1, compress2=compress2,
    #                                                                is_ion=False, verbose_plot=verbose_plot, )
    #         deriv1 = state0.create_like(deriv1_ve0, deriv1_vi0, recalc=False)
    #
    #         deriv_sum = deriv0.add(deriv1, compress_level=0)
    #         new_state = state0.euler(0.5 * dt, deriv0=deriv_sum, inplace=True, do_update_V=False,
    #                                  compress_level=compress)
    #     elif method[:2] == 'SL':
    #
    #         assert all([x_ax.basis.type == BasisType.SPATIAL for x_ax in self.x_axes]), \
    #             'SL time evolution for spatial basis only'
    #
    #         assert(len(self.x_axes)==1),'SL method only valid for 1DnV'
    #
    #         fi_gtn = state0.fi.component
    #         fe_gtn = state0.fe.component
    #
    #         if method == 'SL':
    #             print('SL F advection')
    #             fi_mpo_func = fi_gtn.grid._get_mpo_advection_SL
    #             fe_mpo_func = fe_gtn.grid._get_mpo_advection_SL
    #         elif method == 'SL-pfc':
    #             print('SL-pfc F advection')
    #             fi_mpo_func = fi_gtn.grid._get_mpo_advection_SL_pfc
    #             fe_mpo_func = fe_gtn.grid._get_mpo_advection_SL_pfc
    #         else:
    #             raise ValueError('not valid SL method')
    #
    #         for x_ax in self.x_axes:
    #             C = self.coords_x.get_coord(x_ax.coordinate.type)
    #             force_term = self.compute_force_term(compress_level1=compress1, compress_level2=compress2)
    #
    #             if force_term is None:
    #                 print('force is None')
    #                 continue
    #
    #             force_data = force_term.get_comp_data(C)
    #             if force_data is None:     # no field
    #                 print('force is None')
    #                 continue
    #
    #             if verbose_plot:
    #                 plt.figure()
    #                 plt.plot(force_data)
    #                 plt.title('elc lorentz force')
    #                 plt.show()
    #
    #             ion_coeff = self.ion_params.e * self.ion_params.Z / self.ion_params.mass
    #             elc_coeff = self.elc_params.e * self.elc_params.Z / self.elc_params.mass
    #             elc_to_ion_scale = ion_coeff / elc_coeff
    #             print('elc/ion scale', elc_to_ion_scale)
    #
    #             ## ions
    #             v_ax = self.vi_axes_dict[x_ax]
    #             deriv_params = fi_gtn.ax_deriv_configs[v_ax]
    #             compress_opts = state0.fi.compress_config.get_compress_opts(compress)
    #             fi_mpo_gtn = fi_mpo_func(v_ax, x_ax, dt, deriv_params, advec_coeffs=force_data * elc_to_ion_scale)
    #             fi_gtn.apply(fi_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #
    #             ## elecs
    #             v_ax = self.ve_axes_dict[x_ax]
    #             deriv_params = fe_gtn.ax_deriv_configs[v_ax]
    #             compress_opts = state0.fe.compress_config.get_compress_opts(compress)
    #             fe_mpo_gtn = fe_mpo_func(v_ax, x_ax, dt, deriv_params, advec_coeffs=force_data)
    #             fe_gtn.apply(fe_mpo_gtn, inplace=True, zipup=self.zipup, compress=compress, compress_opts=compress_opts)
    #
    #         new_state = state0
    #
    #     else:
    #         raise NotImplementedError
    #
    #     return new_state
    #
    #
    # def _get_vel_advection(self, dt, inplace=False, method=None, compress=1, compress1=4, compress2=5,
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
    #         state0 = state0.euler(dt, deriv0=dFdt_x, inplace=True, do_update_V=False, compress_level=compress)
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
                           verbose_plot=False, background_only=False,):
        """ compute EM force:  q/m (E + v x B)
        """
        spec_params = self.ion_params if is_ion else self.elc_params

        em_term = None
        if np.abs(spec_params.Z) > 0:
            if background_only:
                em_term = self.background_E0
            else:
                if self._total_E is None:

                    em_term = self.V.gradient(compress_level=0)   # E = -grad(V)
                    em_term.scalar_multiply(-1, inplace=True)

                    if self.background_E0 is not None:
                        em_term = em_term.add( self.background_E0, inplace=True )
                        # add E0 to electric field

                    self._total_E = em_term
                else:
                    em_term = self.total_E


        if em_term is not None:
            em_term = em_term.scalar_multiply(spec_params.e * spec_params.Z / spec_params.mass)
            if self.compress_F:
                # print('compressF', self.compress_F_opts)
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

            # plt.figure()
            # X = self.coords_x.coords[0]
            # plt.plot(self.V.get_comp_data(), label='V')
            # plt.plot(em_term.get_comp_data(X), label='E')
            # plt.legend()
            # plt.show()

        return em_term


    # @profile
    def update_V(self,inplace=False,compress=1,compress1=0) -> 'ScalarField':
        """ div E = div^2 V = 4*pi*rho
            compress:  compression of V
            compress1: compression of terms to create V
        """
        x_axes = self.coords_x.axes
        # ve_axes = self.coords_ve.axes
        # vi_axes = self.coords_vi.axes

        new_sys = self if inplace else self.copy()

        ## particles are not charged
        if self.elc_params.e == 0 or (np.abs(self.elc_params.Z) == 0 and np.abs(self.ion_params.Z) == 0):
            new_sys.V = None
            return new_sys

        if self.inv_laplacian_mpo is None:
            ax_deriv_configs = self.V.component.ax_deriv_configs if self.V is not None \
                                    else {ax: DerivativeConfiguration() for ax in x_axes}
            inv_laplacian_mpo = self.V.grid.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs,
                                                                  compress_opts={'cutoff':1.0e-20,'cutoff_mode':'rsum2'})
            self.inv_laplacian_mpo = inv_laplacian_mpo
            print('inv laplacian', inv_laplacian_mpo.max_bond())
        else:
            inv_laplacian_mpo = self.inv_laplacian_mpo

        ## get Field object after integrating over velocity space
        charge_density = self.compute_charge_density(compress=compress1)
        # fe = self.fe
        # e_density = fe.integrate(axes=ve_axes, new_grid=self.V.grid,
        #                          new_ax_deriv_configs=self.V.component.ax_deriv_configs)
        # e_density.compress_config = self.V.compress_config
        # e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
        #
        # if self.fi is not None:
        #     # if self.fi.is_sqrt:   fi = self.fi.mult_cc(compress=0) # compress+1)
        #     # else:                 fi = self.fi
        #     fi = self.fi
        #     i_density = fi.integrate(axes=vi_axes, new_grid=self.V.grid,
        #                              new_ax_deriv_configs=self.V.component.ax_deriv_configs)
        #     i_density.compress_config = self.V.compress_config
        #     i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)
        #
        #     charge_density = e_density.add(i_density, compress_level=compress1, inplace=False)
        # else:
        #     charge_density = e_density.compress()

        # e_dens_dat = e_density.get_comp_data()
        # i_dens_dat = i_density.get_comp_data()
        # plt.figure()
        # # plt.plot(np.real(i_dens_dat), label='ion')
        # # plt.plot(np.real(e_dens_dat), label='elec')
        # # plt.plot(np.imag(e_dens_dat), label='elec', ls='--')
        # charge_dat = charge_density.get_comp_data()
        # plt.plot(np.real(charge_dat), label='total')
        # # plt.plot(np.imag(charge_dat), label='total', ls='--')
        # plt.title('charge density')
        # plt.legend()
        # plt.show()

        # print('charge density', charge_density)
        if self.background_E0 is not None:
            if self.divE0 is None:
                self.divE0 = self.background_E0.divergence(self.coords_x, compress_level=compress1)
            charge_density.add(self.divE0.scalar_multiply(-1/self.elc_params.eps0, inplace=False), inplace=True)
        charge_density.scalar_multiply(-1, inplace=True)

        bc_values = self.potential_boundary_conditions[1]
        if bc_values is not None:  # ie. 0
            if bc_values.grid.ndim < charge_density.grid.ndim:
                bc_values = bc_values.pad_to_new_grid(new_grid=charge_density.grid)
            soln_gtn = charge_density.component.add(bc_values[0], compress=False)
        else:
            soln_gtn = charge_density.component

        # print('update V', compress, new_sys.V.compress_config[compress])

        new_V_mps = soln_gtn.apply(inv_laplacian_mpo, compress=compress,
                                   compress_opts=new_sys.V.compress_config[compress])
        # print('new V', compress, new_sys.V.get_te_compress_opts(compress), new_V_mps.max_bond())

        # V_ = V.copy()
        new_sys.V.component = new_V_mps  ## update potential field (which is a scalar field)
        new_sys._total_E = None

        # plt.figure()
        # plt.plot(charge_density.get_comp_data(), label='charge')
        # plt.plot(new_sys.V.get_comp_data(), label='new V')
        # plt.legend()
        # plt.show()

        return new_sys


    # def get_collision_term(self, is_ion=False, v_axes:Sequence['Axis']=None, v_grads: 'Field'=None,
    #                        compress1=0, compress2=0):
    #     """ obtain collision term
    #     """
    #     dist = self.fi if is_ion else self.fe
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


    # def next_time_step(self, dt, deriv0: Optional['VlasovPoisson'] = None, inplace: bool = False,
    #                    compress_level: int = 1, max_iter=100, err_tol=None, is_first_time_step=False,
    #                    is_last_time_step=False, do_postprocessing=False, process_kwargs=None,
    #                    verbose_plot=False) -> 'VlasovPoisson':
    #     """ convenience fct btwn different TE methods?
    #         process_kwargs:
    #             'axes':  list of axes to post-process along or 'all'
    #             'sigma': gaussian smoothing stddev (default = 0.3)
    #
    #         comp_levels:
    #         comp1:   compress final state
    #         comp2:   + compress intermediate rk4 states
    #         comp3:   + compress sum of derivatives
    #         comp4:   + compress derivatives
    #         comp5:   + compress during calculation of derivative
    #     """
    #     te_order = self.te_order
    #
    #     ## update density (potential should remain untouched)
    #     if te_order == 1:
    #         state_t = self.euler(dt, deriv0, inplace=inplace, compress_level=compress_level)
    #     elif te_order == 2:
    #         state_t = self.rk2(dt, deriv0, compress_level=compress_level)
    #     elif te_order == 3:
    #         state_t = self.rk3(dt, deriv0, compress_level=compress_level)
    #     elif te_order == 4:
    #         state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot)
    #     elif te_order == 21:
    #         if inplace:
    #             raise NotImplementedError
    #         state_t = self.backwards_euler(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
    #                                        verbose_plot=verbose_plot)
    #     elif te_order == 221:
    #         if inplace:
    #             raise NotImplementedError
    #         state_t = self.implicit_midpoint(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
    #                                          verbose_plot=verbose_plot)
    #     elif te_order == 222:
    #         if inplace:
    #             raise NotImplementedError
    #         state_t = self.crank_nicolson(dt, err_tol=err_tol, max_iter=max_iter, compress_level=compress_level,
    #                                       verbose_plot=verbose_plot)
    #     else:
    #         raise NotImplementedError
    #
    #     if do_postprocessing:
    #         if process_kwargs is None:  process_kwargs = {}
    #         if state_t.fe is not None:  state_t.fe.smooth_data(compIDs=[0],inplace=True,**process_kwargs)
    #         if state_t.fi is not None:  state_t.fi.smooth_data(compIDs=[0],inplace=True,**process_kwargs)
    #
    #
    #     # plt.plot(state_t.fe.get_field_data(comps=0)[:,npts//2], label='post')
    #     # plt.xlabel('x')
    #     # plt.legend()
    #     # plt.show()
    #
    #     # i_tot = state_t.fields[state_t.fi].integrate('all')[0]
    #     if state_t.do_normalization:
    #         state_t.normalize() # target=i_tot)
    #
    #     # plt.figure()
    #     # fe_data = state_t[state_t.fe].get_field_data(comps=0)
    #     # fi_data = state_t[state_t.fi].get_field_data(comps=0)
    #     # diff_data = fe_data - fi_data
    #     # plt.imshow(fe_data-fi_data)
    #     # plt.colorbar()
    #     # plt.xlabel('v')
    #     # plt.ylabel('x')
    #     # plt.title('new charge 2')
    #
    #     # plt.figure()
    #     # plt.plot(diff_data[:,state0.q**state0.L//2])
    #     # plt.xlabel('x')
    #     # plt.ylabel('fe-fi')
    #     # plt.show()
    #
    #     return state_t

