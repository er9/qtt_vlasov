from setup.configs import *
import helper_quimb as helper
import helper_sl as helper_sl
from axis import Axis
from grid1D import Grid1D
from coord.cartesian import CartesianCoordinateSpace
from field import Field, ScalarField
from pde_system import PDE_system
from pde_vlasov import Vlasov
from pde_boltzmann import Boltzmann
from pde_darwin import Darwin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid


class VlasovDarwin(Vlasov):
    """ df/dt + v grad(f) + q/m (E) grad_v(f) = 0
    """
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField', 'Boltzmann']],
                 sys_fi: Optional[Union['ScalarField', 'Boltzmann']],
                 sys_ED: Optional['Darwin'] = None,
                 grid_X: Optional['Grid'] = None,
                 coords_x:  Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 f0_gradv_e=None, f0_gradv_i=None,
                 evolve_ion=True, evolve_EM=True, normalize=True, upwind=False, zipup=False,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 te_order=2, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            boundary_conditions: boundary conditions on V
               tuple of (bc_vectors [m vectors or mps], bc_results [m float]) such that
               bc_vectors (dot) V_data = bc_results
        """
        grid_X = grid_X if sys_ED is None else sys_ED.grid_X

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         te_order=te_order, compress_levels=compress_levels)

        self.sys_ED = sys_ED
        self.evolve_EM = evolve_EM
        self.deriv0: Optional['PDE_system'] = None


    @property
    def sys_ED(self) -> Optional['Darwin']:
        return self._sys_ED

    @sys_ED.setter
    def sys_ED(self, new_EM_sys: Optional['Darwin']):

        self._sys_ED = Darwin(None, None, None, grid_X=self.grid_X, coords_x=self.coords_x) \
                            if new_EM_sys is None else new_EM_sys

        if new_EM_sys is not None:
            assert (new_EM_sys.grid_X == self.grid_X), 'Maxwell grid_X does not match self'

            self.names['ET'] = self._sys_ED.names['ET'] if self._sys_ED.names['ET'] is not None else 'ET'
            self.names['EL'] = self._sys_ED.names['EL'] if self._sys_ED.names['EL'] is not None else 'EL'
            self.names['B'] = self._sys_ED.names['B'] if self._sys_ED.names['B'] is not None else 'B'
            self.field_names = self.field_names[:2] + [self.names['ET'], self.names['EL'], self.names['B']]

            if self.background_fe is not None:
                self.sys_fe.background_force_neg = self.compute_force_term(background_only=True)
            if self.background_fi is not None:
                self.sys_fi.background_force_neg = self.compute_force_term(is_ion=True, background_only=True)
            try:
                self.sys_fe.background_force_neg.scalar_multiply(-1, inplace=True)
            except AttributeError:
                pass

            try:
                self.sys_fi.background_force_neg.scalar_multiply(-1, inplace=True)
            except AttributeError:
                pass

        else:
            self.sys_fe.background_force_neg = None
            self.sys_fi.background_force_neg = None

    @property
    def ET(self) -> Optional['Field']:
        return self.sys_ED.ET

    @ET.setter
    def ET(self, new_field: 'Field'):
        self.sys_ED.ET = new_field

    @property
    def EL(self) -> Optional['Field']:
        return self.sys_ED.EL

    @EL.setter
    def EL(self, new_field: 'Field'):
        self.sys_ED.EL = new_field

    @property
    def E(self) -> Optional['Field']:
        return self.sys_ED.E

    @property
    def B(self) -> Optional['Field']:
        return self.sys_ED.B

    @B.setter
    def B(self, new_field: 'Field'):
        self.sys_ED.B = new_field


    def get_field(self, field_name):
        """ return ith component of the field. for convenience
        """
        if field_name == self.names['ET']:
            return self.ET
        elif field_name == self.names['EL']:
            return self.EL
        elif field_name == self.names['B']:
            return self.B
        else:
            return super().get_field(field_name)

    def set_field(self, field_name, new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        assert (isinstance(new_field, Field) or new_field is None), 'new_field must be Field object or None'

        if field_name == self.names['ET']:
            self.ET = new_field
        elif field_name == self.names['EL']:
            self.EL = new_field
        elif field_name == self.names['B']:
            self.B = new_field
        else:
            self._fields[field_name] = new_field


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
        if len(new_fields) < 5:
            new_fields = new_fields + (None,) *(5-len(new_fields))

        dist_e, dist_i, ET, EL, B = new_fields[:5]
        new_system = super().create_like(dist_e, dist_i, recalc=recalc, deep=deep)

        sys_EM = self.sys_ED.create_like(ET, EL, B, recalc=recalc)
        new_system._sys_ED = sys_EM

        new_system.field_names = self.field_names
        new_system.names = self.names
        new_system.evolve_EM = self.evolve_EM

        new_system.sys_fe.background_force_neg = self.sys_fe.background_force_neg
        new_system.sys_fi.background_force_neg = self.sys_fi.background_force_neg

        return new_system


    def copy(self):
        """ create a new system like this with new fields
        """
        new_system = super().copy()
        # new_system._total_E = self._total_E
        # new_system._total_B = self._total_B
        return new_system


    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['VlasovDarwin'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False,
              **deriv_kwargs) -> 'VlasovDarwin':
        """ perform explicit Euler time evolution
            compress:  final compression of final state, updateV
            compress1: compression of derivative
            compress2: compress1 of calculate derivative, updateV
        """
        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time,compress_level=compress_level,
                                                    compress_level1=compress_level1, compress_level2=compress_level2,
                                                verbose_plot=verbose_plot)

        state1 = super().euler(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               verbose_plot=verbose_plot)
        state1.deriv0 = deriv0

        if np.abs(state1.elc_params.e) > 0 and self.evolve_EM:
            state1.update_EM_sys(inplace=True, deriv_fe=deriv0.sys_fe.f, deriv_fi=deriv0.sys_fi.f,
                                 compress=compress_level, compress1=compress_level2)

        state1.time = self.time + dt if self.time is not None else None

        return state1


    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_force=True, internal_force=True, only_EL=False, **kwargs):
        """ compute EM force:  q/m (E + v x B)
        """
        # print('EM computing force', is_ion)
        coords_v = self.coords_vi if is_ion else self.coords_ve
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params


        background_B0 = self.sys_ED.background_B0 if background_force else None
        background_E0 = self.sys_ED.background_E0 if background_force else None

        # print('compress', compress_level, compress_level1, compress_level2)

        if self.B is not None and internal_force:
            B = self.B.add(background_B0, compress_level=compress_level2, inplace=False)
        else:
            B = background_B0.copy()

        if verbose_plot:
            plt.figure()
            for compID in B.componentIDs:
                B_data = B.get_comp_data(compID)
                if B_data is not None:
                    plt.plot(B_data,label=f'B {compID}')
            plt.legend()
            plt.show()

        # print('background only', background_only)
        # print('em cross', velocities.componentIDs, B.componentIDs)
        em_term = velocities.cross_product(B, self.coords_x,
                                           compress_level=compress_level1, inner_compress_level=compress_level2,
                                           verbose_plot=False,) \
            if B is not None else None

        if matl_params.is_cgs:
            em_term.scalar_multiply( 1./ matl_params.c, inplace=True)

        E = self.EL if only_EL else self.E
        if E is not None and internal_force:
            E = E.add(background_E0, compress_level=compress_level2, inplace=False)
        else:
            E = background_E0

        if verbose_plot:
            if E is not None:
                plt.figure()
                for compID in E.componentIDs:
                    E_data = E.get_comp_data(compID)
                    if E_data is not None:
                        plt.plot(E_data, label=f'E {compID}')
                plt.legend()
                plt.show()

        if em_term is not None:
            em_term = em_term.add(E, compress_level=compress_level2)
        else:
            em_term = E.copy()

        if em_term is not None:
            em_term = em_term.scalar_multiply(matl_params.e * matl_params.Z / matl_params.mass, inplace=False)
            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

        ## set which axes are constant
        if background_force and not internal_force:  ## background only
            # print('background force ONLY')
            for C in em_term.componentIDs:
                E0_C = None if background_E0 is None else background_E0.components.get(C, None)
                B0_C = None if background_B0 is None else background_B0.components.get(C, None)
                if (E0_C is None or E0_C.is_constant) and (B0_C is None or B0_C.is_constant):
                    vC_ax = coords_v.get_axis(C.type)  ## direction of advection (df/dv axis)
                    em_term[C].constant_axes = [ax for ax in self.grid_X.axes] + [vC_ax]
                    # print('set em term constant axes', C, em_term[C].constant_axes)
                    # exit()

        if verbose_plot:
            print('get force term plot')
            x_coords = self.coords_x.coords
            # ax_x, ax_y, ax_vx, ax_vy, ax_vz = em_term[em_term.componentIDs[0]].grid.axes
            ax_x, ax_vx, ax_vy, ax_vz = em_term[em_term.componentIDs[0]].grid.axes
            for x_c in x_coords:
                em_term_data = em_term.get_comp_data(x_c, ax_select={# ax_x: ax_x.npts//2,
                                                                     # ax_vx: ax_vx.npts//2,
                                                                     ax_vy: ax_vy.npts//2,
                                                                     ax_vz: ax_vz.npts//2,
                                                                     })
                if em_term_data is None:
                    print('em term data is None', x_c)
                    continue
                plt.figure()
                plt.imshow(np.real(em_term_data))
                plt.title(f'em {x_c} re')
                plt.xlabel('vx')
                plt.ylabel('x')
                plt.colorbar()

                plt.figure()
                plt.imshow(np.imag(em_term_data))
                plt.title(f'em {x_c} im')
                plt.xlabel('vx')
                plt.ylabel('x')
                plt.colorbar()
            plt.show()

        # print('E + v x B term', em_term.components)

        return em_term


    def compute_force_term_boris(self, dt, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                                 verbose_plot=False, background_force=True, internal_force=True, **kwargs):
        """ compute "advec coeffs" for EM force:  q/m (E + v x B)
            boris update:  semi-implicit update of v:  v[n+1] - v[n] / dt = q/m (E + (v[n+1]+v[n])/2 x B)
            translate v[n+1] into term compatible with force advec coeff = (v[n+1] - v[n]) / dt

            coeff = dt / 2 * q / m
            t = coeff * B
            s = 2 t / (1 + |t|^2)

            v- = v[n] + coeff * E
            v+ = v[n+1] - coeff * E

            v' = v- + v- x t
            v+ = v- + v' x s

            v+ - v- = v' x s = v- x s + v- x t x s
            v[n+1] - v[n] = v+ - v- - 2 * coeff * E

            = [vx] ( vx * (-sy ty - sz tz) + vy * (sz + sy tx) + vz (-sy + tx sz) )
              [vy] ( vx * (-sz + sx ty) + vy (-tx sx - sz tz) + vz (sx + ty sz) )
              [vz] ( vx * (sy + sx tz) + vy (-sx + tz sy) + vz (-tx sx - ty sy) )

            include backward trajectory, because the updates along each v direction are not done simultaneously?
        """
        # print('boris force')
        coords_v = self.coords_vi if is_ion else self.coords_ve
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        background_B0 = self.sys_ED.background_B0 if background_force else None
        background_E0 = self.sys_ED.background_E0 if background_force else None

        # print('compress', compress_level, compress_level1, compress_level2)

        if self.B is not None and internal_force:
            B = self.B.add(background_B0, compress_level=compress_level2)
        else:
            B = background_B0

        if matl_params.is_cgs:
            B = B.scalar_multiply( 1./ matl_params.c, inplace=False)

        if self.E is not None and internal_force:
            E = self.E.add(background_E0, compress_level=compress_level2)
        else:
            E = background_E0

        coeff = dt * matl_params.charge / 2 / matl_params.mass
        tot_V = np.prod([ax.length() + ax.dx for ax in B.grid.axes])
        B_mag = np.sqrt(B.norm() ** 2 / tot_V)
        # print('matl params', matl_params.charge, matl_params.mass)
        # print('B field norm', B.norm()**2, tot_V, B_mag)
        # print('E', E.componentIDs, B.componentIDs, velocities.componentIDs)
        t_field = B.scalar_multiply(-coeff, inplace=False)
        s_field = B.scalar_multiply(2 * coeff / (1 + (coeff*B_mag)**2) )

        ## calculate v^[n+1] from v^[n] using Lorentz force
        vm_field = velocities.copy()

        v1_field = vm_field.add(vm_field.cross_product(t_field, self.coords_x), inplace=False)
        v2_field = vm_field.add(v1_field.cross_product(s_field, self.coords_x), inplace=False)  # vp

        # v_next = v2_field.add( E.scalar_multiply(coeff, inplace=False), inplace=False ) if E is not None else v2_field
        v_next = v2_field
        v_diff = v_next.add(velocities.scalar_multiply(-1))

        if E is not None:
            v_diff = v_diff.add(E.scalar_multiply(2 * coeff, inplace=False))

        v_diff = v_diff.scalar_multiply(1 / dt, inplace=False)

        ## set which axes are constant
        if background_force and not internal_force:  ## background only
            # print('background force ONLY')
            for C in v_diff.componentIDs:
                E0_C = None if background_E0 is None else background_E0.components.get(C,None)
                B0_C = None if background_B0 is None else background_B0.components.get(C,None)
                if (E0_C is None or E0_C.is_constant) and (B0_C is None or B0_C.is_constant):
                    v_diff[C].constant_axes = [ax for ax in self.grid_X.axes]
                    # print('v diff', v_diff[C].constant_axes)
                    ## or, union of E0_C, B0_C constant axes?

        return v_diff



    # def compute_force_term_bg_SL(self, dt, is_ion=False, sl_order=DEFAULT_SL_ORDER, compress_level=1, compress_level1=0,
    #                              compress_level2=0, verbose_plot=False):
    #     """ assumes field is constant in space
    #     """
    #     sys_f = self.sys_fi if is_ion else self.sys_fe
    #     coords_v = self.coords_vi if is_ion else self.coords_ve
    #     velocities = self.velocities_i if is_ion else self.velocities_e
    #     spec_params = self.ion_params if is_ion else self.elc_params
    #     background_B0 = self.sys_ED.background_B0
    #     background_E0 = self.sys_ED.background_E0
    #
    #     em_SL_dict = sys_f.background_force_SL
    #     mult_const = spec_params.e * spec_params.Z / spec_params.mass
    #
    #     if background_E0 is not None or background_B0 is not None:
    #
    #         if em_SL_dict is None:  ## compute these terms
    #
    #             if background_B0 is None:
    #                 em_SL_dict = {}
    #                 for C in background_E0.componentIDs:
    #                     EC = background_E0[C]
    #                     v0_ax = coords_v.get_axis(C)
    #                     CV = v0_ax.coordinate
    #                     if EC.data is not None:
    #                         if EC.is_constant:
    #                             select_gtn = EC.grid.get_select_elems_mps([0])
    #                             E0_val = EC.ovlp(select_gtn) * mult_const
    #                             em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_val, sl_order=sl_order)
    #                         else:
    #                             E0_data = background_E0.get_comp_data(C) * mult_const
    #                             em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_data, sl_order=sl_order)
    #                 sys_f.background_force_SL = em_SL_dict
    #
    #             elif background_E0 is None:
    #
    #                 em_SL_dict = {}
    #                 comps = self.coords_x.coords
    #                 for i in range(3):
    #                     C0 = comps[i]
    #                     C1, C2 = comps[(i + 1) % 3], comps[(i + 2) % 3]
    #
    #                     v0_ax = coords_v.get_axis(C0)
    #                     CV = v0_ax.coordinate
    #
    #                     B1s = background_B0.components.get(C1, None)
    #                     B2s = background_B0.components.get(C2, None)
    #
    #                     v1s = velocities.components.get(C1, None)
    #                     v2s = velocities.components.get(C2, None)
    #
    #                     v1_ax = coords_v.get_axis(C1)
    #                     v2_ax = coords_v.get_axis(C2)
    #
    #                     comp = sys_f.grid.make_gtn_from_dicts([{v1_ax: v1s.elemental_multiply(B2s, compress=4)},
    #                                                            {v2_ax: v2s.elemental_multuply(B1s.scalar_multiply(-1),
    #                                                                                           compress=4)}])
    #                     comp_data = comp.get_data(ax_select={**{x_ax: 0 for x_ax in self.coords_x.axes}, v0_ax: 0})
    #
    #                     em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, comp_data * mult_const, sl_order=sl_order)
    #
    #             else:
    #                 ## E0 + v x B0 -- need to find where E0 > v x B0
    #                 raise NotImplementedError
    #
    #         return em_SL_dict
    #     else:
    #         return {}
    #
    #
    # @profile
    def update_EM_sys(self, dt, inplace=False,
                  deriv_fe: Optional['ScalarField']=None, deriv_fi: Optional['ScalarField']=None,
                  compress=1, compress1=4, compress2=5) -> 'Darwin':
        """ div E = div^2 V = 4*pi*rho
            compress:  compression of V
            compress1: compression of terms to create V
        """
        # ## update EM components
        # if np.abs(state0.matl_params.e) > 0:
        #     state0.sys_EM.charge_density = state0.compute_charge_density()
        #     state0.sys_EM.current_density = state0.compute_current()
        #     state0.sys_EM.current_density_time_deriv = state0.compute_current_time_derivative()
        #     state0.sys_EM.update_EM(inplace=True, compress=compress_level, compress1=comp2)


        new_sys = self if inplace else self.copy()
        new_sys.sys_ED.charge_density = new_sys.compute_charge_density(compress=compress)
        new_sys.sys_ED.current_density = new_sys.compute_current(compress=compress)

        new_sys.sys_ED.update_EM(inplace=True, compress=compress, compress1=compress1)
        new_sys.update_ET(inplace=True, compress=compress, compress1=compress1)
        return new_sys.sys_ED


    def update_ET(self, deriv_fe: Optional['ScalarField']=None, deriv_fi: Optional['ScalarField']=None,
                  inplace=False, compress=1, compress1=4, compress2=5, verbose_plot=False) -> 'Field':
        """ different methods:  compute time derivative of f and then compute its current
        """
        # print('update ET', compress, compress1, compress2)
        # verbose_plot = True
        version = 2     # 2 is preferred esp for sqrt
        do_pseudo = False

        new_sys_ED = self.sys_ED if inplace else self.sys_ED.copy()
        # const = self.matl_params.mu0  ## need to double-check this

        ## LHS:
        ## pseudospectral method to take inverse
        if not isinstance(self.coords_x, CartesianCoordinateSpace):
            raise NotImplementedError('these inverse laplacians only valid for Cartesian Coordinates')

        # ### integrate d/dv f * v
        # ve_axes = [self.ve_axes_dict[C] for C in self.coords_x.coords]
        # fe_gradv = self.sys_fe.f.gradient(deriv_axes=ve_axes, compress_level=compress1)
        # vi_axes = [self.vi_axes_dict[C] for C in self.coords_x.coords]
        # fi_gradv = self.sys_fi.f.gradient(deriv_axes=vi_axes, compress_level=compress1)
        #
        # ## compute current
        # je = fe_gradv.xmultiply(ve_axes, compress_level=compress1)
        # je.replace_compIDs({ve_axes[i].coordinate: self.coords_x.coords[i] for i in range(len(ve_axes))})
        # # print('je compIDs', je.componentIDs, self.sys_fe.velocities.componentIDs)
        # je = je.meas_expec(self.sys_fe.velocities.components, integ_axes=ve_axes, new_grid=self.grid_X,
        #                    new_ax_deriv_configs=self.sys_fe.f.component.ax_deriv_configs)
        # je.scalar_multiply(self.elc_params.charge / self.elc_params.mass, inplace=True)
        #
        # ji = fi_gradv.xmultiply(vi_axes, compress_level=compress1)
        # ji.replace_compIDs({vi_axes[i].coordinate: self.coords_x.coords[i] for i in range(len(vi_axes))})
        # ji = ji.meas_expec(self.sys_fi.velocities.components, integ_axes=vi_axes, new_grid=self.grid_X,
        #                    new_ax_deriv_configs=self.sys_fi.f.component.ax_deriv_configs)
        # ji.scalar_multiply(self.ion_params.charge / self.ion_params.mass, inplace=True)
        #
        # correction = je.add(ji, compress_level=compress1)

        ### integrate f (keep spatially dependent).
        correction_e = self.sys_fe.compute_density()
        correction_e = correction_e.scalar_multiply(self.elc_params.charge ** 2 / self.elc_params.mass)

        correction_i = self.sys_fi.compute_density()
        correction_i = correction_i.scalar_multiply(self.ion_params.charge ** 2 / self.ion_params.mass)

        correction = correction_e.add(correction_i, compress_level=compress1)
        correction = correction.component

        ## integrate f, approximate as spatially constant
        # tot_V = self.grid_X.volume()
        # correction_e = self.sys_fe.f.integrate().component / tot_V * self.elc_params.charge ** 2 / self.elc_params.mass
        # correction_i = self.sys_fi.f.integrate().component / tot_V * self.ion_params.charge ** 2 / self.ion_params.mass
        # correction = self.grid_X.get_ones_mps()
        # correction.scalar_multiply(correction_e + correction_i, inplace=True)

        if self.matl_params.is_cgs:
            correction = correction.scalar_multiply(-4 * np.pi / self.matl_params.c ** 2, inplace=True)
        else:
            correction = correction.scalar_multiply(-self.matl_params.mu0, inplace=True)


        grid_X = self.grid_X

        if grid_X.is_k():

            k2s_list = []
            for ax in grid_X.axes:
                k2s = ax.xpts ** 2
                mps_k2s = ax.map_state_to_mps(k2s)
                helper.scalar_multiply(mps_k2s, -1., inplace=True)
                mps_k2s_dict = {ax_other: ax_other.get_all_ones_mps() for ax_other in grid_X.axes if ax_other != ax}
                mps_k2s_dict[ax] = mps_k2s
                k2s_list += [mps_k2s_dict]

            lapl_mps = grid_X.make_gtn_from_dicts(k2s_list, data_type=DataType.MPS)
            correction_k = grid_X.get_ones_mps()
            coeff = self.elc_params.charge ** 2 / self.elc_params.mass + \
                    self.ion_params.charge ** 2 / self.ion_params.mass
            coeff *= correction     ## assume is a number...
            if self.matl_params.is_cgs:
                coeff *= 4 * np.pi / self.matl_params.c ** 2
            else:
                coeff *= self.matl_params.mu0
            lapl_mps.add(correction_k.scalar_multiply(-1. * coeff), inplace=True, compress=compress1)
            inv_lapl_mpo = grid_X.get_diag_mpo_inverse(lapl_mps,
                                                       compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})

        else:

            ## pseudospectral method. this doesn't work??
            if do_pseudo:
                lapl_mps = grid_X.get_qft_ks(power=2)
                lapl_mps = lapl_mps.scalar_multiply(-1, inplace=False)

                # correction_k = correction.apply_qft(compress=False)
                ### approximate correction as a constant
                correction_k = grid_X.get_ones_mps()
                lapl_mps.add(correction_k.scalar_multiply(-1. * self.matl_params.mu0),
                             inplace=True, compress=True)

                inv_lapl_mpo = grid_X.get_diag_mpo_inverse(lapl_mps, compress_opts=None)
                inv_lapl_mpo.apply_inverse_qft(inplace=True, compress=True,
                                               compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})

                ### o.w.
                # correction_k = correction.mps_to_diag_mpo()
                # correction_k.apply_qft(inplace=True, compress=True)
                ## and can no longer use get_diag_mpo_inverse

                # plt.figure()
                # plt.imshow(np.real(inv_lapl_mpo.get_data()))
                # plt.title('qft inverse (re)')
                # plt.colorbar()
                #
                # plt.figure()
                # plt.imshow(np.imag(inv_lapl_mpo.get_data()))
                # plt.title('qft inverse (im)')
                # plt.colorbar()
                # plt.show()

            else:
                ## naive inversion
                # TODO: be more careful with ax_deriv_configs
                lapl_mpo = grid_X.laplacian_mpo(ax_deriv_configs=
                                                {ax: DerivativeConfiguration(left_bc=BCType.PERIODIC, order=1)
                                                 for ax in self.grid_X.axes})
                correction = correction.mps_to_diag_mpo()
                lapl_mpo = lapl_mpo.add(correction, inplace=False)
                inv_lapl_mpo = grid_X.get_mpo_inverse(lapl_mpo)

                # plt.figure()
                # plt.imshow(inv_lapl_mpo.get_data())
                # plt.title('naive inverse')
                # plt.colorbar()
                # plt.show()



        ### RHS
        if version==1:
            ### method 1:  more expensive but easier to implement?
            if deriv_fe is None:
                store_fe_force = self.sys_fe.force_term
                sys_fe = self.sys_fe
                if sys_fe.f.is_sqrt:
                    sys_fe = sys_fe.copy()
                    fe = sys_fe.f
                    fe_cc = fe.component.conj(mangle_inner=True)
                    fe_cc = fe_cc.apply_elemental_multiply_op()
                    fe.component.apply(fe_cc, zipup=True, compress=compress1, inplace=True)
                    fe.is_sqrt = False
                sys_fe.force_term = self.compute_force_term(is_ion=False, only_EL=True)
                deriv_fe = sys_fe.calculate_time_derivative(compress_level=compress1, compress_level1=compress2)
                sys_fe.force_term = store_fe_force

            if deriv_fi is None:
                store_fi_force = self.sys_fi.force_term
                sys_fi = self.sys_fi
                if sys_fi.f.is_sqrt:
                    sys_fi = sys_fi.copy()
                    fi = sys_fi.f
                    fi_cc = fi.component.conj(mangle_inner=True)
                    fi_cc = fi_cc.apply_elemental_multiply_op()
                    fi.component.apply(fi_cc, zipup=True, compress=compress1, inplace=True)
                    fi.is_sqrt = False
                sys_fi.force_term = self.compute_force_term(is_ion=True, only_EL=True)
                deriv_fi = sys_fi.calculate_time_derivative(compress_level=compress1, compress_level1=compress2)
                sys_fi.force_term = store_fi_force

            if verbose_plot:
                plt.figure()
                ax_ve_x, ax_ve_y, ax_ve_z = [self.ve_axes_dict[C] for C in self.coords_x.coords]
                plt.imshow(deriv_fe.f.get_comp_data(ax_select={ax_ve_x: ax_ve_x.npts // 2, ax_ve_z: ax_ve_z.npts // 2}),
                           )
                plt.title('dfe/dt')
                plt.colorbar()
                plt.show()

                plt.figure()
                ax_vi_x, ax_vi_y, ax_vi_z = [self.vi_axes_dict[C] for C in self.coords_x.coords]
                plt.imshow(deriv_fi.f.get_comp_data(ax_select={ax_vi_y: ax_vi_y.npts // 2, ax_vi_z: ax_vi_z.npts // 2}),
                           )
                plt.title('dfi/dt')
                plt.colorbar()
                plt.show()

            dt_flow_L_e = deriv_fe.compute_flow(compress=compress1, compress1=compress2)
            dt_flow_L_i = deriv_fi.compute_flow(compress=compress1, compress1=compress2)

            dt_flow_L_e.scalar_multiply(self.elc_params.charge, inplace=True)
            dt_flow_L_i.scalar_multiply(self.ion_params.charge, inplace=True)

            rhs = dt_flow_L_e.add(dt_flow_L_i, inplace=False, compress_level=compress1)
            if self.matl_params.is_cgs:
                rhs.scalar_multiply(+4 * np.pi / self.matl_params.c**2 , inplace=True)
            else:
                rhs.scalar_multiply(+self.matl_params.mu0, inplace=True)

            if verbose_plot:
                plt.figure()
                for C in self.coords_x.coords:
                    if C in dt_flow_L_i.componentIDs and dt_flow_L_i[C].data is not None:
                        plt.plot(dt_flow_L_i.get_comp_data(C), label=f'ion flow rhs {C}')
                plt.legend()

                plt.figure()
                for C in self.coords_x.coords:
                    if C in dt_flow_L_e.componentIDs and dt_flow_L_e[C].data is not None:
                        plt.plot(dt_flow_L_e.get_comp_data(C), label=f'elc flow rhs {C}')
                plt.legend()

                je = self.sys_fe.compute_flow(compress=compress1, compress1=compress2)
                plt.figure()
                for C in self.coords_x.coords:
                    if C in je.componentIDs:
                        plt.plot(je.get_comp_data(C), label=f'je {C}')
                plt.legend()
                plt.show()

        elif version == 2:

            ### method 2:  broken down components
            ### lapl(ET) - a^2 w^2 ET = sum_s  grad(rho_s) <v^2>_s + q_s rho_s / m_s ( E_L + <v>_s x B )
            ### a = v0/c (in the paper's normalization)
            ### w^2 = sum_s q_s rho_s / m_s

            def get_rhs(is_ion=False):
                sys_f = self.sys_fi if is_ion else self.sys_fe
                spec_params = self.ion_params if is_ion else self.elc_params
                v_axes_dict = self.vi_axes_dict if is_ion else self.ve_axes_dict

                flows = sys_f.compute_flow(compress=compress1, compress1=0)     # mp charges included
                # force_term = flows.cross_product(self.B, compress_level=compress1, inner_compress_level=0)
                force_term = flows.cross_product(self.sys_ED.total_B, self.coords_x,
                                                 compress_level=compress1, inner_compress_level=0)
                if self.matl_params.is_cgs:
                    force_term.scalar_multiply(1/self.matl_params.c, inplace=True)

                densities = sys_f.compute_density(compress=compress1)   # no charges included
                E_term = self.EL.elemental_multiply(densities.component, zipup=True, inplace=False,
                                                    compress_level=compress1)
                force_term.add(E_term, inplace=True, compress_level=compress1 )
                force_term.scalar_multiply(spec_params.charge**2 / spec_params.mass, inplace=True)

                if verbose_plot:

                    plt.figure()
                    plt.plot(densities.get_comp_data(), label='density')
                    plt.plot(self.EL.get_comp_data(self.coords_x.coords[0]), label='EL[X]')
                    plt.plot(E_term.get_comp_data(self.coords_x.coords[0]), label='F[X] * rho')
                    plt.legend()
                    plt.show()

                    plt.figure()
                    for C in self.coords_x.coords:
                        if C in force_term.componentIDs:
                            force_data = force_term.get_comp_data(C)
                            if force_data is not None:
                                plt.plot(force_data, label=f'force term rhs {C}')
                    plt.legend()
                    plt.show()

                vv_dict = {}
                for C1 in self.coords_x.coords:
                    v_ax1 = v_axes_dict[C1]

                    vv_sum = None
                    for C2 in self.coords_x.coords:
                        v_ax2 = v_axes_dict[C2]
                        x_ax2 = self.coords_x.get_axis(C2.type)

                        if x_ax2 is None:
                            continue

                        # print('C1', C1, v_ax1)
                        # print('C2', C2, x_ax2, v_ax2)
                        # print('C1==C2', C1 == C2)

                        if C1 == C2:
                            vv = sys_f.compute_moment([v_ax1], powers=[2], compress=compress1)
                        else:
                            vv = sys_f.compute_moment([v_ax1, v_ax2], compress=compress1)

                        grad_vv = vv.gradient(deriv_axes=[x_ax2], compress_level=0)

                        if vv_sum is not None:
                            vv_sum.add(grad_vv[C2], inplace=True)
                        else:
                            vv_sum = grad_vv[C2]

                    if compress1:
                        vv_sum.compress(inplace=True,)

                    vv_dict[C1] = vv_sum.scalar_multiply(-1 * spec_params.charge, inplace=True)

                vv_term = force_term.create_like(new_components=vv_dict)
                rhs = force_term.add(vv_term, inplace=True)
                return rhs

            ion_rhs = get_rhs(is_ion=True)
            elc_rhs = get_rhs(is_ion=False)
            rhs = ion_rhs.add(elc_rhs, inplace=False, compress_level=compress1)

            if self.matl_params.is_cgs:
                rhs.scalar_multiply(4 * np.pi / self.matl_params.c ** 2, inplace=True)
            else:
                rhs.scalar_multiply(self.matl_params.mu0, inplace=True)

            if verbose_plot:
                plt.figure()
                for C in self.coords_x.coords:
                    if C in ion_rhs.componentIDs:
                        plt.plot(ion_rhs.get_comp_data(C), label=f'ion rhs {C}')
                plt.legend()

                plt.figure()
                for C in self.coords_x.coords:
                    if C in elc_rhs.componentIDs:
                        plt.plot(elc_rhs.get_comp_data(C), label=f'elc rhs {C}')
                plt.legend()

        else:
            raise NotImplementedError('check desired rhs version')

        if verbose_plot:
            plt.figure()
            for C in self.coords_x.coords:
                if C in rhs.componentIDs and rhs[C].data is not None:
                    plt.plot(rhs.get_comp_data(C), label=f'rhs {C}')
            plt.legend()
            plt.show()


        ### solve for ET (only valid for cartesian coordinates)
        ET = rhs.copy()
        for C in ET.componentIDs:

            ET[C].apply(inv_lapl_mpo, inplace=True, compress=compress,
                         compress_opts=self.ET.compress_config[compress])

        ### project ET onto div(ET) = 0
        Theta_term = ET.divergence(self.coords_x, compress_level=compress1)

        if Theta_term is not None and Theta_term.component is not None:
            inv_lapl_mpo_ET = self.grid_X.inverse_laplacian_mpo(ax_deriv_configs=new_sys_ED.divET_ax_deriv_configs)
            Theta_term.component.apply(inv_lapl_mpo_ET, inplace=True, zipup=self.zipup)  # no correction to laplacian

            if verbose_plot:
                plt.figure()
                Theta_term.component.ax_deriv_configs = {ax: DerivativeConfiguration(left_bc=BCType.PERIODIC, order=1)
                                                         for ax in self.grid_X.axes}
                test = Theta_term.laplacian(self.coords_x, compress_level=0)

                plt.plot(Theta_term.get_comp_data(), label='Theta')
                plt.plot(test.get_comp_data(), '--', label='lapl(Theta)')
                plt.plot(ET.divergence(self.coords_x).get_comp_data(), ':', label='div(ET)')
                plt.plot(ET.get_comp_data(self.coords_x.coords[0]), ':', label=f'ET X')
                plt.legend()
                plt.show()

            Theta_grad = Theta_term.gradient(compress_level=compress1)
            Theta_grad.scalar_multiply(-1, inplace=True)

            if verbose_plot:
                plt.figure()
                plt.plot(Theta_term.get_comp_data(), label=f'inv lapl (divET)')
                for C in self.coords_x.coords:
                    if C in Theta_grad.componentIDs:
                        plt.plot(Theta_grad.get_comp_data(C), label=f'- grad Theta {C}')
                        plt.plot(rhs.get_comp_data(C), label=f'rhs {C}')
                        plt.plot(ET.get_comp_data(C), label=f'ET {C}')
                plt.legend()
                plt.show()

            new_ET = ET.add(Theta_grad, inplace=False, compress_level=compress2)

            # new_divET_term = new_ET.divergence(self.coords_x, compress_level=compress1)
            # if new_divET_term.component is not None and new_divET_term.component.data is not None:
            #     print('not div free', new_divET_term.norm())
            #
            #     if verbose_plot:
            #         plt.figure()
            #         for C in new_ET.componentIDs:
            #             if new_ET[C] is not None and new_ET[C].data is not None:
            #                 plt.plot(new_ET.get_comp_data(C), label=f'new ET {C}')
            #         if new_divET_term.component.data is not None:
            #             plt.plot(new_divET_term.get_comp_data(), label='div new ET')
            #         plt.legend()
            #         plt.show()

        else:
            new_ET = ET

        ## inplace update for ET
        new_sys_ED.ET.clear()
        for C in new_ET.componentIDs:
            new_sys_ED.ET[C] = new_ET[C]

        if verbose_plot:
            plt.figure()
            for C in new_ET.componentIDs:
                if new_ET[C].data is not None:
                    plt.plot(new_ET.get_comp_data(C), label = f'new ET {C}')
            plt.legend()
            plt.title('new ET')
            plt.show()

        return new_sys_ED


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
            ## evolve velocity advection dt/2
            state0 = state0.get_vel_advection(dt/2, inplace=True, method=method_v, verbose_plot=verbose_plot,
                                              compress=comp1, compress1=comp3, compress2=comp5, )

        # else:

        ## update V or EM sys
        if state0.evolve_EM:
            print('update ED sys')
            state0.sys_ED = state0.update_EM_sys(dt, inplace=True, compress=comp1)

        print('force advection')
        state0 = state0.get_force_advection(dt, inplace=True, method=method_f, verbose_plot=verbose_plot,
                                            compress=comp1, compress1=comp3, compress2=comp5,)

        dt_ = dt / 2 if is_last_time_step else dt
        state0 = state0.get_vel_advection(dt_, inplace=True, method=method_v, verbose_plot=verbose_plot,
                                          compress=comp1, compress1=comp3, compress2=comp5, )


        if verbose_plot:
            from pde_vlasovES import VlasovPoisson
            if isinstance(self, VlasovPoisson):
                V_data = state0.V.get_comp_data()
                plt.figure()
                plt.plot(V_data)
                plt.title('V')
                plt.show()

        ## include collisions (cross ion/elc collisions)
        print('collisions', self.collision.coll_type)
        if self.collision.coll_type is not None:
            # raise NotImplementedError('collisions not implemented')
            # deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
            deriv_coll_e = state0.sys_fe.get_collision_term()
            deriv_coll_i = state0.sys_fi.get_collision_term()
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        state0.time = self.time + dt if self.time is not None else None
        return state0
