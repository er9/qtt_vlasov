from setup.configs import *
import helper_quimb as helper
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
                 sys_EM: Optional['Darwin'] = None,
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
        grid_X = grid_X if sys_EM is None else sys_EM.grid_X

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         te_order=te_order, compress_levels=compress_levels)

        self.sys_EM = sys_EM
        self.evolve_EM = evolve_EM
        self.deriv0: Optional['PDE_system'] = None


    @property
    def sys_EM(self) -> Optional['Darwin']:
        return self._sys_EM

    @sys_EM.setter
    def sys_EM(self, new_EM_sys: Optional['Darwin']):

        self._sys_EM = Darwin(None, None, None, grid_X=self.grid_X) if new_EM_sys is None else new_EM_sys

        if new_EM_sys is not None:
            assert (new_EM_sys.grid_X == self.grid_X), 'Maxwell grid_X does not match self'

            self.names['ET'] = self._sys_EM.names['ET'] if self._sys_EM.names['ET'] is not None else 'ET'
            self.names['EL'] = self._sys_EM.names['EL'] if self._sys_EM.names['EL'] is not None else 'EL'
            self.names['B'] = self._sys_EM.names['B'] if self._sys_EM.names['B'] is not None else 'B'
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
        return self.sys_EM.ET

    @ET.setter
    def ET(self, new_field: 'Field'):
        self.sys_EM.ET = new_field

    @property
    def EL(self) -> Optional['Field']:
        return self.sys_EM.EL

    @EL.setter
    def EL(self, new_field: 'Field'):
        self.sys_EM.EL = new_field

    @property
    def E(self) -> Optional['Field']:
        return self.sys_EM.E

    @property
    def B(self) -> Optional['Field']:
        return self.sys_EM.B

    @B.setter
    def B(self, new_field: 'Field'):
        self.sys_EM.B = new_field

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

        sys_EM = self.sys_EM.create_like(ET, EL, B, recalc=recalc)
        new_system._sys_EM = sys_EM

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
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False) \
            -> 'VlasovDarwin':
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
            state1.update_EM(inplace=True, deriv_fe=deriv0.sys_fe.f, deriv_fi=deriv0.sys_fi.f,
                             compress=compress_level, compress1=compress_level2)

        state1.time = self.time + dt if self.time is not None else None

        return state1


    def compute_force_term(self, is_ion=False, only_EL=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_only=False):
        """ compute EM force:  q/m (E + v x B)
        """
        # print('EM computing force', is_ion)
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        background_B0 = self.sys_EM.background_B0
        background_E0 = self.sys_EM.background_E0

        # print('compress', compress_level, compress_level1, compress_level2)

        if self.B is not None and not background_only:
            B = self.B.add(background_B0, compress_level=compress_level2)
        else:
            B = background_B0

        # print('background only', background_only)
        # print('em cross', velocities.componentIDs, B.componentIDs)
        em_term = velocities.cross_product(B, compress_level=compress_level1, inner_compress_level=compress_level2,
                                           verbose_plot=False,) \
            if self.B is not None else None

        E = self.EL if only_EL else self.E
        if E is not None and not background_only:
            E = E.add(self.sys_EM.background_E0, compress_level=compress_level2)
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


    # @profile
    def update_EM_sys(self, inplace=False,
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
        new_sys.sys_EM.charge_density = self.compute_charge_density(compress=compress)
        new_sys.sys_EM.current_density = self.compute_current(compress=compress)

        new_sys.sys_EM.update_EM(inplace=True, compress=compress, compress1=compress1)
        new_sys.update_ET(inplace=True, compress=compress, compress1=compress1)
        return new_sys.sys_EM


    def update_ET(self, deriv_fe: Optional['ScalarField']=None, deriv_fi: Optional['ScalarField']=None,
                  inplace=False, compress=1, compress1=4, compress2=5) -> 'Field':
        """ different methods:  compute time derivative of f and then compute its current
        """
        new_sys_EM = self.sys_EM if inplace else self.sys_EM.copy()

        ### left-hand side
        correction_e = self.sys_fe.compute_density()
        correction_e = correction_e.scalar_multiply( self.elc_params.charge / self.elc_params.mass )

        correction_i = self.sys_fi.compute_density()
        correction_i = correction_i.scalar_multiply(self.ion_params.charge / self.ion_params.mass)

        correction = correction_e.add(correction_i, compress_level=compress1)
        correction = correction.component

        const = self.matl_params.mu0    ## need to double-check this
        grid_X = self.grid_X

        ## pseudospectral method to take inverse
        if isinstance(self.coords_x, CartesianCoordinateSpace):
            raise NotImplementedError('these inverse laplacians only valid for Cartesian Coordinates')

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
            lapl_mps.add(correction.scalar_multiply(-const**2), compress=compress1)
            inv_lapl_mpo = grid_X.get_diag_mpo_inverse(lapl_mps,
                                                       compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})

        else:
            lapl_mps = grid_X.get_qft_ks(power=2)
            lapl_mps.scalar_multiply(-1, inplace=True)
            correction.apply_qft(inplace=True, compress=False)
            lapl_mps.add(correction.scalar_multiply(-const**2, inplace=True), compress=True)

            inv_lapl_mpo = grid_X.get_diag_mpo_inverse(lapl_mps, compress_opts=None)
            inv_lapl_mpo.apply_inverse_qft(inplace=True, compress=True,
                                           compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})


        ### RHS
        version = 2
        if version==1:
            ### method 1:  more expensive but easier to implement?
            if deriv_fe is None:
                store_fe_force = self.sys_fe.force_term
                self.sys_fe.force_term = self.compute_force_term(is_ion=False, only_EL=True)
                deriv_fe = self.sys_fe.calculate_time_derivative(compress_level=compress1, compress_level1=compress2)
                self.sys_fe.force_term = store_fe_force

            if deriv_fi is None:
                store_fi_force = self.sys_fi.force_term
                self.sys_fi.force_term = self.compute_force_term(is_ion=False, only_EL=True)
                deriv_fi = self.sys_fi.calculate_time_derivative(compress_level=compress1, compress_level1=compress2)
                self.sys_fi.force_term = store_fi_force


            dt_flow_L_e = deriv_fe.compute_flow(compress=compress1, compress1=compress2)
            dt_flow_L_i = deriv_fi.compute_flow(compress=compress1, compress1=compress2)

            rhs = dt_flow_L_e.add(dt_flow_L_i, inplace=True, compress_level=compress1)

        elif version == 2:

            ### method 2:  broken down components
            ### lapl(ET) - a^2 w^2 ET = sum_s  grad(rho_s) <v^2>_s + q_s rho_s / m_s ( E_L + <v>_s x B )
            ### a = v0/c (in the paper's normalization)
            ### w^2 = sum_s q_s rho_s / m_s

            def get_rhs(is_ion=False):
                sys_f = self.sys_fi if is_ion else self.sys_fe
                spec_params = self.ion_params if is_ion else self.elc_params
                v_axes_dict = self.vi_axes_dict if is_ion else self.ve_axes_dict

                flows = sys_f.compute_flow(compress=compress1, compress1=0)
                force_term = flows.cross_product(self.B, compress_level=compress1, inner_compress_level=0)

                densities = sys_f.compute_density(compress=compress1)
                E_term = self.EL.elemental_multiply(densities, zipup=True, inplace=False, compress_level=compress1)
                force_term.add(E_term, inplace=True, compress_level=compress1 )
                force_term.scalar_multiply(spec_params.charge**2 / spec_params.mass)

                vv_dict = {}
                for C1 in self.coords_x.coords:
                    v_ax1 = v_axes_dict[C1]

                    vv_sum = None
                    for C2 in self.coords_x.coords:
                        v_ax2 = v_axes_dict[C2]
                        x_ax2 = self.coords_x.get_axis(C2)

                        if C1 == C2:
                            vv = sys_f.compute_moment([v_ax1], powers=[2], compress=compress1)
                        else:
                            vv = sys_f.compute_moment([v_ax1, v_ax2], compress=compress1)

                        vv.gradient(deriv_axes=[x_ax2], compress=0)

                        if vv_sum is not None:
                            vv_sum.add(vv, inplace=True)
                        else:
                            vv_sum = vv

                    if compress1:
                        vv_sum.compress(inplace=True,)

                    vv_dict[C1] = vv_sum.scalar_multiply(-1 * spec_params.charge, inplace=True)

                vv_term = force_term.create_like(new_components=vv_dict)
                rhs = force_term.add(vv_term, inplace=True)
                return rhs

            ion_rhs = get_rhs(is_ion=True)
            elc_rhs = get_rhs(is_ion=False)
            rhs = ion_rhs.add(elc_rhs, inplace=True, compress=compress1)

        else:
            raise NotImplementedError('check desired rhs version')


        ### solve for ET
        for C in rhs.componentIDs:
            rhs[C].apply(inv_lapl_mpo, inplace=True, compress=compress,
                         compress_opts=self.ET.compress_config[compress])

        ### project ET onto div(ET) = 0
        divET_term = rhs.divergence(compress_level=compress1)
        divET_term.component.apply(new_sys_EM.inv_laplacian_mpo, inplace=True)  # no correction to laplacian
        divET_term = divET_term.gradient(compress_level=compress1)
        divET_term.scalar_multiply(-1, inplace=True)

        new_ET = rhs.add(divET_term, inplace=True, compress_level=compress)

        ### update ET
        new_ET.name = self.field_names['ET']
        new_sys_EM.ET = new_ET
        return new_sys_EM