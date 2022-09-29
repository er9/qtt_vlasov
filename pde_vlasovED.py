from setup.configs import *
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
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


    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_v: Optional['Field'] = None,
                   method_f=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1 -> 'VlasovDarwin':
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

        if is_first_time_step:
            ## evolve force advection dt/2
            state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
            state0.sys_fe.get_force_advection(dt / 2, inplace=True, method=method_f,
                                              compress=comp2, compress1=comp4, compress2=comp5,
                                              verbose_plot=verbose_plot)
            state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
            state0.sys_fi.get_force_advection(dt / 2, inplace=True, method=method_f,
                                              compress=comp2, compress1=comp4, compress2=comp5,
                                              verbose_plot=verbose_plot)

        ## evolve velocity advection dt
        state0.sys_fe.get_vel_advection(dt, inplace=True, method=method_v, compress=comp2, compress1=comp4,
                                        compress2=comp5, verbose_plot=verbose_plot)
        state0.sys_fi.get_vel_advection(dt, inplace=True, method=method_v, compress=comp2, compress1=comp4,
                                        compress2=comp5, verbose_plot=verbose_plot)

        ## update EM components
        if np.abs(state0.matl_params.e) > 0:
            state0.sys_EM.charge_density = self.compute_charge_density()
            state0.sys_EM.current_density = self.compute_current()
            state0.sys_EM.current_density_time_deriv = self.compute_current_time_derivative()
            state0.sys_EM.update_EM(inplace=True, compress=compress_level, compress1=comp2)


        ## evolve force advection dt/2
        dt_ = dt / 2 if is_last_time_step else dt
        state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
        state0.sys_fe.get_force_advection(dt_, inplace=True, method=method_f,
                                          compress=comp2, compress1=comp4,
                                          compress2=comp5, verbose_plot=verbose_plot)
        state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
        state0.sys_fi.get_force_advection(dt_, inplace=True, method=method_f,
                                          compress=comp2, compress1=comp4,
                                          compress2=comp5, verbose_plot=verbose_plot)

        ## include collisions
        if self.collision.coll_type is not None:
            raise NotImplementedError

        state0.time = self.time + dt if self.time is not None else None

        if compress_level:
            state0.compress(compress_level=compress_level, verbose=False)

        if state0.do_normalization:
            state0.normalize()

        return state0


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


    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
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

        if self.E is not None and not background_only:
            E = self.E.add(self.sys_EM.background_E0, compress_level=compress_level2)
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
    def update_EM(self, inplace=False,
                  deriv_fe: Optional['ScalarField']=None, deriv_fi: Optional['ScalarField']=None,
                  compress=1, compress1=4, compress2=5) -> 'Darwin':
        """ div E = div^2 V = 4*pi*rho
            compress:  compression of V
            compress1: compression of terms to create V
        """
        new_sys = self if inplace else self.copy()
        new_sys.sys_EM.charge_density = self.compute_charge_density(compress=compress)
        new_sys.sys_EM.current_density = self.compute_current(compress=compress)
        new_sys.sys_EM.current_density_time_deriv = self.compute_current_time_derivative(deriv_fe=deriv_fe,
                                                                                         deriv_fi=deriv_fi,
                                                                                         compress=compress1,
                                                                                         compress1=compress2)

        new_sys.sys_EM.update_EM(inplace=True, compress=compress, compress1=compress1)
        return new_sys.sys_EM


    def compute_current_time_derivative(self,
                                        deriv_fe: Optional['ScalarField']=None, deriv_fi: Optional['ScalarField']=None,
                                        compress=1, compress1=4, compress2=5) -> 'Field':
        """ different methods:  compute time derivative of f and then compute its current
        """
        deriv_fe = self.sys_fe.calculate_time_derivative(compress_level=compress1, compress_level1=compress2)
        deriv_fi = self.sys_fi.calculate_time_derivative(compress)

        dt_current_e = deriv_fe.compute_current(compress=compress1, compress1=compress2)
        dt_current_i = deriv_fi.compute_current(compress=compress1, compress1=compress2)

        dt_current = dt_current_e.add(dt_current_i, inplace=True, compress_level=compress)

        ### alternative method:
        ### lapl(ET) - a^2 w^2 ET = sum_s  grad(rho_s) <v^2>_s + q_s rho_s / m_s ( E_L + <v>_s x B )
        ### a = v0/c (in the paper's normalization)
        ### w^2 = sum_s q_s rho_s / m_s

        return dt_current