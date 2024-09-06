from setup_.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from helper_tdvp import TDVPSolver2
from field import Field, ScalarField
from pde_system import PDE_system

if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from gridTN import GridTN


class VectorPotential(PDE_system):
    """ Maxwell's equations solver
    """
    def __init__(self,
                 A: Optional[Field], phi: Optional[ScalarField],
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 background_A0: Optional['Field'] = None,
                 background_phi0: Optional['ScalarField'] = None,
                 is_electrostatic=False, # is_yee = False, clean=True,
                 normalize=False, upwind=False, zipup=False,
                 te_order=60, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov.
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        self.names = {}
        self.names['A'] = A.name if A is not None else 'A'
        self.names['phi'] = phi.name if phi is not None else 'phi'

        field_names = [self.names[x] for x in ['A', 'phi']]

        super().__init__(A, phi, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         )

        self.coords_x = coords_x

        self.init_norm_A = None
        self.init_norm_phi = None

        if A is not None:
            self.grid_X = A.grid
        elif phi is not None:
            self.grid_X = phi.grid
        else:
            self.grid_X = grid_X

        ## background fields
        self.background_A0 = background_A0
        self.divA0 = None
        self.curlA0 = None

        self.background_phi0 = background_phi0
        self.gradphi0 = None

        ## species parameters
        self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
        self.is_electrostatic = is_electrostatic
        self.upwind = upwind
        self.zipup = zipup

        ## charge and current densities
        self.charge_density = None
        self.charge_density_deriv = None
        self.current_density = None
        self.current_density_deriv = None

        ## history of vector potential (needed to compute E, Lorentz force)
        self.A_history: Sequence[tuple[Numeric, Field]] = []       # list of (dt, Field)
        self.history_length = 1

        self.E: Optional['Field'] = None
        self.B: Optional['Field'] = None
        self.dA_dt: Optional['Field'] = None


    @property
    def A(self) -> Optional['Field']:
        try:
            return self._fields[self.names['A']]
        except KeyError:
            return None

    @A.setter
    def A(self, new_field: 'Field'):
        self._fields[self.names['A']] = new_field


    @property
    def phi(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['phi']]
        except KeyError:
            return None

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        self._fields[self.names['phi']] = new_field

    @property
    def partial_dA_dt(self, compress_level=1):
        if self.dA_dt is None:
            if self.history_length == 1:
                dt, A1 = self.A_history[-1]
                dA_dt = self.A.add(A1.scalar_multiply(-1, inplace=False), inplace=False,
                                   compress_level=compress_level)
            elif self.history_length == 1:
                A0 = self.A.scalar_multiply(1.5, inplace=False)
                dt1, A1 = self.A_history[-1]
                A1 = A1.scalar_multiply(-2, inplace=False)
                dt2, A2 = self.A_history[-2]
                A2 = A2.scalar_multiply(0.5, inplace=False)
                dt = (dt1 + dt2) / 2        ## ideally roughly the same

                dA_dt = A0.add(A1, inplace=True, compress_level=0)
                dA_dt = dA_dt.add(A2, inplace=True, compress_level=compress_level)
            else:
                raise NotImplementedError

            dA_dt.scalar_multiply(dt, inplace=True)
            self.dA_dt = dA_dt

        return self.dA_dt


    def E(self, compress_level=1, inner_compress_level=4) -> Optional['Field']:
        if self.E is None:
            ## dA/dt
            dA_dt = self.partial_dA_dt
            dA_dt = dA_dt.scalar_multiply(-1, inplace=False)

            # ## (v \dot grad) A
            # vgA = self.A.advective_derivative(velocities, self.coords_x, compress_level=inner_compress_level,
            #                                   inner_compress_level=inner_compress_level + 1)
            ## grad phi
            grad_phi = self.phi.gradient(compress_level=inner_compress_level)
            grad_phi.scalar_multiply(-1, inplace=True)

            out = grad_phi.add(dA_dt, compress_level=compress_level, inplace=True)
            # out.add(vgA, compress_level=compress_level, inplace=True)
            self.E = out

        return self.E


    def B(self, compress_level=1, inner_compress_level=4) -> Optional['Field']:
        if self.B is None:
            out = self.A.curl(self.coords_x, compress_level=compress_level, inner_compress_level=inner_compress_level)
            self.B = out

        return self.B


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 2:
            new_fields = new_fields + (None,) * (2 - len(new_fields))

        A, phi, = new_fields[:2]
        if recalc:
            new_system = self.__class__(A, phi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_A0=self.background_A0, background_phi0=self.background_phi0,
                                        normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
        else:
            new_system = self.__class__(A, phi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_A0=self.background_A0, background_phi0=self.background_phi0,
                                        normalize=False, upwind=self.upwind, zipup=self.zipup,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_E = self.init_norm_A
            new_system.init_norm_B = self.init_norm_phi

            new_system.E = self.E
            new_system.B = self.B
            new_system.dA_dt = self.dA_dt

        if A is None:           new_system.names['A'] = self.names['A']
        if phi is None:         new_system.names['phi'] = self.names['phi']

        new_system.divA0 = self.divA0
        new_system.curlA0 = self.curlA0
        new_system.gradphi0 = self.gradphi0

        new_system.charge_density = self.charge_density
        new_system.charge_density_deriv = self.charge_density_deriv
        new_system.current_density = self.current_density
        new_system.current_density_deriv = self.current_density_deriv
        new_system.A_history = self.A_history

        return new_system


    def _tdvp_component(self, comp_gtn: 'GridTN', dt: Numeric, mpo_list: Sequence['GridTN'],
                        source_list: Optional[Sequence['GridTN']] = None, inplace = False, do_adapt=True, te_order=0,
                        compress_config: 'CompressionConfiguration'=None):

        comp_gtn = comp_gtn if inplace else comp_gtn.copy()

        comp_mpx = comp_gtn.data
        tdvp_solver = TDVPSolver2(comp_mpx, [mpo.data for mpo in mpo_list], [mps.data for mps in source_list],
                                  te_order=te_order, compress_config = compress_config)
        tdvp_solver.take_time_step(dt, do_adapt=do_adapt)
        comp_gtn.data = tdvp_solver.ket
        return comp_gtn


    def _tdvp_A(self, dt: Numeric, inplace = False, do_adapt=True, te_order=0):
        """ lapl(A) - 1/c d^2/dt^2 A = -mu_0 J  [SI]
            lapl(A)
        """
        new_A = self.A if inplace else self.A.copy()
        compress_config = new_A.compress_config

        for compID in new_A.componentIDs:
            A_comp = new_A[compID]

            lapl_mpo = new_A.grid.laplacian_mpo(ax_deriv_configs=A_comp.ax_deriv_configs, compID=compID)
            lapl_mpo = lapl_mpo.scalar_multiply(self.matl_params.c * -1)
            mpo_list = [lapl_mpo]

            j_comp = self.current_density[compID] if self.current_density is not None else None
            djc_dt = self.current_density_deriv[compID] if self.charge_density_deriv is not None else None
            j_coeff = 1 if self.matl_params.is_cgs else 2
            if j_comp is not None:
                j_comp = j_comp.scalar_multiply(j_coeff, inplace=False)
            if djc_dt is not None:
                djc_dt = djc_dt.scalar_multiply(j_coeff, inplace=False)
            source_list = [j_comp, djc_dt]      ## [constant in time, linear in time]

            self._tdvp_component(A_comp, dt, mpo_list, source_list, inplace=True, do_adapt=do_adapt, te_order=te_order,
                                 compress_config=compress_config)

        return new_A


    def _tdvp_phi(self, dt: Numeric, inplace=False, do_adapt=True, te_order=0):
        """ lapl(A) - 1/c d^2/dt^2 A = -mu_0 J  [SI]
            lapl(A)
        """
        new_phi = self.phi if inplace else self.phi.copy()
        compress_config = new_phi.compress_config

        phi_comp = new_phi.component
        lapl_mpo = new_phi.grid.laplacian_mpo(ax_deriv_configs=phi_comp.ax_deriv_configs)
        lapl_mpo = lapl_mpo.scalar_multiply(self.matl_params.c * -1)
        mpo_list = [lapl_mpo]

        rho = self.charge_density.component if self.charge_density is not None else None
        drho_dt = self.current_density_deriv.component if self.charge_density_deriv is not None else None
        rho_coeff = 1 if self.matl_params.is_cgs else 2
        if rho is not None:
            rho = rho.scalar_multiply(rho_coeff, inplace=False)
        if drho_dt is not None:
            drho_dt = drho_dt.scalar_multiply(rho_coeff, inplace=False)
        source_list = [rho, drho_dt]        ## [constant in time, linear in time]

        self._tdvp_component(phi_comp, dt, mpo_list, source_list, inplace=True, do_adapt=do_adapt, te_order=te_order,
                             compress_config=compress_config)

        return new_phi


    def time_dependent_variational_principle(self, dt: Numeric, te_order=0, do_adapt: bool = True, inplace=False,
                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                             advec_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True, verbose_plot: bool = False, **kwargs):
        """
        """
        old_A = self.A.copy()
        new_state = self if inplace else self.copy()
        new_state._tdvp_A(dt, inplace=True, do_adapt=do_adapt, te_order=te_order)
        new_state._tdvp_phi(dt, inplace=True, do_adapt=do_adapt, te_order=te_order)
        new_state.E = None
        new_state.B = None
        new_state.A_history = [(dt, old_A)] + self.A_history[:self.history_length]
        return new_state


