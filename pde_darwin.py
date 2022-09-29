from setup.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from coord.cartesian import CartesianCoordinateSpace
from field import Field, ScalarField
from pde_system import PDE_system

if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from gridTN import GridTN


class Darwin(PDE_system):
    """ Maxwell's equations solver
    """
    def __init__(self,
                 ET: Optional[Field], EL: Optional[Field], B: Optional[Field],
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 pot_ax_deriv_configs: Optional[dict['Axis','DerivativeConfiguration']] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 background_E0: Optional['Field'] = None,
                 background_B0: Optional['Field'] = None,
                 normalize=True, upwind=False, zipup=False,
                 te_order=2, compress_levels=None,
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
        self.names['ET'] = ET.name if ET is not None else 'ET'
        self.names['EL'] = EL.name if EL is not None else 'EL'
        self.names['B'] = B.name if B is not None else 'B'

        field_names = [self.names[x] for x in ['ET', 'EL', 'B']]

        super().__init__(ET, EL, B, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         )

        self.coords_x = coords_x

        self.init_norm_E = None
        self.init_norm_B = None

        if ET is not None:
            self.grid_X = ET.grid
        elif EL is not None:
            self.grid_X = EL.grid
        elif B is not None:
            self.grid_X = B.grid
        else:
            self.grid_X = grid_X

        ## background fields
        self.background_E0 = background_E0
        self.divE0 = None
        self.curlE0 = None

        self.background_B0 = background_B0
        self.divB0 = None
        self.curlB0 = None

        ## species parameters
        self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
        self.upwind = upwind
        self.zipup = zipup

        ## charge and current densities
        self.charge_density = None
        self.current_density = None
        self.current_density_time_deriv = None

        ## inverse operators
        self.pot_ax_deriv_configs = {ax: DerivativeConfiguration() for ax in self.coords_x.axes} \
                                        if pot_ax_deriv_configs is None else pot_ax_deriv_configs
        self._inv_laplacian_mpo = None


    @property
    def ET(self) -> Optional['Field']:
        try:
            return self._fields[self.names['ET']]
        except KeyError:
            return None

    @ET.setter
    def ET(self, new_field: 'Field'):
        self._fields[self.names['ET']] = new_field

    @property
    def EL(self) -> Optional['Field']:
        try:
            return self._fields[self.names['EL']]
        except KeyError:
            return None

    @EL.setter
    def EL(self, new_field: 'Field'):
        self._fields[self.names['EL']] = new_field

    @property
    def E(self) -> Optional['Field']:
        if self.ET is None:
            return self.EL
        else:
            return self.ET.add(self.EL)

    @property
    def total_E(self) -> Optional['Field']:
        if self.background_E0 is not None:
            return self.background_E0.add(self.E)
        else:
            return self.E

    @property
    def B(self) -> Optional['Field']:
        try:
            return self._fields[self.names['B']]
        except KeyError:
            return None

    @B.setter
    def B(self, new_field: 'Field'):
        self._fields[self.names['B']] = new_field

    @property
    def total_B(self) -> Optional['Field']:
        if self.background_B0 is not None:
            return self.background_B0.add(self.B)
        else:
            return self.B

    @property
    def inv_laplacian_mpo(self) -> 'GridTN':

        if self.inv_laplacian_mpo is None:
            ax_deriv_configs = self.pot_ax_deriv_configs
            inv_laplacian_mpo = self.grid_X.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs,
                                                                  compress_opts={'cutoff':CUTOFF,
                                                                                 'cutoff_mode':CUTOFF_MODE})
            self._inv_laplacian_mpo = inv_laplacian_mpo
            print('inv laplacian', inv_laplacian_mpo.max_bond())
        else:
            inv_laplacian_mpo = self._inv_laplacian_mpo
        return inv_laplacian_mpo


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 3:
            new_fields = new_fields + (None,) * (3 - len(new_fields))

        ET, EL, B  = new_fields[:3]

        if recalc:
            new_system = Darwin(ET, EL, B, grid_X=self.grid_X,
                                 coords_x=self.coords_x, matl_params=self.matl_params,
                                 background_E0=self.background_E0, background_B0=self.background_B0,
                                 normalize=self.do_normalization,upwind=self.upwind, zipup=self.zipup,
                                 te_order=self.te_order,
                                 compress_levels=self._comp_levels,
                                 )
        else:
            new_system = Darwin(ET, EL, B, grid_X=self.grid_X,
                                 coords_x=self.coords_x, matl_params=self.matl_params,
                                 background_E0=self.background_E0, background_B0=self.background_B0,
                                 normalize=False, upwind=self.upwind, zipup=self.zipup,
                                 te_order=self.te_order,
                                 compress_levels=self._comp_levels,
                                 )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_E = self.init_norm_E
            new_system.init_norm_B = self.init_norm_B

        if ET is None:           new_system.names['ET'] = self.names['ET']
        if EL is None:           new_system.names['EL'] = self.names['EL']
        if B  is None:           new_system.names['B']  = self.names['B']

        new_system.divE0 = self.divE0
        new_system.divB0 = self.divB0
        new_system.curlE0 = self.curlE0
        new_system.curlB0 = self.curlB0

        new_system.charge_density = self.charge_density
        new_system.current_density = self.current_density
        new_system.current_density_time_deriv = self.current_density_time_deriv

        new_system.pot_ax_deriv_configs = self.pot_ax_deriv_configs
        new_system._inv_laplacian_mpo = self._inv_laplacian_mpo
        return new_system


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError


    def update_EM(self, inplace=False, compress=1, compress1=4):
        new_system = self if inplace else self.copy()
        new_system.update_EL(inplace=True, compress=compress, compress1=compress1)
        new_system.update_B(inplace=True, compress=compress, compress1=compress1)
        new_system.update_ET(inplace=True, compress=compress, compress1=compress1)
        return new_system


    def update_ET(self, inplace=False, current_density_time_deriv=None, compress=1, compress1=4) -> 'Darwin':
        """ veclapl(ET) = d/dt curl(B) = mu0 d/dt j - mu0*eps0 d^2/dt^2 EL
            time derivative of current density obtained explicitly
            last term determined by forcing div(ET) = 0
                define TH = lapl^-1 ( div(ET) )   [ looks like a potential ]
            ET <- ET - grad(TH)
        """
        new_system = self if inplace else self.copy()
        current_density_time_deriv: 'Field' = self.current_density_time_deriv if current_density_time_deriv is None \
                                                    else current_density_time_deriv

        new_ET = current_density_time_deriv.scalar_multiply(self.matl_params.mu0, inplace=False)
        divET_term = new_ET.divergence(compress_level=compress1)
        ## missing:  apply bc as necessary
        divET_term.component.apply(new_system.inv_laplacian_mpo, inplace=True)
        divET_term = divET_term.gradient(compress_level=compress1)
        divET_term.scalar_multiply(-1, inplace=True)

        new_ET.add(divET_term, inplace=True, compress_level=compress)
        new_ET.name = self.field_names['ET']
        new_system.ET = new_ET
        return new_system


    def update_EL(self, inplace=False, charge_density: Optional['ScalarField']=None, compress=1, compress1=4):
        """ EL = -grad( V ) where V = lapl(-charge density)
        """
        new_sys = self if inplace else self.copy()
        new_EL = self.charge_density.copy() if charge_density is None else charge_density.copy()

        inv_lapl_mpo = new_sys.inv_laplacian_mpo

        if new_sys.background_E0 is not None:
            if new_sys.divE0 is None:
                new_sys.divE0 = new_sys.background_E0.divergence(new_sys.coords_x, compress_level=compress1)
            new_EL.add(new_sys.divE0.scalar_multiply(-1/new_sys.elc_params.eps0, inplace=False), inplace=True)
        new_EL.scalar_multiply(-1, inplace=True)

        ## apply bc if necessary
        bc_values = new_sys.potential_boundary_conditions[1]
        if bc_values is not None:  # ie. 0
            if bc_values.grid.ndim < charge_density.grid.ndim:
                bc_values = bc_values.pad_to_new_grid(new_grid=charge_density.grid)
            new_EL.component.add(bc_values[0], inplace=True, compress=False)

        new_EL.component.apply(inv_lapl_mpo, inplace=True, compress=compress,
                               compress_opts=new_sys.V.compress_config[compress])
        new_sys.EL = new_EL.gradient(compress_level=compress)
        new_sys.EL.name = self.field_names['EL']
        return new_sys


    def update_B(self, inplace=False, current_density: Optional['Field']=None, compress=1, compress1=4):
        """ veclapl(B) = - mu0 curl(J)
        """
        new_sys = self if inplace else self.copy()
        current_density: 'Field' = self.current_density if current_density is None else current_density

        if isinstance(self.coords_x, CartesianCoordinateSpace):
            raise NotImplementedError('these inverse laplacians only valid for Cartesian Coordinates')

        curl_j = current_density.curl(coord_sys=new_sys.coords_x, compress_level=compress1)
        curl_j.scalar_multiply(-1*new_sys.matl_params.mu0)

        ## take inverse Laplacian to obtain B
        for C in curl_j.componentIDs:
            ## missing:  apply boundary conditions if necessary
            curl_j[C].apply(new_sys.inv_laplacian_mpo, inplace=True, compress=compress,
                            compress_opts=new_sys.V.compress_config[compress])

        curl_j.name = self.field_names['B']
        new_sys.B = curl_j
        return new_sys