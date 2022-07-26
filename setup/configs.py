from setup.defaults import *
from setup.enums import *
from setup.helper import *

from typing import Optional

# class MaterialConfiguration:
#     """ plasma material parameters
#     """
#     def __init__(self, matl_params: dict = None):
#         if matl_params is None:     matl_params = {}
#
#         self.e = matl_params.get('e', 1.0)  ## elementary charge
#         self.Zi = matl_params.get('Zi', 1.0)  ## ion charge (always positive)
#         self.mass_e = matl_params.get('mass_e', 1.0)  ## electron mass
#         self.mass_i = matl_params.get('mass_i', 1.0)  ## ion mass
#         self.eps0 = matl_params.get('eps0', 1.0)
#         self.mu0 = matl_params.get('mu0', 1.0)
#         self.c = 1. / np.sqrt(self.eps0 * self.mu0)
#         self.Te = matl_params.get('Te', 1.0)  ## electron temperature
#         self.Ti = matl_params.get('Ti', 1.0)  ## ion temperature
#         self.eV = matl_params.get('eV', 1.0)  ## kB

# class MaterialConfiguration:
#     """ plasma material parameters
#     """
#     def __init__(self, e:Numeric=1.0, Zi:Numeric=1.0, n0_e:Numeric=1.0, n0_i:Numeric=1.0,
#                  mass_e:Numeric=1.0, mass_i:Numeric=1.0, Te:Numeric=1.0, Ti:Numeric=1.0,
#                  eps0:Numeric=1.0, mu0:Numeric=1.0, eV:Numeric=1.0):
#         self.e = e  ## elementary charge
#         self.Zi = Zi  ## ion charge (always positive)
#         self.mass_e = mass_e  ## electron mass
#         self.mass_i = mass_i  ## ion mass
#         self.eps0 = eps0
#         self.mu0 = mu0
#         self.c = 1. / np.sqrt(self.eps0 * self.mu0)
#         self.Te = Te  ## electron temperature
#         self.Ti = Ti  ## ion temperature
#         self.eV = eV  ## kB
#         self.n0_e = n0_e
#         self.n0_i = n0_i
#
#         self.wp_e = np.sqrt(e ** 2 * n0_e / eps0 / mass_e)  # plasma e' frequency
#         self.wp_i = np.sqrt(e ** 2 * n0_i / eps0 / mass_i)  # plasma i+ frequency
#         self.vth_e = np.sqrt(eV * Te / mass_e)  # electron thermal speed
#         self.vth_i = np.sqrt(eV * Ti / mass_i)  # ion thermal speed
#         self.lamD = self.vth_e / self.wp_e  # Debye length


class UnitsConfiguration:
    """ units
    """
    def __init__(self, e: Numeric=1.0, eps0: Numeric=1.0, mu0: Numeric=1.0, eV:Numeric=1.0):
        self.e = e    ## elementary charge
        self.eV = eV  ## kB
        self.eps0 = eps0
        self.mu0 = mu0
        self.c = 1. / np.sqrt(self.eps0 * self.mu0)


class SpeciesConfiguration:
    """ species material parameters
    """
    def __init__(self, Z:Numeric=1.0, n0:Numeric=1.0, mass:Numeric=1.0, T:Numeric=1.0,
                 units_config: 'UnitsConfiguration' =None):
        self.Z = Z  ## charge (positive for ions, negative for elec)
        self.mass = mass  ## electron mass
        self.T = T  ## electron temperature
        self.n0 = n0
        if units_config is None:
            units_config = UnitsConfiguration()
        self.units_config = units_config

    @property
    def units_config(self):
        return self._units_config

    @units_config.setter
    def units_config(self, units_config:'UnitsConfiguration'):
        self._units_config = units_config
        if units_config is not None:
            self.e = units_config.e  ## elementary charge
            self.eps0 = units_config.eps0
            self.mu0 = units_config.mu0
            self.eV = units_config.eV
            self.c = units_config.c

            self.wp = np.sqrt(self.e ** 2 * self.n0 / self.eps0 / self.mass)  # plasma e' frequency
            self.vth = np.sqrt(self.eV * self.T / self.mass)  # electron thermal speed
            self.lamD = self.vth / self.wp  # Debye length


class IonConfiguration(SpeciesConfiguration):
    """ configuration for ions
    """
    def __init__(self, Z:Numeric=1.0, n0:Numeric=1.0, mass:Numeric=1.0, T:Numeric=1.0,
                 units_config: 'UnitsConfiguration' =None):
        super().__init__(Z=Z, n0=n0, mass=mass, T=T, units_config=units_config)

class ElcConfiguration(SpeciesConfiguration):
    """ configuration for electrons (change sign of Z)
    """
    def __init__(self, Z:Numeric=1.0, n0:Numeric=1.0, mass:Numeric=1.0, T:Numeric=1.0,
                 units_config: 'UnitsConfiguration' =None):
        super().__init__(Z=abs(Z)*-1, n0=n0, mass=mass, T=T, units_config=units_config)



class SubCompressConfigType(Enum):
    TT = 'tt'

class CompressionConfiguration:
    """ class containing compression parameters for different levels
    """
    def __init__(self):
        self.max_bonds = {}
        self.cutoff_modes = {}
        self.cutoffs = {}
        self.do_midpt = False
        self.midpt = None

        self.sub_compress_configs: dict['SubCompressConfigType', 'CompressionConfiguration'] = {}


    def __getitem__(self, compress_level):
        return self.get_compress_opts(compress_level)

    def __str__(self):
        str_bonds = 'bonds: ' + str(self.max_bonds)
        str_cutoffs = 'cutoffs: ' + str(self.cutoffs)
        str_modes = 'cutoff modes: ' + str(self.cutoff_modes)
        str_midpt = 'midpt: ' + str(self.do_midpt)
        if self.midpt is not None:    str_midpt += f',{self.midpt}'
        return str_bonds + '\n' + str_cutoffs + '\n' + str_modes + '\n' + str_midpt

    def set_compress_opts(self, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                          cutoff=CUTOFF):
        self.max_bonds[compress_level] = max_bond
        self.cutoff_modes[compress_level] = cutoff_mode
        self.cutoffs[compress_level] = cutoff

    def get_compress_opts(self, compress_level: int) -> dict:
        max_bond = self.max_bonds.get(compress_level, MAXBOND)
        cutoff = self.cutoffs.get(compress_level, CUTOFF)
        cutoff_mode = self.cutoff_modes.get(compress_level, CUTOFF_MODE)
        compress_opts = {'max_bond': max_bond, 'cutoff': cutoff, 'cutoff_mode': cutoff_mode}
        if self.do_midpt:
            compress_opts['do_midpt'] = True
            if self.midpt is not None:
                compress_opts['midpt'] = self.midpt
        return compress_opts

    def set_sub_compress_opts(self, sub_key, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                          cutoff=CUTOFF):
        if sub_key not in self.sub_compress_configs:
            return None     # s.t. compress_opts is same as the main
        compress_config = self.sub_compress_configs[sub_key]
        compress_config.set_compress_opts(compress_level, max_bond=max_bond, cutoff_mode=cutoff_mode, cutoff=cutoff)
        self.sub_compress_configs[sub_key] = compress_config

    def get_sub_compress_opts(self, sub_key, compress_level: int) -> dict:
        compress_opts = self.sub_compress_configs[sub_key].get_compress_opts(compress_level)
        return compress_opts

    def get_all_sub_compress_opts(self, compress_level: int) -> Optional[dict['SubCompressConfigType', dict]]:
        sub_keys = self.sub_compress_configs.keys()

        if len(sub_keys) == 0:
            return None

        compress_opts = {}
        for sub_key, sub_config in self.sub_compress_configs.keys():
            compress_opts[sub_key] = sub_key.get_compress_opts(compress_level)
        return compress_opts


class DerivativeConfiguration:
    def __init__(self,
                 left_bc: BCType or str = DEFAULT_BC,
                 right_bc: BCType or str = None,
                 order: int = DEFAULT_ORDER,
                 fd_type: FDType or str = DEFAULT_FDTYPE):

        self._left_bc = parse_BCType(left_bc)
        if right_bc is None:
            right_bc = left_bc
        self._right_bc = parse_BCType(right_bc)
        self._order = order
        self._fdtype = parse_FDType(fd_type)

        # # other params only used in some cases
        # self.v_ax: Optional['Axis'] = None

    def __str__(self):
        return 'DerivConfig' + str(vars(self))

    def __repr__(self):
        return 'DerivConfig' + str(vars(self))

    def copy(self):
        return self.__class__(self.left_bc, self.right_bc, self.order, self.fd_type)

    @property
    def left_bc(self) -> BCType:
        return self._left_bc

    @left_bc.setter
    def left_bc(self, left_bc: BCType or str):
        self._update_left_bc(left_bc)

    @property
    def right_bc(self) -> BCType:
        return self._right_bc

    @right_bc.setter
    def right_bc(self, right_bc: BCType or str):
        self._update_right_bc(right_bc)

    @property
    def bc(self) -> tuple[BCType, BCType]:
        return self._left_bc, self._right_bc

    @bc.setter
    def bc(self, bc: BCType or str):
        self.left_bc = bc
        self.right_bc = bc

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, new_order: int):
        self._update_order(new_order)

    @property
    def fd_type(self) -> FDType:
        return self._fdtype

    @fd_type.setter
    def fd_type(self, new_fdtype: FDType or str):
        self._update_fdtype(new_fdtype)


    @property
    def deriv_params(self):
        return {'left_bc': self._left_bc, 'right_bc': self._right_bc, 'order': self._order, 'fd_type': self._fdtype}


    def update(self, left_bc: Optional[BCType or str] = None, right_bc: Optional[BCType or str] = None,
               order:int = None, fd_type:int = None): # , v_ax:'Axis' = None):
        if left_bc is not None:
            self.left_bc = left_bc
            self.right_bc = self.left_bc
        if right_bc is not None:
            self.right_bc = right_bc
        if order is not None:
            self.order = order
        if fd_type is not None:
            self.fd_type = fd_type
        # if v_ax is not None:
        #     self.v_ax = v_ax

    def _update_left_bc(self, new_left_bc):
        self._left_bc = parse_BCType(new_left_bc)

    def _update_right_bc(self, new_right_bc):
        self._right_bc = parse_BCType(new_right_bc)

    def _update_order(self, new_order: int):
        assert (isinstance(new_order, int)), 'order must be an integer'
        self._order = new_order

    def _update_fdtype(self, new_fdtype: FDType or str):
        self._fdtype = parse_FDType(new_fdtype)


    def get_deriv_key(self):
        # if self._fdtype == FDType.UPWIND:
        #     assert(self.v_ax is not None), 'need to set v_axis for upwind FD method'
        #     return (self._left_bc, self._right_bc), self._order, self._fdtype, self.v_ax.axID
        # else:
        return (self._left_bc, self._right_bc), self._order, self._fdtype


class CollisionConfiguration:
    def __init__(self, coll_type, coeff_e=1.0, coeff_i=1.0, rate_e=1.0, rate_i=1.0, v0_e=0.0, v0_i=0.0):
        # {'rate_e':coll_rate_e, 'rate_i':coll_rate_i, 'v0_e':0.0, 'v0_i':0.0}

        self.coll_type = coll_type

        if coll_type is not None:
            self.coeff_e: Union[np.ndarray,Numeric] = coeff_e
            self.coeff_i: Union[np.ndarray,Numeric] = coeff_i
            self.base_rate_e: Union[np.ndarray,Numeric] = rate_e
            self.base_rate_i: Union[np.ndarray,Numeric] = rate_i
            self.v0_e: Union[np.ndarray,Numeric] = v0_e
            self.v0_i: Union[np.ndarray,Numeric] = v0_i
        else:
            self.coeff_e = 0.
            self.coeff_i = 0.
            self.base_rate_e = 0.
            self.base_rate_i = 0.
            self.v0_e = 0.
            self.v0_i = 0.

        if coll_type in ['LB','lb','Lenard-Bernstein','LBO']:
            self.coll_type = CollisionType.LB

    def __str__(self):
        if self.coll_type is None:
            return 'colltype None'
        return self.coll_type.value + ': ' + str(vars(self))

    @property
    def coll_rate_e(self):
        return self.get_coeff_e() * self.get_base_rate_e()

    @property
    def coll_rate_i(self):
        return self.get_coeff_i() * self.get_base_rate_i()

    @property
    def coll_params(self):
        return vars(self)

    def get_base_rate_e(self, dim=0):
        try:
            return self.base_rate_e[dim]
        except (IndexError,TypeError):
            return self.base_rate_e

    def get_base_rate_i(self, dim=0):
        try:
            return self.base_rate_i[dim]
        except (IndexError,TypeError):
            return self.base_rate_i

    def get_coeff_e(self, dim=0):
        try:
            return self.coeff_e[dim]
        except (IndexError,TypeError):
            return self.coeff_e

    def get_coeff_i(self, dim=0):
        try:
            return self.coeff_i[dim]
        except (IndexError,TypeError):
            return self.coeff_i

    def get_coll_rate_e(self, dim=0):
        return self.get_base_rate_e(dim) * self.get_coeff_e(dim)

    def get_coll_rate_i(self, dim=0):
        return self.get_base_rate_e(dim) * self.get_coeff_i(dim)

    def get_v0_e(self, dim=0):
        try:
            return self.v0_e[dim]
        except (IndexError,TypeError):
            return self.v0_e

    def get_v0_i(self, dim=0):
        try:
            return self.v0_i[dim]
        except (IndexError,TypeError):
            return self.v0_i

