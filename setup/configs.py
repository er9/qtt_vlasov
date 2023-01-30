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
    def __init__(self, e: Numeric=1.0, eps0: Numeric=1.0, mu0: Numeric=1.0, eV:Numeric=1.0, is_cgs=False):
        self.e = e    ## elementary charge
        self.eV = eV  ## kB
        self.eps0 = eps0
        self.mu0 = mu0
        self.c = 1. / np.sqrt(self.eps0 * self.mu0)
        self.is_cgs = is_cgs


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
            self.is_cgs = units_config.is_cgs

            # self.wp = np.sqrt(self.charge ** 2 * self.n0 / self.eps0 / self.mass)  # plasma e' frequency
            # self.vth = np.sqrt(self.eV * self.T / self.mass)  # electron thermal speed
            # self.lamD = self.vth / self.wp  # Debye length

    @property
    def charge(self):
        return self.e * self.Z

    @property
    def plasma_frequency(self):
        if self.is_cgs:
            return np.sqrt(4 * np.pi * self.n0 * self.charge ** 2 / self.mass)
        else:
            return np.sqrt(self.n0 * self.charge ** 2 / self.eps0 / self.mass)

    @property
    def wp(self):
        return self.plasma_frequency

    @property
    def vth(self):
        return np.sqrt(np.abs(self.Z) * self.eV * self.T / self.mass)  # thermal speed

    @property
    def debye_length(self):
        return self.vth / self.plasma_frequency

    @property
    def lamD(self):
        return self.debye_length

    def cyclotron_frequency(self, B0):
        if self.is_cgs:
            return self.charge * B0 / self.mass / self.c
        else:
            return self.charge * B0 / self.mass

    def alfven_speed(self, B0):
        if self.is_cgs:
            return B0 / np.sqrt( 4 * np.pi * self.n0 * self.mass )
        else:
            return B0 / np.sqrt(self.mu0 * self.n0 * self.mass)

    @property
    def skin_depth(self):   # c / plasma frequency
        return self.c / self.plasma_frequency

    def beta(self, B0):     #2 vth^2 / vA^2
        return 2 * self.vth ** 2 / self.alfven_speed(B0) ** 2


class IonConfiguration(SpeciesConfiguration):
    """ configuration for ions
    """
    def __init__(self, Z:Numeric=1.0, n0:Numeric=1.0, mass:Numeric=1.0, T:Numeric=1.0,
                 units_config: 'UnitsConfiguration' =None):
        super().__init__(Z=Z, n0=n0, mass=mass, T=T, units_config=units_config)

    # def alfven_speed(self, B0):
    #     return B0 / np.sqrt( self.mu0 * self.n0 * self.mass )
    #
    # @property
    # def skin_depth(self):   # c / ion plasma frequency
    #     return self.c / self.plasma_frequency

    # @property
    # def skin_depth(self):
    #     return np.sqrt( self.mass / self.e ** 2 / self.eps0  / self.n0)


class ElcConfiguration(SpeciesConfiguration):
    """ configuration for electrons (change sign of Z)
    """
    def __init__(self, Z:Numeric=1.0, n0:Numeric=1.0, mass:Numeric=1.0, T:Numeric=1.0,
                 units_config: 'UnitsConfiguration' =None):
        super().__init__(Z=abs(Z)*-1, n0=n0, mass=mass, T=T, units_config=units_config)

    # @property
    # def skin_depth(self):
    #     return self.c / self.plasma_frequency      # ion skin depth



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
        if self.do_midpt and compress_level == 1:
            compress_opts['do_midpt'] = True
            if self.midpt is not None:
                compress_opts['midpt'] = self.midpt
        return compress_opts

    def set_sub_compress_opts(self, sub_key, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                          cutoff=CUTOFF):
        if sub_key not in self.sub_compress_configs:
            compress_config = CompressionConfiguration()
        else:
            # return None     # s.t. compress_opts is same as the main
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
        for sub_key, sub_config in self.sub_compress_configs.items():
            compress_opts[sub_key] = sub_config.get_compress_opts(compress_level)
        return compress_opts


class DerivativeConfiguration:
    def __init__(self,
                 left_bc: BCType or str = DEFAULT_BC,
                 right_bc: BCType or str = None,
                 order: int = DEFAULT_ORDER,
                 fd_type: FDType or str = DEFAULT_FDTYPE,
                 offset: int = 0,):
        """
            offset: offset of boundary condition from left-hand side -- flipped for right_bc
                    (eg. in SYMMETRIC and ASYMMETRIC case)
                    offset = 0: boundary is the point of symmetry
                    offset < 0: point of symmetry is within data (offset specifies distance from boundary)
                    offset > 0 (maximum = 1): point of symmetry is not within data (offset specifies distance)
                    offset value indicates half steps (eg. x_[-1], x_0, x_1 vs x_[-1/2], x_[1/2]_
        """

        self._left_bc = parse_BCType(left_bc)
        if right_bc is None:
            right_bc = left_bc
        self._right_bc = parse_BCType(right_bc)
        self._order = order
        self._offset = offset
        self._fdtype = parse_FDType(fd_type)

        # # other params only used in some cases
        # self.v_ax: Optional['Axis'] = None

    def __str__(self):
        return 'DerivConfig' + str(vars(self))

    def __repr__(self):
        return 'DerivConfig' + str(vars(self))

    def copy(self):
        return self.__class__(left_bc=self.left_bc, right_bc=self.right_bc, order=self.order,
                              fd_type=self.fd_type, offset= self.offset)

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
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, new_offset: int):
        self._update_offset(new_offset)

    @property
    def deriv_params(self):
        return {'left_bc': self._left_bc, 'right_bc': self._right_bc, 'order': self._order, 'fd_type': self._fdtype,
                'offset': self._offset}


    def update(self, left_bc: Optional[BCType or str] = None, right_bc: Optional[BCType or str] = None,
               order:int = None, fd_type:'FDType' = None, offset:int = None): # , v_ax:'Axis' = None):
        if left_bc is not None:
            self.left_bc = left_bc
            self.right_bc = self.left_bc
        if right_bc is not None:
            self.right_bc = right_bc
        if order is not None:
            self.order = order
        if fd_type is not None:
            self.fd_type = fd_type
        if offset is not None:
            self.offset = offset
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

    def _update_offset(self, new_offset: int):
        self._offset = new_offset


    def get_deriv_key(self):
        # if self._fdtype == FDType.UPWIND:
        #     assert(self.v_ax is not None), 'need to set v_axis for upwind FD method'
        #     return (self._left_bc, self._right_bc), self._order, self._fdtype, self.v_ax.axID
        # else:
        return (self._left_bc, self._right_bc), self._order, self._fdtype, self._offset


class CollisionConfiguration:
    ### TODO: adapt to include interspecies collisions
    def __init__(self, coll_type, coeff_e=1.0, coeff_i=1.0, rate_e=1.0, rate_i=1.0, v0_e=0.0, v0_i=0.0):
        # {'rate_e':coll_rate_e, 'rate_i':coll_rate_i, 'v0_e':0.0, 'v0_i':0.0}

        self.collisions_e = SpeciesCollisionConfiguration(coll_type, coeff=coeff_e, rate=rate_e, v0=v0_e)
        self.collisions_i = SpeciesCollisionConfiguration(coll_type, coeff=coeff_i, rate=rate_i, v0=v0_i)

        self.coll_type = coll_type
        if coll_type in ['LB','lb','Lenard-Bernstein','LBO']:
            self.coll_type = CollisionType.LB

    def __str__(self):
        if self.coll_type is None:
            return 'colltype None'
        return self.coll_type.value + ': ' + str(vars(self))

    @property
    def coll_rate_e(self):
        return self.collisions_e.coll_rate

    @property
    def coll_rate_i(self):
        return self.collisions_i.coll_rate

    @property
    def base_rate_e(self):
        return self.collisions_e.base_rate

    @base_rate_e.setter
    def base_rate_e(self, new_base_rate):
        self.collisions_e.base_rate = new_base_rate

    @property
    def base_rate_i(self):
        return self.collisions_i.base_rate

    @base_rate_i.setter
    def base_rate_i(self, new_base_rate):
        self.collisions_i.base_rate = new_base_rate

    @property
    def coll_params(self):
        return vars(self)

    def get_base_rate_e(self, dim=0):
        return self.collisions_e.get_base_rate(dim=dim)

    def get_base_rate_i(self, dim=0):
        return self.collisions_i.get_base_rate(dim=dim)

    def get_coeff_e(self, dim=0):
        return self.collisions_e.get_coeff(dim=dim)

    def get_coeff_i(self, dim=0):
        return self.collisions_i.get_coeff(dim=dim)

    def get_coll_rate_e(self, dim=0):
        return self.collisions_e.get_coll_rate(dim=dim)

    def get_coll_rate_i(self, dim=0):
        return self.collisions_i.get_coll_rate(dim=dim)

    def get_v0_e(self, dim=0):
        return self.collisions_e.get_v0(dim=dim)

    def get_v0_i(self, dim=0):
        return self.collisions_i.get_v0(dim=dim)


class SpeciesCollisionConfiguration:
    def __init__(self, coll_type, coeff=1.0, rate=1.0, v0=0.0):
        # {'rate_e':coll_rate_e, 'rate_i':coll_rate_i, 'v0_e':0.0, 'v0_i':0.0}

        self.coll_type = coll_type

        if coll_type is not None:
            self.coeff: Union[np.ndarray,Numeric] = coeff
            self.base_rate: Union[np.ndarray,Numeric] = rate
            self.v0: Union[np.ndarray,Numeric] = v0
        else:
            self.coeff = 0.
            self.base_rate = 0.
            self.v0 = 0.

        if coll_type in ['LB','lb','Lenard-Bernstein','LBO']:
            self.coll_type = CollisionType.LB

    def __str__(self):
        if self.coll_type is None:
            return 'colltype None'
        return self.coll_type.value + ': ' + str(vars(self))

    @property
    def coll_rate(self):
        return self.get_coeff() * self.get_base_rate()

    @property
    def coll_params(self):
        return vars(self)

    def get_base_rate(self, dim=0):
        try:
            return self.base_rate[dim]
        except (IndexError,TypeError):
            return self.base_rate

    def get_coeff(self, dim=0):
        try:
            return self.coeff[dim]
        except (IndexError,TypeError):
            return self.coeff

    def get_coll_rate(self, dim=0):
        return self.get_base_rate(dim) * self.get_coeff(dim)

    def get_v0(self, dim=0):
        try:
            return self.v0[dim]
        except (IndexError,TypeError):
            return self.v0

