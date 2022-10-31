from setup.configs import *
from dataclasses import dataclass
import helper_quimb as helper
from coord.coord_sys import Coordinate, CoordinateSystem
from coord.cartesian import CartesianCoordinateSpace
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis, RealFourierBasis
from basis.basis_hermite import HermiteBasis
from axis_map.map_binary import BinaryMap
# from axis_map.map_flipbinary import FlipBinaryMap
# from axis_map.map_mirror import MirrorMap

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basis.basis import Basis
    from axis_map.map import AxisMap


""" Class defining a dimension
    Specifies dimension ID, basis representation, discretization/grid points along axis
    Build operators to take differentials, multiply by x (defined by Basis)
"""

def parse_bc_type(bc_id) -> 'BCType':
    """ return BasisType(Enum) given accepted basis_id (strs)
    """
    if isinstance(bc_id, BCType):
        return bc_id

    if bc_id in ['periodic','p','P']:
        return BCType.PERIODIC
    elif bc_id in ['antiperiodic','ap','AP']:
        return BCType.ANTIPERIODIC
    elif bc_id in ['zero gradient', 'zg', 'ZG']:
        return BCType.ZEROGRADIENT
    elif bc_id in ['zero value', 'zv', 'ZV']: # previosuly 'r','R','reflecting','reflect']
        return BCType.ZEROVALUE
    elif bc_id in ['reflecting', 'r', 'R']:
        return BCType.REFLECTING
    elif bc_id in ['absorbing', 'a', 'A']:
        return BCType.ABSORBING
    else:
        return BCType.OPEN


def parse_basis_type(basis_id) -> 'BasisType':
    """ return BasisType(Enum) given accepted basis_id (strs)
    """
    if isinstance(basis_id, BasisType):
        return basis_id

    # if basis_id in ['cart','cartesian','Cartesian']:
    #     return BasisType.CARTESIAN
    # elif basis_id in ['cyl','cylindrical','Cylindrical']:
    #     return BasisType.CYLINDRICAL
    # elif basis_id in ['sph','spherical','Spherical']:
    #     return BasisType.CYLINDRICAL

    if basis_id in ['spatial','Spatial']:
        return BasisType.SPATIAL
    elif basis_id in ['k','fourier','Fourier']:
        return BasisType.FOURIER
    elif basis_id in ['H','Hermite']:
        return BasisType.HERMITE

def get_basis(basis_type) -> 'Basis':
    """ return basis class given BasisType(Enum)
    """
    if not isinstance(basis_type,Enum):
        basis_type = parse_basis_type(basis_type)

    # if basis_type == BasisType.CARTESIAN:
    #     from basis.basis_cart import CartesianBasis
    #     return CartesianBasis
    # if basis_type == BasisType.CYLINDRICAL:
    #     from basis.basis_cyl import CylindricalBasis
    #     return CylindricalBasis
    # if basis_type == BasisType.SPHERICAL:
    #     import basis.basis_sph as SphericalBasis
    #     return SphericalBasis

    if basis_type == BasisType.SPATIAL:
        from basis.basis_spatial import SpatialBasis
        return SpatialBasis()
    elif basis_type == BasisType.FOURIER:
        from basis.basis_k import FourierBasis
        return FourierBasis()
    elif basis_type == BasisType.HERMITE:
        from basis.basis_hermite import HermiteBasis
        return HermiteBasis()


@dataclass(unsafe_hash=True)
class Axis:
    _frozen = False

    def __init__(self, L: int, q: int = 2, coordinate: 'Coordinate' = None,
                 x0: int = 0, dx: Numeric = 1, xpts: Optional[np.ndarray] = None,
                 basis: 'Basis' = None, is_flipped = False, ax_map : 'AxisMap' = None):
        """ Inputs:
                axID:   Coordinate of Axis. Can be shared across different Axis objects
                q, L:   determines vector size (L-dimensional tensor with size of q)
                x0:     value of first point in discretization (ie. xpts[0]) (default = 0)
                dx:     spacing of grid points (default = 1)
                xpts:   an np.array specifying the actual points in the discretization
                        allows for non-even discretizations. must be of length q**L
                        (default = None)
                basis_type:  basis class identifier (BasisType or one of accepted strings)
                        (default = BasisType.CARTESIAN)

            Attributes:
                coordinate:    coordinate identifier
                q, L:    parameters of vector size, q**L
                basis:   basis class
                xpts:    x-points along this dimension. must be monotically increasing, but can be unevenly spaced
                dx:      spacing of xpts. either int (if is_even) or np.array (not is_even)
                is_even: if discretization is evenly spaced
                is_flipped:  if True, tensors are ordered fine to coarse (left to right)
                             else, tensors are ordered coarse to fine (right to left)

        """
        self.coordinate = coordinate
        self.q = q
        self.L = L
        self.npts = q**L
        self._get_xpts(x0,dx,xpts)
        self.is_flipped = is_flipped

        if ax_map is None:
            ax_map = BinaryMap()
        self.map = ax_map

        # self.basis_type = parse_basis_type(basis)
        if basis is None:
            self.basis = SpatialBasis()
        else:
            self.basis = basis

        ## assigned when included in a CoordinateSystem object
        self.coord_sys = CoordinateSystem('_DUMMY', self)

        zero_ind = np.argmax(self.xpts >= 0)   ## assumes xpts are monotonically increasing
        if self.xpts[zero_ind] != 0:
            if zero_ind > 0:                   ## return crossover from neg to positive values
                zero_ind = (zero_ind-1,zero_ind)
            else:                              ## 0 doesn't exist
                zero_ind = None
        self.zero_ind = zero_ind

        self._frozen = True


    def __str__(self):
        return '(' + self.coord_sys.coordID + ',' + str(self.coordinate) + ')'

    def __repr__(self):
        out_str = f'Axis({self.L},{self.q},{self.coordinate})'
        return out_str

    def __delattr__(self, item):
        if self._frozen:
            raise AttributeError('not allowed to alter Axis attributes')
        object.__delattr__(self,item)

    def __setattr__(self, key, value):
        if self._frozen:
            if key == 'coord_sys' and self.coord_sys.coordID == '_DUMMY':
                object.__setattr__(self, key, value)
            else:
                raise AttributeError('not allowed to alter Axis attributes')
        object.__setattr__(self,key,value)


    def __eq__(self, other:'Axis'):
        return self is other

    ### use default
    # def __eq__(self,other:'Axis'):
    #     bool0 = (self.axID == other.axID)
    #     bool1 = (self.basis_type == other.basis_type)
    #     bool2 = (self.q == other.q)
    #     bool3 = (self.L == other.L)
    #     bool4 = (self.xpts[0] == other.xpts[0])
    #     bool5 = (self.dx == self.dx)
    #     bool6 = (self.coord_type == other.coord_type)
    #     return bool0 and bool1 and bool2 and bool3 # and bool4


    def shape(self) -> tuple[int]:
        return (self.q,)*self.L


    def _get_xpts(self, x0, dx, xpts) -> np.ndarray:
        if xpts is None:
            self.is_even = True
            self.dx = dx
            self.xpts = np.arange(self.q**self.L)*dx + x0
        else:
            diff = np.diff(xpts)
            self.xpts = xpts
            assert(len(xpts)==self.q**self.L), f'provided xpts must be of length {self.q**self.L}'

            if np.all( np.abs(diff-diff[0]) < 1.0E-12 ):
                self.is_even = True
                self.dx = diff[0]
            else:
                self.is_even = False
                self.dx = diff
        return self.xpts


    def create_like(self, new_coord: Coordinate): #, new_coord_sys=None, new_coord_type=None):
        new_ax = self.__class__( self.L, self.q, coordinate = new_coord, basis = self.basis, xpts=self.xpts,
                                 is_flipped=self.is_flipped)
        # new_ax.xpts = self.xpts
        # new_ax.dx = self.dx
        # new_ax.is_even = self.is_even
        # new_ax.zero_ind = self.zero_ind
        return new_ax


    def get_val_ind(self, val: Numeric) -> Optional[int]:
        """ find index (int) of grid poin corresponding to desired value val along axis ax
        """
        ind = np.argwhere(np.abs(self.xpts - val) < 1.0 - 10)
        try:
            return ind[0]
        except IndexError:
            print('value',val,'not found in grid')
            return None

    def get_position_inds(self, idx: int) -> list[int]:
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        return self.map.get_position_inds(self.L, self.q, idx)

    def get_inds_position(self, inds_list: Iterable[int], ax_map: 'AxisMap' = None) -> int:
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        return self.map.get_inds_position(self.q, inds_list)

    def is_k(self):
        if isinstance(self.basis, FourierBasis):
            return True
        return False

    def is_real_k(self):
        if isinstance(self.basis, RealFourierBasis):
            return True
        return False

    ######################
    ## build common TNs ##
    ######################

    def get_all_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> MPSType:
        """ build ones vector mps in shape of self
            TODO: rename this and get_ones_mps??
        """
        return get_ones_mps(self.L, self.q, site_ind_id, site_tag_id)

    def get_iden_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> MPSType:
        """ build ones vector mps in shape of self
        """
        return self.basis.get_ones_mps(self, site_ind_id, site_tag_id)  ## takes map into account
        # return get_ones_mps(self.L, self.q, site_ind_id, site_tag_id)

    def get_iden_mpo(self, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})') -> MPOType:
        """ build identity mpo
        """
        return get_iden_mpo(self.L, self.q, upper_ind_id, lower_ind_id, site_tag_id)

    def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})',
                             compress=False, compress_opts: dict = None) -> MPSType:
        """ build MPS to select certain elements specified by inds
        """
        ind_list = self.get_position_inds(inds[0])
        # print('select ind', inds[0], ind_list, self.map)
        mps = get_select_elem_mps(self.L, self.q, ind_list, site_ind_id, site_tag_id)
        for ind in inds[1:]:
            ind_list = self.get_position_inds(ind)
            mps_ = get_select_elem_mps(self.L, self.q, ind_list, site_ind_id, site_tag_id)
            helper.add_MPS(mps, mps_, inplace=True, compress=False)

        if compress:
            if compress_opts is None:   compress_opts = {}
            helper.compress(mps, **compress_opts)

        return mps

    def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})',
                             compress=False, compress_opts: dict = None) -> MPOType:
        """ build MPO to select certain elements specified by inds
        """
        ind_list = self.get_position_inds(inds[0])
        mpo = get_select_elem_mpo(self.L, self.q, ind_list, upper_ind_id, lower_ind_id, site_tag_id,)
        for ind in inds[1:]:
            ind_list = self.get_position_inds(ind)
            mpo_ = get_select_elem_mpo(self.L, self.q, ind_list, upper_ind_id, lower_ind_id, site_tag_id,)
            helper.add_MPO(mpo, mpo_, inplace=True, compress=False)

        if compress:
            if compress_opts is None:   compress_opts = {}
            helper.compress(mpo, **compress_opts)

        return mpo

    def get_all_except_elem_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> MPSType:
        """ build MPO in which all elements are 1 except at element in inds which are 0
        """
        out = self.get_all_ones_mps(site_ind_id, site_tag_id)
        select_mps = self.get_select_elems_mps(inds, site_ind_id, site_tag_id)
        helper.scalar_multiply(select_mps, -1, inplace=True)
        helper.add_MPS(out, select_mps, inplace=True, compress=False)
        return out

    def get_all_except_elem_mpo(self, inds: Iter[int], upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})')\
            -> MPOType:
        """ build MPS in which all elements along diagonal are 1 except at element in inds which are 0
        """
        out = self.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        select_mpo = self.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id)
        helper.scalar_multiply(select_mpo, -1, inplace=True)
        helper.add_MPO(out, select_mpo, inplace=True, compress=False)
        return out

    def get_heaviside_mpo(self, zero_ind: int, get='right') -> MPOType:
        """ get Heaviside step function where side denoted by 'get' is 1. x denotes that zero_ind
            is excluded
        """
        out = get_heaviside_mpo(self.L, self.q, zero_ind, get)
        out = self.map.transform_mpo(out)
        return out

    def get_reverse_diag_mpo(self, upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id: str = 'T{}') \
            -> MPOType:
        """ build MPO with 1's along backward diagonal
        """
        out = get_reverse_diag_mpo(self.L, self.q, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                   site_tag_id=site_tag_id)
        out = self.map.transform_mpo(out)
        return out

    def get_shift_mpo(self, shift: int, boundary_conditions: 'DerivativeConfiguration' = None,
                      upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})') -> MPOType:
        """ build MPO with 1's along diagonal shifted by k
        """
        out = get_shift_mpo(self.L, self.q, shift, boundary_conditions=boundary_conditions,
                            upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        out = self.map.transform_mpo(out)
        return out


    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                         split_opts: dict = None) -> MPSType:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        ## i think transform mps does not require us to transform vector array
        # # state = self.axis_map.transform_array(state)   ## assume data only along axis dimension
        out = map_state_to_mps(self.L, self.q, state, site_ind_id, site_tag_id, split_opts=split_opts)
        out = self.map.transform_mps(out)
        return out


    def map_mps_to_state(self, mps) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        mps = self.map.inverse_transform_mps(mps)
        return map_mps_to_state(mps)


    def map_operator_to_mpo(self, operator, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                        site_tag_id: str = 'T({})', split_opts:dict = None) -> MPOType:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        out = map_operator_to_mpo(self.L, self.q, operator, upper_ind_id, lower_ind_id, site_tag_id,
                                  split_opts=split_opts)
        out = self.map.transform_mpo(out)
        return out


    def map_mpo_to_operator(self, mpo) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        mpo = self.map.inverse_transform_mpo(mpo)
        return map_mpo_to_operator(mpo, flip=self.is_flipped)


    ##################################
    ## build derivatives, xmultiply ##
    ##################################

    def build_firstderivative_mpo(self, deriv_config: 'DerivativeConfiguration') -> MPOType:
        """ build MPO that takes first derivative
            deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis,SpatialBasis):
            raise NotImplementedError('first derivative for uneven dx for spatial basis not supported')
        mpo = self.basis.build_firstderivative_mpo(self, **deriv_config.deriv_params)
        # print('mpo', self, deriv_config.order, deriv_config.fd_type, mpo.max_bond())
        return mpo

    def build_secondderivative_mpo(self, deriv_config: 'DerivativeConfiguration') -> MPOType:
        """ deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):  ## real space method
            raise NotImplementedError('second derivative for uneven dx for spatial basis not supported')
        mpo = self.basis.build_secondderivative_mpo(self, **deriv_config.deriv_params)
        return mpo


    def get_xmultiply_mps(self, x_power:int =1, offset:Numeric =0.0, scale:Numeric =1.0, split_opts:dict = None)\
            -> MPOType:
        """ x * f(x) MPO
            (mpo already flipped/transformed in basis method if need be)
        """
        mpo = self.basis.build_xmultiply_mps(self, x_power=x_power, offset=offset, scale=scale, split_opts=split_opts)
        return mpo


    def get_xmultiply_mpo(self, x_power:int =1, offset:Numeric =0.0, scale:Numeric =1.0, split_opts:dict = None)\
            -> MPOType:
        """ x * f(x) MPO
            (mpo already flipped/transformed in basis method if need be)
        """
        mpo = self.basis.build_xmultiply_mpo(self, x_power=x_power, offset=offset, scale=scale, split_opts=split_opts)
        return mpo


    def get_elemental_multiply_tn(self,in1_ind_id: str = 'i({})[1]', in2_ind_id: str = 'i({})[2]',
                                  out_ind_id: str = 'o({})', site_tag_id='d_ijk({})') -> MPTType:
        """ x * f(x) MPO
            (tn1d already flipped/transformed in basis method if need be)
        """
        tn1d = self.basis.build_elemental_multiply_tn(self, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
        return tn1d


    def apply_elemental_multiply_op(self, mps, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                                    site_tag_id: str = 'T({})', compress=False, compress_opts=None) -> MPOType:

        if mps.site_ind_id == upper_ind_id:
            mps = mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        if mps.site_ind_id == lower_ind_id:
            mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        elem_mult_tn = self.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=mps.site_ind_id,
                                                      out_ind_id=upper_ind_id, site_tag_id=site_tag_id)

        new_mpo = qtn.TensorNetwork([])
        for x in range(self.L):
            tens_x1 = mps[x]
            tens_x2 = elem_mult_tn[x]
            new_tens = tens_x1.contract(tens_x2)  ## includes ancilla inds
            if tens_x1.tags != tens_x2.tags:
                new_tens.drop_tags(tags=tens_x1.tags)
            new_mpo.add(new_tens)

        new_mpo.fuse_multibonds(inplace=True)
        new_mpo = new_mpo.view_as(qtn.MatrixProductOperator, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                  site_tag_id=site_tag_id, inplace=True, cyclic=False, L=self.L)
        new_mpo.exponent = mps.exponent + elem_mult_tn.exponent

        if compress:
            helper.compress(new_mpo, compress_opts=compress_opts)

        return new_mpo


    def get_integral_mps(self, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})') -> MPSType:
        """ integral (reimann sum)
            (mps already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):
            raise NotImplementedError
        return self.basis.build_integral_mps(self, site_ind_id=site_ind_id, site_tag_id=site_tag_id)


    def get_integral_weight(self):
        return self.basis.get_integral_weight(self)


    def get_qft_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'

        qft_mpo = get_qft_mpo(self.L, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                              lower_ind_id=lower_ind_id)
        qft_mpo = self.map.transform_mpo(qft_mpo)
        # helper.scalar_multiply(qft_mpo, np.sqrt(self.npts), inplace=True)

        return qft_mpo


    def get_qft_mpo_pad_k(self, num_pad=1, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'
        qft_mpo = get_qft_mpo(self.L+num_pad, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                              lower_ind_id=lower_ind_id)

        ## removed interpolated finer grid in real space
        for x in range(num_pad):
            tens_x = qft_mpo[self.L+x]
            tens_x.isel({lower_ind_id.format(self.L+x): 0}, inplace=True)

        qft_mpo = self.map.transform_mpo(qft_mpo, upper_L=self.L+num_pad, lower_L=self.L)
        # helper.scalar_multiply(qft_mpo, np.sqrt(self.npts), inplace=True)

        return qft_mpo


    def get_qft_ks(self, num_pad=0):
        """ get k's associated with qft in vector form
        """
        assert num_pad >= 0, 'padding k must be >= 0'

        L = self.L + num_pad
        kvals = np.fft.fftfreq(self.q**L) * 2 * np.pi / self.dx

        ## set padded k's to zero
        if num_pad > 0:
            kmax = (self.npts / 2) * 2 * np.pi / self.dx
            kvals = np.where(kvals >= kmax, np.zeros_like(kvals), kvals)
            kvals = np.where(kvals < -kmax, np.zeros_like(kvals), kvals)

        kvals = kvals.reshape((self.q,) * L).transpose().reshape(-1)   # for binary rep
        ## kvals = self.map.transform_vector(kvals)
        return kvals


    def get_inverse_qft_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'

        qft_mpo = get_inverse_qft_mpo(self.L, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                      lower_ind_id=lower_ind_id)
        qft_mpo = self.map.inverse_transform_mpo(qft_mpo)
        # helper.scalar_multiply(qft_mpo, 1./np.sqrt(self.npts), inplace=True)

        return qft_mpo


    def get_inverse_qft_mpo_pad_k(self, num_pad=1, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})',
                                  lower_ind_id: str = 'i({})'):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'
        qft_mpo = get_inverse_qft_mpo(self.L + num_pad, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                      lower_ind_id=lower_ind_id)

        ## removed interpolated finer grid in real space
        for x in range(num_pad):
            tens_x = qft_mpo[self.L + x]
            tens_x.isel({lower_ind_id.format(self.L + x): 0}, inplace=True)

        qft_mpo = self.map.transform_mpo(qft_mpo, lower_L=self.L + num_pad, upper_L=self.L)
        # helper.scalar_multiply(qft_mpo, np.sqrt(self.npts), inplace=True)

        return qft_mpo

######################
## static functions ##
######################

def get_ones_mps(L: int, q: int, site_ind_id: str = 'i({})', site_tag_id: str = 'X({})') -> MPSType:
    """ build ones vector mps
    """
    ones = np.ones((q,))
    mps_0 = None
    if L >= 2:
        mps_0 = qtn.MatrixProductState(
            [np.array([ones])] + [np.array([[ones]])] * (L - 2) + [np.array([ones])],
            shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 1:
        mps_0 = qtn.TensorNetwork([qtn.Tensor(ones, inds=('i(0)',), tags=('X(0)',))])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 0:
        mps_0 = qtn.TensorNetwork([])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    return mps_0


def get_iden_mpo(L: int, q: int, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                 site_tag_id: str = 'X({})') -> MPOType:
    """ build identity mpo
    """
    iden = np.eye(q)
    mpo_0 = None
    if L >= 2:
        mpo_0 = qtn.MatrixProductOperator(
            [np.array([iden])] + [np.array([[iden]])] * (L - 2) + [np.array([iden])],
            shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 1:
        mpo_0 = qtn.TensorNetwork([qtn.Tensor(iden, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                              tags=(site_tag_id.format(0),))])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 0:
        mpo_0 = qtn.TensorNetwork([])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    return mpo_0


def get_select_elem_mps(L: int, q: int, inds_list: Sequence[int], site_ind_id: str ='i({})', site_tag_id: str = 'X({})')\
        -> MPSType:
    """ build MPS to select certain elements specified by inds
        inds_list:  binary strings
    """
    mdict = {}
    for b in range(q):
        mvec = np.zeros((q,))
        mvec[b] = 1.
        mdict[b] = mvec

    mps_0 = None
    if L >= 2:
        ms = [mdict[b] for b in inds_list]
        mps_0 = qtn.MatrixProductState(
            [np.array([ms[0]])] + [np.array([[m]]) for m in ms[1:-1]] + [np.array([ms[-1]])],
            shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 1:
        m = mdict[inds_list[0]]
        mps_0 = qtn.TensorNetwork([ qtn.Tensor(m, inds=(site_ind_id.format(0),), tags=(site_tag_id.format(0),) )])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 0:
        mps_0 = qtn.TensorNetwork([])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    return mps_0


def get_select_elem_mpo(L: int, q: int, inds_list: Sequence[int], upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                        site_tag_id: str = 'X({})') -> MPOType:
    """ build MPO to select certain elements specified by inds
    """
    mdict = {}
    for b in range(q):
        mvec = np.zeros((q,))
        mvec[b] = 1.
        mdict[b] = np.diag(mvec)

    assert(len(inds_list) == L), 'inds_list needs to be desired index in q-nary'

    mpo_0 = None
    if L >= 2:
        ms = [mdict[b] for b in inds_list]
        mpo_0 = qtn.MatrixProductOperator(
            [np.array([ms[0]])] + [np.array([[m]]) for m in ms[1:-1]] + [np.array([ms[-1]])],
            shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 1:
        m = mdict[inds_list[0]]
        mpo_0 = qtn.TensorNetwork([ qtn.Tensor(m, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                               tags=(site_tag_id.format(0),) )])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 0:
        mpo_0 = qtn.TensorNetwork([])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    return mpo_0


def map_state_to_mps(L: int, q: int, state: np.ndarray, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                     split_opts: dict = None) -> MPSType:
    """ convert state represented as vector converted to mps state
        here, returns original vec as a single tensor as quimb.TensorNetwork object
    """
    if split_opts is None:   split_opts = {}
    state = state.reshape((q,) * L)
    state_tensor = qtn.Tensor(data=state, inds=[site_ind_id.format(x) for x in range(L)])
    state_mps = helper.mpx_from_dense(state_tensor, L, (site_ind_id,), site_tag_id=site_tag_id, split_opts=split_opts)
    return state_mps


def map_operator_to_mpo(L: int, q: int, operator: np.ndarray, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                        site_tag_id: str = 'T({})', split_opts: dict = None) -> MPOType:
    """ convert operator represented as a matrix (output x input) as an MPO
            here, returns original vec as a single tensor as quimb.TensorNetwork object
    """
    if split_opts is None:   split_opts = {}
    operator = operator.reshape((q,) * L * 2)
    state_tensor = qtn.Tensor(data=operator, inds=[upper_ind_id.format(x) for x in range(L)] +
                                                  [lower_ind_id.format(x) for x in range(L)])
    state_mpo = helper.mpx_from_dense(state_tensor, L, (upper_ind_id, lower_ind_id), site_tag_id=site_tag_id,
                                      **split_opts)
    return state_mpo


def map_mps_to_state(mps: MPSType, flip=False) -> np.ndarray:
    """ convert MPS into 1-D np.ndarray
    """
    L = mps.L
    q = mps.phys_dim()

    tensor = mps.contract()  # contract all tensors
    if flip:
        tensor.transpose(*[mps.site_ind_id.format(i) for i in range(L-1,-1,-1)], inplace=True)
    else:
        tensor.transpose(*[mps.site_ind_id.format(i) for i in range(L)], inplace=True)

    ## need to reshape to ndim-dimensional tensor
    out_data = tensor.data  # self.axis_map.array_reshape(tensor.data, (self.q,)*self.L*ndim)
    out_data = out_data.reshape((q**L,))

    return out_data * (10 ** mps.exponent)


def map_mpo_to_operator(mpo: MPOType, flip=False) -> np.ndarray:
    """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
    """
    L = mpo.L
    q = mpo.phys_dim()

    tensor = mpo.contract()  # contract all tensors
    if flip:
        out_inds = [mpo.upper_ind_id.format(i) for i in range(L-1,-1,-1)] + \
                   [mpo.lower_ind_id.format(i) for i in range(L-1,-1,-1)]
    else:
        out_inds = [mpo.upper_ind_id.format(i) for i in range(L)] + \
                   [mpo.lower_ind_id.format(i) for i in range(L)]
    tensor.transpose(*out_inds, inplace=True)

    ## need to reshape to ndim-dimensional tensor
    out_data = tensor.data
    out_data = out_data.reshape((q**L,)*2)

    return out_data * (10 ** mpo.exponent)


def get_heaviside_mpo(L: int, q: int, zero_ind: int, get='right', upper_ind_id: str ='o({})',
                      lower_ind_id: str = 'i({})', site_tag_id: str = 'T({})') -> MPOType:
    """ get mpo representing heaviside mpo function
        zero_ind specifies index at which the step occurs
        get: 'right','xright','left','xleft'
           right: inds >= zero_ind = 1 else 0
           left:  inds <= zero_ind = 1 else 0
           x:  excludes zero_ind
    """
    npts = q**L

    if isinstance(zero_ind, tuple):      # zero grid pt not included
        zero_ind_L, zero_ind_R = zero_ind
        zero_ind_L1, zero_ind_R1 = zero_ind_L, zero_ind_R
    else:
        zero_ind_L = zero_ind_R = zero_ind
        zero_ind_L1, zero_ind_R1 = zero_ind_L-1, zero_ind_R+1

    if (zero_ind_R == -1 and get == 'right') or (zero_ind_L == npts and get == 'left'):
        return get_iden_mpo(L, q, upper_ind_id, lower_ind_id, site_tag_id)
    elif (zero_ind_L == -1 and get == 'left') or (zero_ind_R == npts and get == 'right'):
        return None
    else:
        pass

    if   get=='right':    vec = np.append(np.zeros((zero_ind_R,)),np.ones((npts-zero_ind_R,)))
    elif get=='xright':   vec = np.append(np.zeros((zero_ind_R1,)),np.ones((npts-zero_ind_R1,)))
    elif get=='xleft':    vec = np.append(np.ones((zero_ind_L1+1,)),np.zeros((npts-zero_ind_L1-1,)))
    elif get=='left':     vec = np.append(np.ones((zero_ind_L+1,)),np.zeros((npts-zero_ind_L-1,)))
    else:
        raise ValueError('need valid input for get parameter')

    # vec = vec.reshape((q,)*L)
    # vec_tensor = qtn.Tensor(data=vec,inds=[lower_ind_id.format(x) for x in range(L)])
    # heaviside_mps = helper.mpx_from_dense(vec_tensor,L,(lower_ind_id,),site_tag_id=site_tag_id)
    heaviside_mps = map_state_to_mps(L, q, vec, lower_ind_id, site_tag_id) # default compress opts
    heaviside_mpo = helper.mps_to_diag_mpo(heaviside_mps, upper_ind_id=upper_ind_id,
                                           lower_ind_id=lower_ind_id)
    return heaviside_mpo


def get_reverse_diag_mpo(L: int, q:int, upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}',
                  site_tag_id:str = 'T{}') -> 'MPOType':
    """ mpo that flip inds:  SX * SX * ...
    """
    sx = np.zeros((q,q))
    for i in range(q):
        sx[i,-1-i] = 1.0

    tens_list = [np.array([sx])] + [np.array([[sx]])] * (L-2) + [np.array([sx])]
    return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                         lower_ind_id=lower_ind_id)


# def get_shift_mpo(L: int, q: int, shift: int, left_bc: 'BCType'=None, right_bc: 'BCType'=None,
#                   upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id:str = 'T{}') -> 'MPOType':
#     """ make an MPO with 1's shifted from the diagonal
#     """
#     if shift == 0:
#         tens_list = [np.array([np.eye(q)])] + [np.array([[np.eye(q)]])] * (L - 2) + [np.array([np.eye(q)])]
#         return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
#                                          lower_ind_id=lower_ind_id)
#
#     if q != 2:
#         raise NotImplementedError
#
#     sp = np.array([[0., 0.], [1., 0.]])
#     sm = np.array([[0., 1.], [0., 0.]])
#     zero = np.zeros((q, q))
#
#     mat1 = sm if shift > 0 else sp
#     mat2 = sp if shift > 0 else sm
#
#     # nbits = np.round(np.log(shift) / np.log(q))  # number of bits to encode shift value
#     nbits = int(np.floor(np.log(np.abs(shift)) / np.log(q)) + 1)  # number of bits to encode shift value
#
#     if abs(shift) == 1:
#         last_tens = [np.array([mat1, mat2])]
#     elif abs(shift) == q:
#         last_tens = [np.array([[np.eye(q), mat1], [zero, mat2]]),
#                      np.array([zero, np.eye(q)])]
#     else:
#         raise NotImplementedError
#
#     if left_bc == BCType.PERIODIC:
#         tens_list = [np.array([np.eye(q), mat1 + mat2])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#     elif left_bc == BCType.ANTIPERIODIC:
#         tens_list = [np.array([np.eye(q), mat1 - mat2])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#     else:   # OPEN, no boundaries
#         tens_list = [np.array([np.eye(q), mat1])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#
#     tens_list += last_tens
#
#     out = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
#                                      lower_ind_id=lower_ind_id)
#     return out


def get_shift_mpo(L: int, q: int, shift: int, boundary_conditions: 'DerivativeConfiguration' = None,
                  upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id: str = 'T{}') -> 'MPOType':
    """ make an MPO with 1's shifted from the diagonal
    """
    if shift == 0:
        tens_list = [np.array([np.eye(q)])] + [np.array([[np.eye(q)]])] * (L - 2) + [np.array([np.eye(q)])]
        return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                         lower_ind_id=lower_ind_id)

    if q != 2:
        raise NotImplementedError

    if boundary_conditions is None:
        boundary_conditions = DerivativeConfiguration()

    left_bc, right_bc = boundary_conditions.bc
    bc_offset = boundary_conditions.offset

    sp = np.array([[0., 0.], [1., 0.]])
    sm = np.array([[0., 1.], [0., 0.]])
    zero = np.zeros((q, q))

    mat1 = sm if shift > 0 else sp
    mat2 = sp if shift > 0 else sm

    # nbits = np.round(np.log(shift) / np.log(q))  # number of bits to encode shift value
    nbits = int(np.floor(np.log(np.abs(shift)) / np.log(q)) + 1)  # number of bits to encode shift value

    if abs(shift) == 1:
        last_tens = [np.array([mat1, mat2])]
    elif abs(shift) == q:
        last_tens = [np.array([[np.eye(q), mat1], [zero, mat2]]),
                     np.array([zero, np.eye(q)])]
    else:
        raise NotImplementedError


    if left_bc == BCType.PERIODIC:
        tens_list = [np.array([np.eye(q), mat1 + mat2])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
    elif left_bc == BCType.ANTIPERIODIC:
        tens_list = [np.array([np.eye(q), mat1 - mat2])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
    else:  # OPEN, no boundaries
        tens_list = [np.array([np.eye(q), mat1])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)

    tens_list += last_tens

    out = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                    lower_ind_id=lower_ind_id)

    if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC:
        nbits_offset = int(np.floor(np.log(np.abs(-shift - min(0,bc_offset))) / np.log(q)) + 1)
        end_mat_0 = np.zeros((q**nbits_offset, q**nbits_offset))
        for i in range(-shift):
            try:
                end_mat_0[i, -shift - bc_offset - i] = 1.0 * np.sign(left_bc.value)
            except IndexError:
                pass

        if not np.linalg.norm(end_mat_0) < 1.0e-16:
            bc0_tens = qtn.Tensor( end_mat_0.reshape((q,)* nbits * 2), inds=tuple([f'o({i})' for i in range(nbits)] + \
                                                                                  [f'i({i})' for i in range(nbits)]) )
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nbits, ('o({})', 'i({})'), site_tag_id='X({})')

            mpo_0 = get_select_elem_mpo(L-nbits, q, [0]*(L-nbits))
            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.add_MPO(out, mpo_0, inplace=True)

    elif left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC or left_bc == BCType.OPEN:
        pass
    else:
        raise NotImplementedError

    if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC:
        nbits_offset = int(np.floor(np.log(np.abs(-shift + min(0,bc_offset))) / np.log(q)) + 1)
        end_mat_0 = np.zeros((q ** nbits_offset, q ** nbits_offset))
        for i in range(shift):
            try:
                end_mat_0[-i-1, q**nbits - shift - bc_offset + i - 1] = 1.0 * np.sign(right_bc.value)
            except IndexError:
                pass

        if not np.linalg.norm(end_mat_0) < 1.0e-16:
            bc0_tens = qtn.Tensor(end_mat_0.reshape((q,) * nbits * 2), inds=tuple([f'o({i})' for i in range(nbits)] + \
                                                                                  [f'i({i})' for i in range(nbits)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nbits, ('o({})', 'i({})'), site_tag_id='X({})')

            mpo_0 = get_select_elem_mpo(L - nbits, q, [1]*(L-nbits))
            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.add_MPO(out, mpo_0, inplace=True)

    elif left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC or left_bc == BCType.OPEN:
        pass
    else:
        raise NotImplementedError

    return out


def get_qft_mpo(L, q, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
    """ obtain MPO that takes discrete Fourier transform
    """
    if L == 1:
        mat = np.eye(q)
        qft_mat = np.fft.fft(mat,axis=0)
        qft_tens = qtn.Tensor(qft_mat, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                              tags=(site_tag_id.format(0),))
        qft_mpo = qtn.TensorNetwork([qft_tens])
        qft_mpo = qft_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                  upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        return qft_mpo


    if q != 2:
        raise NotImplementedError

    zero = np.zeros((q,q), dtype=complex)
    delta = np.eye(q, dtype=complex)
    # hadamard = 1. / np.sqrt(2) * np.array([[1., 1.],[1., -1.]])
    hadamard_0 = 1. / np.sqrt(2) * np.array([[1., 1.], [0., 0.]])
    hadamard_1 = 1. / np.sqrt(2) * np.array([[0., 0.], [1.,-1.]])

    qft_mpo = None
    for i in range(L):

        tens_list = []

        for j in range(i):
            if j==0:
                tens_list += [np.array([delta])]
            else:
                tens_list += [np.array([[delta]])]

        if i == 0:
            tens_list += [ np.array([hadamard_0, hadamard_1]) ]
        elif i == L-1:
            tens_list += [ np.array([hadamard_0+hadamard_1]) ]
        else:
            tens_list += [ np.array([[hadamard_0, hadamard_1]]) ]

        for j in range(i+1,L):
            op00 = delta
            op11 = np.array([[1., 0.],[0., np.exp(-1.j*np.pi/(2**(j-i))) ]])
            if j != L-1:
                tens_list += [ np.array([[op00, zero],[zero, op11]]) ]
            else:
                tens_list += [ np.array([op00, op11]) ]

        mpo_i = qtn.MatrixProductOperator( tens_list, shape='lrud', site_tag_id=site_tag_id,
                                           upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id )
        if qft_mpo is None:
            qft_mpo = mpo_i
        else:
            qft_mpo = helper.apply(mpo_i, qft_mpo, compress=True,
                                   compress_opts={'cutoff':1.0e-20, 'cutoff_mode':'rsum2'})

    return qft_mpo


def get_inverse_qft_mpo(L, q, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
    """ obtain MPO that takes discrete Fourier transform
    """
    if L == 1:
        mat = np.eye(q)
        qft_mat = np.fft.ifft(mat,axis=1)
        qft_tens = qtn.Tensor(qft_mat, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                              tags=(site_tag_id.format(0),))
        qft_mpo = qtn.TensorNetwork([qft_tens])
        qft_mpo = qft_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                  upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        return qft_mpo


    if q != 2:
        raise NotImplementedError

    zero = np.zeros((q, q), dtype=complex)
    delta = np.eye(q, dtype=complex)
    # hadamard = 1. / np.sqrt(2) * np.array([[1., 1.],[1., -1.]])
    hadamard_0 = 1. / np.sqrt(2) * np.array([[1., 1.], [0., 0.]]).T
    hadamard_1 = 1. / np.sqrt(2) * np.array([[0., 0.], [1., -1.]]).T

    qft_mpo = None
    for i in range(L):

        tens_list = []

        for j in range(i):
            if j == 0:
                tens_list += [np.array([delta])]
            else:
                tens_list += [np.array([[delta]])]

        if i == 0:
            tens_list += [np.array([hadamard_0, hadamard_1])]
        elif i == L - 1:
            tens_list += [np.array([hadamard_0 + hadamard_1])]
        else:
            tens_list += [np.array([[hadamard_0, hadamard_1]])]

        for j in range(i + 1, L):
            op00 = delta
            op11 = np.array([[1., 0.], [0., np.exp(1.j * np.pi / (2 ** (j - i)))]])
            if j != L - 1:
                tens_list += [np.array([[op00, zero], [zero, op11]])]
            else:
                tens_list += [np.array([op00, op11])]


        mpo_i = qtn.MatrixProductOperator( tens_list, shape='lrud', site_tag_id=site_tag_id,
                                           upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id )
        if qft_mpo is None:
            qft_mpo = mpo_i
        else:
            qft_mpo = helper.apply(qft_mpo, mpo_i, compress=True,
                                   compress_opts={'cutoff':1.0e-20, 'cutoff_mode':'rsum2'})
    return qft_mpo