from setup.configs import *
# from enums import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN


"""
Basis Class (Interface):
    Specifies how to take derivatives and perform multiplications on given basis set.
    A property of Grid class, which contains attribute axes which is a list of Axis objects

Make all kwargs into args
"""

class Basis:

    def __init__(self):
        self.type = None

    def get_realspace_1D(self, data: 'np.ndarray', ax_ind: int, x0: Numeric=0.0, xL: Numeric=1.0, npts: Numeric=128):
        """ ie. coeffs of delta fcts """
        raise NotImplementedError

    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], x0: Numeric=0.0, xL: Numeric=1.0,
                         npts: Numeric=128):
        """ ie. coeffs of delta fcts """
        raise NotImplementedError

    # @classmethod
    # def elemental_multiply(cls, gtn1, gtn2, inplace=False, **compress_opts):
    #     #### DOESN"T REALLY FIT IN THIS CLASS...
    #     raise NotImplementedError

    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})') -> 'MPSType':
        raise NotImplementedError

    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id, in2_ind_id, out_ind_id, site_tag_id) -> MPTType:
        """ elemental multiply:  d_ijk in real space, convolution op in Fourier space
        """
        raise NotImplementedError

    def build_xmultiply_mps(self, ax: 'Axis', x_power: int=1, offset=0.0, scale=1.0, split_opts:dict = None) -> MPOType:
        """ x * H_m(x)
        """
        raise NotImplementedError

    def build_xmultiply_mpo(self, ax: 'Axis', x_power: int=1, offset=0.0, scale=1.0, split_opts:dict = None) -> MPOType:
        """ x * H_m(x)
        """
        raise NotImplementedError

    def build_firstderivative_mpo(self, ax: 'Axis', left_bc: BCType=DEFAULT_BC, right_bc: BCType=DEFAULT_BC,
                                  order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE, offset=0,
                                  compress_opts: dict = None) -> MPOType:
        """ get first derivative along specified axis of tn_grid (MPO along 1D)
        """
        raise NotImplementedError

    def build_secondderivative_mpo(self, ax: 'Axis', left_bc: BCType=DEFAULT_BC, right_bc: BCType=DEFAULT_BC,
                                   order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE, offset=0,
                                   compress_opts: dict = None) -> MPOType:
        """ get second derivative along specified axis
        """
        raise NotImplementedError

    def get_integral_weight(self, ax: 'Axis'):
        """ normalization for integral_mps
        """
        raise NotImplementedError

    def build_integral_mps(self, ax: 'Axis', is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})') -> MPSType:
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError

    def build_indefinite_integral_mps(self, ax: 'Axis', order=1) -> MPOType:
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError
