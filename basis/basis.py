from setup.defaults import *
# from enums import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


"""
Basis Class (Interface):
    Specifies how to take derivatives and perform multiplications on given basis set.
    A property of Grid class, which contains attribute axes which is a list of Axis objects

Make all kwargs into args
"""

class Basis:

    @classmethod
    def get_realspace_1D(cls, coeffs):   # mps already expanded from mps to vecs
        """ ie. coeffs of delta fcts """
        raise NotImplementedError

    # @classmethod
    # def elemental_multiply(cls, gtn1, gtn2, inplace=False, **compress_opts):
    #     #### DOESN"T REALLY FIT IN THIS CLASS...
    #     raise NotImplementedError

    @classmethod
    def build_elemental_multiply_tn(cls, ax: 'Axis', in1_ind_id, in2_ind_id, out_ind_id, site_tag_id) -> TNType:
        """ elemental multiply:  d_ijk in real space, convolution op in Fourier space
        """
        raise NotImplementedError

    @classmethod
    def build_xmultiply_mpo(cls, ax: 'Axis', offset=0.0, scale=1.0, split_opts:dict = None) -> MPOType:
        """ x * H_m(x)
        """
        raise NotImplementedError

    @classmethod
    def build_firstderivative_mpo(cls, ax: 'Axis', left_bc: BCType=DEFAULT_BC, right_bc: BCType=DEFAULT_BC,
                                  order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE, compress_opts: dict = None) -> MPOType:
        """ get first derivative along specified axis of tn_grid (MPO along 1D)
        """
        raise NotImplementedError

    @classmethod
    def build_secondderivative_mpo(cls, ax: 'Axis', left_bc: BCType=DEFAULT_BC, right_bc: BCType=DEFAULT_BC,
                                   order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE, compress_opts: dict = None) -> MPOType:
        """ get second derivative along specified axis
        """
        raise NotImplementedError

    @classmethod
    def build_integral_mps(cls, ax: 'Axis', site_ind_id='i({})', site_tag_id='T({})') -> MPSType:
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError

    @classmethod
    def build_indefinite_integral_mps(cls, ax: 'Axis', order=1) -> MPOType:
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError