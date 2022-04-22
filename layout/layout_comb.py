import numpy as np
import quimb.tensor as qtn
import helper_quimb as helper
from ..grid1D import Grid1D

""" Layout_comb:  tensor train structure, where the legs are factorized into MPS
                  each leg/branch can be an Axis or GridLayout object
"""

class LayoutComb:

    @classmethod
    def L(cls, axes) -> int:
        mpx_L = sum([ax.L for ax in axes])
        return mpx_L

    @classmethod
    def get_inds_in_axis(cls, axes, axID, ax_ind=None) -> list:
        if ax_ind is None:
            axIDs = [ax.coordinate for ax in axes]
            ax_ind = axIDs.index(axID)
        return list(range(axes[ax_ind].L))

    @classmethod
    def shape(cls, axes) -> tuple:
        """ get shape assuming sequential ordering
        """
        out_shape = ()
        for ax in axes:
            out_shape += (ax.shape(),)
        return out_shape    # nested tuple

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    @classmethod
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', **split_opts):
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    @classmethod
    def map_mps_to_state(cls, axes, mps):
        """ convert MPS into 1-D np.ndarray
        """
        raise NotImplementedError

    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})',
                            site_tag_id='T({})', **split_opts):
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    @classmethod
    def map_mpo_to_operator(cls, axes, mpo):
        """ convert MPO into 1-D np.ndarray
        """
        raise NotImplementedError


