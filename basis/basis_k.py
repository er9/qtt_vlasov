from setup.configs import *
import numpy as np
import quimb.tensor as qtn
import helper_quimb as helper
from basis.basis import Basis

if TYPE_CHECKING:
    from axis import Axis


class FourierBasis(Basis):

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ coeffs:  coefficients of Hermite polynomials (-K, -K+1, ..., 0, ..., K-1)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        data = np.moveaxis(data, ax_ind, -1)
        coeffs_  = np.fft.fftshift(data, axes=-1)   # moves 0 frequency term (at npts//2) to 0
        out_data = np.fft.ifftn(coeffs_, axes=[-1]) * data.shape[-1]
        out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data

    # @classmethod
    # def elemental_multiply(cls,tn1,tn2,ax1=None,ax2=None,inplace=False,**compress_opts):
    #     tn1 = tn1 if inplace else tn1.copy()
    #     tn1.convolve(tn2,ax1=ax1,ax2=ax2,**compress_opts)
    #     return tn1


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})') -> 'MPSType':
        return ax.get_select_elems_mps([ax.zero_ind], site_ind_id=site_ind_id, site_tag_id=site_tag_id)


    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        L, q = ax.L, ax.q
        if q != 2:  raise NotImplementedError('convolution for q!=2 not implemented')

        conv_op_tn = qtn.TensorNetwork([])

        # if ax.is_flipped:
        #     inds_list = list(range(L-1,-1,-1))
        # else:
        #     inds_list = list(range(L))
        inds_list = list(range(L))

        i0 = inds_list[0]
        T0 = qtn.Tensor(np.array([[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
                                  [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]]),
                        inds=(out_ind_id.format(i0), f'b{i0}', in1_ind_id.format(i0), in2_ind_id.format(i0)),
                        tags=(site_tag_id.format(i0),))
        conv_op_tn.add(T0)

        for x in range(1, L - 1):
            ix_, ix = inds_list[x-1], inds_list[x]
            T1 = qtn.Tensor(np.array([[[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]],
                                      [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],
                                     [[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
                                      [[[0.,0.],[0.,0.]],[[0.,0.],[0.,1.]]]]]),
                            inds=(out_ind_id.format(ix), f'b{ix_}', f'b{ix}',
                                  in1_ind_id.format(ix), in2_ind_id.format(ix)),
                            tags=(site_tag_id.format(ix),))
            conv_op_tn.add(T1)

        ix_, iL = inds_list[-2], inds_list[-1]
        TL = qtn.Tensor(np.array([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,0.]]],
                                  [[[0.,0.],[0.,1.]],[[0.,0.],[0.,0.]]]]),
                        inds=(f'b{ix_}', out_ind_id.format(iL), in1_ind_id.format(iL), in2_ind_id.format(iL)),
                        tags=(site_tag_id.format(iL),))
        conv_op_tn.add(TL)

        conv_op_tn = conv_op_tn.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                                        upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
        # conv_op_tn = ax.map.transform_tn1d(conv_op_tn, out_ind_id, in1_ind_id, in2_ind_id, inplace=True)
        conv_op_tn = ax.map.transform_tn1d(conv_op_tn, inplace=True)
        return conv_op_tn


    def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        raise NotImplementedError


    def build_firstderivative_mpo(self, ax, **kwargs):
        ks = ax.xpts
        mps_ks = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_ks,-1.j,inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_ks)
        return mpo


    def build_secondderivative_mpo(self, ax, **kwargs):
        ks = ax.xpts**2
        mps_k2s = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_k2s,-1.,inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_k2s)
        return mpo


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        # domain is from -np.pi/dk to np.pi/dk
        if is_sqrt:
            raise NotImplementedError
            ## this is wrong
            # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
        else:
            integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * 2 * np.pi / ax.dx

        return integ_mps


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError




class RealFourierBasis(FourierBasis):
    """ assumes data is real and that a_k = a_{-k}^* for some wavevector k
        (and a_0 is real)
    """

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ coeffs:  coefficients of wavevector (0, k, 2k, ...)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        data = np.moveaxis(data, ax_ind, -1)
        out_data = np.fft.irfftn(data, axes=[-1]) # 0 frequency term at 0
        out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data


    def build_elemental_multiply_tn(self, ax, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        """ requires taking complex conjugates of MPSs via MPOs, which i don't think is possible
            no reduction in number of operations in real representation
            just memory and enforcing of real-ness
        """
        raise NotImplementedError


    def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        raise NotImplementedError

    def build_firstderivative_mpo(self, ax, **kwargs):
        return super().build_firstderivative_mpo(ax, **kwargs)

    def build_secondderivative_mpo(self, ax, **kwargs):
        return super().build_secondderivative_mpo(ax, **kwargs)

    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        # domain is from -np.pi/dk to np.pi/dk
        if is_sqrt:
            raise NotImplementedError
            ## this is wrong
            # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
        else:
            # ax.zero_ind should be 0
            integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * 2 * np.pi / ax.dx

        return integ_mps

    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError
