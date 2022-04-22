import numpy as np
import quimb.tensor as qtn
import helper_quimb as helper
from basis import BasisInterface


class FourierBasis(BasisInterface):

    @classmethod
    def get_realspace_1D(cls,coeffs,x_window,npts,x_offset=0.,x_scale=1.):
        """ coeffs:  coefficients of Hermite polynomials (-K, -K+1, ..., 0, ..., K-1)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        x0,xL = x_window
        domain = ((x0-x_offset)/x_scale, (xL-x_offset)/x_scale)
        domain_grid = np.linspace(domain[0],domain[1],npts)
        coeffs_ = np.ifftshift(coeffs)  # moves 0 frequency term (at npts//2) to 0
        return np.fft.ifft(coeffs_)

    # @classmethod
    # def elemental_multiply(cls,tn1,tn2,ax1=None,ax2=None,inplace=False,**compress_opts):
    #     tn1 = tn1 if inplace else tn1.copy()
    #     tn1.convolve(tn2,ax1=ax1,ax2=ax2,**compress_opts)
    #     return tn1

    @classmethod
    def build_elemental_multiply_tn(cls, ax, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        L, q = ax.L, ax.q
        if q != 2:  raise NotImplementedError('convolution for q!=2 not implemented')

        conv_op_tn = qtn.TensorNetwork([])

        T0 = qtn.Tensor(np.array([[[[1., 0.], [0., 0.]], [[0., 1.], [1., 0.]]],
                                  [[[0., 1.], [1., 0.]], [[0., 0.], [0., 1.]]]]),
                        inds=(out_ind_id.format(0), 'b0', in1_ind_id.format(0), in2_ind_id.format(0)),
                        tags=(site_tag_id.format(0),))
        conv_op_tn.add(T0)

        for x in range(1, L - 1):
            T1 = qtn.Tensor(np.array([[[[[0., 1.], [1., 0.]], [[0., 0.], [0., 1.]]],
                                       [[[0., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]],
                                      [[[[0., 0.], [0., 1.]], [[0., 0.], [0., 0.]]],
                                       [[[1., 0.], [0., 0.]], [[0., 1.], [1., 0.]]]]]),
                            inds=(out_ind_id.format(x), f'b{x - 1}', f'b{x}', in1_ind_id.format(x), in2_ind_id.format(x)),
                            tags=(site_tag_id.format(x),))
            conv_op_tn.add(T1)

        TL = qtn.Tensor(np.array([[[[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]],
                                  [[[0., 0.], [0., 1.]], [[1., 0.], [0., 0.]]]]),
                        inds=(out_ind_id.format(L - 1), f'b{L - 2}', in1_ind_id.format(L - 1), in2_ind_id.format(L - 1)),
                        tags=(site_tag_id.format(L - 1),))
        conv_op_tn.add(TL)
        return conv_op_tn


    @classmethod
    def build_xmulitply_mpo(cls,tn_grid,ax):
        """ x * exp(ikx) for all k...
        """
        raise NotImplementedError

    @classmethod
    def build_firstderivative_mpo(cls, ax):
        ks = ax.xpts
        mps_ks = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_ks,-1.j,inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_ks)
        return mpo

    @classmethod
    def build_secondderivative_mpo(cls, ax):
        ks = ax.xpts**2
        mps_k2s = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_k2s,-1.,inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_k2s)
        return mpo

    @classmethod
    def build_integral_mps(cls, ax):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        k_ind = ax.get_val_ind(0.0)
        integ_mps = ax.get_select_elems_mps([k_ind])
        return integ_mps
    