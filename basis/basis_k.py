from setup.configs import *
import numpy as np
import quimb.tensor as qtn
import helper_quimb as helper
# from axis import Axis
# import axis_map
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
        coeffs_ = np.fft.fftshift(data, axes=ax_ind)  # moves 0 frequency term (at npts//2) to 0
        out_data = np.fft.ifft(coeffs_, axis=ax_ind) * data.shape[ax_ind]
        out_data = np.fft.ifftshift(out_data, axes=ax_ind)

        # data = np.moveaxis(data, ax_ind, -1)
        # coeffs_  = np.fft.fftshift(data, axes=-1)   # moves 0 frequency term (at npts//2) to 0
        # out_data = np.fft.ifftn(coeffs_, axes=[-1]) * data.shape[-1]
        # out_data = np.fft.ifftshift(out_data, axes=-1)
        # out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data

    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], **kwargs):
        """ ie. coeffs of delta fcts """
        data_size = np.prod([data.shape[x] for x in ax_inds])
        coeffs_ = np.fft.fftshift(data, axes=ax_inds)  # moves 0 frequency term (at npts//2) to 0
        out_data = np.fft.ifftn(coeffs_, axes=ax_inds) * data_size
        out_data = np.fft.ifftshift(out_data, axes=ax_inds)
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
        conv_op_tn = ax.map.transform_tn1d(conv_op_tn)
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


    def get_integral_weight(self, ax: 'Axis'):
        """ normalization for integral_mps
        """
        return 2 * np.pi / ax.dx


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        # domain is from -np.pi/dk to np.pi/dk
        if is_sqrt:
            raise NotImplementedError
            ## this is wrong
            # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
        else:
            print('integ ax zero ind', ax.zero_ind)
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
        # data = np.moveaxis(data, ax_ind, -1)
        # npts = data.shape[-1]
        # out_data = np.fft.irfftn(data, axes=[-1], s=[2*npts]) # 0 frequency term at 0
        # out_data = np.fft.ifftshift(out_data, axes=-1) * data.shape[-1] * 2
        # out_data = np.moveaxis(out_data, -1, ax_ind)

        print('real real space', ax_ind)
        npts = data.shape[ax_ind]
        out_data = np.fft.irfft(data, axis=ax_ind, n=2 * npts)  # 0 frequency term at 0
        out_data = np.fft.ifftshift(out_data, axes=ax_ind) * npts * 2
        return out_data


    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], **kwargs):
        """ ie. coeffs of delta fcts """
        data_size = np.prod([data.shape[x] for x in ax_inds])
        out_data = np.fft.irfftn(data, axes=ax_inds) * data_size * (2**len(ax_inds))
        return out_data


    def build_elemental_multiply_tn(self, ax, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        """ requires taking complex conjugates of MPSs via MPOs, which i don't think is possible
            no reduction in number of operations in real representation
            just memory and enforcing of real-ness
        """
        L, q = ax.L, ax.q

        full_ax = ax.__class__(L+1, q=ax.q, ax_map=axis_map.MirrorMap())
        conv_op_tn = super().build_elemental_multiply_tn(full_ax, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
        ## Mirror mapping --> # (-kmax,-kmax+1,...,-1) (kmax-1, kmax-2, ..., 1, 0)

        from axis import get_reverse_diag_mpo

        # apply SX's on all qubits --> (0, 1, ... kmax-1) (-1, -2, ... -kmax)
        flip_ind_mpo = get_reverse_diag_mpo(L+1, q, site_tag_id=site_tag_id, upper_ind_id=out_ind_id,
                                            lower_ind_id=out_ind_id+'_tmp_')         ## MPO of SX's
        ### conv_op_tn = conv_op_tn.apply(flip_ind_mpo)     # applies SX's on output leg (upper_ind_id)
            ### TODO: above should work but doesn't. not sure why yet
        conv_op_tn.upper_ind_id = out_ind_id + '_tmp_'
        conv_op_tn.add(flip_ind_mpo)
        for i in range(L+1):
            conv_op_tn.contract(site_tag_id.format(i), inplace=True)
        conv_op_tn.fuse_multibonds(inplace=True)
        conv_op_tn.view_like(conv_op_tn, upper_ind_id=out_ind_id, extra_ind_ids=conv_op_tn.extra_ind_ids, inplace=True)
            ### TODO: fix above so that not need to specify extra_ind_ids
        # print('conv op tn', conv_op_tn)
        # print('conv_op tn', conv_op_tn.upper_ind_id, conv_op_tn.lower_ind_id, conv_op_tn.extra_ind_ids)


        ## after apply SX's on input legs (will do later)
        ## shift to avoid repeating the k=0 element for anc=1 (when applying to two (0, 1, ..., kmax-1) MPS)
        ## apply SX's on all qubits --> (-2, -3, ... -kmax-1)  (but input is (0, 1, ..., kmax-1)
        bin_ax = ax.__class__(L, q=ax.q, ax_map=axis_map.BinaryMap())
        shift_mpo = bin_ax.get_shift_mpo(1, upper_ind_id=in1_ind_id, lower_ind_id=in1_ind_id+'_tmp_',
                                     site_tag_id=site_tag_id)
        helper.renumber_mpo(shift_mpo, list(range(L)), list(range(1, L + 1)), inplace=True)
        tens_sel_1 = qtn.Tensor(np.array([[0.,0.],[0.,1.]]),
                              inds=(shift_mpo.upper_ind_id.format(0), shift_mpo.lower_ind_id.format(0)),
                              tags=(shift_mpo.site_tag_id.format(0),))
        shift_mpo.add(tens_sel_1)
        shift_mpo.view_like(shift_mpo, L=L+1, inplace=True)
        shift_mpo[0].new_bond(shift_mpo[1])

        ## inputs for anc=0 remain unchanged
        iden = bin_ax.get_iden_mpo(upper_ind_id=in1_ind_id, lower_ind_id=in1_ind_id+'_tmp_', site_tag_id=site_tag_id)
        helper.renumber_mpo(iden, list(range(L)), list(range(1, L + 1)), inplace=True)
        tens_sel_0 = qtn.Tensor(np.array([[1., 0.], [0., 0.]]),
                                inds=(shift_mpo.upper_ind_id.format(0), shift_mpo.lower_ind_id.format(0)),
                                tags=(shift_mpo.site_tag_id.format(0),))
        iden.add(tens_sel_0)
        iden.view_like(iden, L=L+1, inplace=True)
        iden[0].new_bond(iden[1])

        helper.add_MPO(shift_mpo, iden, inplace=True)
        # print('shift mpo', shift_mpo)

        ## apply to TN
        adj_inds_mpo = helper.apply(flip_ind_mpo, shift_mpo)
        # adj_inds_mpo = shift_mpo
        # print('adj inds mpo', adj_inds_mpo)


        adj_inds_mpo2 = adj_inds_mpo.copy(virtual=True)
        adj_inds_mpo2.upper_ind_id = in2_ind_id
        adj_inds_mpo2.lower_ind_id = in2_ind_id + '_tmp_'

        conv_op_tn.add([adj_inds_mpo, adj_inds_mpo2], virtual=True)   # still a MatrixProductTensor
        for i in range(L+1):
            conv_op_tn.contract((site_tag_id.format(i)), inplace=True)
            tens = conv_op_tn.select_tensors(site_tag_id.format(i))[0]
            tens.reindex({in1_ind_id.format(i)+'_tmp_': in1_ind_id.format(i),
                          in2_ind_id.format(i)+'_tmp_': in2_ind_id.format(i)}, inplace=True)

        conv_op_tn.fuse_multibonds(inplace=True)
        conv_op_tn.exponent += adj_inds_mpo.exponent + adj_inds_mpo2.exponent


        ## take first tensor and treat input legs as ancilla
        conv_op_tn_rest, conv_op_tn_0 = conv_op_tn.partition(site_tag_id.format(0))

        conv_op_tn_0 = conv_op_tn_0.tensors[0]
        conv_op_tn_0.isel({out_ind_id.format(0):0}, inplace=True)   # keep only positive ks
        conv_op_tn_0.reindex({in1_ind_id.format(0):f'anc_elem_mult_1_{ax}',
                              in2_ind_id.format(0):f'anc_elem_mult_2_{ax}'}, inplace=True)

        for i in range(1,L+1):
            tens = conv_op_tn_rest.select_tensors(site_tag_id.format(i))[0]
            tens.reindex({in1_ind_id.format(i): in1_ind_id.format(i-1),
                          in2_ind_id.format(i): in2_ind_id.format(i-1),
                          out_ind_id.format(i): out_ind_id.format(i-1)}, inplace=True)
            tens.retag({site_tag_id.format(i): site_tag_id.format(i-1)}, inplace=True)

            if i == 1:
                tens_new = tens.contract(conv_op_tn_0)
                tens.modify(data=tens_new.data, inds=tens_new.inds)

        # print('conv op tn rest exp', conv_op_tn_rest.exponent)
        conv_op_tn_rest.exponent = conv_op_tn.exponent
        conv_op_tn_rest.view_like(conv_op_tn, L=L, inplace=True)
        ## should be in binary ordering from k = (0, 1, ..., kmax)
        conv_op_tn = ax.map.transform_tn1d(conv_op_tn_rest)
        helper.compress(conv_op_tn)

        # print('tot conv op', conv_op_tn)
        # print(type(conv_op_tn))

        # ## check
        # # check = ax.get_ones_mps(site_ind_id=in2_ind_id, site_tag_id=site_tag_id)
        # check = ax.get_select_elems_mps([1], site_ind_id=in2_ind_id, site_tag_id=site_tag_id)
        # check[0].new_ind(f'anc_elem_mult_2_{ax}', size=1)
        # check = helper.add_MPS(check, check)
        # conv_op_tn_check = conv_op_tn.copy()
        # conv_op_tn_check.add(check)
        # for i in range(L):
        #     conv_op_tn_check.contract(site_tag_id.format(i), inplace=True)
        # conv_op_tn_check = conv_op_tn_check.view_as(qtn.MatrixProductOperator, upper_ind_id=out_ind_id,
        #                                             lower_ind_id=in1_ind_id)
        # conv_op_tn_check.isel({f'anc_elem_mult_1_{ax}': 0}, inplace=True)
        # conv_op_data = ax.map_mpo_to_operator(conv_op_tn_check)
        # print(type(conv_op_tn))
        #
        # plt.figure()
        # plt.imshow(conv_op_data.T)
        # plt.colorbar()
        # plt.show()
        # exit()

        return conv_op_tn


    @classmethod
    def convert_mps_to_full(cls, grid, mps_data, anc_ind=1, inplace=False):
        """ mps + mps.conj()
        """
        mps_data = mps_data if inplace else mps_data.copy()

        for ax in grid.axes:
            if not ax.is_real_k():  continue

            ax_inds = grid.get_inds_in_axis(ax)
            if ax.map.is_flipped():
                mps_data[ax_inds[-1]].new_ind(f'anc_elem_mult_{anc_ind}_{ax}')
            else:
                mps_data[ax_inds[0]].new_ind(f'anc_elem_mult_{anc_ind}_{ax}')

        conj_data = mps_data.conj()  ## should be ok to take complex conj. of entire state
        mps_data = helper.add_MPS(mps_data, conj_data, inplace=True, compress=False)
        return mps_data


    def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        raise NotImplementedError

    # def build_firstderivative_mpo(self, ax, **kwargs):
    #     return super().build_firstderivative_mpo(ax, **kwargs)
    #
    # def build_secondderivative_mpo(self, ax, **kwargs):
    #     return super().build_secondderivative_mpo(ax, **kwargs)

    def get_integral_weight(self, ax: 'Axis'):
        """ normalization for integral_mps
            integ F[k] F*[k'] dk dk' = sum k,k' a[k] a*[k'] delta(k,k') * 2*np.pi/dk for k = (-kmax,kmax)
            -->  a[0] + 2 * ( a[1] + a[2] + ... ) * 2*np.pi/dk
        """
        # integ_mps = ax.get_ones_mps()
        # helper.scalar_multiply(integ_mps, 2., inplace=True)
        # select_zero_mps = ax.get_select_elems_mps([0])
        # helper.scalar_multiply(select_zero_mps, -1., inplace=True)
        # integ_mps = helper.add_MPS( integ_mps, select_zero_mps, inplace=True)

        integ_mps = ax.map_state_to_mps(np.array([1.] + [2.]*(ax.npts-1)))
        helper.scalar_multiply(integ_mps, super().get_integral_weight(ax), inplace=True)
        return integ_mps
        # return 2 * super().get_integral_weight(ax)

    # def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
    #     """ get MPS that integrates along Axis ax. multiply by dx here
    #     """
    #     integ_mps = super().build_integral_mps(ax, is_sqrt=is_sqrt, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
    #
    #     # # domain is from -np.pi/dk to np.pi/dk
    #     # if is_sqrt:
    #     #     raise NotImplementedError
    #     #     ## this is wrong
    #     #     # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
    #     # else:
    #     #     # ax.zero_ind should be 0
    #     #     integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * 2 * np.pi / ax.dx
    #
    #     return integ_mps
    #
    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError
