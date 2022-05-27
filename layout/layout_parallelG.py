from setup.defaults import *
import helper_quimb as helper
from layout.layout import Layout
from layout.layout_parallelF import LayoutParallelF

""" LayoutParallellG:   parallel: n-dimensions are ordered in parallel (x0,y0,z0),(x1,y1,z1),...
                                  ie. interwoven
                        G:  tensors corresponding to same TN position are grouped into 1
    Note that all Axis objects must have the same length
"""

class LayoutParallelG(Layout):

    @classmethod
    def L(cls, axes):
        # return axes[0].L
        return max([ax.L for ax in axes])

    @classmethod
    def get_inds_in_axis(cls, axes, ax, ax_ind=None) -> list:
        """ get indices corresponding to tensors along a given axis (for factorized TNs)
            assumes axes are aligned at index 0
        """
        # return list(range(axes[0].L))
        return list(range(ax.L))

    @classmethod
    def shape(cls, axes) -> tuple:
        """ get shape
        """
        Lmax = max([ax.L for ax in axes])
        qs = [1] * Lmax
        for i in range(Lmax):
            for ax_ in axes:
                if ax_.L > i:
                    qs[i] *= ax_.q
        return tuple(qs)

        # q = np.prod([ax.q for ax in axes])
        # return (q,)*axes[0].L


    #########################
    ## convert array to TN ##
    #########################

    @classmethod
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, ancilla_right=(),
                         ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(), split_opts=None):
        """ state is initially written as an ndim-dimensional tensor of length q**L
            dim 0 x dim 1 x dim 2 ....
        """
        L = cls.L(axes)
        Ls = tuple([ax_.L for ax_ in axes])

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim
        shape_seq = super().shape(axes)   ## shape with sequential ordering of axes
        num_pts = np.prod(shape_seq)

        assert (state.size == num_pts), \
            ('state size should be ' + num_pts + ' not ' + str(state.shape))

        assert (state.ndim - len(ancilla_left) - len(ancilla_right) == len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            state = ax.map.transform_vector(state, axis=len(ancilla_left)+i)

        if ancilla_left or ancilla_right:
            raise NotImplementedError

        else:
            new_tens = state.reshape(shape_seq)

            ## group dims for same TN position together
            ax_inds = super()._transpose_sequential_to_parallel_inds(axes)
            tens_inds = super().get_tensor_inds(axes)
            axT = [tens_inds[x] for x in ax_inds]
            # axT = []
            # for x in range(self.L):
            #     axT += [x + i * self.L for i in range(ndim)]
            new_tens = new_tens.transpose(axT)

            ## combine legs for each spatial position
            dims = cls.shape(tuple(axes))
            new_tens = new_tens.reshape(dims)

            ## decompose into an MPS using quimb
            # if split_opts is None:      split_opts = {'absorb': 'right'}
            # else:                       split_opts = {**split_opts, 'absorb': 'right'}
            new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in range(L)])
            new_mps = helper.mpx_from_dense(new_tens, L, [site_ind_id], site_tag_id=site_tag_id,
                                            return_mpx=True, split_opts=split_opts)
        return new_mps


    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            direction=0, ancilla_right=(), ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(),
                            split_opts=None):
        """ state is initially written as an ndim-dimensional tensor of length q**L
            n-dim operator is (i0 x i1 x i2 ...) x (o0 x o1 x o2 ...)
        """
        L = cls.L(axes)
        Ls = tuple([ax_.L for ax_ in axes])

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim for out x in
        shape_seq = super().shape(axes)   ## shape with sequential ordering of axes
        num_pts = np.prod(shape_seq)

        assert (operator.size == num_pts**2), \
            ('state size should be ' + num_pts**2 + 'not ' + str(operator.shape))

        assert (operator.ndim - len(ancilla_left) - len(ancilla_right) == 2 * len(axes)), \
            'operator should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            operator = ax.map.transform_operator(operator, axis1=len(ancilla_left) + i,
                                                 axis2=len(ancilla_left) + len(axes) + i)

        if ancilla_left or ancilla_right:
            raise NotImplementedError

        new_tens = operator.reshape(shape_seq*2)

        ## group dims for same TN position together
        ax_inds = super()._transpose_sequential_to_parallel_inds(axes)
        tens_inds = super().get_tensor_inds(axes)
        axTo = [tens_inds[x] for x in ax_inds]
        axTi = [sum(Ls) + axTo[x] for x in range(sum(Ls))]
        new_tens = new_tens.transpose(np.append(axTo,axTi))

        ## combine legs for each spatial position
        dims = cls.shape(axes)
        new_tens = new_tens.reshape(dims*2)

        ## decompose into an MPS using quimb
        # if split_opts is None:      split_opts = {'absorb': 'right'}
        # else:                       split_opts = {**split_opts, 'absorb': 'right'}
        new_tens = qtn.Tensor(new_tens, inds=[upper_ind_id.format(i) for i in range(L)] + \
                                             [lower_ind_id.format(i) for i in range(L)])
        new_mpo = helper.mpx_from_dense(new_tens, L, [upper_ind_id, lower_ind_id],
                                        site_tag_id=site_tag_id, return_mpx=True, split_opts=split_opts)
        return new_mpo


    @classmethod
    def map_mps_to_state(cls, axes, mps):
        """ convert MPS into 1-D np.ndarray
        """
        L = cls.L(axes)
        Ls = tuple([ax_.L for ax_ in axes])
        factored_qs = LayoutParallelF.shape(tuple(axes))

        tensor: qtn.Tensor = mps.contract()  # contract all tensors
        tensor.transpose(*[mps.site_ind_id.format(i) for i in range(L)], inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(factored_qs)

        ## separate pts for different dimensions
        # axT = []
        # for x in range(ndim):
        #     axT += [x + i*ndim for i in range(self.L)]
        ax_inds = super()._transpose_parallel_to_sequential_inds(axes)
        tens_inds = super().get_tensor_inds(axes)
        axT = [ax_inds[x] for x in tens_inds]
        out_data = out_data.transpose(axT)
        out_data = out_data.reshape([ax.npts for ax in axes])

        # print('parallel to sequential', ax_inds)
        # print('tens inds', tens_inds)
        # print('axis_map mps to state', axT)

        for i in range(len(axes)):
            ax = axes[i]
            out_data = ax.map.inverse_transform_vector(out_data, axis=i)

        return out_data * (10 ** mps.exponent)


    @classmethod
    def map_mpo_to_operator(cls, axes, mpo):
        """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
        """
        L = cls.L(axes)
        Ls = tuple([ax_.L for ax_ in axes])
        factored_qs = LayoutParallelF.shape(tuple(axes))

        tensor: qtn.Tensor = mpo.contract()  # contract all tensors
        out_inds = [mpo.upper_ind_id.format(i) for i in range(L)] + \
                   [mpo.lower_ind_id.format(i) for i in range(L)]
        tensor.transpose(*out_inds, inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(factored_qs * 2)

        ## separate pts for different dimensions
        # axTo = []
        # for x in range(ndim):
        #     axTo += [x + i * ndim for i in range(self.L)]
        ax_inds = super()._transpose_parallel_to_sequential_inds(axes)
        tens_inds = super().get_tensor_inds(axes)
        axTo = [ax_inds[x] for x in tens_inds]
        axTi = [sum(Ls) + axTo[x] for x in range(sum(Ls))]
        out_data = out_data.transpose(np.append(axTo,axTi))
        out_data = out_data.reshape([ax.npts for ax in axes]*2)

        for i in range(len(axes)):
            ax = axes[i]
            out_data = ax.map.inverse_transform_operator(out_data, axis1=i, axis2=len(axes) + i)

        return out_data * (10 ** mpo.exponent)


    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    @classmethod
    def make_mps_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        ndim = len(axes)
        Lmax = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        ref_tag_id = ref_mps.site_tag_id
        ref_ind_id = ref_mps.site_ind_id

        if ndim == 1:   return ref_mps

        new_mps = super().make_mps_ndim(axes, mps_1d_dict)  ### technically a TN object
        new_mps.drop_tags([f'dim_{ax}' for ax in axes])

        ## contract dims together
        if Lmax == 1:
            new_exponent = new_mps.exponent
            new_tens = new_mps.contract(ref_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mps = qtn.TensorNetwork([new_tens])
            new_mps.exponent = new_exponent
        else:
            for i in range(Lmax):
                new_mps.contract(ref_tag_id.format(i), inplace=True)
            new_mps.fuse_multibonds(inplace=True)
            # new_mps already contains exponent

        ## fuse physical bonds
        for i in range(Lmax):
            tens = new_mps.select_tensors(ref_tag_id.format(i))[0]
            active_axes = [ax for ax in axes if ax.L > i]
            tens.fuse({ref_ind_id.format(i): [f'i({i}),{ax}' for ax in active_axes]},
                      inplace=True)
        new_mps.view_like(ref_mps, L=Lmax, inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ndim = len(axes)
        Lmax = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        ref_tag_id = ref_mps.site_tag_id
        ref_ind_id = ref_mps.site_ind_id

        if ndim == 1:   return ref_mps

        # out_axes = [axID for axID in axIDs if axID not in mps_1d_dict]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]
        if len(out_axes) == 0:   return cls.make_mps_ndim(axes,mps_1d_dict)

        new_mpx = super().make_mpx_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mpx.drop_tags([f'dim_{ax}' for ax in axes])

        ## contract dims together
        if Lmax == 1:
            new_exponent = new_mpx.exponent
            new_tens = new_mpx.contract(ref_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpx = qtn.TensorNetwork([new_tens])
            new_mpx.exponent = new_exponent
        else:
            for i in range(Lmax):
                new_mpx.contract(ref_tag_id.format(i), inplace=True)
            new_mpx.fuse_multibonds(inplace=True)
            # new_mpx already contains exponent

        ## fuse physical bonds
        for i in range(Lmax):
            tens = new_mpx.select_tensors(ref_tag_id.format(i))[0]
            active_out_axes = [ax for ax in out_axes if ax.L > i]
            active_axes = [ax for ax in axes if ax.L > i]
            tens.fuse({f'o({i})': tuple([f'o({i}),{ax}' for ax in active_out_axes]),
                       ref_ind_id.format(i): tuple([f'i({i}),{ax}' for ax in active_axes])},
                       inplace=True)

        new_mpx.view_as(qtn.MatrixProductOperator, inplace=True, site_tag_id=ref_tag_id,
                        lower_ind_id=ref_ind_id, upper_ind_id='o({})',
                        L=Lmax, cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        Lmax = cls.L(axes)

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]
        ref_tag_id = ref_mpo.site_tag_id
        ref_upper_id = ref_mpo.upper_ind_id
        ref_lower_id = ref_mpo.lower_ind_id

        if ndim == 1:   return ref_mpo

        new_mpo = super().make_mpo_ndim(axes,mpo_1d_dict)
        new_mpo.drop_tags([f'dim_{ax}' for ax in axes])

        ## contract dims together
        if Lmax == 1:
            new_exponent = new_mpo.exponent
            new_tens = new_mpo.contract(ref_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpo = qtn.TensorNetwork([new_tens])
            new_mpo.exponent = new_exponent
        else:
            for i in range(Lmax):
                new_mpo.contract(ref_tag_id.format(i), inplace=True)
            new_mpo.fuse_multibonds(inplace=True)
            # new_mpo already contains exponent

        ## fuse physical bonds
        for i in range(Lmax):
            tens = new_mpo.select_tensors(ref_tag_id.format(i))[0]
            active_axes = [ax for ax in axes if ax.L > i]
            tens.fuse({ref_upper_id.format(i): tuple([f'o({i}),{ax}' for ax in active_axes]),
                       ref_lower_id.format(i): tuple([f'i({i}),{ax}' for ax in active_axes])},
                       inplace=True)

        new_mpo.view_like(ref_mpo, L=Lmax, inplace=True)
        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        Lmax = cls.L(axes)
        dict_axes = [ax for ax in axes if ax in tn1D_1d_dict]

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids, site_tag_id)
        new_tn.drop_tags([f'dim_{ax}' for ax in axes])

        ## contract dims together
        if Lmax == 1:
            new_tens = new_tn.contract(site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_tn = qtn.TensorNetwork([new_tens])
        else:
            for i in range(Lmax):
                new_tn.contract(site_tag_id.format(i), inplace=True)
            new_tn.fuse_multibonds(inplace=True)

        ## fuse physical bonds
        for i in range(Lmax):
            tens = new_tn.select_tensors(site_tag_id.format(i))[0]
            active_axes = [ax for ax in axes if ax.L > i]
            active_dict_axes = [ax for ax in dict_axes if ax.L > i]
            ind_map = {upper_ind_id.format(i): tuple([upper_ind_id.format(i) + f',{ax}' for ax in active_axes]),
                       lower_ind_id.format(i): tuple([lower_ind_id.format(i) + f',{ax}' for ax in active_axes])}
            ind_map.update({xtra_id.format(i): tuple([xtra_id.format(i)+f',{ax}' for ax in active_dict_axes])
                              for xtra_id in extra_ind_ids})
            tens.fuse(ind_map, inplace=True)

        return new_tn


    @classmethod
    def pad_mps_to_grid(cls, axes, scalar_field_mps, mps_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        Lmax = cls.L(axes)

        pad_axes = [ax for ax in axes if ax not in mps_axes]
        ones = np.ones([ax.q for ax in pad_axes])

        new_mps = []
        for i in range(Lmax):
            tag_i = scalar_field_mps.site_tag_id.format(i)
            ind_i = scalar_field_mps.site_ind_id.format(i)
            active_pad_axes = [ax for ax in pad_axes if ax.L > i]

            if len(active_pad_axes) > 0:
                ones_i = qtn.Tensor(ones, inds=[ind_i + f',{ax}' for ax in active_pad_axes])
            else:
                ones_i = None

            try:
                tens_i = scalar_field_mps.select_tensors(tag_i)[0]
                new_tens = tens_i.unfuse({ind_i: [ind_i + f',{ax}' for ax in mps_axes if ax.L > i]},
                                         {ind_i: [ax.q for ax in mps_axes if ax.L > i]})
                if ones_i is not None:
                    new_tens = qtn.tensor_contract(new_tens, ones_i)
            except KeyError:
                # assert (ones_i is not None), f'neither ones or field tens is defined for site {i}'
                new_tens = ones_i
                new_tens.add_tag(tag_i)
            new_tens.fuse({ind_i: [ind_i + f',{ax}' for ax in axes if ax.L > i]}, inplace=True)
            # for correct ordering
            new_mps += [new_tens]

        new_mps = qtn.TensorNetwork(new_mps)
        new_mps = new_mps.view_like(scalar_field_mps, L=Lmax)
        new_mps.exponent = scalar_field_mps.exponent

        return new_mps


    @classmethod
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        Lmax = cls.L(axes)

        pad_axes = [ax for ax in axes if ax not in mpo_axes]
        pad_qs = [ax.q for ax in pad_axes]
        iden = np.eye( np.prod(pad_qs) ).reshape(pad_qs*2)

        new_mpo = []
        for i in range(Lmax):
            tag_i = scalar_field_mpo.site_tag_id.format(i)
            ind1_i = scalar_field_mpo.upper_ind_id.format(i)
            ind2_i = scalar_field_mpo.lower_ind_id.format(i)
            active_pad_axes = [ax for ax in pad_axes if ax.L > i]
            active_mpo_axes = [ax for ax in mpo_axes if ax.L > i]
            active_axes = [ax for ax in axes if ax.L > i]

            if len(active_pad_axes) > 0:
                iden_i = qtn.Tensor(iden, inds=[ind1_i + f',{ax}' for ax in active_pad_axes] +
                                               [ind2_i + f',{ax}' for ax in active_pad_axes])
            else:
                iden_i = None

            try:
                tens_i = scalar_field_mpo.select_tensors(tag_i)[0]
                new_tens = tens_i.unfuse({ind1_i: [ind1_i + f',{ax}' for ax in active_mpo_axes],
                                          ind2_i: [ind2_i + f',{ax}' for ax in active_mpo_axes],},
                                         {ind1_i: [ax.q for ax in active_mpo_axes],
                                          ind2_i: [ax.q for ax in active_mpo_axes]})
                if iden_i is not None:
                    new_tens = qtn.tensor_contract(new_tens, iden_i)
            except KeyError:
                # assert(iden_i is not None), f'neither iden or field tens is defined for site {i}'
                new_tens = iden_i
                new_tens.add_tag(tag_i)

            new_tens.fuse({ind1_i: [ind1_i + f',{ax}' for ax in active_axes],
                           ind2_i: [ind2_i + f',{ax}' for ax in active_axes],}, inplace=True)
            # for correct ordering
            new_mpo += [new_tens]

        new_mpo = qtn.TensorNetwork(new_mpo)
        new_mpo = new_mpo.view_like(scalar_field_mpo, L=Lmax)
        new_mpo.exponent = scalar_field_mpo.exponent
        return new_mpo

