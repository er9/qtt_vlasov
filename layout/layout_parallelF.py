import re
from setup.defaults import *
import helper_quimb as helper
from layout.layout import Layout

""" LayoutParallellF:   parallel: n-dimensions are ordered in parallel (x0,y0,z0),(x1,y1,z1),...
                                  ie. interwoven
                        F:  tensors corresponding to different dimensions remain factorized
"""

class LayoutParallelF(Layout):

    @classmethod
    def L(cls, axes: Sequence['Axis']) -> int:
        L0 = axes[0].L
        ndim = len(axes)
        return L0 * ndim

    @classmethod
    def get_inds_in_axis(cls, axes: Sequence['Axis'], ax: 'Axis', ax_ind=None) -> list:
        """ get indices corresponding to tensors along a given axis (for factorized TNs)
        """
        if ax_ind is None:
            # axIDs = [ax.axID for ax in axes]
            ax_ind = axes.index(ax)

        ndim = len(axes)
        L0 = axes[0].L
        return [ndim * i + ax_ind for i in range(L0)]

    @classmethod
    def shape(cls, axes: Sequence['Axis']) -> tuple:
        """ get shape
        """
        qs = [ax.q for ax in axes]
        L0 = axes[0].L
        return tuple(qs)*L0


    #########################
    ## convert array to TN ##
    #########################

    @classmethod
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, ancilla_right=(),
                         ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(), split_opts=None):
        """ state is initially written as an ndim-dimensional tensor of length q**L
            dim 0 x dim 1 x dim 2 ....
        """
        ndim = len(axes)
        L0 = axes[0].L

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim
        shape_seq = super().shape(axes)
        num_pts = np.prod(shape_seq)

        assert (state.size % num_pts == 0), \
            ('state size should be multiple of' + num_pts + ' not ' + str(state.shape))

        if ancilla_left or ancilla_right:
            raise NotImplementedError

        else:
            new_tens = state.reshape(shape_seq)

            ## group dims for same TN position together
            axT = super()._transpose_sequential_to_parallel_inds(L0, ndim)
            # axT = []
            # for x in range(self.L0):
            #     axT += [x + i * self.L0 for i in range(ndim)]
            new_tens = new_tens.transpose(axT)

            ## decompose into an MPS using quimb
            # if split_opts is None:   split_opts = {}
            # else:                    split_opts = split_opts.copy()
            # split_opts['absorb'] = 'right'
            new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in range(L0*ndim)])
            new_mps = helper.mpx_from_dense(new_tens, L0*ndim, [site_ind_id],
                                            site_tag_id=site_tag_id, return_mpx=True, split_opts=split_opts)

        return new_mps


    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            direction=0, ancilla_right=(), ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(),
                            split_opts=None):
        """ state is initially written as an ndim-dimensional tensor of length q**L
            n-dim operator is (i0 x i1 x i2 ...) x (o0 x o1 x o2 ...)
        """
        ndim = len(axes)
        L0 = axes[0].L

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim for out x in
        shape_seq = super().shape(axes)   ## shape with sequential ordering of axes
        num_pts = np.prod(shape_seq)

        assert (operator.size == num_pts**2), \
            ('state size should be ' + num_pts**2 + 'not ' + str(operator.shape))

        if ancilla_left or ancilla_right:
            raise NotImplementedError

        new_tens = operator.reshape(shape_seq*2)

        ## group dims for same spatial position together
        # axTo = []
        # for x in range(self.L0):
        #     axTo += [x + i * self.L0 for i in range(ndim)]
        axTo = super()._transpose_sequential_to_parallel_inds(L0,ndim)
        axTi = [axTo[-1] + 1 + axTo[x] for x in range(L0*ndim)]
        new_tens = new_tens.transpose(axTo + axTi)

        ## decompose into an MPS using quimb
        # if split_opts is None:   split_opts = {}
        # else:                    split_opts = split_opts.copy()
        # split_opts['absorb'] = 'right'
        new_tens = qtn.Tensor(new_tens, inds=[upper_ind_id.format(i) for i in range(L0*ndim)] + \
                                             [lower_ind_id.format(i) for i in range(L0*ndim)])
        new_mpo = helper.mpx_from_dense(new_tens, L0*ndim, [upper_ind_id, lower_ind_id],
                                        site_tag_id=site_tag_id, return_mpx=True, split_opts=split_opts)
        return new_mpo


    @classmethod
    def map_mps_to_state(cls, axes, mps):
        """ convert MPS into 1-D np.ndarray
        """
        ndim = len(axes)
        L0 = axes[0].L

        # axT = []
        # for x in range(ndim):
        #     axT += [x + i*ndim for i in range(self.L0)]

        ## contract and reorder to sequential ordering
        axT = super()._transpose_parallel_to_sequential_inds(L0, ndim)
        tensor: qtn.Tensor = mps.contract()  # contract all tensors
        tensor.transpose(*[mps.site_ind_id.format(i) for i in axT], inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape([ax.npts for ax in axes])

        return out_data * (10 ** mps.exponent)


    @classmethod
    def map_mpo_to_operator(cls, axes, mpo):
        """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
        """
        ndim = len(axes)
        L0 = axes[0].L

        # axT = []
        # for x in range(ndim):
        #     axT += [x + i * ndim for i in range(self.L0)]

        ## contract and reorder to sequential ordering
        axT = super()._transpose_parallel_to_sequential_inds(L0, ndim)
        tensor: qtn.Tensor = mpo.contract()  # contract all tensors
        out_inds = [mpo.upper_ind_id.format(i) for i in axT] + \
                   [mpo.lower_ind_id.format(i) for i in axT]
        tensor.transpose(*out_inds, inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape([ax.npts for ax in axes] * 2)

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
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]

        if ndim == 1:
            return ref_mps

        new_mps = super().make_mps_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mps.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mps.exponent

        site_tag_id = ref_mps.site_tag_id
        site_ind_id = ref_mps.site_ind_id

        ## contract dims together
        if L0 == 1:
            new_tens = new_mps.contract(ref_mps.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mps = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mps.contract(ref_mps.site_tag(i), inplace=True)
            new_mps.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mps.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}',),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': site_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mps = qtn.TensorNetwork(tens_list)
        new_mps.exponent = new_exponent
        new_mps.view_like(ref_mps, L=L0 * ndim, inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ndim = len(axes)
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        # out_axIDs = [axID for axID in axIDs if axID not in mps_1d_dict]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]

        if len(out_axes) == 0:
            return cls.make_mps_ndim(axes, mps_1d_dict)

        if ndim == 1:
            return ref_mps

        site_tag_id = ref_mps.site_tag_id
        lower_ind_id = ref_mps.site_ind_id
        upper_ind_id = 'o({})'

        new_mpx = super().make_mpx_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mpx.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpx.exponent

        ## contract dims together
        if ref_mps.L == 1:
            new_tens = new_mpx.contract(ref_mps.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpx = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mpx.contract(ref_mps.site_tag(i), inplace=True)
            new_mpx.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mpx.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes:
                if ax not in out_axes:  ## no output leg, make a dummy one of size 1
                    tens.unfuse({f'i({i}),{ax}': (f'i({i}),{ax}', f'o({i}),{ax}')},
                                {f'i({i}),{ax}': (tens.ind_size(f'i({i}),{ax}'), 1)}, inplace=True)
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': lower_ind_id.format(new_ind),
                            f'o({i}),{axes[dim]}': upper_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mpx = qtn.TensorNetwork(tens_list)
        new_mpx.exponent = new_exponent

        new_mpx.view_as(qtn.MatrixProductOperator, inplace=True, site_tag_id=ref_mps.site_tag_id,
                        lower_ind_id=ref_mps.site_ind_id, upper_ind_id='o({})',
                        L=L0*ndim, cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L0 = axes[0].L

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]

        if ndim == 1:
            return ref_mpo

        new_mpo = super().make_mpo_ndim(axes, mpo_1d_dict)
        new_mpo.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpo.exponent

        site_tag_id = ref_mpo.site_tag_id
        upper_ind_id = ref_mpo.upper_ind_id
        lower_ind_id = ref_mpo.lower_ind_id

        ## contract dims together
        if ref_mpo.L == 1:
            new_tens = new_mpo.contract(ref_mpo.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpo = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mpo.contract(ref_mpo.site_tag(i), inplace=True)
            new_mpo.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mpo.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': lower_ind_id.format(new_ind),
                            f'o({i}),{axes[dim]}': upper_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mpo = qtn.TensorNetwork(tens_list)
        new_mpo.exponent = new_exponent
        new_mpo.view_like(ref_mpo, L=L0 * ndim, inplace=True)

        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids,
                                        site_tag_id)
        new_tn.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_tn.exponent

        ## contract dims together
        if L0 == 1:
            new_tens = new_tn.contract(site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_tn = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_tn.contract(site_tag_id.format(i), inplace=True)
            new_tn.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_tn.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:

                if ax in tn1D_1d_dict:    # is general tensor
                    inds = ind_ids
                else:                       # is identity MPO
                    inds = (upper_ind_id, lower_ind_id)

                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + tuple([iid.format(i) + f',{ax}' for iid in inds]),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim

            if axes[dim] in tn1D_1d_dict:  # is general tensor
                inds = ind_ids
            else:  # is identity MPO
                inds = (upper_ind_id, lower_ind_id)

            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({iid.format(i) + f',{axes[dim]}': iid.format(new_ind) for iid in inds}, inplace=True)
            new_ind += 1

        new_tn = qtn.TensorNetwork(tens_list)
        new_tn.exponent = new_exponent
        new_tn.view_as(qtn.TensorNetwork1D, inplace=True, L=new_tn.num_tensors, site_tag_id=site_tag_id)
        return new_tn


    @classmethod
    def pad_mps_to_grid(cls, axes, scalar_field_mps, mps_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        L0 = axes[0].L
        ndim = len(axes)

        pad_axes = [ax for ax in axes if ax not in mps_axes]
        ones = np.ones([ax.q for ax in pad_axes])

        site_tag_id = scalar_field_mps.site_tag_id
        site_ind_id = scalar_field_mps.site_ind_id

        new_mps = scalar_field_mps.copy()
        for dim in range(len(mps_axes)):
            ax = mps_axes[dim]
            old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=dim)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            new_mps.retag({site_tag_id.format(old_i): site_tag_id.format(new_i)
                           for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)
            new_mps.reindex({site_ind_id.format(old_i): site_ind_id.format(new_i)
                             for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)

        ## tag for finding matching site_tag_id for arbitrary integer label
        search_tag = site_tag_id
        search_tag = re.sub('\)', '\)', search_tag)
        search_tag = re.sub('\(', '\(', search_tag)
        search_tag = search_tag.format('[\d]+')

        for ax in pad_axes:   # will be in increasing dim order
            new_inds = cls.get_inds_in_axis(axes, ax)

            for i in new_inds:      # need to add this missing tensor into mps, sorted
                try:
                    tensL = new_mps.select_tensors(site_tag_id.format(i - 1))[0]
                except KeyError:
                    tensL = None
                try:
                    tensR = new_mps.select_tensors(site_tag_id.format(i + 1))[0]
                except KeyError:
                    tensR = None

                if tensL is None and tensR is None:  ## must be i=0
                    new_mps.add(qtn.Tensor(data=ones.reshape(1, ax.q),
                                           inds=(f'_h{i}_', site_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))
                elif tensR is None:  ## no immediate existing neighbor to the right
                    ## find bond that needs to be inserted into
                    tagL = site_tag_id.format(i - 1)
                    neighbors = new_mps.select_neighbors(tagL)
                    tagR, indR = None, None
                    for t in neighbors:
                        for t_tag in t.tags:
                            match = re.search(search_tag, t_tag)    # f
                            try:
                                tagR = match[0]
                                indR = int(re.search('[\d]+', tagR)[0])
                                if indR < i:  raise IndexError
                                break
                            except(IndexError, TypeError):
                                indR = None
                                tagR = None     # was to the right of tens

                        if tagR is not None:
                            break

                    if tagR is not None:
                        tensR = new_mps.select_tensors(tagR)[0]
                        shared_ind = tuple(tensL.bonds(tensR))[0]  # should only be one bond
                        ind_size = tensL.ind_size(shared_ind)

                        ## change virtual bond names
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{indR - 1}_'}, inplace=True)

                        one_id = np.einsum('ij,k->ijk', np.eye(ind_size), ones)
                        new_mps.add(qtn.Tensor(data=one_id,
                                               inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))
                    else:  ## add ind to tagL (was at the end of the mps)
                        tensL.new_ind(f'_h{i - 1}_', axis=1)
                        new_mps.add(qtn.Tensor(data=ones.reshape(1, ax.q),
                                               inds=(f'_h{i - 1}_', site_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))

                else:  ## left and right neighbors exist
                    try:  ## need to insert new tens along bond
                        shared_ind = tuple(tensL.bonds(tensR))[0]
                        ind_size = tensL.ind_size(shared_ind)
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
                    except IndexError:  ## virtual bonds should already be indexed by _h{i}_
                        ind_size = tensL.ind_size(f'_h{i - 1}_')

                    one_id = np.einsum('ij,k->ijk', np.eye(ind_size), ones)
                    new_mps.add(qtn.Tensor(data=one_id,
                                           inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))

        new_mps = new_mps.view_like(scalar_field_mps, L=L0*ndim)
        # new_mps.exponent = scalar_field_mps.exponent
        return new_mps


    @classmethod
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes):
        raise NotImplementedError


