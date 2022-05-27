from setup.defaults import *
from quimb.tensor import MatrixProductState as MPS
from quimb.tensor import MatrixProductOperator as MPO

import helper_quimb as helper
from layout.layout import Layout

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

""" Layout_sequentialF:  sequential: n-dimensions are ordered sequentailly (x0,x1,...), (y0,y1...), ...,
                         F:  tensors corresponding to different dimensions remain factorized
"""

class LayoutSequentialF(Layout):

    @classmethod
    def L(cls, axes: Sequence['Axis']) -> int:
        return super().L(axes)
        # mpx_L = sum([ax.L for ax in axes])
        # return mpx_L

    @classmethod
    def get_inds_in_axis(cls, axes: tuple['Axis'], ax: 'Axis', ax_ind=None) -> list:
        """ get indices corresponding to tensors along a given axis (for factorized TNs)
        """
        if ax_ind is None:
            # axIDs = [ax.axID for ax in axes]
            ax_ind = axes.index(ax)
        Ls = [ax.L for ax in axes]
        return [sum(Ls[:ax_ind]) + i for i in range(axes[ax_ind].L)]

    @classmethod
    def shape(cls, axes: tuple['Axis']) -> tuple:
        """ get shape, axes are sequentially ordered
        """
        return super().shape(axes)


    #########################
    ## convert array to TN ##
    #########################

    @classmethod
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, ancilla_right=(),
                         ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(), split_opts=None) -> MPSType:
        """ state is initially written as an ndim-dimensional tensor of length q**L
            ancilla_right, and ancilla_left: add ancilla legs (ints = size of leg)
                to the right and left sides of MPS. only one of them can be -1
        """
        tens_shape = cls.shape(axes)
        num_pts = np.prod(tens_shape)
        L = len(tens_shape)
        inds_list = cls.get_tensor_inds(axes)

        assert (state.size % num_pts == 0), \
            (f'state size should be a multiple of {num_pts} not ' + str(tens_shape))

        assert (state.ndim - len(ancilla_left) - len(ancilla_right) == len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            state = ax.map.transform_vector(state, axis=len(ancilla_left)+i)

        if len(ancilla_right) > 0 or len(ancilla_left) > 0:   # add ancillas

            new_tens = state.reshape(ancilla_left+tens_shape+ancilla_right)
            inds = ancilla_left_inds + tuple([site_ind_id.format(i) for i in inds_list]) +\
                   ancilla_right_inds
            if len(ancilla_left) == 0:           # only ancilla on right
                site_nlegs = (1,) * L + (len(ancilla_right),)
            elif len(ancilla_right) == 0:        # only ancilla on left
                site_nlegs = (len(ancilla_left),) + (1,) * L
            else:                           # both exist
                site_nlegs = (len(ancilla_left),) + (1,) * L + (len(ancilla_right),)

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=inds)
            new_mps = helper.tn1D_from_dense(new_tens, L, site_nlegs, site_tag_id=site_tag_id,
                                            direction=direction, **split_opts)
            new_mps =  MPS.from_TN(new_mps, inplace=True, cyclic=False, L=L,
                                           site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        else:
            ## fold each axis into n-ary form, each dimension considered serially
            ## eg. (q x q x q ...) ** self.dim
            new_tens = state.reshape(tens_shape)
            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in inds_list])
            new_mps = helper.mpx_from_dense(new_tens, L, [site_ind_id], site_tag_id=site_tag_id,
                                            return_mpx=True, split_opts=split_opts)
        return new_mps


    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            direction=0, ancilla_right=(), ancilla_left=(), ancilla_right_inds=(), ancilla_left_inds=(),
                            split_opts=None ) -> MPOType:
        """ state is initially written as an ndim-dimensional tensor of length q**L
            n-dim operator is (i0 x i1 x i2 ...) x (o0 x o1 x o2 ...)
        """
        tens_shape = cls.shape(axes)
        num_pts = np.prod(tens_shape)
        L = len(tens_shape)
        inds_list = cls.get_tensor_inds(axes)

        assert (operator.size % num_pts**2 == 0), \
            (f'state size should be multiple of {num_pts**2} not ' + str(operator.shape))

        assert (operator.ndim - len(ancilla_left) - len(ancilla_right) == 2*len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            operator = ax.map.transform_operator(operator, axis1=len(ancilla_left)+i, axis2=len(ancilla_left)+len(axes)+i)

        if len(ancilla_right) > 0 or len(ancilla_left) > 0:   # add ancillas
            new_tens = operator.reshape(ancilla_left+tens_shape*2+ancilla_right)

            inds = tuple([upper_ind_id.format(i) for i in inds_list]) + \
                   tuple([lower_ind_id.format(i) for i in inds_list])
            new_inds = tuple([inds[i] for i in range(0,2*L,2)]) + \
                       tuple([inds[i] for i in range(1,2*L,2)])

            if len(ancilla_left) == 0:           # only ancilla on right
                site_nlegs = (2,) * L + (len(ancilla_right),)
            elif len(ancilla_right) == 0:        # only ancilla on left
                site_nlegs = (len(ancilla_left),) + (2,) * L
            else:                           # both exist
                site_nlegs = (len(ancilla_left),) + (2,) * L + (len(ancilla_right),)

            inds = ancilla_left_inds + inds + ancilla_right_inds
            new_inds = ancilla_left_inds + new_inds + ancilla_right_inds

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=inds)
            new_tens.transpose(*new_inds, inplace=True)
            new_mpo = helper.tn1D_from_dense(new_tens, L, site_nlegs, site_tag_id=site_tag_id,
                                            direction=direction, compress_opts=split_opts)
            new_mpo =  MPO.from_TN(new_mpo, inplace=True, cyclic=False, L=L,
                                    site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)

        else:
            ## fold each axis into n-ary form, each dimension considered serially
            ## eg. (q x q x q ...) ** self.dim for out x in
            # new_tens = self.axis_map.array_reshape(operator, tens_shape*2)
            new_tens = operator.reshape(tens_shape*2)

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=[upper_ind_id.format(i) for i in inds_list] + \
                                                 [lower_ind_id.format(i) for i in inds_list])
            new_mpo = helper.mpx_from_dense(new_tens, L, [upper_ind_id, lower_ind_id],
                                            site_tag_id=site_tag_id, return_mpx=True, split_opts=split_opts)
        return new_mpo


    @classmethod
    def map_mps_to_state(cls, axes, mps) -> np.ndarray:
        """ convert MPS into 1-D np.ndarray
            TO DO: include ancilla?
        """
        L = sum([ax.L for ax in axes])
        out_shape = [ax.npts for ax in axes]
        inds_list = cls.get_tensor_inds(axes)

        tensor: qtn.Tensor = mps.contract()  # contract all tensors
        tensor.transpose(*[mps.site_ind_id.format(i) for i in inds_list], inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(out_shape)

        for i in range(len(axes)):
            ax = axes[i]
            out_data = ax.map.inverse_transform_vector(out_data, axis=i)

        return out_data * (10 ** mps.exponent)


    @classmethod
    def map_mpo_to_operator(cls, axes, mpo: 'qtn.MatrixProductOperator') -> np.ndarray:
        """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
            TO DO: include ancilla?
        """
        L = sum([ax.L for ax in axes])
        out_shape = [ax.npts for ax in axes]
        inds_list = cls.get_tensor_inds(axes)

        tensor: qtn.Tensor = mpo.contract()  # contract all tensors
        out_inds = [mpo.upper_ind_id.format(i) for i in inds_list] + \
                   [mpo.lower_ind_id.format(i) for i in inds_list]
        tensor.transpose(*out_inds, inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(out_shape * 2)

        for i in range(len(axes)):
            ax = axes[i]
            out_data = ax.map.inverse_transform_operator(out_data, axis1=i, axis2=len(axes)+i)

        return out_data * (10 ** mpo.exponent)


    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    @classmethod
    def make_mps_ndim(cls, axes, mps_1d_dict) -> MPSType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        ndim = len(axes)
        L = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]

        if ndim == 1:   return ref_mps

        new_mps = super().make_mps_ndim(axes,mps_1d_dict)  ### technically a TN object, all tags same as ref

        site_ind_id = ref_mps.site_ind_id
        site_tag_id = ref_mps.site_tag_id

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mps.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim + 1]
            tR = new_mps.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]
            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mps.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({f'i({i}),{ax}': site_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mps already contains new exponent
        new_mps.view_like(ref_mps, L=L, inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict) -> MPOType:
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ndim = len(axes)
        L = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]

        if ndim == 1:   return ref_mps
        if len(out_axes) == 0:   return cls.make_mps_ndim(axes, mps_1d_dict)

        new_mpx = super().make_mpx_ndim(axes, mps_1d_dict)  ### technically a TN object

        site_tag_id = ref_mps.site_tag_id
        lower_ind_id = ref_mps.site_ind_id
        upper_ind_id = 'o({})'

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mpx.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim+1]
            tR = new_mpx.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]
            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mpx.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]

                if ax not in out_axes:  ## no output leg, make a dummy one of size 1
                    tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind)}, inplace=True)
                    tens.new_ind(upper_ind_id.format(new_ind))
                else:
                    tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                                  f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mpx already contains new exponent
        new_mpx.view_as(MPO, inplace=True, site_tag_id=ref_mps.site_tag_id,
                        lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id,
                        L=L, cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict) -> MPOType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L = cls.L(axes)

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]

        if ndim == 1:
            return ref_mpo

        new_mpo = super().make_mpo_ndim(axes,mpo_1d_dict)

        site_tag_id = ref_mpo.site_tag_id
        lower_ind_id = ref_mpo.lower_ind_id
        upper_ind_id = ref_mpo.upper_ind_id

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mpo.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim + 1]
            tR = new_mpo.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]
            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mpo.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                              f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mpo already contains new exponent
        new_mpo.view_like(ref_mpo, L=L, inplace=True)
        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})') -> TNType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids,
                                        site_tag_id)
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        # ## add bond between last tens of dim and first tens of dim+1
        # for dim in range(ndim - 1):
        #     axL = axes[dim]
        #     tL = new_tn.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL.axID}'))[0]
        #     axR = axes[dim + 1]
        #     tR = new_tn.select_tensors((site_tag_id.format(0), f'dim_{axR.axID}'))[0]
        #     tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)

            if ax in tn1D_1d_dict:  # is general tensor
                inds = ind_ids
            else:  # is identity MPO
                inds = (upper_ind_id, lower_ind_id)

            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_tn.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({ind_.format(i)+f',{ax}': ind_.format(new_ind) for ind_ in inds}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        new_tn.view_as(qtn.TensorNetwork1D, inplace=True, site_tag_id=site_tag_id, L=new_tn.num_tensors)

        # new_tn already contains new exponent
        return new_tn


    @classmethod
    def pad_mps_to_grid(cls, axes: tuple['Axis'], scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']) -> MPSType:
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        L = cls.L(axes)
        pad_axes = [ax for ax in axes if ax not in mps_axes]

        site_ind_id = scalar_field_mps.site_ind_id
        site_tag_id = scalar_field_mps.site_tag_id

        new_mps = scalar_field_mps.view_as(qtn.TensorNetwork)
        for x in range(len(mps_axes)):
            ax = mps_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            for i in range(len(old_ax_inds)):
                old_ind = old_ax_inds[i]
                new_ind = new_ax_inds[i]
                new_mps.reindex({site_ind_id.format(old_ind): f'_tmp{new_ind}_'}, inplace=True)
                new_mps.retag({site_tag_id.format(old_ind): f'_TMP{new_ind}_'}, inplace=True)

        for ax in pad_axes:
            one_mps = ax.get_ones_mps(site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            for i in range(ax.L):
                new_ind = new_ax_inds[i]
                one_mps.reindex({site_ind_id.format(i): f'_tmp{new_ind}_'}, inplace=True)
                one_mps.retag({site_tag_id.format(i): f'_TMP{new_ind}_'}, inplace=True)
            new_mps.add(one_mps)

        L0 = 0
        for subL in [ax.L for ax in axes[:-1]]:
            L0 = L0 + subL
            tens1 = new_mps.select_tensors(f'_TMP{L0 - 1}_')[0]
            tens2 = new_mps.select_tensors(f'_TMP{L0}_')[0]
            qtn.new_bond(tens1, tens2)

        new_mps.view_like(scalar_field_mps, inplace=True, L=L,
                          site_tag_id='_TMP{}_', site_ind_id='_tmp{}_')
        new_mps.site_ind_id = site_ind_id
        new_mps.site_tag_id = site_tag_id
        # new_mps.exponent = scalar_field_mps.exponent  ## not needed
        return new_mps


    @classmethod
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes) -> MPOType:
        """ pad scalar_field_mpo, which exists on axes labeled by axIDs
                    in such that they exist in (target_ndim)-dimensional space
                    fields are constant along all dimensions not specified by self.axes
                    need either new_grid or target_ndim
        """
        L = cls.L(axes)
        pad_axes = [ax for ax in axes if ax not in mpo_axes]

        upper_ind_id = scalar_field_mpo.upper_ind_id
        lower_ind_id = scalar_field_mpo.lower_ind_id
        site_tag_id = scalar_field_mpo.site_tag_id


        new_mpo = scalar_field_mpo.view_as(qtn.TensorNetwork)
        for x in range(len(mpo_axes)):
            ax = mpo_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mpo_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            for i in range(len(old_ax_inds)):
                old_ind = old_ax_inds[i]
                new_ind = new_ax_inds[i]
                new_mpo.reindex({upper_ind_id.format(old_ind): f'_tmpU{new_ind}_',
                                 lower_ind_id.format(old_ind): f'_tmpL{new_ind}_',}, inplace=True)
                new_mpo.retag({site_tag_id.format(old_ind): f'_TMP{new_ind}_'}, inplace=True)

        for ax in pad_axes:
            iden_mpo = ax.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            for i in range(ax.L):
                new_ind = new_ax_inds[i]
                iden_mpo.reindex({upper_ind_id.format(i): f'_tmpU{new_ind}_',
                                  lower_ind_id.format(i): f'_tmpL{new_ind}_',}, inplace=True)
                iden_mpo.retag({site_tag_id.format(i): f'_TMP{new_ind}_'}, inplace=True)
            new_mpo.add(iden_mpo)

        L0 = 0
        for subL in [ax.L for ax in axes[:-1]]:
            L0 = L0 + subL
            tens1 = new_mpo.select_tensors(f'_TMP{L0 - 1}_')[0]
            tens2 = new_mpo.select_tensors(f'_TMP{L0}_')[0]
            qtn.new_bond(tens1, tens2)

        new_mpo.view_like(scalar_field_mpo, inplace=True, L=L,
                          site_tag_id='_TMP{}_', upper_ind_id='_tmpU{}_', lower_ind_id='_tmpL{}_')
        new_mpo.upper_ind_id = upper_ind_id
        new_mpo.lower_ind_id = lower_ind_id
        new_mpo.site_tag_id = site_tag_id
        # new_mps.exponent = scalar_field_mps.exponent  ## not needed
        return new_mpo


    @classmethod
    def apply_partial_mps(cls, mps, mpx, axes):
        raise NotImplementedError