from scipy import linalg as spla

import axis
from setup.configs import *
from setup.quimb_TN1D import MatrixProductStateTN

import helper_quimb as helper
import helper_dmrg
import helper_mg

from basis.basis_k import RealFourierBasis
from grid import Grid
from gridTN_1D import GridTN1D

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from layout.layout import Layout

""" Defines the multi-dimensional grid, built from Axis objects
    layout is the geometry used to represent high-dimensional TNs
    contains information like those involving differential operators (laplacian, cross-product)
    and common mathematical operations (like elemental multiply and xmultiply
"""

def get_layout(layout_type) -> 'Layout':
    """ return basis class given BasisType(Enum)
    """
    if layout_type == LayoutType.SEQUENTIAL:
        from layout.layout_sequentialF import LayoutSequentialF
        return LayoutSequentialF
    if layout_type == LayoutType.PARALLEL:
        from layout.layout_parallelF import LayoutParallelF
        return LayoutParallelF
    if layout_type == LayoutType.PARALLEL_GROUP:
        from layout.layout_parallelG import LayoutParallelG
        return LayoutParallelG


class Grid1D(Grid):

    def __init__(self, gridID, axes: Iter['Axis'], layout_type=LayoutType.SEQUENTIAL):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?
            layout_type: LayoutType that specifies Grid class defining geometry of TN of multi-dimensional system.
                Defines mathematical operations applied to TN given said geometry
            order:  2*order+1 - point stencil for finite difference methods
            compress_opts: see quimb. eg. max_bond: max virtual bond dimension of TNs,
                                          cutoff: cutoff tolerance for singular values

        Attributes:
            gridID: ID for this GridLayout object
            axes:   list of Axis objects
            ndim:   number of Axis objects
            layout: Layout of the ndim-object mapped to a 1D grid
            order:  sets finite difference accuracy
        """
        self.layout_type = layout_type
        self.layout = get_layout(layout_type)
        self.L = self.layout.L(axes)

        super().__init__(gridID, axes)



    def get_inds_in_axis(self, ax, ax_ind=None) -> list:
        """ get inds in 1D TN corresponding to axis axID
        """
        return self.layout.get_inds_in_axis(self.axes, ax, ax_ind)


    def shape(self) -> tuple:
        """ get grid shape
        """
        return self.layout.shape(self.axes)


    def create_like(self, new_axes=None, new_gridID=None):
        """ create a new grid object with potentially new Axis objects and new gridID
        """
        if new_axes is None:     new_axes = self.axes
        if new_gridID is None:   new_gridID = self.gridID

        # print('new axes', new_axes, self.axes)
        new_grid = self.__class__(new_gridID, new_axes, self.layout_type)
        return new_grid


    def get_coarsen_inds(self, coarsen_level):
        coarsen_inds = []
        for ax in self.axes:
            c_ind = ax.get_coarseness_ind(coarsen_level)
            if c_ind is not None:
                axis_inds = self.get_inds_in_axis(ax)
                coarsen_inds += [axis_inds[c_ind]]
        return coarsen_inds


    # ######################
    # ## axis information ##
    # ######################
    #
    # def get_inds_in_axes(self,ax):
    #     """ get indices corresponding to tensors along a given axis (for factorized TNs)
    #     """
    #     return None
    #
    #
    # def get_ax_vals(self,axes=None):
    #     """ get axis values for axes specified by dims
    #         eg. get x, v data on the full grid
    #     """
    #     return_single = False
    #     if isinstance(axes,int):
    #         axes = [axes]
    #         return_single = True
    #
    #     if axes is None:   axes = range(self.ndim)
    #
    #     npts = self.q**self.L
    #
    #     out = {}
    #     for ax in axes:
    #         out[ax] = self.zero[ax] + np.arange(npts)*self.dx[ax]
    #
    #     if return_single:   return out[ax]
    #     else:               return out
    #
    #
    # def get_val_ind(self,ax,val):
    #     """ find index (int) of grid poin corresponding to desired value val along axis ax
    #     """
    #     grid_pts = self.get_ax_vals(ax)
    #     ind = np.argwhere(np.abs(grid_pts-val)<1.0-10)
    #     if len(ind) == 0:
    #         raise IndexError('get_val_ind: grid does not contain desired value',val)
    #     return ind[0]

    #
    # def get_ax_mesh(self,axes=None):
    #     """ get self.ndim -dimensional grid with axis values for axes specified by dims
    #         eg. get x, v data on the full grid
    #     """
    #     return_single = False
    #     if isinstance(axes,int):
    #         axes = [axes]
    #         return_single = True
    #
    #     if axes is None:
    #         axes = range(self.ndim)
    #
    #     ax_vals = self.get_ax_vals(axes)
    #
    #     out = {}
    #     for ax in axes:
    #         val_mps = self.map_state_to_mps(ax_vals[ax],ndim=1)
    #         out[ax] = self.make_mps_ndim({ax:val_mps})
    #
    #     if return_single:   return out[ax]
    #     else:               return out


    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs=None) -> 'GridTN1D':
        return GridTN1D(self, ax_deriv_configs=ax_deriv_configs)


    def make_gridTN(self, data: TNType = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) \
            -> 'GridTN1D':
        return GridTN1D(self, data, ax_deriv_configs)


    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1D:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {ax: ax.get_iden_mps(site_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mps_ndim(mps_dict)
        # return gtn_cls(self, data=new_mps)


    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1D:
        """ build identity mpo
        """
        mpo_dict = {ax: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mpo_ndim(mpo_dict)
        # return gtn_cls(self, data=self.make_mps_ndim(mpo_dict))


    # def get_select_elems_mps(self, inds, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1D:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {ax: ax.get_select_elems_mps(ix, site_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes,inds)}
    #     # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))
    #     return self.make_mps_ndim(mps_dict)


    # def get_select_elems_mpo(self, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1D:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {ax: ax.get_select_elems_mpo(ix, upper_ind_id, lower_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes,inds)}
    #     return self.make_mps_ndim(mps_dict)
    #     # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))


    # def get_shift_mpo(self, shifts: dict['Axis',int], ax_boundary_conditions: dict['Axis', 'DerivativeConfiguration'] = None,
    #                   upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'GridTN1D':
    #     """ build MPO to shift elements along each axes by specified amount
    #     """
    #     mpo_dict = {ax: ax.get_shift_mpo(shifts[ax], boundary_conditions=ax_boundary_conditions[ax],
    #                                      upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
    #                 for ax in shifts.keys()}
    #     return self.make_mpo_ndim(mpo_dict)


    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', direction=0,
                         split_opts=None, ax_deriv_configs=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) -> GridTN1D:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        mps = self.layout.map_state_to_mps(self.axes, state, site_ind_id=site_ind_id, site_tag_id=site_tag_id,
                                           direction=direction, split_opts=split_opts, ancilla_right=ancilla_right,
                                           ancilla_right_inds=ancilla_right_inds,
                                           ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)
        return GridTN1D(self, data=mps, ax_deriv_configs=ax_deriv_configs)


    def map_mps_to_state(self, gtn_mps: 'GridTN1D', ax_select=None, ancilla_right = (),
                         ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()) \
            -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPS into ndim-dim np.ndarray
        """
        return self.layout.map_mps_to_state(self.axes, gtn_mps.data, ax_select=ax_select,
                                            ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                            ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)


    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> GridTN1D:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        mpo = self.layout.map_operator_to_mpo(self.axes, operator,
                                              upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                              site_tag_id=site_tag_id, direction=direction, split_opts=split_opts,
                                              ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                              ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)
        return GridTN1D(self, data=mpo)

    def map_mpo_to_operator(self, gtn_mpo: 'GridTN1D', ax_select=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        return self.layout.map_mpo_to_operator(self.axes, gtn_mpo.data, ax_select=ax_select,
                                               ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                               ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)


    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_mps_ndim(self, mps_1d_dict) -> Union[GridTN1D,Numeric]:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        mult_constant = 1.0
        keys = list(mps_1d_dict.keys())
        for ax in keys:
            state = mps_1d_dict[ax]
            if np.isscalar(state):
                mult_constant *= state
                mps_1d_dict.pop(ax)
            elif isinstance(state, np.ndarray):
                mps_state = ax.map_state_to_mps(state)
                mps_1d_dict[ax] = mps_state

        if len(mps_1d_dict) == 0:
            return mult_constant

        mps = self.layout.make_mps_ndim(self.axes, mps_1d_dict)
        helper.scalar_multiply(mps, mult_constant, inplace=True)
        return GridTN1D(self, mps)


    def make_mpx_ndim(self, mps_1d_dict) -> GridTN1D:
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        for ax, state in mps_1d_dict.items():
            if isinstance(state, np.ndarray):
                mps_state = ax.map_state_to_mps(state)
                mps_1d_dict[ax] = mps_state

        mpx = self.layout.make_mpx_ndim(self.axes, mps_1d_dict)
        return GridTN1D(self, mpx)


    def make_mpo_ndim(self, mpo_1d_dict) -> GridTN1D:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        if len(mpo_1d_dict) == 0:
            return None

        for ax, operator in mpo_1d_dict.items():
            # if isinstance(operator, np.ndarray):
            if not isinstance(operator, qtn.TensorNetwork):
                mpo_state = ax.map_operator_to_mpo(operator)
                mpo_1d_dict[ax] = mpo_state

        mpo = self.layout.make_mpo_ndim(self.axes, mpo_1d_dict)
        return GridTN1D(self, mpo)


    def make_tn1d_ndim(self, tn_1d_dict, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                       site_tag_id='D({})') -> GridTN1D:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        tn = self.layout.make_tn1D_ndim(self.axes, tn_1d_dict, upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id,
                                              extra_ind_ids=(in2_ind_id,), site_tag_id=site_tag_id)
        return GridTN1D(self, tn)


    def pad_gtn_to_grid(self, gtn):
        if gtn.data_type==DataType.MPS:
            return self.pad_mps_to_grid(gtn.data, gtn.grid.axes)
        elif gtn.data_type==DataType.MPO:
            return self.pad_mpo_to_grid(gtn.data, gtn.grid.axes)
        elif gtn.data_type is None:
            return None
        else:
            raise NotImplementedError


    def pad_mps_to_grid(self, scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']) -> GridTN1D:
        """ pad MPS on mps_grid onto self
        """
        mps = self.layout.pad_mps_to_grid(self.axes, scalar_field_mps, mps_axes)
        return GridTN1D(self, mps)


    def pad_mpo_to_grid(self, scalar_field_mpo: 'MPOType', mpo_axes: tuple['Axis']) -> GridTN1D:
        """ pad MPO on mpo_grid onto self.grid
        """
        mpo = self.layout.pad_mpo_to_grid(self.axes, scalar_field_mpo, mpo_axes)
        return GridTN1D(self, mpo)


    def apply_elemental_multiply_op(self, mps_gtn: 'GridTN1D', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                                    site_tag_id: str = 'T({})', compress=False, compress_opts=None) -> 'GridTN':
        """
        """
        mps = mps_gtn.data

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



    def add_gtns(self, *mps_objs: Union['GridTN1D', Sequence['GridTN1D']], weights=None, zipup=True,
                 init_guess: 'qtn.MatrixProductState' = None, compress=True, sub_compress=True,
                 compress_type=CompressType.SVD, compress_opts=None, sub_compress_opts=None):
        """ add series of x, A*x using specified compress type
        """
        compress_opts = {} if compress_opts is None else compress_opts
        sub_compress_opts = {} if sub_compress_opts is None else sub_compress_opts
        weights = [1.0] * len(mps_objs) if weights is None else weights
        assert(len(weights)==len(mps_objs)), f'weight coefficients ({len(weights)})' + \
                                             f' needs to match mps objects ({len(mps_objs)})'

        if compress_type is CompressType.SVD:
            return super().add_gtns(*mps_objs, weights=weights, compress=compress, sub_compress=sub_compress,
                                    compress_type=compress_type, compress_opts=compress_opts,
                                    sub_compress_opts=sub_compress_opts)

        else:       ## DMRG or MG
            targets = []
            ref_gtn = None
            # print('weights', weights)
            for weight, item in zip(weights,mps_objs):
                if isinstance(item, GridTN1D):
                    assert (item.data_type is DataType.MPS), f'add gtn data type needs to MPS not {mpx.data_type}'
                    if item.data is not None:
                        targets += [helper.scalar_multiply(item.data, weight, inplace=False)]
                    init_guess = item.data if init_guess is None else init_guess
                    ref_gtn = item.copy() if ref_gtn is None else ref_gtn
                elif isinstance(item, qtn.MatrixProductState):
                    targets += [helper.scalar_multiply(item, weight, inplace=False)]
                    init_guess = item if init_guess is None else init_guess
                elif isinstance(item, (list, tuple)):
                    ## ordered how it's written:  [A, B, C, x]
                    ket = item[-1]
                    if isinstance(ket, GridTN1D):
                        assert (ket.data_type is DataType.MPS), \
                            f'add_gtn: last element in list needs to be MPS, not {ket.data_type}'
                        ref_gtn = ket.copy() if ref_gtn is None else ref_gtn
                        ket = ket.data
                        if ket is not None:
                            ket = helper.scalar_multiply(ket, weight, inplace=False)
                            init_guess = ket if init_guess is None else init_guess
                    elif isinstance(ket, qtn.MatrixProductState):
                        ket = helper.scalar_multiply(ket, weight, inplace=False)
                        init_guess = ket if init_guess is None else init_guess
                    else:
                        raise TypeError('incorrect ket data type', type(ket))

                    ops = []
                    for op in item[-2::-1]:
                        if isinstance(op, GridTN1D):
                            assert (op.data_type is DataType.MPO), \
                                f'add_gtn: last element in list needs to be MPS, not {ket.data_type}'
                            ops += [op.data]
                        elif isinstance(op, qtn.MatrixProductOperator):
                            ops += [op]
                        else:
                            raise TypeError('incorrect ket data type', type(op))

                    if ket is not None:
                        targets += [MatrixProductStateTN(mps=ket, mpos=ops)]

            # print('targets', targets)
            # print('target norms', [helper.norm(t) for t in targets])
            # print('init guess', helper.norm(init_guess), init_guess.max_bond())

            if compress_type is CompressType.DMRG:
                solver = helper_dmrg.LinearSolver(init_guess, targets=targets,
                                                  max_bond=compress_opts.get('max_bond', None))
            elif compress_type is CompressType.MG:
                coarseness_mps_inds = ref_gtn.grid.get_coarseness_levels()
                solver = helper_mg.MultigridLinearSolver(init_guess, targets=targets,
                                                         max_bond=compress_opts.get('max_bond', None),
                                                         coarseness_mps_inds=coarseness_mps_inds)
            else:
                raise TypeError('not valid compress type', compress_type)

            solver.solve(2)
            # print('solver', solver.err, solver.is_conv)
            out_gtn = ref_gtn.create_like(new_data=solver.ket.view_as(qtn.MatrixProductState))
            return out_gtn

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    # def get_xmultiply_mpo(self, x_axes = None, recalc=False, store=True, compress_opts=None) -> GridTN1D:
    #     """ multiply mps by axis value  (x * f(x))
    #         x_axes: axes over which operation is applied
    #     """
    #     if x_axes is None:   x_axes = self.axes
    #     axIDs = tuple([ax.axID for ax in x_axes])
    #
    #     try:
    #         if recalc:   raise KeyError
    #         return self._mpo_xmultiply[frozenset(axIDs)]
    #     except KeyError:
    #         mpos = {ax.axID: ax.build_xmultiply_mpo(split_opts=compress_opts) for ax in x_axes}
    #         mpo_full = self.make_mpo_ndim(mpos)
    #         if store:  self._mpo_xmultiply[frozenset(axIDs)] = mpo_full
    #         return mpo_full


    # def get_elemental_multiply_tn(self, x_axes: Iter['Axis'] or None = None, recalc=False, store=True,
    #                               in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
    #                               site_tag_id='d_ijk({})') -> GridTN1D:
    #     """ multiply mps by axis value  (x * f(x))
    #         x_axes: axes over which operation is applied
    #     """
    #     if x_axes is None:   x_axes = self.axes
    #     axIDs = tuple([ax.axID for ax in x_axes])
    #
    #     try:
    #         if recalc:   raise KeyError
    #         return self._mpo_xmultiply[frozenset(axIDs)]
    #     except KeyError:
    #         mpos = {ax.axID: ax.build_elemental_multiply_mpo(in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
    #                 for ax in x_axes}
    #         mpo_full = self.layout.make_tn1D_ndim(self.axes, mpos, upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id,
    #                                               extra_ind_ids=(in2_ind_id,), site_tag_id=site_tag_id)
    #         if store:  self._mpo_xmultiply[frozenset(axIDs)] = mpo_full
    #         return mpo_full


    # def get_firstderivative_mpo(self, ax: 'Axis', recalc=False, store=False, deriv_params=None) -> GridTN1D:
    #     """ obtain MPO taking second derivative
    #         ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
    #         bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
    #         order: finite difference expansion order
    #         extra_params:  v_ax for upwind
    #     """
    #     bc = deriv_params.get('bc', DEFAULT_BC)
    #     order = deriv_params.get('order', DEFAULT_ORDER)
    #     fd_type = deriv_params.get('fd_type', DEFAULT_FDTYPE)
    #     upwind = deriv_params.get('upwind', False)
    #     v_ax = deriv_params.get('v_ax', None)
    #
    #     try:
    #         if recalc:   raise KeyError
    #         if upwind:
    #             # v_ax = extra_params['v_ax']
    #             return self._mpo_firstderivatives[(ax.axID, bc, order, 'upwind', v_ax.axID)]
    #         else:
    #             return self._mpo_firstderivatives[(ax.axID, bc, order, fd_type)]
    #     except KeyError:
    #         if upwind:
    #             print('upwind')
    #             # v_ax = extra_params['v_ax']
    #             mpo = self._build_mpo_firstderivative_upwind(ax,v_ax,bc,order=order)
    #             helper.scalar_multiply(mpo, 1. / ax.dx, inplace=True)
    #             self._mpo_firstderivatives[(ax.axID, bc, order, 'upwind', v_ax.axID)] = mpo
    #         else:
    #             print('normal')
    #             mpo = ax.build_firstderivative_mpo(bc=bc, order=order, fd_type=fd_type)
    #             helper.scalar_multiply(mpo, 1. / ax.dx, inplace=True)
    #             self._mpo_firstderivatives[(ax.axID, bc, order, fd_type)] = mpo
    #
    #         return mpo
    #
    #
    # def get_secondderivative_mpo(self, ax1, ax2, recalc=False, store=False, deriv_params1=None,
    #                              deriv_params2=None) -> GridTN1D:
    #     """ obtain MPO taking second derivative
    #         ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
    #         bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
    #         order: finite difference expansion order
    #     """
    #     if deriv_params2 is None:   deriv_params2 = deriv_params1
    #
    #     bc1 = deriv_params1.get('bc', DEFAULT_BC)
    #     order1 = deriv_params1.get('order', DEFAULT_ORDER)
    #     fd_type1 = deriv_params1.get('fd_type', DEFAULT_FDTYPE)
    #
    #     bc2 = deriv_params2.get('bc', DEFAULT_BC)
    #     order2 = deriv_params2.get('order', DEFAULT_ORDER)
    #     fd_type2 = deriv_params2.get('fd_type', DEFAULT_FDTYPE)
    #
    #     if order2 is None:    order2 = order1
    #     if fd_type2 is None:  fd_type2 = fd_type1
    #
    #     try:
    #         if recalc:   raise KeyError
    #         return self._mpo_secondderivatives[((ax1.axID, bc1, order1, fd_type1),
    #                                             (ax2.axID, bc2, order2, fd_type2))]
    #     except KeyError:
    #         if ax2 is None or ax1.axID == ax2.axID:
    #             mpo = ax1.build_secondderivative_mpo(bc=bc1, order=order1, fd_type=fd_type1)
    #         else:
    #             mpo1 = ax1.build_firstderivative_mpo(bc=bc1, order=order1, fd_type=fd_type1)
    #             mpo2 = ax2.build_firstderivative_mpo(bc=bc2, order=order2, fd_type=fd_type2)
    #             mpo = self.make_mpo_ndim({ax1.axID: mpo1, ax2.axID: mpo2})
    #         if store:  self._mpo_secondderivatives[((ax1.axID, bc1, order1, fd_type1),
    #                                                 (ax2.axID, bc2, order2, fd_type2))] = mpo
    #         return mpo
    #
    #
    # def _build_mpo_firstderivative_upwind(self, x_ax, v_ax, bc=DEFAULT_BC, order=DEFAULT_ORDER) -> GridTN1D:
    #     """ S+|x> = |x+1> , S-|x> = |x-1>
    #         df/dx = \sum_i (S- - S+)|x_i>
    #         ax:  Axis object along which to take derivative
    #         v_ax:  Axis object whose xpts (monotonically increasing) determine when
    #                forward or backward finite differences is se
    #         scale by 1/dt later
    #         boundary_condition:  boundary condition to use when taking derivative
    #             BCType or Tuple of BCTypes (left, right boundary)
    #         order:  order of finite difference method
    #     """
    #     deriv_f = x_ax.build_firstderivative_mpo(bc=bc, fd_type=FDType.FORWARD,  order=order)
    #     deriv_b = x_ax.build_firstderivative_mpo(bc=bc, fd_tyep=FDType.BACKWARD, order=order)
    #     vels = v_ax.xpts
    #     zero_ind = np.argmax(vels>=0)
    #     if vels[-1] < 0:  zero_ind = v_ax.npts
    #     elif vels[0] > 0: zero_ind = -1
    #
    #     # print('upwind',vdim,vels,zero_ind)
    #
    #     if zero_ind < v_ax.npts:
    #         v_pos = v_ax.get_heaviside_mpo(zero_ind,get='right')  ## inds >= zero_ind
    #         mpo1 = self.make_mpo_ndim({x_ax.axID:deriv_b,v_ax.axID:v_pos})
    #     else:    mpo1 = None
    #
    #     if zero_ind > 0:
    #         v_neg = v_ax.get_heaviside_mpo(zero_ind,get='xleft')  ## exclusive left
    #         mpo2 = self.make_mpo_ndim({x_ax.axID:deriv_f,v_ax.axID:v_neg})
    #     else:    mpo2 = None
    #     # mpo2 = None
    #
    #     helper.add_MPO(mpo1, mpo2, inplace=True, compress=True)
    #     # mpo1.add(mpo2, inplace=True, compress=True)
    #     mpo1.data.upper_ind_id = 'o({})'  ## also resets ind names in actually mpo
    #     mpo1.data.lower_ind_id = 'i({})'
    #     mpo1.data.site_tag_id = 'B({})'
    #
    #     return mpo1


    #######################
    ## partial integrals ##
    #######################

    # def get_integral_mps(self, axes = None, order=1, recalc=False, store=True):
    #     """ get MPS taking integral along axis ax
    #         order*2: degree of polynomial expansion in Newton-Cotes
    #         for definite integrals over the entire space
    #     """
    #     raise NotImplementedError


    # def get_integrals_mpx(self, axes = None, order=1, recalc=False, store=False):
    #     """ get MPS taking integral along axes specified by axes (ie. partial integration
    #         order*2: degree of polynomial expansion in Newton-Cotes
    #         for definite integrals over the entire space
    #         PROBABLY NOT A NECESSARY METHOD?
    #     """
    #     if axes is None:   axes = self.axes
    #     axIDs = [ax.axIDs for ax in axes]
    #
    #     try:
    #         if recalc:  raise KeyError
    #         return self._mpx_integrals[frozenset(axIDs)]
    #
    #     except KeyError:
    #
    #         integs_1d = {}
    #         for ax in axes:
    #             integs_1d[ax.axID] = ax.basis.build_integral_mps(ax)
    #
    #         integ = self.make_mpx_ndim(integs_1d)   ## pad other axes with identity MPOs
    #         self._mpx_integrals[frozenset(axes)] = integ
    #         return integ

