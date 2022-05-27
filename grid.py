from scipy import linalg as spla

from setup.configs import *

import helper_quimb as helper
import helper_inverse as helper_inv
from axis import Axis
from gridTN import GridTN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from gridTN_1D import GridTN1D
    GTN_TYPE = 'GridTN' or GridTN1D
    # from coord.coord_sys import CoordinateSystem

""" Defines the multi-dimensional grid, built from Axis objects
    layout is the geometry used to represent high-dimensional TNs
    contains information like those involving differential operators (laplacian, cross-product)
    and common mathematical operations (like elemental multiply and xmultiply
"""

class Grid:

    def __init__(self, gridID, axes: Iter['Axis']): #, ax_deriv_configs=None):
                 # axis_coordinate_systems: dict[Any, 'CoordinateSystem'] = None):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?
            axis_coordinate_systems:  dict in which axIDs key a CoordinateSystem
                in which the corresponding Axis belongs

        Attributes:
            gridID: ID for this GridLayout object
            axes:   list of Axis objects
            ndim:   number of Axis objects
        """

        self.gridID = gridID
        self.axes: tuple['Axis'] = tuple(axes)
        self.ndim = len(axes)
        self.npts = np.prod([ax.npts for ax in axes])
        self.axIDs = tuple([ax.coordinate for ax in self.axes])

        # if ax_deriv_configs is None:
        #     self.ax_deriv_configs = {ax.axID: DerivativeConfiguration() for ax in self.axes}
        # else:
        #     self.ax_deriv_configs = ax_deriv_configs

        # if axis_coordinate_systems is None:
        #     from coord.cartesian import CartesianCoordinateSpace
        #     cartesian_coords = CartesianCoordinateSpace('default', {i: self.axes[i] for i in range(self.ndim)})
        #     self.ax_coordsys_map = {ax.axID: cartesian_coords for ax in self.axes}
        # else:
        #     self.ax_coordsys_map = axis_coordinate_systems

        ## to calculate
        self._mpo_xmultiply = {}
        self._tn_elem_multiply = {}
        self._mpo_firstderivatives = {}
        self._mpo_secondderivatives = {}
        self._mpx_integrals = {}
        self._mpo_integrals_indef = {}
        self._mpo_inverses = {}
        self._mpo_laplacian = {}


    def __eq__(self, other: 'Grid') -> bool:
        bool0 = (self.gridID == other.gridID)
        bool1 = all([a1 == a2 for a1, a2 in zip(self.axes, other.axes)])
        return bool0 and bool1


    # def _get_axID_pos(self) -> dict[Any,int]:
    #     return {self.axes[x].axID: x for x in range(self.ndim)}
    #
    # axID_pos = property(fget=_get_axID_pos)


    # def get_ax(self, axID):
    #     return self.axes[self.axID_pos[axID]]


    def shape(self) -> tuple:
        """ get grid shape
        """
        return tuple(ax.npts for ax in self.axes)


    # def update_deriv_params(self, axID, left_bc: Optional[BCType or str] = None,
    #                         right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
    #                         fd_type: Optional[FDType or str] = None, v_ax: Optional['Axis'] = None):
    #     """ kwargs:  'order', 'fdtype', 'bc', 'upwind', 'v_ax'
    #     """
    #     dp = self.ax_deriv_configs[axID]
    #     dp.update(left_bc, right_bc, order, fd_type, v_ax)


    def create_like(self, new_axes: Iter['Axis']=None, new_gridID=None) -> 'Grid':
        """ create a new grid object with potentially new Axis objects and new gridID
        """
        if new_axes is None:
            new_axes = self.axes

        if new_gridID is None:
            new_gridID = self.gridID

        new_grid = self.__class__(new_gridID, new_axes) #, self.ax_deriv_configs)
        return new_grid


    def copy(self, deep: bool = False, new_gridID = None) -> 'Grid':
        """ make a copy of these Axes. can redefine gridID while not making
            deep copies of the saved MPOs
        """
        new_grid = self.create_like(self.axes,new_gridID)

        if deep:
            new_grid._mpo_xmultiply = {k: mpo.copy() for k, mpo in self._mpo_xmultiply.items()}
            new_grid._mpo_firstderivatives = {k: mpo.copy() for k, mpo in self._mpo_firstderivatives}
            new_grid._mpo_secondderivatives = {k: mpo.copy() for k, mpo in self._mpo_secondderivatives}
            new_grid._mpx_integrals = {k: mpo.copy() for k, mpo in self._mpx_integrals}
        else:
            new_grid._mpo_xmultiply = self._mpo_xmultiply
            new_grid._mpo_firstderivatives = self._mpo_firstderivatives
            new_grid._mpo_secondderivatives = self._mpo_secondderivatives
            new_grid._mpx_integrals = self._mpx_integrals

        return new_grid


    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self) -> 'GridTN':
        return GridTN(self)

    def make_gridTN(self, data=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None) -> 'GridTN':
        return GridTN(self, data, ax_deriv_configs)

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {ax: ax.get_ones_mps(site_ind_id, site_tag_id) for ax in self.axes}
        return GridTN(self, data=[mps_dict])

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'GridTN':
        """ build identity mpo
        """
        mpo_dict = {ax: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for ax in self.axes}
        return GridTN(self, data=[mpo_dict])

    def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mps_dict = {ax: ax.get_select_elems_mps(ix, site_ind_id, site_tag_id)
                    for ax, ix in zip(self.axes, inds)}
        return GridTN(self, data=[mps_dict])

    def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})')\
            -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mpo_dict = {ax: ax.get_select_elems_mpo(ix, upper_ind_id, lower_ind_id, site_tag_id)
                    for ax, ix in zip(self.axes, inds)}
        return GridTN(self, data=[mpo_dict])
        # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', split_opts=None,
                         ax_deriv_configs: Optional[dict['Axis','DerivativeConfiguration']]=None) -> 'GridTN':
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mps_to_state(self, mps: 'GridTN') -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        raise NotImplementedError

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='o({})', site_tag_id='B({})',
                            split_opts=None) -> 'GridTN':
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mpo_to_operator(self, mpo: 'GridTN') -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        raise NotImplementedError

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_mps_ndim(self, mps_1d_dict: dict['Axis','MPSType']) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        return GridTN(self, data=[mps_1d_dict])


    def make_mpx_ndim(self, mps_1d_dict: dict['Axis','MPSType']) -> 'GTN_TYPE':
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        raise NotImplementedError


    def make_mpo_ndim(self, mpo_1d_dict: dict['Axis','MPOType']) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        return GridTN(self, data=[mpo_1d_dict])


    def make_tn1d_ndim(self, tn_1d_dict: dict['Axis','TN1Type'], **kwargs) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        return GridTN(self, data=[tn_1d_dict])


    def make_gtn_from_dicts(self, data:Iter[dict['Axis',Optional['GridTN']]], data_type=DataType.MPO, compress=False,
                            compress_opts=None):
        """ return GridTN object
        """
        # out = self.make_empty_gridTN()
        out = None
        out_compress = False

        for data_dict in data:
            if data_type is DataType.MPS:
                new_out = self.make_mps_ndim(data_dict)
            elif data_type is DataType.MPO:
                new_out = self.make_mpo_ndim(data_dict)
            elif data_type is DataType.MPX:
                new_out = self.make_mpx_ndim(data_dict)
            elif data_type is DataType.TN3:
                new_out = self.make_tn1d_ndim(data_dict)
            else:
                raise TypeError(f'data type is currently {data_type}')
                # raise TypeError(f'data is currently type {type(data_dict)}')

            if out is None:
                out = new_out         # don't compress bc input would have already been compressed
            else:
                out.add(new_out, inplace=True)
                out_compress = compress

            if out_compress:
                out.compress(inplace=True, compress_opts=compress_opts)

        return out


    def pad_mps_to_grid(self, scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']):
        """ pad MPS on mps_grid onto self.grid
        """
        raise NotImplementedError


    def pad_mpo_to_grid(self, scalar_field_mpo: 'MPOType', mpo_axes: tuple['Axis']):
        """ pad MPO on mpo_grid onto self.grid
        """
        raise NotImplementedError

    #######################################################
    ## taking inverse. can probably be moved elsewhere? ##
    #######################################################

    def get_mpo_inverse(self, mpo: 'MPOType', boundary_conditions=None, use_dmrg=False, compress_opts=None) -> 'GridTN':
        """ mpo:  operator to take the inverse of
            boundary condition:  at the moment, an m x q**K**L array such that
                                 for each m, vector.T * data = mth boundary condition
            expensive method right now -- take inverse directly
            optimization method:
        """
        npts = self.npts
        if compress_opts is None:   compress_opts = {}

        if use_dmrg:
            mpo = mpo.copy()

            print('get inverse dmrg')
            if boundary_conditions is not None:
                if isinstance(boundary_conditions, np.ndarray):
                    num_bc = len(boundary_conditions)
                    tens = np.zeros((npts,) * self.ndim * 2)
                    tens[:num_bc, :] += boundary_conditions
                    bc = self.map_operator_to_mpo(tens)
                    mpo = helper.add_MPO(mpo, bc, inplace=True, compress=True, **compress_opts)
                elif isinstance(boundary_conditions, list) or isinstance(boundary_conditions, tuple):
                    for bc in boundary_conditions:
                        if not isinstance(bc, qtn.MatrixProductOperator):
                            bc = self.map_operator_to_mpo(bc)
                        mpo = helper.add_MPO(mpo, bc, inplace=True, compress=True, **compress_opts)

            inverse_mpo = helper_inv.get_mpo_inverse_dmrg(mpo, verbose=1, cutoff=1.0e-15, max_bond=48)
            # **self.get_compress_opts(-1))

        else:
            if self.npts < 2 ** 12:  # catch for some matrix size
                tens = self.map_mpo_to_operator(mpo.copy()).reshape(npts, npts)
                if boundary_conditions is not None:
                    if isinstance(boundary_conditions, np.ndarray):
                        num_bc = len(boundary_conditions)
                        tens[:num_bc, :] = tens[:num_bc, :] + boundary_conditions
                    elif isinstance(boundary_conditions, list) or isinstance(boundary_conditions, tuple):
                        for bc in boundary_conditions:
                            if isinstance(bc, qtn.MatrixProductOperator):
                                bc = self.map_mpo_to_operator(bc)
                            tens += bc

                inverse = spla.pinv(tens)
                # ans = np.dot(inverse,tens)
                # print('inverse',np.linalg.norm( np.dot(inverse,tens)-np.eye(len(tens)) ),
                #                 np.linalg.norm( np.dot(tens,inverse)-np.eye(len(tens)) ))
                inverse_mpo = self.map_operator_to_mpo(inverse.reshape(self.shape() * 2))
            else:
                raise NotImplementedError
                inverse_mpo = helper_inv.do_newton_schultz_inv(mpo, conv_tol=5.0e-7, num_it=500,
                                                               compress=2, max_bond=150, cutoff=1.0e-30)
        return inverse_mpo

    #####################
    ## post-processing ##
    #####################

    def interpolate_data(self):
        """ process the data in some way
        """
        pass

    def gaussian_smooth_mpo(self, sigma=0.3, proc_axes: Optional[Sequence['Axis']]=None, mode='wrap',
                            split_opts: dict = None) -> 'GTN_TYPE':
        """ apply gaussian smoothing to data along proc_axes (axes to process)
            mode = 'wrap' or 'mirror' or whatever other options
        """
        ## current method uses wrap bcs
        if proc_axes is None:             proc_axes = self.axes
        if isinstance(proc_axes, Axis):   proc_axes = [proc_axes]

        mpo1d_dict = {}

        for ax in proc_axes:

            ## smoothing MPO
            nx = max(1, int(np.round(4 * sigma)))
            nb = int(np.ceil(np.log2(nx * 2 + 1)))  ## number of bits req to encode gaussian smoothing
            xpts = nx * 2 + 2
            # print('processing',nb,mode)

            xs = np.arange(xpts) - xpts // 2
            blur = np.exp(-xs ** 2 / 2 / sigma ** 2)
            blur = blur / np.sum(blur)

            blur_mat = np.zeros((ax.npts, ax.npts))
            if mode == 'wrap':
                for x0 in range(xpts // 2):
                    blur_mat[x0, :xpts // 2 + x0] = blur[xpts // 2 - x0:]
                    blur_mat[x0, -xpts // 2 + x0:] = blur[:xpts // 2 - x0]
            elif mode == 'mirror':
                for x0 in range(xpts // 2):
                    blur_mat[x0, :xpts // 2 + x0] = blur[xpts // 2 - x0:]
                    blur_mat[x0, 2 * x0 + 1:xpts // 2 + x0 + 1] += blur[:xpts // 2 - x0][::-1]

            for x0 in range(ax.npts - xpts + 1):
                blur_mat[xpts // 2 + x0, x0:x0 + xpts] = blur

            if mode == 'wrap':
                # print('blurring mode = wrap?')
                for x0 in range(xpts // 2 - 1):
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:] = blur[:-xpts // 2 + x0 + 1]
                    blur_mat[-xpts // 2 + x0 + 1, :x0 + 1] = blur[-x0 - 1:]
            elif mode == 'mirror':
                for x0 in range(xpts // 2 - 1):
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:] = blur[:-xpts // 2 + x0 + 1]
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:-2 * x0 - 1] += blur[:-xpts // 2 - x0]

            blur_mpo = ax.map_operator_to_mpo(blur_mat, split_opts=split_opts)
            mpo1d_dict[ax] = blur_mpo

        gtn_mpo = self.make_gtn_from_dicts([mpo1d_dict], data_type=DataType.MPO)
        return gtn_mpo


    def absorbing_bc_mpo(self, x_ax: 'Axis', v_ax: 'Axis', left_bc: BCType, right_bc: BCType, compress=True,
                            compress_opts: dict = None) -> 'GTN_TYPE':
        """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
            (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
            returns list of MPO dicts to add together for final MPO
        """
        mpo_list = []

        v_vals = v_ax.xpts
        zero_ind = np.argmax(v_vals >= 0)
        if zero_ind == 0:
            if np.all(v_vals < 0):
                zero_ind = v_ax.npts
            elif np.all(v_vals > 0):
                zero_ind = -1

        if left_bc == BCType.ABSORBING:

            if zero_ind < 0:              ## all v positive (away from left boundary), nothing absorbed
                absorb_bc_L = None
            elif zero_ind == v_ax.npts:   ## all v negative (towards boundary), all absorbed
                mpo_L = x_ax.get_select_elems_mpo([0])
                helper.scalar_multiply(mpo_L, -1, inplace=True)
                absorb_bc_L = {x_ax: mpo_L}
            else:
                v_neg = v_ax.get_heaviside_mpo(zero_ind, get='left')  ## inds <= zero_ind
                mpo_L = x_ax.get_select_elems_mpo([0])
                helper.scalar_multiply(mpo_L, -1, inplace=True)
                absorb_bc_L = {x_ax: mpo_L, v_ax: v_neg}

            mpo_list += [absorb_bc_L]

        if right_bc == BCType.ABSORBING:

            if zero_ind == v_ax.npts:     ## all v negative (away from right boundary), nothing absorbed
                absorb_bc_R = None
            elif zero_ind < 0:            ## all v positive (towards boundary), all absorbed
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                helper.scalar_multiply(mpo_R, -1, inplace=True)
                absorb_bc_R = {x_ax: mpo_R}
            else:
                v_pos = v_ax.get_heaviside_mpo(zero_ind, get='right')  ## inds >= zero_ind
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                helper.scalar_multiply(mpo_R, -1, inplace=True)
                absorb_bc_R = {x_ax: mpo_R, v_ax: v_pos}

            mpo_list += [absorb_bc_R]

        if len(mpo_list) > 0:
            iden_mpo = {x_ax: x_ax.get_iden_mpo(), v_ax: v_ax.get_iden_mpo()}
            mpo_list += [iden_mpo]
            gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)
        else:
            gtn_mpo = None

        return gtn_mpo



    def reflecting_v_bc_mpo(self, x_ax: 'Axis', v_ax: 'Axis', left_bc: BCType, right_bc: BCType, compress=True,
                            compress_opts: dict = None) -> 'GTN_TYPE':
        """ reflecting_v bc MPO that takes v -> -v
        """
        mpo_list = []
        except_inds = []   # 0, npts-1 for left, right boundary

        v_vals = v_ax.xpts  # corresponding velocity axis
        vpos_0 = np.argmin(v_vals >= 0)  # location of the zero in v_ax
        vneg_0 = np.nonzero(np.abs(v_vals - (-1 * v_vals[vpos_0])) < 1.0e-12)[0]

        if left_bc == BCType.REFLECTING:

            v_neg_to_pos = np.zeros((len(v_vals),) * 2)
            for i in range(len(v_vals) - vpos_0 + 1):
                try:
                    v_neg_to_pos[vpos_0 + i, vneg_0 - i] = 1.0
                except IndexError:
                    break

            if not np.all(v_neg_to_pos == 0):
                v_mpo = v_ax.map_operator_to_mpo(v_neg_to_pos, split_opts=compress_opts)
                mpo_0 = x_ax.get_select_elems_mpo([0])
                reflect_bc_L = {x_ax: mpo_0, v_ax: v_mpo}
                except_inds += [0]
                mpo_list += [reflect_bc_L]

        if right_bc == BCType.REFLECTING:

            v_pos_to_neg = np.zeros((len(v_vals),) * 2)
            for i in range(len(v_vals) - vpos_0 + 1):
                try:
                    v_pos_to_neg[vneg_0 - i, vpos_0 + i] = 1.0
                except IndexError:
                    break

            if not np.all(v_pos_to_neg == 0):
                v_mpo = v_ax.map_operator_to_mpo(v_pos_to_neg, split_opts=compress_opts)
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                reflect_bc_R = {x_ax: mpo_R, v_ax: v_mpo}
                except_inds += [x_ax.npts-1]
                mpo_list += [reflect_bc_R]

        if len(except_inds) > 0:
            except_iden = {x_ax: x_ax.get_all_except_elem_mpo(except_inds),
                           v_ax: v_ax.get_iden_mpo()}
            mpo_list += [except_iden]

        gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)
        return gtn_mpo


    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def get_xmultiply_mpo(self, x_axes: Sequence['Axis'] or None = None, x_power=1,
                          offsets: dict['Axis',Numeric]=None, scales: dict['Axis',Numeric]=None,
                          recalc=False, store=True, compress_opts: dict = None) -> 'GTN_TYPE':
        """ multiply mps by axis value  (x * f(x))
            x_axes: axes over which operation is applied
        """
        if x_axes is None:   x_axes = self.axes
        # axIDs = tuple([ax.axID for ax in x_axes])

        if isinstance(offsets,int) or isinstance(offsets,float):
            offsets_dict = {ax: offsets for ax in x_axes}
            offsets_ = offsets
        elif isinstance(offsets,dict):
            offsets_dict = {}
            offsets_ = []
            for ax in x_axes:
                try:
                    offsets_dict[ax] = offsets[ax]
                except KeyError:  offsets_dict[ax] = 0.0
                offsets_ += [offsets_dict[ax]]
            offsets_ = tuple(offsets)
        else:  # if offsets is None:
            offsets_dict = {ax: 0.0 for ax in x_axes}
            offsets_ = offsets

        if isinstance(scales, int) or isinstance(scales, float):
            scales_dict = {ax: scales for ax in x_axes}
            scales_ = scales
        elif isinstance(scales, dict):
            scales_dict = {}
            scales_ = []
            for ax in x_axes:
                try:
                    scales_dict[ax] = scales[ax]
                except KeyError:
                    scales_dict[ax] = 1.0
                scales += [scales_dict[ax]]
            scales_ = tuple(scales_)
        else:  # scales is None:
            scales_dict = {ax: 1.0 for ax in x_axes}
            scales_ = scales

        key = (tuple(x_axes), x_power, offsets_, scales_)

        try:
            if recalc:   raise KeyError
            return self._mpo_xmultiply[key]
        except KeyError:
            mpo = {ax: ax.get_xmultiply_mpo(x_power=x_power, offset=offsets_dict[ax], scale=scales_dict[ax],
                                            split_opts=compress_opts)
                   for ax in x_axes}
            gtn_mpo = self.make_gtn_from_dicts([mpo], data_type=DataType.MPO)
            if store:
                self._mpo_xmultiply[key] = gtn_mpo
            print('xmultiply bond dimension',gtn_mpo.max_bond())
            # print('xmultiply singular values', helper.singular_values_all(gtn_mpo.data))
            return gtn_mpo


    def get_elemental_multiply_tn(self, x_axes: Sequence['Axis'] = None, recalc=False, store=True,
                                  in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                  site_tag_id='d_ijk({})') -> 'GridTN':
        """ multiply mps by axis value  (x * f(x))
            x_axes: axes over which operation is applied
        """
        if x_axes is None:   x_axes = self.axes
        # axIDs = tuple([ax.axID for ax in x_axes])

        try:
            if recalc:   raise KeyError
            return self._tn_elem_multiply[frozenset(x_axes)]
        except KeyError:
            tns = {ax: ax.get_elemental_multiply_tn(in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
                    for ax in x_axes}
            # gtn_obj = GridTN(self,data=[tns])
            gtn_obj = self.make_tn1d_ndim(tns, in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id, out_ind_id=out_ind_id,
                                          site_tag_id=site_tag_id)
            if store:  self._tn_elem_multiply[frozenset(x_axes)] = gtn_obj
            return gtn_obj


    def get_firstderivative_mpo(self, ax: 'Axis', deriv_config: 'DerivativeConfiguration' = None,
                                upwind_ax: Optional['Axis'] = None,
                                recalc=False, store=True) -> 'GTN_TYPE':
        #  bc: tuple[BCType] or BCType = DEFAULT_BC, order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE,
        #  upwind=False, v_ax:Axis=None
        """ obtain MPO taking first derivative
            ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
            order: finite difference expansion order
            extra_params:  v_ax for upwind
        """
        if deriv_config is None:
            deriv_config = DerivativeConfiguration()

        # fd_type = deriv_config.fd_type
        key = (ax,) + deriv_config.get_deriv_key() + (upwind_ax,)
        ## key = (ax.axID, bc, order, fd_type, [v_ax.axID])

        # print('grid 1st deriv', key)

        try:
            if recalc:   raise KeyError
            return self._mpo_firstderivatives[key]
        except KeyError:
            if isinstance(upwind_ax,Axis):  # fd_type == FDType.UPWIND:
                print('upwind', upwind_ax)
                gtn_mpo = self._build_mpo_firstderivative_upwind(ax, upwind_ax, deriv_config)
                print('upwind gtn_mpo', gtn_mpo.max_bond())
            else:
                print('normal')
                coordsys = ax.coord_sys
                mpo_dict = coordsys.build_firstderivative_mpo(ax, deriv_config)
                gtn_mpo = self.make_gtn_from_dicts([mpo_dict], data_type=DataType.MPO)
            if store:   self._mpo_firstderivatives[key] = gtn_mpo
            return gtn_mpo


    def get_secondderivative_mpo(self, ax1: 'Axis', ax2: 'Axis' or None,
                                 deriv_config1: 'DerivativeConfiguration' = None,
                                 deriv_config2: 'DerivativeConfiguration' = None,
                                 recalc=False, store=True) -> 'GTN_TYPE':
        """ obtain MPO taking second derivative
            ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
            order: finite difference expansion order
        """
        if deriv_config1 is None:   deriv_config1 = DerivativeConfiguration()
        if deriv_config2 is None:   deriv_config2 = deriv_config1
        if ax2 is None:             ax2 = ax1

        key1 = (ax1,) + deriv_config1.get_deriv_key()
        key2 = (ax2,) + deriv_config2.get_deriv_key()
        # (ax1.axID, bc1, order1, fd_type1)
        # (ax2.axID, bc2, order2, fd_type2)

        try:
            if recalc:   raise KeyError
            return self._mpo_secondderivatives[(key1, key2)]
        except KeyError:
            coordsys1 = ax1.coord_sys
            coordsys2 = ax2.coord_sys

            if coordsys1 is coordsys2 or coordsys2 is None:
                mpo_dict = coordsys1.build_secondderivative_mpo(ax1, None, deriv_config1, None)
            else:
                mpo_dict1 = coordsys1.build_firstderivative_mpo(ax1, deriv_config1)
                mpo_dict2 = coordsys2.build_firstderivative_mpo(ax2, deriv_config2)
                mpo_dict = {**mpo_dict1, **mpo_dict2}
                # mpo_dict = {ax1.axID: mpo1[ax1.axID], ax2.axID: mpo2[ax2.axID]}
                # mpo = self.make_mpo_ndim({ax1.axID: mpo1, ax2.axID: mpo2})

            gtn_mpo = self.make_gtn_from_dicts([mpo_dict], data_type=DataType.MPO)
            if store:   self._mpo_secondderivatives[(key1, key2)] = gtn_mpo
            return gtn_mpo


    def _build_mpo_firstderivative_upwind(self, x_ax: 'Axis', v_ax: 'Axis', deriv_params: 'DerivativeConfiguration')\
            -> 'GTN_TYPE':
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative
                BCType or Tuple of BCTypes (left, right boundary)
            order:  order of finite difference method
        """
        mpo_list = []

        # v_ax = deriv_params.v_ax
        order = deriv_params.order
        left_bc, right_bc = deriv_params.bc

        coordsys = x_ax.coord_sys
        deriv_config_F = DerivativeConfiguration(left_bc, right_bc, order, FDType.FORWARD)
        deriv_config_B = DerivativeConfiguration(left_bc, right_bc, order, FDType.BACKWARD)
        print('upwind F', deriv_config_F)
        print('upwind B', deriv_config_B)
        mpo_dict_f = coordsys.build_firstderivative_mpo(x_ax, deriv_config_F)
        mpo_dict_b = coordsys.build_firstderivative_mpo(x_ax, deriv_config_B)

        vels = v_ax.xpts
        zero_ind = np.argmax(vels >= 0)
        if vels[-1] < 0:
            zero_ind = v_ax.npts
        elif vels[0] > 0:
            zero_ind = -1

        # print('upwind',v_ax.axID,vels,zero_ind)

        if zero_ind < v_ax.npts:
            v_pos = v_ax.get_heaviside_mpo(zero_ind, get='right')  ## inds >= zero_ind
            mpo1 = {**mpo_dict_b, v_ax: v_pos}
            mpo_list += [mpo1]   ## this one is ok

        if zero_ind > 0:
            v_neg = v_ax.get_heaviside_mpo(zero_ind, get='xleft')  ## exclusive left
            mpo2 = {**mpo_dict_f, v_ax: v_neg}
            mpo_list += [mpo2]   ## this one is problematic

        print('mpo f', mpo_dict_f[x_ax].max_bond())
        print('mpo b', mpo_dict_b[x_ax].max_bond())
        gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO, compress=True)

        verbose_plot = False
        if verbose_plot:
            gtn_data = gtn_mpo.get_data()

            plt.figure()
            plt.imshow(gtn_data[0,:,0,:])
            plt.colorbar()

            plt.figure()
            plt.imshow(gtn_data[:,-1,:,-1])
            plt.colorbar()

            mpo_data = x_ax.map_mpo_to_operator(mpo_dict_f[x_ax])
            plt.figure()
            plt.imshow(mpo_data)
            plt.title('forward')
            plt.colorbar()

            mpo_data = x_ax.map_mpo_to_operator(mpo_dict_b[x_ax])
            plt.figure()
            plt.imshow(mpo_data)
            plt.title('back')
            plt.colorbar()

            plt.show()

        return gtn_mpo


    def laplacian_mpo(self, axes: Sequence['Axis']=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None,
                      recalc=False, store=True) -> 'GTN_TYPE':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            mps_scalar_field:  mps representing the scalar field
        """
        if axes is None:
            axes = self.axes

        if ax_deriv_configs is None:
            ax_deriv_configs = {ax: DerivativeConfiguration() for ax in axes}

        # axIDs = [ax.axID for ax in axes]
        try:
            if recalc:  raise KeyError
            return self._mpo_laplacian[frozenset(axes)]

        except KeyError:
            mpo_list = []
            for ax in axes:
                coord_sys = ax.coord_sys
                # deriv_E_config = self.ax_deriv_configs[ax.axID]
                mpo_list += coord_sys.build_laplacian_mpo([ax], ax_deriv_configs)

            gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)
            if store:   self._mpo_laplacian[frozenset(axes)] = gtn_mpo
            return gtn_mpo


    #######################
    ## partial integrals ##
    #######################

    def get_integrals_mpx(self, axes: list['Axis'] or None = None, order=1, is_sqrt=False, recalc=False, store=True) \
            -> 'GTN_TYPE':
        """ get MPS taking integral along axis ax
            order*2: degree of polynomial expansion in Newton-Cotes
            for definite integrals over the entire space
        """
        if axes is None:   axes = self.axes
        # axIDs = [ax.axID for ax in axes]
        try:
            if recalc:  raise KeyError
            return self._mpx_integrals[(frozenset(axes),is_sqrt)]

        except KeyError:
            out_mps_dict = {}
            for ax in axes:
                coordsys = ax.coord_sys
                integ = coordsys.build_integral_mps(ax, is_sqrt=is_sqrt)
                # integ = ax.basis.build_integral_mps()
                ## if sqrt:  integ = helper.mps_to_diag_mpo(self[i].conj())
                out_mps_dict[ax] = integ
            gtn_mps = self.make_gtn_from_dicts([out_mps_dict], data_type=DataType.MPX)
            if store:   self._mpx_integrals[(frozenset(axes),is_sqrt)] = gtn_mps
            return gtn_mps


    # def get_integral_mps(self, axes: list['Axis'] or None = None, order=1, recalc=False, store=False):
    #     """ get MPS taking integral along axes specified by axes (ie. partial integration
    #         order*2: degree of polynomial expansion in Newton-Cotes
    #         for definite integrals over the entire space
    #         PROBABLY NOT A NECESSARY METHOD?
    #     """
    #     raise NotImplementedError
