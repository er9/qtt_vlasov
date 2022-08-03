from setup.configs import *

import pickle
import helper_quimb as helper
from axis import Axis
from grid import Grid
from grids_composite import CompositeGrid
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb

""" Group of Grid objects (which are groups of Axis objects) or other Grids objects
"""

class GridsComb(CompositeGrid):

    def __init__(self, gridID, grids: Iter[Grid]):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?

        Attributes:
            grids:  list of Grid or Grids objects
            axes:   list of Axis objects associated with Grids
            ngrids: number of grids
            ndim:   number of Axis objects
        """
        self.layout_type = LayoutType.COMB
        super().__init__(gridID, grids)


    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs=None) -> 'GridTN1DComb':
        return GridTN1DComb(self, ax_deriv_configs=ax_deriv_configs)

    def make_gridTN(self, data=None, ax_deriv_configs=None) -> 'GridTN1DComb':
        gtnc = GridTN1DComb(self, ax_deriv_configs=ax_deriv_configs)
        gtnc.data = data
        return gtnc

    def load_gtn_data(self, fstr, gtn=None, ax_deriv_configs=None) -> 'GridTN1DComb':
        if gtn is None:
            gtn = self.make_empty_gridTN(ax_deriv_configs=ax_deriv_configs)
        exponent, sign, spine, branch_data = pickle.load(open(fstr + '.pkl','rb'))
 
        gtn._exponent = exponent
        gtn._sign = sign
        gtn.spine = spine
        branches = {}
        assert(len(branch_data) == self.ngrids), 'num grids from loaded data incorrect'
        for i in range(self.ngrids):
            gr = self.grids[i]
            branches[gr] = gr.make_gridTN(branch_data[i], ax_deriv_configs=gtn.ax_deriv_configs)
        gtn.branches = branches
        print('loaded data', fstr + '.pkl')
        return gtn
 

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {gr: gr.get_ones_mps(site_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mps_dict)

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1DComb:
        """ build identity mpo
        """
        mpo_dict = {gr: gr.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mpo_dict)

    # def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {gr.gridID: gr.get_select_elems_mps([ix], site_ind_id, site_tag_id)
    #                 for gr, ix in zip(self.grids, inds)}
    #     return GridTN1DComb(self, mps_dict)
    #
    # def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') \
    #         -> GridTN1DComb:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mpo_dict = {gr.gridID: gr.get_select_elems_mpo([ix], upper_ind_id, lower_ind_id, site_tag_id)
    #                 for gr, ix in zip(self.grids, inds)}
    #     return GridTN1DComb(self, mpo_dict)

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', direction=0,
                         split_opts=None, ax_deriv_configs=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) -> 'GridTN1DComb':
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_state(state, self, site_ind_id, site_tag_id, split_opts=split_opts)

    def map_mps_to_state(self, gtn_mps: GridTN1DComb, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        return gtn_mps.get_data(ax_select=ax_select)

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> GridTN1DComb:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_operator(operator, self, upper_ind_id, lower_ind_id, site_tag_id,
                                                split_opts=split_opts)

    def map_mpo_to_operator(self, gtn_mpo, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                            ancilla_left=(), ancilla_left_inds=()) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        return gtn_mpo.get_data(ax_select=ax_select)

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    # def make_ndim(self, mpx_ax_dict):
    #     """ mpx dict is defined with axIDs as keys
    #     """
    #     new_branches = {}
    #     for gr in self.grids:
    #         active_axIDs = [axID for axID in gr.axIDs if axID in mpx_ax_dict]
    #         active_dict = {axID: mpx_ax_dict[axID] for axID in active_axIDs}
    #         if isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductState):
    #             out_gr = gr.make_mps_ndim(active_dict)
    #         elif isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductOperator):
    #             out_gr = gr.make_mpo_ndim(active_dict)
    #         else:
    #             # out_gr = gr.make_tn_ndim(active_dict)
    #             raise TypeError('legs of combs should be 1D TNs')
    #         new_branches[gr.gridID] = out_gr
    #
    #     mps_comb = GridTN1DComb(self, branches=new_branches)
    #     return mps_comb


    def make_mps_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            pad empty grids with ones
        """
        new_branches = {}
        active_grids = self.get_active_grids(mps_ax_dict.keys())
        for grid in self.grids:
            if grid in active_grids:
                active_axes = [ax for ax in grid.axes if ax in mps_ax_dict]
                grid_mps_dict = {ax: mps_ax_dict[ax] for ax in active_axes}
                grid_mps = grid.make_mps_ndim(grid_mps_dict)
                new_branches[grid] = grid_mps
            else:
                new_branches[grid] = grid.get_ones_mps()

        mps_comb = GridTN1DComb(self, branches = new_branches)
        return mps_comb


    def make_mpx_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            does not pad remaining axes
        """
        new_branches = {}
        for grid in self.get_active_grids(mps_ax_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in mps_ax_dict]
            grid_mps_dict = {ax: mps_ax_dict[ax] for ax in active_axes}
            grid_mpx = grid.make_mpx_ndim(grid_mps_dict)
            new_branches[grid] = grid_mpx

        mpo_comb = GridTN1DComb(self, branches = new_branches)
        return mpo_comb


    def make_mpo_ndim(self, mpo_ax_dict) -> 'GridTN1DComb':
        new_branches = {}
        for grid in self.get_active_grids(mpo_ax_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in mpo_ax_dict]
            grid_mpo_dict = {ax: mpo_ax_dict[ax] for ax in active_axes}
            grid_mpo = grid.make_mpo_ndim(grid_mpo_dict)
            new_branches[grid] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb


    def make_tn1d_ndim(self, tn_1d_dict, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                       site_tag_id='D({})') -> 'GridTN1DComb':
        new_branches = {}
        for grid in self.get_active_grids(tn_1d_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in tn_1d_dict]
            grid_tn1_dict = {ax: tn_1d_dict[ax] for ax in active_axes}
            grid_mpo = grid.make_tn1d_ndim(grid_tn1_dict, in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id,
                                           out_ind_id=out_ind_id, recalc=True, store=False)
            new_branches[grid] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb


    def pad_gtn_to_grid(self, gtn):
        if gtn.grid == self:
            return gtn
        elif isinstance(gtn, GridTN1D):
            if gtn.grid in self.grids:
                return GridTN1DComb(self, branches={gtn.grid: gtn})
            else:
                raise NotImplementedError('GridTN1D.grid must be subgrid of GridComb')
        elif isinstance(gtn, GridTN1DComb):
            assert all([subgr in self.grids for subgr in gtn.grid.grids])
            return gtn.pad_self_to_new_grid(self)
            # if gtn.spine is not None:
            #     mps_inds = [self.get_grid_ind(subgr) for subgr in gtn.grid.grids]
            #     spine = helper.pad_mps(gtn.spine, mps_inds, self.ngrids, pad_ind_size=1)
            # else:
            #     spine = None
            # return GridTN1DComb(self, branches=gtn.branches, spine=spine, exponent=gtn.exponent)
        else:
            raise NotImplementedError

    # def pad_mps_to_grid(self, tn_data, tn_grid) -> 'GridTN1DComb':
    #     """ pad data on some grid onto self
    #         cgrid:  Composite grid on which tn_data exists
    #     """
    #     if tn_grid == self:
    #         return GridTN1DComb(self, *tn_data)     # branches and maybe spine
    #     if tn_grid in self.grids:
    #         return GridTN1DComb(self, branches={tn_grid: tn_data})
    #     if tn_grid in self.axes:
    #         return self.make_mps_ndim({tn_grid: tn_data})
    #     raise TypeError

    # def pad_mpo_to_grid(self, tn_data, tn_grid) -> GridTN1DComb:
    #     """ pad MPO on mpo_grid onto self.grid
    #     """
    #     if tn_grid == self:
    #         return GridTN1DComb(self, *tn_data)  # branches and maybe spine
    #     if tn_grid in self.grids:
    #         return GridTN1DComb(self, branches={tn_grid: tn_data})
    #     if tn_grid in self.axes:
    #         return self.make_mpo_ndim({tn_grid: tn_data})
    #     print(tn_grid, self, self.grids, self.axes)
    #     raise TypeError


    def pad_mps_to_grid(self, scalar_field_mps, mps_axes) -> 'GridTN1DComb':
        """ pad data on some grid onto self
            cgrid:  Composite grid on which tn_data exists
        """
        subgr = self.get_active_grids(mps_axes)

        if len(subgr) > 1:
            raise NotImplementedError
        else:
            branch = subgr[0].pad_mps_to_grid(scalar_field_mps, mps_axes)
            return GridTN1DComb(self, branches={subgr[0]: branch})


    def pad_mpo_to_grid(self, scalar_field_mpo, mpo_axes) -> GridTN1DComb:
        """ pad MPO on mpo_grid onto self.grid
        """
        subgr = self.get_active_grids(mpo_axes)

        if len(subgr) > 1:
            raise NotImplementedError
        else:
            branch = subgr[0].pad_mps_to_grid(scalar_field_mpo, mpo_axes)
            return GridTN1DComb(self, branches={subgr[0]: branch})


    def take_outerproduct(self, *gtns):
        """ take outer product of gtns provided
            assert that the active grids of the gtns are not overlapping
        """
        raise NotImplementedError

    #####################
    ## post-processing ##
    #####################

    # def interpolate_data(self):
    #     """ process the data in some way
    #     """
    #     pass
    #
    # def gaussian_smooth_mpo(self, sigma=0.3, proc_axes=None, mode='wrap', compress=True, split_opts=None) \
    #         -> GridTN1DComb:
    #     """ apply gaussian smoothing to data along proc_axes (axes to process)
    #         mode = 'wrap' or 'mirror' or whatever other options
    #     """
    #     gtn_mpo1d = super().gaussian_smooth_mpo(sigma, proc_axes, mode, compress=compress, split_opts=split_opts)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     out_mpo.compress(compress_opts=split_opts)
    #     return out_mpo
    #
    #
    # def absorbing_bc_mpo(self, x_ax: Axis, v_ax: Axis, left_bc: BCType, right_bc: BCType, compress=True,
    #                      compress_opts=None) -> GridTN1DComb:
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #         returns list of MPO dicts to add together for final MPO
    #     """
    #     gtn_mpo1d = super().absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     if compress:
    #         out_mpo.compress(compress_opts=compress_opts)
    #     return out_mpo
    #
    #
    # def reflecting_v_bc_mpo(self, x_ax: Axis, v_ax: Axis, left_bc: BCType, right_bc: BCType, compress=True,
    #                         compress_opts=None) -> GridTN1DComb:
    #     """ reflecting_v bc MPO that takes v -> -v
    #     """
    #     gtn_mpo1d = super().reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     if compress:
    #         out_mpo.compress(compress_opts=compress_opts)
    #     return out_mpo


