from setup.defaults import *

from axis import Axis
from grid import Grid
from grids_composite import CompositeGrid
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
        super().__init__(gridID, grids)


    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self) -> 'GridTN1DComb':
        return GridTN1DComb(self)

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {gr.gridID: gr.get_ones_mps(site_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mps_dict)

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1DComb:
        """ build identity mpo
        """
        mpo_dict = {gr.gridID: gr.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mpo_dict)

    def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mps_dict = {gr.gridID: gr.get_select_elems_mps([ix], site_ind_id, site_tag_id)
                    for gr, ix in zip(self.grids, inds)}
        return GridTN1DComb(self, mps_dict)

    def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') \
            -> GridTN1DComb:
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mpo_dict = {gr.gridID: gr.get_select_elems_mpo([ix], upper_ind_id, lower_ind_id, site_tag_id)
                    for gr, ix in zip(self.grids, inds)}
        return GridTN1DComb(self, mpo_dict)

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', split_opts=None,
                         ax_deriv_configs=None) \
            -> GridTN1DComb:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_state(state, self, site_ind_id, site_tag_id)

    def map_mps_to_state(self, mps: GridTN1DComb) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        raise mps.get_data()

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='o({})', site_tag_id='B({})',
                            split_opts=None) -> MPOType:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_operator(operator, self, upper_ind_id, lower_ind_id, site_tag_id)

    def map_mpo_to_operator(self, mpo) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        raise mpo.get_data()

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_ndim(self, mpx_ax_dict):
        """ mpx dict is defined with axIDs as keys
        """
        new_branches = {}
        for gr in self.grids:
            active_axIDs = [axID for axID in gr.axIDs if axID in mpx_ax_dict]
            active_dict = {axID: mpx_ax_dict[axID] for axID in active_axIDs}
            if isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductState):
                out_gr = gr.make_mps_ndim(active_dict)
            elif isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductOperator):
                out_gr = gr.make_mpo_ndim(active_dict)
            else:
                # out_gr = gr.make_tn_ndim(active_dict)
                raise TypeError('legs of combs should be 1D TNs')
            new_branches[gr.gridID] = out_gr

        mps_comb = GridTN1DComb(self, branches=new_branches)
        return mps_comb


    def make_mps_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            pad empty grids with ones
        """
        axIDs = mps_ax_dict.keys()
        new_branches = {}
        active_grids = self.get_active_grids(axIDs)
        for grid in self.grids:
            if grid in active_grids:
                active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
                grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
                grid_mps = grid.make_mps_ndim(grid_mps_dict)
                new_branches[grid.gridID] = grid_mps
            else:
                new_branches[grid.gridID] = grid.get_ones_mps()

        mps_comb = GridTN1DComb(self, branches = new_branches)
        return mps_comb


    def make_mpx_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            does not pad remaining axes
        """
        axIDs = mps_ax_dict.keys()
        new_branches = {}
        for grid in self.get_active_grids(axIDs):
            active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
            grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
            grid_mpx = grid.make_mpx_ndim(grid_mps_dict)
            new_branches[grid.gridID] = grid_mpx

        mpo_comb = GridTN1DComb(self, branches = new_branches)
        return mpo_comb


    def make_mpo_ndim(self, mpo_ax_dict) -> 'GridTN1DComb':
        axIDs = mpo_ax_dict.keys()
        new_branches = {}
        for grid in self.get_active_grids(axIDs):
            active_axIDs = [axID for axID in grid.axIDs if axID in mpo_ax_dict]
            grid_mpo_dict = {axID: mpo_ax_dict[axID] for axID in active_axIDs}
            grid_mpo = grid.make_mpo_ndim(grid_mpo_dict)
            new_branches[grid.gridID] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb


    def make_tn1d_ndim(self, mpo_ax_dict) -> 'GridTN1DComb':
        axIDs = mpo_ax_dict.keys()
        new_branches = {}
        for grid in self.get_active_grids(axIDs):
            active_axIDs = [axID for axID in grid.axIDs if axID in mpo_ax_dict]
            grid_mpo_dict = {axID: mpo_ax_dict[axID] for axID in active_axIDs}
            grid_mpo = grid.make_tn1d_ndim(grid_mpo_dict)
            new_branches[grid.gridID] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb


    def pad_mps_to_grid(self, tn_data, cgrid) -> GridTN1DComb:
        """ pad data on some grid onto self
            cgrid:  Composite grid
        """
        if cgrid == self:
            return GridTN1DComb(self, *tn_data)     # branches and maybe spine
        if cgrid in self.grids:
            return GridTN1DComb(self, branches={cgrid.gridID: tn_data})
        if cgrid in self.axes:
            return self.make_mps_ndim({cgrid.coordinate: tn_data})
        raise TypeError


    def pad_mpo_to_grid(self, tn_data, cgrid) -> GridTN1DComb:
        """ pad MPO on mpo_grid onto self.grid
        """
        if cgrid == self:
            return GridTN1DComb(self, *tn_data)  # branches and maybe spine
        if cgrid in self.grids:
            return GridTN1DComb(self, branches={cgrid.gridID: tn_data})
        if cgrid in self.axes:
            return self.make_mpo_ndim({cgrid.coordinate: tn_data})
        raise TypeError


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


    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    # def get_xmultiply_mpo(self, x_axes: list[Axis] or None = None, recalc=False, store=True, compress_opts=None)\
    #         -> GridTN1DComb:
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
    #         gtn_mpo = super().get_xmultiply_mpo(x_axes, store=False)
    #         mpo = self.make_mpo_ndim(gtn_mpo.data[0])
    #         if store:  self._mpo_xmultiply[frozenset(axIDs)] = mpo
    #         return mpo


    def get_elemental_multiply_tn(self, x_axes: list[Axis] or None = None, recalc=False, store=True,
                                  in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                  site_tag_id='d_ijk({})') -> GridTN1DComb:
        """ multiply mps by axis value  (x * f(x))
            x_axes: axes over which operation is applied
        """
        if x_axes is None:   x_axes = self.axes
        axIDs = tuple([ax.coordinate for ax in x_axes])

        try:
            if recalc:   raise KeyError
            return self._mpo_xmultiply[frozenset(axIDs)]
        except KeyError:
            gtn_3tn = super().get_elemental_multiply_tn(x_axes, store=False, in1_ind_id=in1_ind_id,
                                                        in2_ind_id=in2_ind_id, out_ind_id=out_ind_id,
                                                        site_tag_id=site_tag_id)
            tns = self.make_tn1d_ndim(gtn_3tn.data[0])
            if store:  self._mpo_xmultiply[frozenset(axIDs)] = tns
            return tns


    # def get_firstderivative_mpo(self, ax: Axis, recalc=False, store=False, deriv_params=None):
    #     """ obtain MPO taking second derivative
    #         ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
    #         bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
    #         order: finite difference expansion order
    #         extra_params:  v_ax for upwind
    #     """
    #     gtn_deriv = super().get_firstderivative_mpo(ax, recalc, store, deriv_params)
    #     if isinstance(gtn_deriv, GridTN1DComb):
    #         return gtn_deriv
    #     else:
    #         out_mpo = GridTN1DComb(self)
    #         for mpo_ax_dict in gtn_deriv.data:
    #             tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #             out_mpo.add(tn_comb, inplace=True)
    #
    #         self._mpo_firstderivatives[key] = out_mpo
    #         return out_mpo
    #
    #
    # def get_secondderivative_mpo(self, ax1: Axis, ax2: Axis, recalc=False, store=False, deriv_params1=None,
    #                              deriv_params2=None) -> GridTN1DComb:
    #     """ obtain MPO taking second derivative
    #         ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
    #         bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
    #         order: finite difference expansion order
    #     """
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
    #     key1 = (ax1.axID, bc1, order1, fd_type1)
    #     key2 = (ax2.axID, bc2, order2, fd_type2)
    #
    #     try:
    #         if recalc:   raise KeyError
    #         return self._mpo_secondderivatives[(key1, key2)]
    #     except KeyError:
    #         gtn_deriv = super().get_secondderivative_mpo(ax1, ax2, recalc=True, store=False,
    #                                                      bc1=bc1, order1=order1, fd_type1=fd_type1,
    #                                                      bc2=bc2, order2=order2, fd_type2=fd_type2)
    #         gtn_mpo = self.make_mpo_ndim(gtn_deriv.data[0])
    #         if store:   self._mpo_secondderivatives[(key1, key2)] = gtn_mpo
    #         return gtn_mpo


    #######################
    ## partial integrals ##
    #######################

    # def get_integrals_mpx(self, axes=None, order=1, recalc=False, store=False):
    #     """ get MPS taking integral along axis ax
    #         order*2: degree of polynomial expansion in Newton-Cotes
    #         for definite integrals over the entire space
    #     """
    #     axIDs = [ax.axID for ax in axes]
    #     key = frozenset(axIDs)
    #
    #     try:
    #         if recalc:  raise KeyError
    #         return self._mpx_integrals[key]
    #
    #     except KeyError:
    #         mps_dict = super().get_integrals_mpx(axes, order=order, recalc=True, store=False)
    #         gtn_mpx = self.make_mpx_ndim(mps_dict)
    #         if store:   self._mpx_integrals[key] = gtn_mpx
    #         return gtn_mpx

