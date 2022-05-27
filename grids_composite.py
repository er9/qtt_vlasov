from setup.defaults import *

from grid import Grid
from grid1D import Grid1D
from gridTN import GridTN

""" Group of Grid objects (which are groups of Axis objects) or other Grids objects
"""

class CompositeGrid(Grid):

    def __init__(self, gridID, grids: Iter[Grid or 'CompositeGrid']):
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

        self.gridID = gridID
        self.grids: tuple[Union[Grid,Grid1D,'CompositeGrid']] = tuple(grids)
        self.gridIDs = tuple([gr.gridID for gr in self.grids])
        self.ngrids = len(grids)

        axes = []
        ax_to_grid_ind = {}
        for x in range(self.ngrids):
            axes += [ax for ax in grids[x].coords_ax]
            ax_to_grid_ind.update({ax.coordinate: x for ax in grids[x].coords_ax})

        super().__init__(gridID, axes)
        # defines self.axes, self.ndim, self.npts, self.axIDs.
        # and private storage of MPOs
        # will overwrite ax_coordsys_map

        self.ax_coordsys_map = {}
        for gr in self.grids:
            self.ax_coordsys_map.update(gr.ax_coordsys_map)

        self.ax_map = ax_to_grid_ind


    def __getitem__(self, item: int):
        return self.grids[item]

    def __len__(self):
        return self.ngrids

    def __eq__(self, other: 'CompositeGrid') -> bool:
        bool0 = (self.gridID == other.gridID)
        bool1 = all([g1 == g2 for g1, g2 in zip(self.grids, other.grids)])
        return bool0 and bool1

    def get_active_grids(self, axIDs) -> list[Grid]:
        act_inds = set([self.ax_map[axID] for axID in axIDs])
        return [self.grids[i] for i in act_inds]


    #################

    def _get_axID_pos(self) -> dict[Any, int]:
        return {self.axes[x].coordinate: x for x in range(self.ndim)}

    axID_pos = property(fget=_get_axID_pos)


    def _get_gridID_pos(self) -> dict[Any, int]:
        return {self.grids[x].gridID: x for x in range(self.ndim)}

    gridID_pos = property(fget=_get_gridID_pos)


    def copy(self, deep: bool = False, new_gridID=None) -> 'CompositeGrid':
        """ make a copy of these Axes. can redefine gridID while not making
            deep copies of the saved MPOs
        """
        if new_gridID is None:   new_gridID = self.gridID
        new_grid = self.__class__(new_gridID, self.grids)

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

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {}
        for grid in self.grids:
            mps_dict[grid.gridID] = grid.get_ones_mps(site_ind_id, site_tag_id).data
        # mps_dict = {ax.axID: ax.get_ones_mps(site_ind_id, site_tag_id) for ax in self.axes}
        return GridTN(self, data=[mps_dict])

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build identity mpo
        """
        mpo_dict = {}
        for grid in self.grids:
            mpo_dict[grid.gridID] = grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id).data
        # mpo_dict = {ax.axID: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for ax in self.axes}
        return GridTN(self, data=[mpo_dict])

    def get_select_elems_mps(self, inds, site_ind_id='i({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mps_dict = {ax.coordinate: ax.get_select_elems_mps(ix, site_ind_id, site_tag_id)
                    for ax, ix in zip(self.axes, inds)}
        return GridTN(self, data=[mps_dict])

    def get_select_elems_mpo(self, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mpo_dict = {ax.coordinate: ax.get_select_elems_mpo(ix, upper_ind_id, lower_ind_id, site_tag_id)
                    for ax, ix in zip(self.axes, inds)}
        return GridTN(self, data=[mpo_dict])
        # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', split_opts=None,
                         ax_deriv_configs=None) -> MPSType:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mps_to_state(self, mps: MPSType) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        raise NotImplementedError

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='o({})', site_tag_id='B({})',
                            split_opts=None) -> MPOType:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mpo_to_operator(self, mpo) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        raise NotImplementedError

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    #### KEEP EVERYTHING STILL IN TERMS OF DICTS WRT ax.axID??

    # def make_ndim(self, mpx_1d_dict):
    #     out = {}
    #     for gr in self.grids:
    #         active_axIDs = [axID for axID in gr.axIDs if axID in mpx_1d_dict]
    #         active_dict = {axID: mpx_1d_dict[axID] for axID in active_axIDs}
    #         if isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductState):
    #             out_gr = gr.make_mps_ndim(active_dict)
    #         elif isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductOperator):
    #             out_gr = gr.make_mpo_ndim(active_dict)
    #         else:
    #             out_gr = gr.make_tn_ndim(active_dict)
    #         out[gr.gridID] = out_gr
    #     return GridTN(self, data=out)

    def make_mps_ndim(self, mps_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        out = {}
        for gr in self.grids:
            active_axIDs = [axID for axID in gr.axIDs if axID in mps_1d_dict]
            active_dict = {axID: mps_1d_dict[axID] for axID in active_axIDs}
            out[gr.gridID] = gr.make_mps_ndim(active_dict)
        return GridTN(self, data=out)

    def make_mpx_ndim(self, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        raise NotImplementedError

    def make_mpo_ndim(self, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        out = {}
        for gr in self.grids:
            active_axIDs = [axID for axID in gr.axIDs if axID in mpo_1d_dict]
            active_dict = {axID: mpo_1d_dict[axID] for axID in active_axIDs}
            out[gr.gridID] = gr.make_mpo_ndim(active_dict)
        return GridTN(self, data=out)

    def pad_mps_to_grid(self, scalar_field_mps, mps_axes):
        """ pad MPS on mps_grid onto self.grid
            return TN/MPS associated with a particular grid level via padding
        """
        mps_axIDs = [ax.coordinate for ax in mps_axes]
        active_grids = self.get_active_grids(mps_axIDs)
        raise NotImplementedError

    def pad_mpo_to_grid(self, scalar_field_mpo, mpo_axes):
        """ pad MPO on mpo_grid onto self.grid
        """
        raise NotImplementedError

