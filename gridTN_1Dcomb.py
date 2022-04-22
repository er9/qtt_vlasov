from setup.defaults import *
import helper_quimb as helper
from grid_comb import GridsComb
from grid1D import Grid1D
from gridTN import GridTN
from gridTN_1D import GridTN1D

""" class to work with TN's that look like multiple MPS connected at one end by another MPS
    ie. o---o---o---o---o
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
"""

CombDataType = tuple[dict[Any, GridTN1D], MPSType] or dict[Any, GridTN1D]

class GridTN1DComb(GridTN):

    # def __init__(self, grid: CompositeGrid, data: Optional[CombDataType] = None, virtual = True):
    def __init__(self, grid: GridsComb, branches: dict[Any,GridTN1D] = None, spine: MPSType = None, virtual=True):
        """ branches: List of nbranch GridTN1Ds (each of some length) that are the vertical branches
                can have Nones as placeholders, in which that dim is not represented by an MPS
                each GridTN1D has an associated GridLayout object
            spine: MPS of length nbranch that connects the branches (the 'o's)
                Note: not associated with a grid
                if is None, assumes outer product between the branches
                otherwise, assumes all tensors are in correct final form
                          (ie. branches have bonds with spine)
        """
        super().__init__(grid)
        # self.grid = grid
        ## Attributes: grid, _data, # ngrids, axes, axIDs, ax_map, ndim

        ## self._data replacement

        if virtual:
            branches = {gk: b.copy() for gk, b in branches.items()}
            spine = spine.copy if spine is not None else None

        self.data = (branches,spine)
        ## defines self.spine, self.branches

        if self.spine is not None:
            assert(self.spine.L == self.grid.ngrids), 'spine length and num branches need to be equal'

        self.exponent = 0.0
        self.gtn_types = [(type(b._components) if b is not None else None) for b in self.branches]
        # here will be MPS or MPO, more generally could be tuple (ie. if nested combs)


    def __getitem__(self, i: int):
        """ return tn branch corresponding to axis i
        """
        return self.branches[i]   ## allow KeyError to be raised

    def __setitem__(self, i: int, new_branch: GridTN1D):
        """ set ith component of the field. for convenience
        """
        # helper.mpx_add_ind_label(new_branch,f'd{i}',check_exist=True)
        self.branches[i] = new_branch


    def _get_data(self):
        return self.branches, self.spine

    def _set_data(self, data: CombDataType):
        if isinstance(data, tuple):
            self.branches: dict[Any,GridTN1D] = data[0]
            if len(data) > 0:   self.spine: MPSType = data[1]
            else:               self.spine: MPSType = None

        elif isinstance(data, dict):        # Dictionary of {gridID: GridTN1Ds}
            self.branches: dict[Any,GridTN1D] = data
            self.spine: MPSType = None

        else:
            raise NotImplementedError

    data = property(fget=_get_data,fset=_set_data)

    #
    # def _get_gridIDs(self):
    #     """ get all axes spanned by tncomb
    #     """
    #     return [grid.gridID for grid in self.grids]
    #
    # gridIDs = property(fget=_get_gridIDs)


    def create_like(self) -> 'GridTN1DComb':
        """ create a copy but without any data
        """
        new_tnc = self.__class__(self.grid)
        return new_tnc


    def copy(self, deep=False) -> 'GridTN1DComb':
        """ return a copy (copy MPO data)
            if not deep, don't copy the data. default is deep=True
        """
        branches = {gk: b.copy(deep) for gk, b in self.branches.items()}
        spine = self.spine.copy() if (deep and self.spine is not None) else self.spine
        new_tnc = self.__class__(self.grid, branches=branches, spine=spine)
        return new_tnc


    # def collect_exponents(self):
    #     for branch in self.branches:
    #         self.exponent += branch.data.exponent
    #         branch.data.exponent = 0.0
    #     if self.spine is not None:
    #         self.exponent += spine.exponent
    #         spine.exponent = 0.0


    ############################
    # fcts to generate tncombs #
    ############################

    @classmethod
    def get_ones_mps(cls, cgrid: GridsComb, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN1DComb':
        """ build ones vector mps on sepcified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_ones_mps(site_ind_id=site_ind_id,site_tag_id=site_tag_id)
            branches[gl.gridID] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_iden_mpo(cls, cgrid: GridsComb, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})')\
            -> 'GridTN1DComb':
        """ build identity mpo on specified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id)
            branches[gl.gridID] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elems_mps(cls, cgrid: GridsComb, inds: list[int], site_ind_id='i({})', site_tag_id='X({})')\
            -> 'GridTN1DComb':
        """ build MPO to select certain elements specified by inds
            grids:  list of GridLayout objects to build MPS from
            inds:   list of ints specifying position for each GridLayout obj. Position for multi-dim grids should
                    be tuples or lists of the inds along each Axis. Indices for 1-dimensional grids can be ints
        """
        branches = {}
        assert(cgrid.ndim == len(inds)), 'number of inds needs to match total number of Axis objects in grids'

        ind0 = 0
        for gr in cgrid.grids:
            active_inds = inds[ind0:ind0+gr.ndim]
            ind0 += gr.ndim
            branch = gr.get_select_elems_mps(active_inds,site_ind_id=site_ind_id,site_tag_id=site_tag_id)
            branches[gr.gridID] = branch

        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elem_mpo(cls, cgrid: GridsComb, inds: list[int], upper_ind_id='i({})', lower_ind_id='o({})',
                            site_tag_id='X({})') -> 'GridTN1DComb':
        """ build MPO to select certain elements specified by inds
            grids:  list of GridLayout objects to build MPS from
            inds:   list of ints specifying position for each GridLayout obj. Position for multi-dim grids should
                    be tuples or lists of the inds along each Axis. Indices for 1-dimensional grids can be ints
        """
        branches = {}
        assert(cgrid.ndim == len(inds)), 'number of inds needs to match total number of axis in grids'

        ind0 = 0
        for gr in cgrid.grids:
            active_inds = inds[ind0:ind0+gr.ndim]
            ind0 += gr.ndim
            branch = gr.get_select_elems_mpo(active_inds,upper_ind_id=upper_ind_id,lower_ind_id=lower_ind_id,
                                                  site_tag_id=site_tag_id)
            branches[gr.gridID] = branch

        return cls(cgrid, branches=branches)


    @classmethod
    def from_dense_state(cls, data, grid: GridsComb, site_ind_id='i({})', site_tag_id='T({})',
                         tt_tag_id='TT({})', tt_ind_id='tt({})', split_opts=None) -> 'GridTN1DComb':
        """ decompose dense nb-dimensional array along each dimension (Tensor Train decomposition)
            data: nbranches-dimensional data
            grids: list of GridLayout objects corresponding to each dimension
            site_ind_id: root site_ind_id for the branches
            site_tag_id: root tag id for the TNComb
            **tt_split_opts: parameters for tensor_split
        """
        L = grid.ngrids
        if split_opts is None:   split_opts = {}
        tt_split_opts = split_opts.pop('tt_compress_opts', {})
        branch_split_opts = split_opts

        ## decompose into an MPS using quimb
        dims = [gr.npts for gr in grid]
        data = data.reshape(dims)
        tt_format = cls.construct_spine(data, site_tag_id=tt_tag_id, site_ind_id=tt_ind_id, **tt_split_opts)

        spine = qtn.TensorNetwork([])
        branches = {}
        for nb in range(L):

            gr: Grid1D = grid.grids[nb]

            if gr is None:   ## no grid corresponding to this dim
                ## should probably be some placeholder subclass of Grid
                if nb < L - 1:
                    helper.canonize_tens_list(tt_format[nb], tt_format[nb+1], inplace=True)
                spine.add( [tt_format[nb]] )
                continue

            if nb == 0:
                nlegs = 1       ## left-most TT tensor has only 1 leg along spine
                inds = (tt_format.bond(nb,nb+1), tt_format.site_ind_id(nb))
            elif nb == L-1:
                nlegs = 1       ## right-most TT tensor has only 1 leg along spine
                inds = (tt_format.bond(nb-1,nb), tt_format.site_ind_id(nb))
            else:
                nlegs = 2
                inds = (tt_format.bond(nb-1,nb), tt_format.bond(nb,nb+1),
                        tt_format.site_ind_id(nb))

            branch_tens = tt_format[nb].transpose(*inds,inplace=False)
            branch_data = branch_tens._components
            shape_lr = branch_data.shape[:nlegs]  # l/r legs in spine mps
            anc_ind_ids = tuple([f'anc{i}' for i in range(nlegs)])

            ## fold each axis into d-ary form eg. (q x q x q ...) ** self.dim
            branch_data = branch_data.reshape(shape_lr+gr.shape)

            # new_inds = [site_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds )

            gtn_mps = gr.map_state_to_mps(branch_data, site_ind_id=site_ind_id + f',g({gr.gridID})',
                                          site_tag_id=site_tag_id + f',g({gr.gridID})')
            tens0 = gtn_mps.data[0]
            TL, TR = tens0.split(anc_ind_ids, get='tensors', bond=tt_ind_id.format(nb),
                                 absorb='left', **branch_split_opts)
            tens0.modify(data=TR._components, inds=TR.inds)
            branches[gr.gridID] = gtn_mps

            TL.drop_tags()
            TL.add_tag(tt_tag_id.format(nb))

            if nb < L-1:    # push singular values into next tensor in spine
                helper.compress_tens_list(TL, tt_format[nb+1], inplace=True, **tt_split_opts)

            spine.add([TL])     # the remaining tensor containing L,R


            # mps = qtn.TensorNetwork([])
            # for x in range(gr.L-1,-1,-1):
            #     new_tens, TR = new_tens.split(shape_lr + new_inds[:x], get='tensors',
            #                                   rtags=site_tag_id.format(x)+f',g({gr.gridID})',
            #                                   absorb='left', **branch_split_opts)
            #     mps.add(TR)
            #
            # mps.view_as(qtn.MatrixProductState, L=gr.L, cyclic=False, inplace=True,
            #                                     site_tag_id=site_tag_id+f',g({gr.gridID})',
            #                                     site_ind_id=site_ind_id+f',g({gr.gridID})')

            # grid_mps = GridTN1D(gr,data=mps)
            # branches += [grid_mps]
            #
            # new_tens.drop_tags()
            # new_tens.add_tag(tt_tag_id.format(nb))
            # bond_tt, = new_tens.bonds(mps[0])
            # new_tens.reindex({bond_tt:tt_ind_id.format(nb)},inplace=True)

            # if nb < L-1:
            #     helper.compress_tens_list(new_tens, tt_format[nb+1], inplace=True, **tt_split_opts)
            #
            # spine.add([new_tens])     # the remaining tensor containing L,R

        state_tn = cls(grid, branches=branches, spine=spine)
        return state_tn


    @classmethod
    def from_dense_operator(cls, data, grid: GridsComb, upper_ind_id='o({})', lower_ind_id='o({})', site_tag_id='T({})',
                            tt_tag_id='TT({})', tt_ind_id='tt({})', split_opts=None) -> 'GridTN1DComb':
        """ decompose dense nbranch-dimensional array along each dimension (Tensor Train decomposition)
                operator: o0 x o1 x ... i0 x i1 x ... where 0,1,.. refers to dimension/branch
                each branch is then decomposed into MPO (o00 x o01 x... i00 x i01 x...) for 0
                mpo shape of o (or i) given by dims_list[dim]
            data: nb*2-dimensional data
            grids: list of GridLayout objects corresponding to each dimension
            upper_ind_id:  root upper_ind_id for branches
            lower_ind_id:  root lower_ind_id for branches
            site_tag_id: root tag id for the TNComb
            **split_opts: parameters for tensor_split
                also 'tt_compress_opts' is key for tt_compress_opts (eg. {'max_bond': D,'cutoff': 1.0e-10})
                compression parameters used for initial splitting into tensor train / spine
        """
        L = grid.ngrids
        if split_opts is None:   split_opts = {}
        tt_split_opts = split_opts.pop('tt_compress_opts',{})
        branch_split_opts = split_opts

        ## decompose operator data into an MPS using quimb
        axT = [2*x for x in range(L)] + [2*x+1 for x in range(L)]
        data = data.transpose(axT)
        dims = [gl.npts ** 2 for gl in grid]
        data = data.reshape(dims)
        tt_format = cls.construct_spine(data, site_tag_id=tt_tag_id, site_ind_id=tt_ind_id, **tt_split_opts)  # right canon

        spine = qtn.TensorNetwork([])
        branches = {}
        for nb in range(L):

            gr = grid[nb]

            if gr is None:   ## no branch corresponding to this dim
                if nb < L - 1:
                    helper.canonize_tens_list(tt_format[nb], tt_format[nb+1], inplace=True)
                spine.add([tt_format[nb]])
                continue

            if nb == 0:
                nlegs = 1
                inds = (tt_format.bond(nb,nb+1), tt_format.site_ind_id(nb))
            elif nb == L-1:
                nlegs = 1
                inds = (tt_format.bond(nb-1,nb), tt_format.site_ind_id(nb))
            else:
                nlegs = 2
                inds = (tt_format.bond(nb-1,nb), tt_format.bond(nb,nb+1),
                        tt_format.site_ind_id(nb))

            branch_tens = tt_format[nb].transpose(*inds,inplace=False)
            branch_data = branch_tens._components
            shape_lr = branch_data.shape[:nlegs]  # l/r legs in spine mps
            anc_ind_ids = tuple([f'anc{i}' for i in range(nlegs)])

            ## fold each axis into d-ary form eg. (q x q x q ...) * 2
            branch_data = branch_data.reshape(shape_lr+gr.shape()*2)
            # new_inds_1 = [upper_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # new_inds_2 = [lower_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds_1+new_inds_2 )

            gtn_mpo = gr.map_state_to_mpo(branch_data, upper_ind_id=upper_ind_id + f',g({gr.gridID})',
                                          lower_ind_id=lower_ind_id + f',g({gr.gridID})',
                                          site_tag_id=site_tag_id+f',g({gr.gridID})',
                                          direction=1, ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids,
                                          **branch_split_opts)
            tens0 = gtn_mpo._components[0]
            TL, TR = tens0.split(anc_ind_ids, get='tensors', bond=tt_ind_id.format(nb),
                                 absorb='left', **branch_split_opts)
            tens0.modify(data=TR._components, inds=TR.inds)
            branches[gr.gridID] = gtn_mpo

            TL.drop_tags()
            TL.add_tag(tt_tag_id.format(nb))

            if nb < L - 1:  # push singular values into next tensor in spine
                helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, **tt_split_opts)

            spine.add([TL])  # the remaining tensor containing L,R

            # mpo = qtn.TensorNetwork([])
            # for x in range(gl.L-1,-1,-1):
            #     new_tens, TR = new_tens.split(shape_lr + new_inds_1[:x] + new_inds_2[:x],
            #                                   get='tensors', rtags=site_tag_id+f',g({gl.gridID})',
            #                                   absorb='left', **branch_split_opts)
            #     mpo.add(TR)
            # mpo.view_as(qtn.MatrixProductOperator, L=gl.L, cyclic=False, inplace=True,
            #                                        site_tag_id=site_tag_id+f',g({gl.gridID})',
            #                                        upper_ind_id=upper_ind_id+f',g({gl.gridID})',
            #                                        lower_ind_id=lower_ind_id+f',g({gl.gridID})')
            #
            # ## add mpo to branches list
            # grid_mpo = GridTN1D(gl, data=mpo)
            # branches += [grid_mpo]
            #
            # ## update and canonicalize spine
            # new_tens.drop_tags()
            # new_tens.add_tag(tt_tag_id.format(nb))
            # bond_tt, = new_tens.bonds(mpo[0])
            # new_tens.reindex({bond_tt: tt_ind_id.format(nb)}, inplace=True)
            #
            # if nb < L-1:
            #     helper.compress_tens_list(new_tens, tt_format[nb+1], inplace=True, **tt_split_opts)
            #
            # spine.add([new_tens])  # the remaining tensor containing L,R

        state_tn = cls(grid, branches=branches, spine=spine) # , tt_compress_opts_dict=tt_split_opts_dict)
        return state_tn


    # @classmethod
    # def pad_mps_to_grid(cls, mps_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     """ takes dictionary with axIDs as keys to create GridTN1DComb object
    #         pad empty grids with ones
    #     """
    #     axIDs = mps_ax_dict.keys()
    #     new_branches = {}
    #     active_grids = cgrid.get_active_grids(axIDs)
    #     for grid in cgrid.grids:
    #         if grid in active_grids:
    #             active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
    #             grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
    #             grid_mps = grid.make_mps_ndim(grid_mps_dict)
    #             new_branches[grid.gridID] = GridTN1D(grid, grid_mps)
    #         else:
    #             new_branches[grid.gridID] = grid.get_ones_mps()
    #
    #     mps_comb = cls(cgrid, branches = new_branches)
    #     return mps_comb
    #
    #
    # @classmethod
    # def pad_mpx_to_grid(cls, mps_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     """ takes dictionary with axIDs as keys to create GridTN1DComb object
    #         does not pad remaining axes
    #     """
    #     axIDs = mps_ax_dict.keys()
    #     new_branches = {}
    #     for grid in cgrid.get_active_grids(axIDs):
    #         active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
    #         grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
    #         grid_mpx = grid.make_mpx_ndim(grid_mps_dict)
    #         new_branches[grid.gridID] += [GridTN1D(grid, grid_mpx)]
    #
    #     mpo_comb = cls(cgrid, branches = new_branches)
    #     return mpo_comb
    #
    #
    # @classmethod
    # def pad_mpo_to_grid(cls, mpo_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     axIDs = mpo_ax_dict.keys()
    #     new_branches = {}
    #     for grid in cgrid.get_active_grids(axIDs):
    #         active_axIDs = [axID for axID in grid.axIDs if axID in mpo_ax_dict]
    #         grid_mpo_dict = {axID: mpo_ax_dict[axID] for axID in active_axIDs}
    #         grid_mpo = grid.make_mpo_ndim(grid_mpo_dict)
    #         new_branches[grid.gridID] += [GridTN1D(grid, grid_mpo)]
    #
    #     mpo_comb = cls(cgrid, branches=new_branches)
    #     return mpo_comb


    @classmethod
    def construct_spine(cls, L: int, data: np.ndarray, tt_ind_id='tt({})', tt_tag_id='TT({})', **tt_split_opts)\
            -> qtn.MatrixProductState:
        """ get mps connecting the dimensions together
            data is n-dimensional in desired order; no reshaping done
                 if data is None, defaults to data = 1.
            if provided, makes a TT decomposition (decompose by axis)
            Ootput MPS is in RIGHT CANONICAL form
        """
        new_tens = qtn.Tensor(data,inds=[tt_ind_id.format(i) for i in range(L)])
        spine_mps = helper.mpx_from_dense(new_tens,L,[tt_ind_id],
                                          site_tag_id=tt_tag_id, direction=1,
                                          return_mpx=True,**tt_split_opts)  ## right canonical
        return spine_mps


    def get_data(self) -> np.ndarray:
        """ convert to data
            axes ordered like output (0,1,2..) x input (0,1,2...)
        """
        out = self.contract(fuse=True)
        inds_list_o = []
        inds_list_i = []
        for branch in self.branches:
            if isinstance(branch._components, qtn.MatrixProductOperator):
                inds_list_o += [f'o({branch.grid.gridID})']
                inds_list_i += [f'i({branch.grid.gridID})']
            elif isinstance(branch._components, qtn.MatrixProductState):
                inds_list_o += [f'i({branch.grid.gridID})']

        out.transpose(*(inds_list_o+inds_list_i),inplace=True)
        return out.data


    #############################
    ## operations with TN comb ##
    #############################

    def apply(self, mpo: 'GridTN1DComb', inplace: Optional[GridsComb] = False, compress=False,
              compress_opts=None) -> 'GridTN1DComb':
        """ apply smaller MPO TNcomb to this one
            other: MPO TNcomb
            new_grids:  final grid that the TN should occupy. Note that the grid should not
                        require reordering of the original grids
                        self is padded with ones if other components are MPS
        """
        gtnc = self if inplace else self.copy(deep=False)  # original data will be replaced

        self_is_mps = all([isinstance(b._components, qtn.MatrixProductState) for b in gtnc.branches])
        new_branches = {}

        if new_grid is None:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl.gridID not in gtnc.grid.gridIDs]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                new_grid = GridsComb(gtnc.grid.gridID + other.grid.gridID, grids_list)
            else:
                new_grid = gtnc.grid

        new_gridIDs = [gl.gridID for gl in new_grid.grids]
        nbranches = len(new_gridIDs)

        branch_inds1 = [new_gridIDs.index(gl.gridID) for gl in gtnc.grid.grids]
        assert (np.diff(branch_inds1) > 0), 'new grid must not reorder self.grids'

        branch_inds2 = [new_gridIDs.index(gl.gridID) for gl in other.grid.grids]
        assert (np.diff(branch_inds2) > 0), 'new grid must not reorder other.grids'

        ## combine the spines
        if gtnc.spine is not None:
            spine_ind_id = gtnc.spine.site_ind_id
            spine1 = helper.pad_mps(gtnc.spine, branch_inds1, nbranches)

            if other.spine is not None:
                spine2 = helper.pad_mps( other.spine, branch_inds2, nbranches )
                new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
            else:
                new_spine = spine1
        else:
            if other.spine is not None:
                spine_ind_id = other.spine.site_ind_id
                new_spine = helper.pad_mps(other.spine, branch_inds2, nbranches)
            else:
                spine_ind_id = None
                new_spine = None

        ## apply branches to each other
        for nb in range(nbranches):

            try:
                ind1 = branch_inds1.index(nb)
                b_self = gtnc.branches[ind1]
            except IndexError:
                ind1 = None
                if self_is_mps:
                    b_self = new_grid[nb].get_ones_mps()       # pad with ones
                else:
                    b_self = None

            try:
                ind2 = branch_inds2.index(nb)
                b_other = other.branches[ind2]
            except IndexError:
                ind2 = None
                b_other = None

            if b_self is None and b_other is None:
                new_branch = None
            elif b_self is None and b_other is not None:
                new_branch = b_other.copy()
            elif b_self is not None and b_other is None:
                new_branch = b_self.copy()
            else:
                tt_ind_self = gtnc.spine[ind1].bonds(b_self[0]) if ind1 is not None \
                                else None
                tt_ind_other = other.spine[ind2].bonds(b_other[0]) if ind2 is not None \
                                else None

                if tt_ind_other == tt_ind_self:
                    b_other[0].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                    other.spine[ind2].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                    tt_ind_other += '_'

                new_branch = b_self.apply(b_other, compress=False)

                if tt_ind_self and tt_ind_other:  # not both None
                    new_branch.data[0].fuse({spine_ind_id.format(nb):(tt_ind_self, tt_ind_other)},
                                                  inplace=True)  # ordering consistent w/ mps_outerproduct

            new_grid_mpx = GridTN1D(new_grid[nb],data=new_branch)
            new_branches[new_grid[nb].gridID] = new_grid_mpx

        new_comb = GridTN1DComb(new_grid, new_branches, spine=new_spine)
        if compress:
            new_comb.compress(compress_opts=compress_opts)

        return new_comb


    def add(self, gtn_mpx2: 'GridTN1DComb', inplace: 'GridsComb' = False, compress=False,
            compress_opts=None) -> 'GridTN1DComb':
        """ add two TNcombs. can be two separate grids. will pad with ones
            if MPS or identity if MPO
        """
        gtnc = self if inplace else self.copy()

        if new_grid is None:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl.gridID not in gtnc.grid.gridIDs]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                new_grid = gtnc.grid.__class__(gtnc.grid.gridID + other.grid.gridID, grids_list)
            else:
                new_grid = gtnc.grid

        new_gridIDs = [gl.gridID for gl in new_grid.grids]
        nbranches = len(new_gridIDs)

        branch_inds1 = [new_gridIDs.index(gl.gridID) for gl in gtnc.grid.grids]
        assert (np.diff(branch_inds1) > 0), 'new grid must not reorder self.grids'

        branch_inds2 = [new_gridIDs.index(gl.gridID) for gl in other.grid.grids]
        assert (np.diff(branch_inds2) > 0), 'new grid must not reorder other.grids'

        gridtn1 = gtnc.pad_self_to_new_grid(new_grid)
        gridtn2 = other.pad_self_to_new_grid(new_grid)

        spine1 = gridtn1.spine
        if spine1 is None:
            spine1 = helper.ones_mps(nbranches,1,site_ind_id='tt{}',site_tag_id='TT{}')

        spine2 = gridtn2.spine
        if spine2 is None:
            spine2 = helper.ones_mps(nbranches,1,site_ind_id='tt{}',site_tag_id='TT{}')
        else:
            ## make spine2 match spine inds
            for nb in range(nbranches):
                branch = gridtn2.branches[nb]
                old_ind_id = spine2.site_ind(nb)
                new_ind_id = spine1.site_ind(nb)
                if branch is not None:
                    branch[0].reindex({old_ind_id: new_ind_id}, inplace=True)

            spine2.site_ind_id = spine1.site_ind_id
            for nb in range(nbranches-1):
                old_bond_ij = spine2.bond(nb,nb+1)
                new_bond_ij = spine1.bond(nb,nb+1)
                tens_i = spine2[nb]
                tens_j = spine2[nb+1]
                tens_i.reindex({old_bond_ij:new_bond_ij},inplace=True)
                tens_j.reindex({old_bond_ij:new_bond_ij},inplace=True)

        ## combine the spines
        new_spine = spine1
        for nb in range(nbranches):
            new_spine[nb].direct_product(spine2[nb],inplace=True)

            branch1 = gridtn1.branches[nb]
            branch2 = gridtn2.branches[nb]

            if branch1 is None and branch2 is None:
                continue
            elif branch1 is None:  ## branch 2 is not None
                if isinstance(branch2.data, qtn.MatrixProductState):
                    branch1 = branch2.grid.get_ones_mps()
                elif isinstance(branch2.data, qtn.MatrixProductOperator):
                    branch1 = branch2.grid.get_iden_mpo()
            elif branch2 is None:
                if isinstance(branch1.data, qtn.MatrixProductState):
                    branch2 = branch1.grid.get_ones_mps()
                elif isinstance(branch1.data, qtn.MatrixProductOperator):
                    branch2 = branch1.grid.get_iden_mpo()

            branch1.add(branch2, inplace=True)
            # the tt{} bond should be direct producted and fused, but not summed
            # tn.exponent is determined by exponent in branch1

        gridtn1.spine = new_spine
        if compress:
            gridtn1.compress(compress_opts=compress_opts)
        return gridtn1


    def scalar_multiply(self,scalar_val: Numeric,inplace=False):
        tnc = self if inplace else self.copy()
        if tnc.spine is not None:
            helper.scalar_multiply(tnc.spine, scalar_val, inplace=True)
        else:
            nbranches = sum([(b is not None) for b in tnc.branches])
            val = np.abs(scalar_val)**(1./nbranches)
            sign = -1 if scalar_val < 0 else 1
            ind = 0
            for branch in tnc.branches:
                if branch is None:   continue
                if ind == 0:   helper.scalar_multiply(branch._components, val * sign, inplace=True)
                else:          helper.scalar_multiply(branch._components, val, inplace=True)
                ind += 1
        return tnc


    def elemental_multiply(self, gtn_mps2: 'GridTN1DComb', inplace=False, compress=False, compress_opts=None):
        """ compute h(x) = g(x) * f(x) where f(x) is self, g(x) is tn2
        """
        new_tnc = self if inplace else self.copy()
        tn2_mpo = tn2_mps.apply_elemental_multiply_op()
        new_tnc.apply(tn2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return new_tnc


    def canonicalize_around_i(self, i):
        """ canonicalizes the comb such that spine is center canonical at branch/site i
        """
        if self.spine is None:    ## branches are independent
            return

        ## canonicalize branches
        for nb in range(self.grid.ngrids):
            if nb == i:  continue

            branch = self.branches[nb]
            helper.canonize(branch,i=0)

            ## canonicalize into spine
            helper.canonize_tens_list(branch[0], self.spine[nb], inplace=True)

        ## canonicalize spine
        helper.canonize(self.spine,i=i)

        return


    def compress(self, branch_inds=None, direction=0, compress_opts=None):
        """ compress the comb. inplace operation
            compress_level: sets level of compression determined by grid on each branch,
                and in get_tt_compress_opts for the spine
            compress_opts: overwrite default. 'tt_compress_opts' can be used to key a
                dict that contains compress_opts for the tensor-train/spine part.
        """
        if branch_inds is None:
            if direction%2 == 0:     branch_inds = range(self.grid.ngrids - 1)
            else:                    branch_inds = range(self.grid.ngrids - 1, 0, -1)
        else:
            if direction%2 ==0:      branch_inds = np.sort(list(branch_inds))
            else:                    branch_inds = np.sort(list(branch_inds))[::-1]

        self.canonicalize_around_i(branch_inds[0])
        canon_site = branch_inds[0]  # canonicalized around this site

        if compress_opts is None:   compress_opts = {}
        tt_compress_opts = compress_opts.pop('tt_compress_opts', {})

        # canonicalize the branch
        for nb in branch_inds:

            ## move canonicalization to new site if necessary
            if canon_site != nb:
                if nb > canon_site:
                    helper.compress_tens_list(*self.spine[canon_site:nb+1],
                                              inplace=True, compress_opts=tt_compress_opts)
                else:
                    helper.compress_tens_list(*self.spine[canon_site:nb-1:-1],
                                              inplace=True, compress_opts=tt_compress_opts)

            branch = self.branches[nb]

            ## canonicalize spine into branch 
            spine_tens = self.spine[nb]
            helper.canonize_tens_list(spine_tens,branch[0],inplace=True)

            ## canonicalize and then compress branch
            mod_compress_opts = compress_opts.copy()
            mod_compress_opts['form'] = 'right'
            helper.compress(branch,compress_opts=mod_compress_opts)

            ## move canon from branch into spine
            helper.compress_tens_list(branch[0], spine_tens, inplace=True, compress_opts=tt_compress_opts)
            canon_site = nb

        return self


    def compress_branch(self, branch_ind, cur_orthog=None, compress_opts=None):
        """ compress the branch positioned at branch_num
        """
        branch = self.branches[branch_ind]

        if compress_opts is None:   compress_opts = {}
        else:                       compress_opts = compress_opts.copy()
        tt_compress_opts = compress_opts.pop('tt_compress_opts', {})

        if self.spine is not None:

            if cur_orthog is None:
                self.canonicalize_around_i(branch_ind)
            elif cur_orthog != branch_ind:
                helper.canonize(self.branches[branch_ind], i=0)
                if branch_ind < cur_orthog:
                    helper.canonize_tens_list(*self.spine[cur_orthog:branch_ind - 1:-1], inplace=True)
                else:
                    helper.canonize_tens_list(*self.spine[cur_orthog:branch_ind + 1], inplace=True)
            else:
                pass

            ## canonicalize spine into branch
            spine_tens = self.spine[branch_ind]
            helper.canonize_tens_list(spine_tens, branch[0], inplace=True)

        mod_compress_opts = compress_opts.copy()
        mod_compress_opts['form'] = 'right'
        helper.compress(branch, compress_opts=compress_opts)

        if self.spine is not None:
            helper.compress_tens_list(branch[0], self.spine[branch_ind], inplace=True, compress_opts=tt_compress_opts)

        return self


    def contract(self,check_collisions=True,fuse=False):
        """ contract comb TN
            check_collisions: check collisions of ind labels between branches
            fuse: whether or not to fuse inds of a single branch
            out_ind_id:  ind_id to use when fusing indices of branches
        """
        tens = None
        exponent = self.exponent

        for k in range(self.grid.ngrids):

            branch = self.branches[k]
            if branch is None:
                tens_k = None
                continue

            if check_collisions and not fuse:
                helper.mpx_add_label(branch,f'g({branch.grid.gridID})')

            tens_k = branch.data.contract()
            exponent += branch.data.exponent

            if fuse:
                try:
                    tens_k.fuse({f'i({branch.grid.gridID})':
                             tuple([branch.data.site_ind_id.format(i) for i in range(branch.data.L)])},
                                inplace=True)
                except AttributeError:   # branch was an MPO
                    tens_k.fuse(
                        {f'o({branch.grid.gridID})':
                             tuple([branch.data.upper_ind_id.format(i) for i in range(branch.data.L)]),
                         f'i({branch.grid.gridID})':
                             tuple([branch.data.lower_ind_id.format(i) for i in range(branch.data.L)])},
                        inplace=True)

            if self.spine is not None:
                if tens_k is not None:   tens_k.contract(self.spine[k])
                else:                    tens_k = self.spine[k]

            if tens is not None:
                if tens_k is not None:   tens.contract(tens_k)
            else:                        tens = tens_k

        return tens*(10**exponent)

            
    #######################
    ## change dimensions ##
    #######################

    def pad_self_to_new_grid(self, new_grid: GridsComb, inplace=False, explicit_pad_mps=False, explicit_pad_mpo=False):
        """ new_grids: list of grids to expand TN to
            explicit_pad:  whether to explicitly add ones mps/iden mpos
        """
        tnc = self if inplace else self.copy()

        new_gridIDs = [gl.gridID for gl in new_grid]
        branch_inds1 = [new_gridIDs.index(gl.gridID) for gl in self.grid.grids]
        if self.spine is not None:
            assert (np.diff(branch_inds1) > 0), 'new grid must not reorder self.grids'
            new_spine = helper.pad_mps(tnc.spine, branch_inds1, len(new_grid))
        else:
            new_spine = None

        new_branches = []
        for gl in new_grid.grids:
            try:
                ind1 = self.grid.gridIDs.index(gl.gridID)
                new_branches += [tnc.branches[ind1]]
            except IndexError:
                if explicit_pad_mps:
                    new_mps = gl.get_ones_mps()
                    new_branches += [GridTN1D(gl,data=new_mps)]
                elif explicit_pad_mpo:
                    new_mpo = gl.get_iden_mpo()
                    new_branches += [GridTN1D(gl,data=new_mpo)]
                else:
                    new_branches += [None]

        tnc.branches = new_branches
        tnc.spine = new_spine
        return tnc


    def pad_branch_to_axes(self, new_axes, inplace=False):
        """ Map data on a branch to a new set of axes
        """
        raise NotImplementedError


    #####################
    ## data processing ##
    #####################

    # def interpolate_data(self, nsites, inplace=False) -> 'GridTN1DComb':
    #     """ extend grid s.t. data is interpolated between existing grid points
    #     """
    #     raise NotImplementedError
    #
    # def gaussian_smooth_data(self, inplace=False, sigma=0.3, proc_axes: list[Axis] = None, mode='wrap', compress=True,
    #                          **compress_opts) -> 'GridTN1DComb':
    #     """ process the data in some way
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     blur_mpo = self.grid.gaussian_smooth_mpo(sigma=sigma, proc_axes=proc_axes, mode=mode)
    #     if blur_mpo is not None:
    #         gtn.apply(blur_mpo, inplace=True, compress=True, **compress_opts)
    #     return gtn
    #
    #
    # def apply_absorbing_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     absorb_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False, **compress_opts)
    #     if absorb_mpo is not None:
    #         gtn.apply(absorb_mpo, inplace=True, compress=True, **compress_opts)
    #     return gtn
    #
    #
    # def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply reflecting_v bc to state.
    #         apply to derivative MPO?
    #     """
    #     raise NotImplementedError

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def apply_elemental_multiply_op(self, lower_ind_id='i{{})', upper_ind_id='o({})', compress=False,
                                    compress_opts=None):
        """ apply elemental multiply operator to self (e.g. convolution or d_ijk fct)
        """
        new_branches = {}
        for k, b in self.branches.items():
            new_b = b.apply_elemental_multiply_op(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id,
                                                  compress=compress, compress_opts=compress_opts)
            new_branches[k] = new_b

        spine = self.spine.copy() if self.spine is not None else None
        gtn_mpo = self.__class__(self.grid, branches=new_branches, spine=spine)
        return gtn_mpo

    # def apply_xmultiply_mpo(self, x_axes=None, inplace=False, compress=True, **compress_opts):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...)
    #     """
    #     raise NotImplementedError
    #
    # def take_firstderivative(self, ax, bc, deriv_params=None, recalc=False, compress=True, **compress_opts):
    #     """ self.order: finite difference expansion order """
    #     raise NotImplementedError
    #
    # def take_secondderivative(self,ax1,ax2,bc1,bc2,recalc=False,compress=True,**compress_opts):
    #     """ self.order: finite difference expansion order """
    #     raise NotImplementedError

    def integrate(self, integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None, compress=False,
                  compress_opts=None):
        new_branches = {}

        if self.spine is not None:
            spine = self.spine if new_ax_deriv_configs else self.spine.copy()
        else:
            spine = None

        for k, b in self.branches:
            active_axes = [ax for ax in integ_axes if ax in b.grid.axes]
            if len(active_axes) == 0:
                new_branches[k] = b if new_ax_deriv_configs else b.copy()
                continue

            new_b = b.integrate(active_axes, is_sqrt=is_sqrt, **compress_opts)
            if isinstance(new_b,Numeric):
                self.exponent += np.log10(new_b)
            elif isinstance(new_b, qtn.Tensor):
                ind = self.grid.gridID_pos[k]
                spine_tens = self.spine[ind]
                spine_tens.contract(new_b)
                spine_tens.new_ind(spine.site_ind(ind), size=1, axis=-1)
            elif isinstance(new_b, GridTN1D):
                new_branches[k] = new_b
            else:
                raise NotImplementedError

        gtn_mpo = self.__class__(self.grid, branches=new_branches, spine=spine)
        return gtn_mpo


    def meas_expec(self, obs_gtn, is_sqrt=False) -> Numeric:
        raise NotImplementedError