from setup.configs import *
import pickle
import helper_quimb as helper
from grid import Grid
from gridTN import GridTN
from gridTN_composite import GridTN_Composite
from gridTN_1D import GridTN1D

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from grid import Grid
    from grid1D import Grid1D
    from grid_comb import GridsComb

""" class to work with TN's that look like multiple MPS connected at one end by another MPS
    ie. o---o---o---o---o
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
"""

UnitGridType = Union['Grid','Axis']
CombDataType = tuple[dict['Grid', 'GridTN1D'], MPSType] or dict['Grid', 'GridTN1D']

class GridTN1DComb(GridTN_Composite):

    def __init__(self, grid: 'GridsComb', branches: dict['Grid','GridTN'] = None, spine: 'MPSType' = None,
                 exponent = 0.0, sign=1.0, ax_deriv_configs: DerivativeConfiguration = None, virtual = False):
        """ branches: List of nbranch GridTN1Ds (each of some length) that are the vertical branches
                can have Nones as placeholders, in which that dim is not represented by an MPS
                each GridTN1D has an associated GridLayout object
            spine: MPS of length nbranch that connects the branches (the 'o's)
                Note: not associated with a grid
                if is None, assumes outer product between the branches
                otherwise, assumes all tensors are in correct final form
                          (ie. branches have bonds with spine)
        """
        super().__init__(grid, ax_deriv_configs=ax_deriv_configs)
        # self.grid = grid
        ## Attributes: grid, _data, # ngrids, axes, axIDs, ax_map, ndim

        ## self._data replacement

        self._branches: dict['Grid','GridTN'] = {}
        self._spine: 'MPSType' = None
        self._data_type: Optional['DataType'] = None
        if branches is not None:
            self._active_grids = [gk for gk, b in branches.items() if b is not None and b.data is not None]
        else:
            self._active_grids = []

        if not virtual:
            branches = {gk: b.copy(deep=True) for gk, b in branches.items()} if branches is not None else None
            spine = spine.copy() if spine is not None else None

        self.spine = spine
        self.branches = branches
        # self.data = (self.branches, self.spine)
        ## defines self.spine, self.branches

        self._exponent = exponent
        self._sign = sign


    def __repr__(self):
        return '\nbranches: ' +  str(self.branches) + '\nspine: ' + str(self.spine)


    def __getitem__(self, gr: Union['Grid',int]) -> 'GridTN':
        """ return tn branch corresponding to axis i
        """
        return self.get_branch(gr)
        # return self.branches[gr]   ## allow KeyError to be raised

    def __setitem__(self, gr: Union['Grid',int], new_branch: 'GridTN'):
        """ set ith component of the field. for convenience
        """
        self.set_branch(gr, new_branch)
        # if isinstance(gr,int):
        #     gr = self.grid.grids[gr]
        # self._branches[gr] = new_branch
        # self._update_data_type([new_branch.data_type])

    @property
    def exponent(self):
        return self._exponent

    @property
    def sign(self):
        return self._sign

    def _get_data(self):
        if len(self.branches)==0 and self.spine is None:
            return None
        return self.branches, self.spine

    def _set_data(self, data: CombDataType):
        if data is None:
            self._branches = {}
            self._spine = None
            self._active_grids = []
            self._exponent = 0.
            self._sign = 1.
        elif isinstance(data, tuple):
            self.branches: dict['Grid',GridTN1D] = data[0]
            if len(data) > 0:   self.spine = data[1]
            else:               self.spine = None
            self._update_active_grids()
        elif isinstance(data, dict):        # Dictionary of {gridID: GridTN1Ds}
            self.branches: dict['Grid',GridTN1D] = data
            self.spine: MPSType = None
            self._update_active_grids()
        else:
            raise NotImplementedError

    data = property(fget=_get_data, fset=_set_data)

    ## define self.spine
    def _set_spine(self, spine: 'MPSType'):
        if spine is None:
            spine = helper.ones_mps(self.grid.ngrids, 1, site_ind_id='tt({})', site_tag_id='TT({})')
            for gr in self.grid.grids:
                if gr not in self._active_grids:
                    idx = self.grid.get_grid_ind(gr)
                    spine[idx].isel({spine.site_ind_id.format(idx):0}, inplace=True)
        else:
            assert (spine.L == self.grid.ngrids), 'spine length and num branches need to be equal'
        self._spine = spine

    def _get_spine(self):
        return self._spine

    spine = property(fget=_get_spine, fset=_set_spine)

    ## define self.branches
    def _get_branches(self):
        return self._branches

    def _set_branches(self, branches: dict['Grid','GridTN']):
        ## branches = None equivalent to empty dictionary
        if branches is not None:
            for gk, branch in branches.items():
                self.set_branch(gk, branch)

    branches = property(fget=_get_branches, fset=_set_branches)


    ## update spine site_ind_ids
    def _get_spine_ind_id(self):
        return self.spine.site_ind_id if self.spine is not None else None

    def _update_spine_ind_id(self, new_spine_ind):
        """ reindex spine inds and update branch inds. an inplace operation
        """
        old_ind = self.spine.site_ind_id
        self.spine.site_ind_id = new_spine_ind
        for br in self._get_active_branches():
            tens0 = br.get_anchor_tens()
            idx = self.grid.get_grid_ind(br.grid)
            tens0.reindex({old_ind.format(idx): new_spine_ind.format(idx)}, inplace=True)

    spine_ind_id = property(fget=_get_spine_ind_id, fset=_update_spine_ind_id)


    def _update_active_grids(self):
        self._active_grids = [gk for gk, b in self._branches.items() if b is not None and b.data is not None]

    def _get_active_branches(self) -> list['GridTN']:
        return [self.branches[gr] for gr in self._active_grids]

    def _get_active_grids(self) -> list['Grid']:
        return self._active_grids

    def _get_nbranches(self):
        return len(self._active_grids)

    num_active_branches = property(fget=_get_nbranches)
    active_branches = property(fget=_get_active_branches)
    active_grids = property(fget=_get_active_grids)


    def _get_data_type(self):
        return self._data_type

    def _set_data_type(self, data_types):
        is_mps = [dc == DataType.MPS for dc in data_types]
        is_mpo = [dc == DataType.MPO for dc in data_types]

        if all(is_mps):
            self._data_type = DataType.MPS
        elif all(is_mpo):
            self._data_type = DataType.MPO
        elif np.all( np.array(is_mpo)+np.array(is_mps) ):
            self._data_type = DataType.MPX
        else:
            self._data_type = DataType.TN3
            ## not the best practice...

    def _update_data_type(self, data_types):
        orig_data_type = self._data_type

        self._set_data_type(data_types)
        new_data_type = self._data_type

        if orig_data_type is None or orig_data_type == new_data_type:
            pass
        elif (orig_data_type == DataType.MPS and new_data_type == DataType.MPO) or \
             (orig_data_type == DataType.MPO and new_data_type == DataType.MPS):
            self._data_type = DataType.MPX
        else:
            raise TypeError

    data_type = property(fget=_get_data_type)


    def get_branch(self, nb: Union[int,'Grid']) -> Optional['GridTN']:
        try:
            if isinstance(nb, int):
                nb = self.grid.grids[nb]
            return self._branches[nb]
        except KeyError:
            return None

    def set_branch(self, nb: Union[int,'Grid'], new_branch:'GridTN'):
        if new_branch is None or new_branch.data is None:
            return

        if isinstance(nb,int):
            idx = nb
            gr = self.grid.grids[idx]
        else:
            gr = nb
            idx = self.grid.get_grid_ind(gr)

        site_ind = self.spine.site_ind(idx)
        tens0 = new_branch.get_anchor_tens()

        ## adding new branch
        if gr not in self._active_grids:
            spine_tens = self.spine[idx]
            tens0 = new_branch.get_anchor_tens()
            spine_tens.new_bond(tens0, name=site_ind)
            self._active_grids += [gr]
        else:
            if site_ind not in tens0.inds:  ## ie. adding new branch
                tens0.new_ind(site_ind, size=self.spine.phys_dim(idx))
            else:   ## replacing existing branch
                assert(tens0.ind_size(site_ind)), 'spine and branch need same ind size'

        self._branches[gr] = new_branch
        self._update_data_type([new_branch.data_type])


    def create_like(self, new_data=None, exponent=0.0, sign=1.0) -> 'GridTN1DComb':
        """ create a copy but without any data
        """
        if isinstance(new_data,tuple):
            new_branches, new_spine = new_data
        else:
            new_branches = new_data
            new_spine = None

        new_tnc = self.__class__(self.grid, branches=new_branches, spine=new_spine, exponent=exponent, sign=sign,
                                 ax_deriv_configs=self.ax_deriv_configs)
        return new_tnc


    def copy(self, deep=True) -> 'GridTN1DComb':
        """ return a copy (copy MPO data)
            if not deep, don't copy the data. default is deep=True
        """
        branches = {gk: b.copy(deep) for gk, b in self.branches.items()}
        spine = self.spine.copy() if (deep and self.spine is not None) else self.spine
        new_tnc = self.create_like(new_data=(branches, spine), exponent=self._exponent, sign=self._sign)
        # new_tnc = self.__class__(self.grid, branches=branches, spine=spine, exponent=self.exponent, sign=self.sign,
        #                          ax_deriv_configs=self.ax_deriv_configs)
        return new_tnc


    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN1DComb':
        new_gtn = self if inplace else self.copy()
        new_gtn._sign *= -1
        return new_gtn


    def save_data(self, fstr):
        branch_data = []
        for gk in self.grid.grids:
           branch_data += [self.get_branch(gk).data]
        
        data = (self._exponent, self._sign, self.spine, branch_data)
        pickle.dump(data, open(fstr + '.pkl','wb'))
        print('saved data', fstr + '.pkl')


    def max_bond(self) -> int:
        max_bonds = [b.max_bond() for b in self.active_branches]
        if self.spine is not None:
            max_bonds += [self.spine.max_bond()]
        return max(max_bonds)


    def entanglement_entropy_all(self) -> list[Numeric]:
        EEs = []
        self.canonize(inplace=True, i=0)
        cur_orthog = 0

        EEs += helper.entanglement_entropy_all(self.spine.copy())
        for b in self.active_branches:
            gk_idx = self.grid.get_grid_ind(b.grid)
            helper.canonize(self.spine, i=gk_idx, cur_orthog=cur_orthog)
            cur_orthog = gk_idx
            _, left_inds = self.spine[gk_idx].filter_bonds(b.get_anchor_tens())

            spine_singvals = self.spine[gk_idx].singular_values(left_inds=left_inds)
            spine_singvals *= 1. / np.linalg.norm(spine_singvals)
            EE = -1 * np.sum(spine_singvals ** 2 * np.log(spine_singvals ** 2))
            EEs += [EE]

            # print('spine cur orthog', gk_idx, self.spine.calc_current_orthog_center())
            helper.canonize_tens_list(self.spine[gk_idx], b.get_anchor_tens(), inplace=True)
            EEs += helper.entanglement_entropy_all(b.data.copy())[1:]
            helper.canonize_tens_list(b.get_anchor_tens(), self.spine[gk_idx], inplace=True)
            # print('branch cur orthog', b.data.calc_current_orthog_center())
        return EEs


    def collect_exponents(self):
        """ in place collection of all exponents
        """
        for gk, branch in self.branches.items():
            if branch.data is not None:
                # print('branch exponent', branch.data.exponent)
                self._exponent += branch.data.exponent
                branch.data.exponent = 0.0
        if self.spine is not None:
            # print('spine exponent', self.spine.exponent)
            self._exponent += self.spine.exponent
            self.spine.exponent = 0.0
        return self._exponent


    def distribute_exponents(self, val=None):
        """ absorb exponent into self
        """
        if val is None:
            val = self._exponent

        active = self.active_branches
        b_val = val*(1./len(active))
        for branch in active:
            branch.data.exponent += b_val
            branch.data.distribute_exponent()

        self._exponent -= val


    ############################
    # fcts to generate tncombs #
    ############################

    @classmethod
    def get_ones_mps(cls, cgrid: 'GridsComb', site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN1DComb':
        """ build ones vector mps on specified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_ones_mps(site_ind_id=site_ind_id,site_tag_id=site_tag_id)
            branches[gl] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_iden_mpo(cls, cgrid: 'GridsComb', upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})')\
            -> 'GridTN1DComb':
        """ build identity mpo on specified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id)
            branches[gl] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elems_mps(cls, cgrid: 'GridsComb', inds: Sequence[int], site_ind_id='i({})', site_tag_id='X({})')\
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
            branches[gr] = branch

        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elem_mpo(cls, cgrid: 'GridsComb', inds: list[int], upper_ind_id='i({})', lower_ind_id='o({})',
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
            branches[gr] = branch

        return cls(cgrid, branches=branches)


    @classmethod
    def from_dense_state(cls, data, grid: 'GridsComb', site_ind_id='i({})', site_tag_id='T({})',
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
        tt_split_opts = split_opts.pop('tt_compress_opts', split_opts)
        branch_split_opts = split_opts
        # print('tt_split opts', tt_split_opts)
        # print('branch split opts', branch_split_opts)

        ## decompose into an MPS using quimb
        dims = [gr.npts for gr in grid]
        data = data.reshape(dims)
        tt_format = cls.construct_spine(L, data, tt_tag_id=tt_tag_id, tt_ind_id=tt_ind_id, split_opts=tt_split_opts)

        spine = qtn.TensorNetwork([])
        branches = {}
        for nb in range(L):

            gr: 'Grid' = grid.grids[nb]

            if gr is None:   ## no grid corresponding to this dim
                ## should probably be some placeholder subclass of Grid
                if nb < L - 1:
                    helper.canonize_tens_list(tt_format[nb], tt_format[nb+1], inplace=True)
                spine.add( [tt_format[nb]] )
                continue

            if nb == 0 and L==1:
                nlegs = 0       ## no legs along spine
                inds = (tt_format.site_ind_id.format(nb),)
            elif nb == 0:
                nlegs = 1       ## left-most TT tensor has only 1 leg along spine
                inds = (tt_format.bond(nb,nb+1), tt_format.site_ind_id.format(nb))
            elif nb == L-1:
                nlegs = 1       ## right-most TT tensor has only 1 leg along spine
                inds = (tt_format.bond(nb-1,nb), tt_format.site_ind_id.format(nb))
            else:
                nlegs = 2
                inds = (tt_format.bond(nb-1,nb), tt_format.bond(nb,nb+1),
                        tt_format.site_ind_id.format(nb))

            branch_tens = tt_format[nb].transpose(*inds,inplace=False)
            branch_data = branch_tens.data
            shape_lr = branch_data.shape[:nlegs]  # l/r legs in spine mps
            # anc_ind_ids = tuple([f'anc{i}' for i in range(nlegs)])
            anc_ind_ids = inds[:nlegs]

            ## fold each axis into d-ary form eg. (q x q x q ...) ** self.dim
            # branch_data = branch_data.reshape(shape_lr+gr.shape)

            # # new_inds = [site_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds )

            gtn_mps = gr.map_state_to_mps(branch_data, site_ind_id=site_ind_id + f',g({gr.gridID})',
                                          site_tag_id=site_tag_id + f',g({gr.gridID})',
                                          direction=1, split_opts=branch_split_opts,
                                          ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids)
            if len(anc_ind_ids) > 0:
                tens0 = gtn_mps.data[0]
                TL, TR = tens0.split(anc_ind_ids, get='tensors', bond_ind=tt_ind_id.format(nb),
                                     absorb='left', **branch_split_opts)
                tens0.modify(data=TR.data, inds=TR.inds)
                TL.drop_tags()
                TL.add_tag(tt_tag_id.format(nb))

                if nb < L - 1:  # push singular values into next tensor in spine
                    helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, compress_opts=tt_split_opts)

                spine.add([TL])  # the remaining tensor containing L,R

            branches[gr] = gtn_mps

        if spine.num_tensors > 0:
            spine = spine.view_like(tt_format, inplace=True)
        else:
            spine = None

        state_tn = cls(grid, branches=branches, spine=spine, exponent=tt_format.exponent)
        return state_tn


    @classmethod
    def from_dense_operator(cls, data, grid: 'GridsComb', upper_ind_id='o({})', lower_ind_id='i({})',
                            site_tag_id='T({})', split_opts=None, tt_tag_id='TT({})', tt_ind_id='tt({})',
                            **kwargs) -> 'GridTN1DComb':
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
        K = grid.ngrids
        if split_opts is None:   split_opts = {}
        tt_split_opts = split_opts.pop('tt_compress_opts',split_opts)
        branch_split_opts = split_opts

        ## decompose operator data into an MPS using quimb
        # axT = [2*x for x in range(K)] + [2*x+1 for x in range(K)]
        axT = [2 * x for x in range(K)] + [2 * x + 1 for x in range(K)]
        axT = np.argsort(axT)
        # print('axT', axT)
        data = data.transpose(axT)
        dims = [gl.npts ** 2 for gl in grid]
        data = data.reshape(dims)
        tt_format = cls.construct_spine(K, data, tt_tag_id=tt_tag_id, tt_ind_id=tt_ind_id, split_opts=tt_split_opts)

        spine = qtn.TensorNetwork([])
        branches = {}
        for nb in range(K):

            gr: 'Grid' = grid.grids[nb]

            if gr is None:   ## no branch corresponding to this dim
                if nb < K - 1:
                    helper.canonize_tens_list(tt_format[nb], tt_format[nb+1], inplace=True)
                spine.add([tt_format[nb]])
                continue

            if nb == 0 and K==1:
                nlegs = 0       ## no legs along spine
                inds = (tt_format.site_ind_id.format(nb),)
            elif nb == 0:
                nlegs = 1
                inds = (tt_format.bond(nb,nb+1), tt_format.site_ind_id.format(nb))
            elif nb == K-1:
                nlegs = 1
                inds = (tt_format.bond(nb-1,nb), tt_format.site_ind_id.format(nb))
            else:
                nlegs = 2
                inds = (tt_format.bond(nb-1,nb), tt_format.bond(nb,nb+1),
                        tt_format.site_ind_id.format(nb))

            # print('inds', inds)
            branch_tens = tt_format[nb].transpose(*inds,inplace=False)
            shape_lr = branch_tens.shape[:nlegs]  # l/r legs in spine mps
            branch_data = branch_tens.data.reshape(shape_lr + (gr.npts, gr.npts))
            # anc_ind_ids = tuple([tt_format.site_ind_id.format(i) for i in range(nlegs)])
            anc_ind_ids = inds[:nlegs]

            ## fold each axis into d-ary form eg. (q x q x q ...) * 2
            # branch_data = branch_data.reshape(shape_lr+gr.shape()*2)
            # # new_inds_1 = [upper_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # # new_inds_2 = [lower_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds_1+new_inds_2 )

            gtn_mpo = gr.map_operator_to_mpo(branch_data, upper_ind_id=upper_ind_id + f',g({gr.gridID})',
                                             lower_ind_id=lower_ind_id + f',g({gr.gridID})',
                                             site_tag_id=site_tag_id+f',g({gr.gridID})',
                                             direction=1, split_opts=branch_split_opts,
                                             ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids)
            if len(anc_ind_ids) > 0:
                tens0 = gtn_mpo.data[0]
                TL, TR = tens0.split(anc_ind_ids, get='tensors', bond_ind=tt_ind_id.format(nb),
                                     absorb='left', **branch_split_opts)
                tens0.modify(data=TR.data, inds=TR.inds)
                TL.drop_tags()
                TL.add_tag(tt_tag_id.format(nb))

                if nb < K - 1:  # push singular values into next tensor in spine
                    helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, compress_opts=tt_split_opts)

                spine.add([TL])  # the remaining tensor containing L,R

            branches[gr] = gtn_mpo

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

        if spine.num_tensors > 0:
            spine = spine.view_like(tt_format, inplace=True)
        else:
            spine = None

        state_tn = cls(grid, branches=branches, spine=spine, exponent=tt_format.exponent)
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
    def construct_spine(cls, L: int, data: np.ndarray, tt_ind_id='tt({})', tt_tag_id='TT({})', split_opts=None)\
            -> qtn.MatrixProductState:
        """ get mps connecting the dimensions together
            data is n-dimensional in desired order; no reshaping done
                 if data is None, defaults to data = 1.
            if provided, makes a TT decomposition (decompose by axis)
            Ootput MPS is in RIGHT CANONICAL form
        """
        new_tens = qtn.Tensor(data,inds=[tt_ind_id.format(i) for i in range(L)])
        spine_mps = helper.mpx_from_dense(new_tens, L, [tt_ind_id], site_tag_id=tt_tag_id, direction=1, return_mpx=True,
                                          split_opts=split_opts)  ## right canonical
        return spine_mps


    def get_data(self, ax_select=None, pad_data=False) -> np.ndarray:
        """ convert to data
            axes ordered like output (0,1,2..) x input (0,1,2...)
        """
        out = self.contract(ax_select=ax_select)
        if ax_select is None:
            ax_select = {}

        inds_list_o = []
        inds_list_i = []
        for subgr in self.grid.grids:
            branch = self.get_branch(subgr)
            if branch is None or branch.data is None:
                if pad_data:
                    if self.data_type == DataType.MPO or self.data_type == DataType.MPX:
                        o_inds = [f'o({ax})' for ax in subgr.axes]
                        i_inds = [f'i({ax})' for ax in subgr.axes]
                        pad_iden = np.eye(subgr.npts).reshape( [ax.npts for ax in subgr.axes]*2 )
                        pad_tens = qtn.Tensor(data=pad_iden, inds=tuple(o_inds + i_inds))
                        inds_list_o += o_inds
                        inds_list_i += i_inds
                    elif self.data_type == DataType.MPS:
                        out_axes = [ax for ax in subgr.axes if ax not in ax_select]
                        o_inds = [f'i({ax})' for ax in out_axes]
                        pad_ones = np.ones([ax.npts for ax in out_axes])
                        pad_tens = qtn.Tensor(data=pad_ones, inds=tuple(o_inds))
                        inds_list_o += o_inds
                    else:
                        raise TypeError('branch data needs to by MPS or MPO')
                    out = out.contract(pad_tens)
            elif isinstance(branch.data, qtn.MatrixProductOperator):
                inds_list_o += [f'o({ax})' for ax in subgr.axes if ax not in ax_select]
                inds_list_i += [f'i({ax})' for ax in subgr.axes if ax not in ax_select]
            elif isinstance(branch.data, qtn.MatrixProductState):
                inds_list_o += [f'i({ax})' for ax in subgr.axes if ax not in ax_select]


        out.transpose(*(inds_list_o+inds_list_i),inplace=True)
        return out.data


    #############################
    ## operations with TN comb ##
    #############################
    # @profile
    def apply(self, other: 'GridTN1DComb', inplace=False, zipup=False, compress=False, compress_opts=None,
              add_cc=False) -> 'GridTN1DComb':
        """ mpo: MPO TNcomb; mpo.grid.grids can be a subset of self.grid.grids
            new_grids:  final grid that the TN should occupy. Note that the grid should not
                        require reordering of the original grids
                        self is padded with ones if other components are MPS
        """
        # print('apply self', self)
        # print('apply other', other)
        # print('add cc', add_cc)

        # zipup = False
        # zipup = True
        # print('zipup?', zipup)

        # zipup = False
        if zipup:
            return self.apply_zipup(other, inplace=inplace, add_cc=add_cc,
                                    compress=compress, compress_opts=compress_opts)

        gtnc = self if inplace else self.copy(deep=False)  # original data will be replaced

        new_branches = {}

        if not gtnc.grid == other.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl not in gtnc.grid.grids]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = GridsComb(gtnc.grid.gridID + other.grid.gridID, grids_list)

                branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
                # assert (np.diff(branch_inds1) > 0), 'new grid must not reorder self.grids'
            else:
                cgrid = gtnc.grid
                branch_inds1 = None  # range(len(cgrid.cgrids))

            grids_list = cgrid.grids
            branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
            # assert (np.diff(branch_inds2) > 0), 'new grid must not reorder other.grids'
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None  # range(len(cgrid.grids))


        nbranches = len(cgrid.grids)

        ## combine the spines
        spine1 = gtnc.spine
        spine2 = other.spine

        if add_cc:    # gtnc -> gtnc + complex conj
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine1.L):
                tens = spine1[i]
                tens.direct_product(tens.conj(), inplace=True)

        if gtnc.spine is not None:
            spine_ind_id = spine1.site_ind_id
            spine1 = helper.pad_mps(spine1, branch_inds1, nbranches)

            if spine2 is not None:
                # site_ind_id2 = spine2.site_ind_id
                if spine_ind_id == other.spine_ind_id:
                    other.spine_ind_id = other.spine_ind_id + '_'
                # print('change spine2', spine2)
                # if site_ind_id2 == spine1.site_ind_id:
                #     spine2.site_ind_id = site_ind_id2 + '_'
                spine2 = helper.pad_mps(spine2, branch_inds2, nbranches)
                new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
                new_spine.fuse_multibonds(inplace=True)
                # spine2.site_ind_id = site_ind_id2
            else:
                new_spine = spine1
        else:
            if spine2 is not None:
                spine_ind_id = spine2.site_ind_id
                new_spine = helper.pad_mps(spine2, branch_inds2, nbranches)
            else:
                spine_ind_id = None
                new_spine = None

        ## apply branches to each other
        for nb in range(nbranches):

            subgrid = cgrid.grids[nb]

            try:
                b_self = gtnc.branches[subgrid]
            except KeyError:
                if gtnc.data_type == DataType.MPS:
                    b_self = subgrid.get_ones_mps()
                else:
                    b_self = None

            try:
                b_other = other.branches[subgrid]
            except KeyError:
                b_other = None

            if b_self is None and b_other is None:
                new_branch = None
            elif b_self is None and b_other is not None:
                new_branch = b_other.copy()
            elif b_self is not None and b_other is None:
                new_branch = b_self.copy()
                if add_cc:  # gtnc -> gtnc + complex conj
                    new_branch.add(new_branch.conj(), inplace=True)
            else:
                tt_inds_self = spine1[nb].bonds(b_self.get_anchor_tens())
                tt_ind_self = next(iter(tt_inds_self))
                # if spine1 is not None:
                #     tt_inds_self = spine1[nb].bonds(b_self[0])
                #     tt_ind_self = next(iter(tt_inds_self))
                # else:
                #     tt_ind_self = None

                tt_inds_other = spine2[nb].bonds(b_other.get_anchor_tens())
                tt_ind_other = next(iter(tt_inds_other))
                # if spine2 is not None:
                #     tt_inds_other = spine2[nb].bonds(b_other[0])
                #     tt_ind_other = next(iter(tt_inds_other))
                # else:
                #     tt_ind_other = None

                # if tt_ind_other == tt_ind_self and tt_ind_self is not None:
                #     b_other[0].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                #     spine2[nb].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                #     tt_ind_other += '_'

                # print(b_self.max_bond(), b_other.max_bond())
                new_branch = b_self.apply(b_other, compress=False, add_cc=add_cc)
                # print('new branch', new_branch)

                if tt_ind_self and tt_ind_other:  # not both None
                    new_branch.data[0].fuse({spine_ind_id.format(nb):(tt_ind_self, tt_ind_other)}, inplace=True)
                    # ordering consistent w/ helper.mps_outerproduct

            # new_grid_mpx = GridTN1D(subgrid, data=new_branch)
            new_branches[subgrid] = new_branch  # new_grid_mpx

        # gtnc = GridTN1DComb(cgrid, new_branches, spine=new_spine)
        gtnc.branches = new_branches
        gtnc.spine = new_spine
        if compress:
            gtnc.compress(compress_opts=compress_opts)

        gtnc._exponent += other._exponent
        gtnc._sign *= other._sign
        gtnc.collect_exponents()

        return gtnc


    # @profile
    def apply_zipup(self, other: 'GridTN1DComb', inplace=False, add_cc=False, compress=False, compress_opts=None)\
            -> 'GridTN1DComb':
        """ mpo: MPO TNcomb; mpo.grid.grids can be a subset of self.grid.grids
            new_grids:  final grid that the TN should occupy. Note that the grid should not
                        require reordering of the original grids
                        self is padded with ones if other components are MPS
        """
        gtnc = self if inplace else self.copy(deep=False)  # original data will be replaced
        if compress_opts is None:
            compress_opts = {}

        new_branches = {}

        if not gtnc.grid == other.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl not in gtnc.grid.grids]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = GridsComb(gtnc.grid.gridID + other.grid.gridID, grids_list)

                branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            else:
                cgrid = gtnc.grid
                branch_inds1 = None  # range(len(cgrid.cgrids))

            grids_list = cgrid.grids
            branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None  # range(len(cgrid.grids))


        nbranches = len(cgrid.grids)

        ## combine the spines
        spine1 = gtnc.spine
        spine2 = other.spine

        spine_ind_id = spine1.site_ind_id
        spine1 = helper.pad_mps(spine1, branch_inds1, nbranches)

        if spine_ind_id == other.spine_ind_id:
            other.spine_ind_id = other.spine_ind_id + '_'
        spine2 = helper.pad_mps(spine2, branch_inds2, nbranches)


        # add_cc = False
        # print('apply spine', spine1, check_elem_mult_anc)
        if add_cc:    # gtnc -> gtnc + complex conj
            # print('here?')
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine1.L):
                tens = spine1[i]
                tens.direct_product(tens.conj(), inplace=True)
        # print('apply spine', spine1)


        # new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
        # new_spine.fuse_multibonds(inplace=True)
        new_spine = qtn.TensorNetwork([])

        ## apply branches to each other
        prev_spine_tens = None
        # prev_spine_mat = None
        for nb in range(nbranches):
            subgrid = cgrid.grids[nb]

            b_self = gtnc.get_branch(subgrid)
            b_other = other.get_branch(subgrid)
            if b_self is None:
                if gtnc.data_type == DataType.MPS:
                    b_self = subgrid.get_ones_mps()
                else:
                    b_self = None

            # if prev_spine_mat is not None:     # mat from canonicalization decomp of previous spine tens
            #     spine_tens = qtn.tensor_contract(prev_spine_mat, spine1[nb], spine2[nb])
            #     # print(nb, spine1[nb].norm(), spine2[nb].norm())
            # else:
            #     spine_tens = qtn.tensor_contract(spine1[nb], spine2[nb])
            #     # print(nb, spine1[nb].norm(), spine2[nb].norm())
            spine_tens = qtn.tensor_contract(spine1[nb], spine2[nb])

            if b_self is None and b_other is None:
                new_branch = None
            elif b_self is None and b_other is not None:
                new_branch = b_other.copy()
            elif b_self is not None and b_other is None:
                new_branch = b_self.copy()
                if add_cc:  # gtnc -> gtnc + complex conj
                    # print('z', new_branch.max_bond(), new_branch[0].shape)
                    new_branch.add(new_branch.conj(), inplace=True)
            else:
                tt_inds_self = spine1[nb].bonds(b_self.get_anchor_tens())
                tt_ind_self = next(iter(tt_inds_self))

                tt_inds_other = spine2[nb].bonds(b_other.get_anchor_tens())
                tt_ind_other = next(iter(tt_inds_other))

                ## apply zip-up with -> canon_site = 0
                # print('apply zipup comp', add_cc)
                # print('other', b_other)
                # print('self', b_self)
                # print('z', b_self.max_bond(), b_other.max_bond())
                if b_self.get_anchor_ind() == 0:
                    canon_form = 'right'
                else:
                    raise NotImplementedError
                new_branch = b_self.apply(b_other, zipup=True, compress=False, compress_opts={'form': canon_form},
                                          add_cc=add_cc)

                # print('apply zipup new branch', new_branch)
                # print('spine', spine1, spine2)
                # print('spine tens', spine_tens)

                # if tt_ind_self and tt_ind_other:  # not both None
                #     new_branch.data[0].fuse({spine_ind_id.format(nb):(tt_ind_self, tt_ind_other)}, inplace=True)
                b0 = new_branch.get_anchor_tens()
                # print('b0', nb, new_branch.exponent)
                # print('b0', b0.norm(), [new_branch.data[i].norm() for i in range(new_branch.L)])
                new_s0, new_b0 = b0.split((tt_ind_self, tt_ind_other), cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                        absorb='left', ltags=(), bond_ind=f'_temp{nb}_')
                # print('new s0', nb, new_s0.norm(), new_b0.norm())

                b0.modify(data=new_b0.data, inds=new_b0.inds)
                b0.reindex({f'_temp{nb}_': spine_ind_id.format(nb)}, inplace=True)

                # print('apply zipup', nb, spine_tens.shape, new_s0.shape)
                spine_tens = qtn.tensor_contract( spine_tens, new_s0 )
                spine_tens.reindex({f'_temp{nb}_':spine_ind_id.format(nb)}, inplace=True)
                spine_tens.modify(tags=spine1[nb].tags)

            # new_grid_mpx = GridTN1D(subgrid, data=new_branch)
            new_branches[subgrid] = new_branch  # new_grid_mpx

            ## canonicalize/decompose spine_tens
            # if nb == nbranches-1:
            #     # spine_norm = spine_tens.norm()
            #     # gtnc._exponent += np.log10(spine_norm)
            #     # spine_tens.modify(data=spine_tens.data/spine_norm)
            #     new_spine.add(spine_tens, virtual=True)
            # else:
            #     # print('canon spine', nb, spine_tens.norm())
            #     # print('canon spine', spine1[nb+1], spine2[nb+1])
            #     s1_idx, open_idx = spine_tens.filter_bonds(spine1[nb+1])    # shared, unshared with next tensor
            #     s2_idx, open_idx = spine_tens.filter_bonds(spine2[nb+1])
            #     lix = [idx for idx in open_idx if idx not in s1_idx + s2_idx]
            #     # print('idx', lix, s1_idx+s2_idx)
            #     tensL, tensR = spine_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE,
            #                                     right_inds=s2_idx + s1_idx,
            #                                     cutoff=compress_opts.get('cutoff', CUTOFF), ltags=spine_tens.tags)

            #     new_spine.add(tensL, virtual=True)
            #     tensR.drop_tags()
            #     prev_spine_mat = tensR

            new_spine.add(spine_tens, virtual=True)
            if prev_spine_tens is not None:
                helper.compress_tens_list(prev_spine_tens, spine_tens, inplace=True)
                spine_norm = spine_tens.norm()
                spine_tens.modify(data=spine_tens.data/spine_norm)
                gtnc._exponent += np.log10(spine_norm)
            prev_spine_tens = spine_tens

        # gtnc = GridTN1DComb(cgrid, new_branches, spine=new_spine)
        gtnc.branches = new_branches
        gtnc.spine = new_spine.view_like(spine1, inplace=True)
        gtnc.canon_site = nbranches-1

        if compress:
            gtnc.compress(compress_opts=compress_opts, canonize=False)

        gtnc._exponent += other._exponent
        gtnc._sign *= other._sign
        gtnc.collect_exponents()
        # print('apply zipup', gtnc.canon_site)
        # print('apply zip up gtnc', gtnc)

        # print('zipup gtnc', gtnc)

        return gtnc

    # @profile
    def add(self, other: 'GridTN1DComb', inplace=False, compress=False, compress_opts=None) -> 'GridTN1DComb':
        """ add two TNcombs. can be two separate grids. will pad with ones
            if MPS or identity if MPO
        """
        gtnc = self if inplace else self.copy()

        if other.grid != self.grid:
            if set(other.grid.axes).issubset(self.grid.axes):
                other = self.grid.pad_gtn_to_grid(other)
            else:
                raise ValueError(f'gtn_mpx2 ong grid {other.grid} not compatible with self on grid {self.grid}')
        else:
            other = other.copy()

        gtnc.collect_exponents()
        other.collect_exponents()

        other.distribute_exponents(val=other._exponent - gtnc._exponent)

        if gtnc.grid != self.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl.gridID not in gtnc.grid.gridIDs]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = gtnc.grid.__class__(gtnc.grid.gridID + other.grid.gridID, grids_list)
                branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            else:
                cgrid = gtnc.grid
                branch_inds1 = None

            grids_list = cgrid.grids
            branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None

        nbranches = len(cgrid)
        if branch_inds1:
            gtnc = gtnc.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds1)
        if branch_inds2:
            other = other.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds2)

        spine1 = gtnc.spine
        spine2 = other.spine

        ## make spine2 match spine inds and interior bonds
        for nb in range(nbranches):
            branch = other.get_branch(cgrid.grids[nb])
            old_ind_id = spine2.site_ind(nb)
            new_ind_id = spine1.site_ind(nb)
            if branch is not None:
                branch.get_anchor_tens().reindex({old_ind_id: new_ind_id}, inplace=True)

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

            ## add branches together
            branch1 = gtnc.get_branch(cgrid.grids[nb])
            branch2 = other.get_branch(cgrid.grids[nb])

            if branch1 is None and branch2 is None:
                new_spine[nb].direct_product(spine2[nb], inplace=True)
                continue

            elif branch1 is None:  ## branch 2 is not None
                if isinstance(branch2.data, qtn.MatrixProductState):
                    branch1 = branch2.grid.get_ones_mps(site_ind_id=branch2.data.site_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                elif isinstance(branch2.data, qtn.MatrixProductOperator):
                    branch1 = branch2.grid.get_iden_mpo(upper_ind_id=branch2.data.upper_ind_id,
                                                        lower_ind_id=branch2.data.lower_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                # tens0 = branch1.get_anchor_tens()
                # tens0.new_ind(new_spine.site_ind(nb))
                # gtnc._active_grids += [cgrid.grids[nb]]
                gtnc.set_branch(nb, branch1)
            elif branch2 is None:
                if isinstance(branch1.data, qtn.MatrixProductState):
                    branch2 = branch1.grid.get_ones_mps(site_ind_id=branch1.data.site_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                elif isinstance(branch1.data, qtn.MatrixProductOperator):
                    branch2 = branch1.grid.get_iden_mpo(upper_ind_id=branch1.data.upper_ind_id,
                                                        lower_ind_id=branch1.data.lower_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                # tens0 = branch2.get_anchor_tens()
                # tens0.new_ind(new_spine.site_ind(nb))
                other.set_branch(nb, branch2)

            branch1.add(branch2, inplace=True)
            if nb == 0:
                spine2_tens = spine2[nb].copy() * (other._sign / self._sign)
            else:
                spine2_tens = spine2[nb]
            new_spine[nb].direct_product(spine2_tens, inplace=True)

        gtnc.canon_site = -1
        if compress:
            gtnc.compress(compress_opts=compress_opts)

        return gtnc


    def scalar_multiply(self, scalar_val: Numeric, inplace=False):
        tnc = self if inplace else self.copy()

        abs_val = np.abs(scalar_val)
        if np.isreal(scalar_val):
            if scalar_val < 0:
                sign = -1
            else:
                sign = 1
        else:
            sign = np.exp(-1.j * np.angle(scalar_val))

        tnc._exponent += np.log10(abs_val)
        tnc._sign *= sign

        # helper.scalar_multiply(tnc.spine, sign, inplace=True)
        # branch = tnc.active_branches[0]
        # helper.scalar_multiply(branch.data, sign, inplace=True)

        return tnc


    # def elemental_multiply(self, gtn_mps2: 'GridTN1DComb', inplace=False, compress=False, compress_opts=None):
    #     """ compute h(x) = g(x) * f(x) where f(x) is self, g(x) is tn2
    #     """
    #     new_tnc = self if inplace else self.copy()
    #     tn2_mpo = gtn_mps2.apply_elemental_multiply_op()
    #     new_tnc.apply(tn2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
    #     return new_tnc


    def canonize(self, inplace=True, scale=True, form='right',i: Union[int,'Grid']=None):
        """ put in canonical form: all branches right canonical, spine mixed canon form centered at i
        """
        if i is None:
            i = 0
        elif isinstance(i, Grid):
            i = self.grid.get_grid_ind(i)

        gtnc = self if inplace else self.copy()

        gtnc.canonicalize_around_i(i)
        branch = gtnc.get_branch(i)
        branch.canonize(inplace=True, i=branch.get_anchor_ind())
        helper.canonize_tens_list(branch.get_anchor_tens(), gtnc.spine[i], inplace=True)
        gtnc.collect_exponents()
        return gtnc


    # @profile
    def canonicalize_around_i(self, i: Union[int,'Grid']):
        """ canonicalizes the comb such that spine is center canonical at branch/site i
        """
        # print('canonicalize around i', i)
        # if self.spine is None:    ## branches are independent
        #     return

        ## TODO: take into account current self.canon_site

        if isinstance(i, Grid):
            i = self.grid.get_grid_ind(i)

        ## canonicalize branches
        for gr in self.active_grids:
            nb = self.grid.get_grid_ind(gr)
            if nb == i:  continue

            branch = self.get_branch(gr)
            anchor_idx = branch.get_anchor_ind()
            if branch.canon_site != anchor_idx:
                branch.canonize(inplace=True, i=anchor_idx)
            ## canonicalize into spine
            helper.canonize_tens_list(branch.data[anchor_idx], self.spine[nb], inplace=True)

            # print('check branch right orthog C', nb, gr,
            #       helper.check_right_orthog([self.spine[nb]] + [tens for tens in branch.data]))


        ## canonicalize spine
        helper.canonize(self.spine, i=i, cur_orthog=None)
        # print('check spine orthog', i, helper.check_orthog(self.spine))
        self.canon_site = i
        return self


    # @profile
    def compress(self, inplace=True, compress_opts=None, sub_compress_opts=None, branch_keys=None, direction=0,
                 canonize=True):
        """ compress the comb. inplace operation
            compress_level: sets level of compression determined by grid on each branch,
                and in get_tt_compress_opts for the spine
            compress_opts: overwrite default. 'tt_compress_opts' can be used to key a
                dict that contains compress_opts for the tensor-train/spine part.
        """
        gtnc = self if inplace else self.copy()
        gtnc.collect_exponents()

        if branch_keys is None:
            if direction%2 == 0:     branch_inds = range(self.grid.ngrids)
            else:                    branch_inds = range(self.grid.ngrids - 1, -1, -1)
        else:
            if isinstance(branch_keys[0],int):
                branch_inds = branch_keys
            else:
                branch_inds = [self.grid.get_grid_ind(subgr) for subgr in branch_keys]

            if direction%2 ==0:      branch_inds = np.sort(list(branch_inds))
            else:                    branch_inds = np.sort(list(branch_inds))[::-1]

        canon_site = gtnc.canon_site  # canonicalized around this site

        if canon_site != branch_inds[-1]:
            gtnc.canonicalize_around_i(branch_inds[-1])
            canon_site = branch_inds[-1]  # canonicalized around this site
        # print('CANON AROUND check spine orthog', branch_inds[-1], helper.check_orthog(gtnc.spine))


        if compress_opts is None:
            compress_opts = {}

        if sub_compress_opts is None:
            tt_compress_opts = compress_opts
        else:
            tt_compress_opts = sub_compress_opts.get(SubCompressConfigType.TT, compress_opts)

        # ### compress spine first
        # ### if done, need to canonicalize all branches first
        # mod_tt_compress_opts = tt_compress_opts.copy()
        # mod_tt_compress_opts['form'] = 'left' if direction%2==0 else 'right'
        # helper.compress(self.spine, canonize=False, compress_opts=mod_tt_compress_opts)

        for bi in branch_inds[::-1]:

            ## move canonicalization + compress on spine to new site if necessary
            if canon_site != bi:

                if bi > canon_site:
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi + 1],
                                              inplace=True, compress_opts=tt_compress_opts)
                elif bi < canon_site:
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi - 1:-1],
                                              inplace=True, compress_opts=tt_compress_opts)

            branch = gtnc.get_branch(bi)

            if branch is None:
                continue

            mod_compress_opts = compress_opts.copy()
            mod_compress_opts['form'] = 'right'

            # print('check spine orthog A', bi, helper.check_orthog(gtnc.spine))

            ## canonicalize spine into branch
            spine_tens = gtnc.spine[bi]
            helper.canonize_tens_list(spine_tens, branch.get_anchor_tens(), inplace=True)

            ## canonicalize and then compress branch
            branch.compress(inplace=True, compress_opts=mod_compress_opts)
            # print('branch compress', mod_compress_opts)

            ## move canon from branch into spine
            helper.compress_tens_list(branch.get_anchor_tens(), spine_tens, inplace=True, # two_site=True,
                                      compress_opts=tt_compress_opts)
            # print('branch compress', tt_compress_opts)

            # print('check spine orthog C', bi, helper.check_orthog(gtnc.spine))
            # print('check branch right orthog C', bi,
            #       helper.check_right_orthog([spine_tens] + [tens for tens in branch.data]))


            canon_site = bi

        gtnc.canon_site = canon_site
        gtnc.collect_exponents()
        return gtnc


    def compress_branch(self, branch_ind, inplace=True, cur_orthog=None, compress_opts=None):
        """ compress the branch positioned at branch_num
        """
        gtnc = self if inplace else self.copy()

        branch = gtnc.branches[branch_ind]

        if compress_opts is None:   compress_opts = {}
        else:                       compress_opts = compress_opts.copy()
        tt_compress_opts = compress_opts.pop('tt_compress_opts', {})

        if gtnc.spine is not None:

            if cur_orthog is None:
                gtnc.canonicalize_around_i(branch_ind)
            elif cur_orthog != branch_ind:
                branch.canonize(inplace=True, i=0)
                if branch_ind < cur_orthog:
                    helper.canonize_tens_list(*gtnc.spine[cur_orthog:branch_ind - 1:-1], inplace=True)
                else:
                    helper.canonize_tens_list(*gtnc.spine[cur_orthog:branch_ind + 1], inplace=True)
            else:
                pass

            ## canonicalize spine into branch
            spine_tens = gtnc.spine[branch_ind]
            helper.canonize_tens_list(spine_tens, branch[0], inplace=True)

        mod_compress_opts = compress_opts.copy()
        mod_compress_opts['form'] = 'right'
        branch.compress(compress_opts=compress_opts)

        if gtnc.spine is not None:
            helper.compress_tens_list(branch[0], gtnc.spine[branch_ind], inplace=True, compress_opts=tt_compress_opts)

        return gtnc


    def contract(self, ax_select: dict['Axis',int] = None) -> 'qtn.Tensor':
        """ contract comb TN
            check_collisions: check collisions of ind labels between branches
            fuse: whether or not to fuse inds of a single branch
            out_ind_id:  ind_id to use when fusing indices of branches
        """
        tens = None
        self.collect_exponents()    # not necessary?

        if ax_select is None:
            ax_select = {}

        # for branch in self.active_branches:
        #     gk = self.grid.get_grid_ind(branch.grid)

        for gk in range(self.grid.ngrids)[::-1]:

            branch = self.get_branch(gk)
            # gk = self.grid.get_grid_ind(branch.grid)

            if branch is None:
                tens_k = self.spine[gk]

            else:
                tens_k = branch.get_data(ax_select=ax_select)
                # exponent += branch.data.exponent

                left_anc = (self.spine.site_ind_id.format(gk),)
                if branch.data_type == DataType.MPS:
                    ## actually split by axes
                    out_axes = [ax for ax in branch.grid.axes if ax not in ax_select]
                    tens_k = qtn.Tensor(data=tens_k, inds=left_anc + tuple([f'i({ax})' for ax in out_axes]))
                elif branch.data_type == DataType.MPO:
                    out_axes = [ax for ax in branch.grid.axes if ax not in ax_select]
                    tens_k = qtn.Tensor(data=tens_k, inds=left_anc + tuple([f'o({ax})' for ax in out_axes] +
                                                                           [f'i({ax})' for ax in out_axes]))
                # if fuse:
                    # try:
                    #     tens_k.fuse({f'i({branch.grid.gridID})':
                    #              tuple([branch.data.site_ind_id.format(i) for i in range(branch.data.L)])},
                    #                 inplace=True)
                    # except AttributeError:   # branch was an MPO
                    #     tens_k.fuse(
                    #         {f'o({branch.grid.gridID})':
                    #              tuple([branch.data.upper_ind_id.format(i) for i in range(branch.data.L)]),
                    #          f'i({branch.grid.gridID})':
                    #              tuple([branch.data.lower_ind_id.format(i) for i in range(branch.data.L)])},
                    #         inplace=True)

                # if self.spine is not None:
                # if tens_k is not None:
                tens_k = tens_k.contract(self.spine[gk])
                # else:                    tens_k = self.spine[gk]

            if tens is not None:
                # print('contract tens', tens)
                if tens_k is not None:   tens = tens.contract(tens_k)
            else:                        tens = tens_k

        return tens * (10 ** self._exponent) * self._sign


    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':

        assert (self.data_type == DataType.MPS), f'gtn needs to MPS type not {self.data_type}'

        gtnc = self if inplace else self.copy(deep=False)

        # if not inplace:
        #     gtnc.spine = self.spine.copy()
        #     gtnc.exponent = self.exponent
        #     gtnc.sign = self.sign

        for gk, branch in gtnc.branches.items():
            if branch is not None:
                branch.mps_to_diag_mpo(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id, inplace=True)

        return gtnc


    #######################
    ## change dimensions ##
    #######################

    def pad_self_to_new_grid(self, new_grid: 'GridsComb', branch_inds: Sequence[int]=None, inplace=False,
                             explicit_pad_mps=False, explicit_pad_mpo=False):
        """ new_grids: list of grids to expand TN to
            explicit_pad:  whether to explicitly add ones mps/iden mpos
        """
        tnc = self if inplace else self.copy()

        if branch_inds is None:
            branch_inds = [new_grid.get_grid_ind(gl) for gl in tnc.grid.grids]

        if tnc.spine is not None:
            # assert (np.diff(branch_inds) > 0), 'new grid must not reorder self.grids'
            new_spine = helper.pad_mps(tnc.spine, branch_inds, len(new_grid))
        else:
            new_spine = None

        new_branches = {}
        for gl in new_grid.grids:
            tnc_branch = tnc.get_branch(gl)
            if tnc_branch is not None:
                new_branches[gl] = tnc_branch
            else:
                if explicit_pad_mps:
                    new_mps = gl.get_ones_mps()
                    new_branches[gl] = new_mps
                elif explicit_pad_mpo:
                    new_mpo = gl.get_iden_mpo()
                    new_branches[gl] = new_mpo
                else:
                    idx = new_grid.get_grid_ind(gl)
                    new_spine[idx].isel({new_spine.site_ind_id.format(idx): 0}, inplace=True)
                    # new_branches[gl] = None

        tnc.grid = new_grid
        tnc._branches = new_branches
        tnc._spine = new_spine
        return tnc


    def pad_branch_to_axes(self, new_axes, inplace=False):
        """ Map data on a branch to a new set of axes
        """
        raise NotImplementedError


    def take_outerproduct(self, other_gtn, virtual=False):
        """ take outerproduct with another gridTN
        """
        if isinstance(other_gtn, GridTN):
            self.set_branch(other_gtn.grid, other_gtn if virtual else other_gtn.copy())
        elif isinstance(other_gtn, GridTN1DComb):
            if [gr in self.grid.grids for gr in other_gtn.grid.grids]:
                assert([gr not in self.active_branches for gr in other_gtn.active_branches]),\
                    'cannot take outerproduct with overlapping grids'
                raise NotImplementedError
                # pad other to self, take outerproduct of the spines, combine branches
            elif [gr in other_gtn.grid.grids for gr in self.grid.grids]:
                assert([gr not in self.active_branches for gr in other_gtn.active_branches]),\
                    'cannot take outerproduct with overlapping grids'
                raise NotImplementedError
                # pad self to other, take outerproduct of the spines, combine branches
            else:
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

    # @profile
    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', add_cc=False, compress=False,
                                    compress_opts=None):
        """ apply elemental multiply operator to self (e.g. convolution or d_ijk fct)
        """
        add_cc = self.grid.has_basis_k_real()

        new_branches = {}
        for gk, b in self.branches.items():
            new_b = b.apply_elemental_multiply_op(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id, add_cc=add_cc,
                                                  compress=compress, compress_opts=compress_opts)
            new_branches[gk] = new_b

        spine = self.spine.copy() if self.spine is not None else None
        spine.mangle_inner_()

        if add_cc:    # gtnc -> gtnc + complex conj
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine.L):
                tens = spine[i]
                tens.direct_product(tens.conj(), inplace=True)

        gtn_mpo = self.__class__(self.grid, branches=new_branches, spine=spine, exponent=self._exponent, sign=self._sign)
        if compress:
            gtn_mpo.compress(inplace=True, compress_opts=compress_opts)

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

    # @profile
    def integrate(self, integ_axes=None, is_sqrt=False, ancilla_reindex=None, new_grid=None, new_ax_deriv_configs=None,
                  compress=False, compress_opts=None):
        """ integrate over the specified axes
            ancilla_reindex:  if is_sqrt, reindex ancilla inds such that they are not contracted together
                dict[old str: new str]
        """
        if new_ax_deriv_configs is None:
            new_ax_deriv_configs = self.ax_deriv_configs

        if new_grid is None:
            new_grid = self.grid
            # new_grid = self.grid.create_like(new_axes=out_axes) ## used in TN1D?

        if self.data is None:
            return new_grid.make_empty_gridTN(ax_deriv_configs=new_ax_deriv_configs)


        new_branches = {}
        new_spine = qtn.TensorNetwork([])

        self.collect_exponents()
        exponent = self._exponent if not is_sqrt else self._exponent * 2
        sign = self._sign if not is_sqrt else 1.

        spine = self.spine

        for gk, b in self.branches.items():

            gk_idx = self.grid.get_grid_ind(gk)

            if integ_axes is not None:
                active_axes = [ax for ax in integ_axes if ax in b.grid.axes]
            else:
                active_axes = b.grid.axes

            if len(active_axes) == 0:   # no part of branch is integrated over
                if is_sqrt:
                    raise NotImplementedError
                    ## need to square spine_tens, square the branch (like elemental multiply?)
                else:
                    new_ind = new_grid.get_grid_ind(gk)
                    spine_tens = spine[gk_idx].copy()
                    spine_tens.drop_tags()
                    spine_tens.add_tag(spine.site_tag_id.format(new_ind))
                    new_spine.add(spine_tens)
                    new_branches[gk] = b.copy()
                continue

            ## will be (at least partially) integrated
            if is_sqrt:
                tt_ind = spine.site_ind_id.format(gk_idx)
                ancilla_reindex = {tt_ind: tt_ind+'_'}
            else:
                ancilla_reindex = None

            new_b = b.integrate(integ_axes=active_axes, is_sqrt=is_sqrt, ancilla_reindex=ancilla_reindex,
                                compress_opts=compress_opts)

            if isinstance(new_b,float) or isinstance(new_b,complex) or isinstance(new_b,int):
                exponent += np.log10(np.abs(new_b))
                sign *= np.exp(-1.j * np.angle(new_b))
            elif isinstance(new_b, qtn.Tensor):
                spine_tens = spine[gk_idx].contract(new_b)
                if is_sqrt:
                    sqrt_tens = spine[gk_idx].reindex({t_ind: t_ind+'_' for t_ind in spine[gk_idx].inds})
                    sqrt_tens.conj(inplace=True)
                    spine_tens = spine_tens.contract(sqrt_tens)

                if isinstance(spine_tens, float) or isinstance(spine_tens,complex) or isinstance(spine_tens, int):
                    exponent += np.log10(np.abs(spine_tens))
                    sign *= np.exp(-1.j * np.angle(spine_tens))
                    continue

                spine_tens.drop_tags()
                if gk in new_grid.grids:
                    new_ind = new_grid.get_grid_ind(gk)
                    spine_tens.new_ind(spine.site_ind_id.format(new_ind), size=1, axis=-1)
                    spine_tens.add_tag(spine.site_tag_id.format(new_ind))
                    new_spine.add(spine_tens)
                else:   ## combine spine tens with other spine tensors
                    try:
                        # tens_str = spine.site_tag_id.format(new_ind)
                        # spine_tens.add_tag(tens_str)
                        # new_spine.add(spine_tens)
                        # print('new spine pre', new_spine, tens_str)
                        # print(new_spine.select_tensors(tens_str))
                        # out = new_spine.contract_tags([tens_str], inplace=True)
                        # print('new spine post', new_spine, out)
                        tmp = new_spine.select_tensors(spine.site_tag_id.format(new_ind))[0]
                        out = tmp.contract(spine_tens)
                        tmp.modify(data=out.data, inds=out.inds)
                    except NameError:
                        spine[gk_idx + 1].contract(spine_tens)

            elif isinstance(new_b, GridTN1D):
                new_branches[gk] = new_b
                spine_tens = spine[gk_idx].copy()
                if is_sqrt:
                    sqrt_tens = spine_tens.reindex({t_ind: t_ind + '_' for t_ind in spine_tens.inds})
                    sqrt_tens.conj(inplace=True)
                    spine_tens = spine_tens.contract(sqrt_tens)
                    spine_tens.fuse({tt_ind: (tt_ind, tt_ind+'_')}, inplace=True)
                new_spine.add(spine_tens)
            else:
                raise NotImplementedError


        if len(new_branches) > 0:
            if new_spine.num_tensors > 0:
                new_spine = new_spine.view_like(spine, inplace=True, L=new_grid.ndim)
            else:
                new_spine = None

            gtn_mpo = self.__class__(new_grid, branches=new_branches, spine=new_spine,
                                     exponent = exponent, sign = sign,
                                     ax_deriv_configs=new_ax_deriv_configs)
            # gtn_mpo.exponent = exponent
            # gtn_mpo.sign = sign
            return gtn_mpo
        else:       ## is number
            val = 10**exponent * sign
            if new_spine.num_tensors > 0:
                val *= np.asscalar(new_spine.contract().data)
            return val

    # @profile
    def meas_expec(self, obs_gtn: Union[dict['Grid','GridTN'], 'GridTN1DComb'], is_sqrt=False, integ_all=True):
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if self.data is None:
            return 0.0

        new_branches = {}
        new_spine = qtn.TensorNetwork([])
        exponent = self._exponent if not is_sqrt else self._exponent * 2
        sign = self._sign if not is_sqrt else 1.0

        # print('gtn', self)
        # print('obs', obs_gtn)

        if isinstance(obs_gtn, GridTN1DComb):
            obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
            exponent += obs_gtn._exponent
        else:
            obs_gtn_spine = None

        spine = self.spine.copy()
        if obs_gtn_spine is not None:
            obs_gtn.spine_ind_id = obs_gtn.spine_ind_id + '_'

        # for gk, b in self.branches.items():

        for gk in self.grid.grids:
            b = self.get_branch(gk)
            gk_idx = self.grid.get_grid_ind(gk)

            try:

                obs_gtn_gk = obs_gtn[gk]
                if b is None:
                    b = gk.get_ones_mps()

                expec = b.meas_expec(obs_gtn_gk, is_sqrt=is_sqrt)
                if isinstance(expec,float) or isinstance(expec,complex) or isinstance(expec,int):
                    exponent += np.log10(np.abs(expec))
                    sign *= np.exp(-1.j * np.angle(expec) )
                elif isinstance(expec, qtn.Tensor):
                    spine_tens = spine[gk_idx].contract(expec)
                    if obs_gtn_spine is not None:
                        spine_tens = spine_tens.contract(obs_gtn_spine[gk_idx])

                    if is_sqrt:
                        sqrt_tens = spine[gk_idx].reindex({t_ind: t_ind + '_' for t_ind in spine[gk_idx].inds})
                        sqrt_tens.conj(inplace=True)
                        spine_tens = spine_tens.contract(sqrt_tens)

                    spine_tens.new_ind(spine.site_ind_id.format(new_spine.num_tensors+1), size=1, axis=-1)
                    new_spine.add(spine_tens)
                else:
                    raise NotImplementedError

            except KeyError:
                if integ_all:
                    if b is None:
                        expec = 1.0
                    else:
                        expec = b.integrate(is_sqrt=is_sqrt)

                    if isinstance(expec, float) or isinstance(expec, complex) or isinstance(expec, int):
                        exponent += np.log10(np.abs(expec))
                        sign *= np.exp(-1.j * np.angle(expec))
                    elif isinstance(expec, qtn.Tensor):
                        ind = self.grid.grid_inds[gk]
                        spine_tens = spine[ind].contract(expec)
                        if obs_gtn_spine is not None:
                            spine_tens = spine_tens.contract(obs_gtn_spine[ind])
                        spine_tens.new_ind(spine.site_ind_id.format(new_spine.num_tensors + 1), size=1, axis=-1)
                        new_spine.add(spine_tens)
                else:
                    if b is not None:
                        new_branches[gk] = b.copy()

        # print('new branches', new_branches)
        # print('new spine', new_spine)

        if len(new_branches) > 0:
            if new_spine.num_tensors > 0:
                new_spine = new_spine.view_like(spine, inplace=True, L=new_spine.num_tensors)
            else:
                new_spine = None
            gtn_mpo = self.__class__(self.grid, branches=new_branches, spine=new_spine,
                                     exponent=exponent, sign=sign,
                                     ax_deriv_configs=self.ax_deriv_configs)
            # gtn_mpo.exponent = exponent
            # gtn_mpo.sign = sign
            return gtn_mpo
        else:
            val = 10 ** exponent * sign
            if new_spine.num_tensors > 0:
                val *= np.asscalar(new_spine.contract().data)
            return val


    def ovlp(self, other) -> Numeric:
        """ TODO: need to check this is correct more carefully
        """
        if self.data is not None and other.data is not None:
            out = None

            rename = False
            if other.spine_ind_id == self.spine_ind_id:
                self.spine_ind_id = self.spine_ind_id + '_tmp'
                rename = True

            for nb in range(self.grid.ngrids):
                ovlp_b = helper.ovlp( self.get_branch(nb), other.get_branch(nb).conj())
                if out is None:
                    out = qtn.tensor_contract(self.spine[nb], other.spine[nb], ovlp_b)
                else:
                    out = qtn.tensor_contract(out, self.spine[nb], other.spine[nb], ovlp_b)

            out *= 10**(self.exponent + other.exponent)

            if rename:
                self.spine_ind_id = self.spine_ind_id[:-4]

            return out
        else:
            return 0.0
