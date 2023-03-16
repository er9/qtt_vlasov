import numpy as np

from setup.configs import *
from setup.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT

import helper_quimb as helper
from gridTN import GridTN
from basis.basis_k import FourierBasis
from basis.basis_k import RealFourierBasis
from basis.basis_spatial import SpatialBasis

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from grid1D import Grid1D


""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
"""

########################################
## static methods to create GridTN1Ds ##
########################################

################
## common TNs ##
################

####################
## GridTN1D class ##
####################

class GridTN1D(GridTN):

    def __init__(self, grid: 'Grid1D', data: Optional[qtn.TensorNetwork1D]=None, ax_deriv_configs=None):
        """
        Attributes:
            grid:  GridLayout object
            data:  MPS or MPO object consistent with grid
        """
        super().__init__(grid, ax_deriv_configs=ax_deriv_configs)
        # self.grid = grid
        ## Attributes: grids, ngrids, _data, axes, axIDs, ax_map, ndim

        if data is not None:
            self.data = data
            # self._set_data(data)

    def _get_data(self) -> TN1Type:
        return self._data

    def _set_data(self, data):
        if data is None:
            self._data = data
        else:
            if isinstance(data, np.ndarray):
                try:
                    gtn = self.grid.map_state_to_mps(data)
                    new_data = gtn.data
                except AssertionError:
                    gtn = self.grid.map_operator_to_mpo(data)
                    new_data = gtn.data
            elif isinstance(data, dict):
                try:
                    gtn = self.grid.make_mps_ndim(data)
                    new_data = gtn.data
                except AssertionError:
                    gtn = self.grid.make_mpo_ndim(data)
                    new_data = gtn.data
            elif isinstance(data, qtn.TensorNetwork1D):
                new_data = data
            elif isinstance(data, GridTN1D):
                new_data = data.data
            else:
                raise TypeError(type(data))

            self._data = new_data
            self.data_type = type(new_data)
            # self.is_MPS = isinstance(new_data,qtn.MatrixProductState)

    data = property(fget=_get_data,fset=_set_data)


    def __getitem__(self,ind):
        """ imitates indexing MPS/MPO object
        """
        return self._data[ind]


    def get_inds_in_axis(self, ax: 'Axis', ax_ind=None) -> list:
        """ get inds in 1D TN corresponding to axis axID
        """
        return self.grid.get_inds_in_axis(ax, ax_ind)


    def shape(self) -> tuple:
        """ get grid shape
        """
        return self.grid.shape()


    @property
    def L(self):
        return self.data.L


    def create_like(self, new_data=None) -> 'GridTN1D':
        new_tn1D = self.__class__(self.grid, data=new_data, ax_deriv_configs=self.ax_deriv_configs)
        return new_tn1D

    def copy(self, deep=True) -> 'GridTN1D':
        new_tn1D = self.create_like()
        if self.data is not None:
            new_tn1D.data = self.data.copy() if (deep and self.data is not None) else self.data
        new_tn1D.is_constant = self.is_constant
        new_tn1D.constant_axes = self.constant_axes
        return new_tn1D

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN1D':
        new_tn1D = self if inplace else self.copy()
        new_tn1D.data.conj(mangle_inner=mangle_inner, inplace=True)
        return new_tn1D


    # def norm(self, is_sqrt=False) -> Numeric:
    #     if self.data is not None:
    #         out = self.integrate(is_sqrt=is_sqrt)
    #         if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
    #             out = np.real(out)
    #         return out

    def ovlp(self, other) -> Numeric:
        if self.data is not None and other.data is not None:
            out = helper.ovlp(self.data, other.data)
            return out
        else:
            return 0.0

    # def distance(self, other) -> Numeric:
    #     if self.data is not None and other.data is not None:
    #         out = helper.distance(self.data, other.data)
    #         return out
    #     elif self.data is None: # other data is not None
    #         return np.inf
    #     else:
    #         return np.nan

    def max_bond(self) -> int:
        if self.data is not None:
            return self.data.max_bond()
        else:
            return np.nan

    def entanglement_entropy_all(self) -> list[Numeric]:
        return helper.entanglement_entropy_all(self.data)

    def get_anchor_tens(self) -> qtn.Tensor:
        if self.data is not None:
            return self.data[self.get_anchor_ind()]

    def get_anchor_ind(self) -> int:
        return 0


    #######################################
    ## class methods to build common TNs ##
    #######################################

    @classmethod
    def get_ones_mps(cls, grid, site_ind_id='i({})', site_tag_id='X({})'):
        """ build ones vector mps on sepcified grid
        """
        # return cls(grid, grid.get_ones_mps(site_ind_id,site_tag_id))
        return grid.get_ones_mps(site_ind_id,site_tag_id)
        # mps_dict = {ax.axID: ax.get_ones_mps(site_ind_id, site_tag_id) for ax in grid.axes}
        # # return cls(grid, data=grid.layout.make_mps_ndim(mps_dict))
        # new_mps = grid.make_mps_ndim(mps_dict)
        # return cls(grid, data=new_mps)

    @classmethod
    def get_iden_mpo(cls, grid, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build identity mpo on specified grid
        """
        # return cls(grid, grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id))
        return grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        # mpo_dict = {ax.axID: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        #             for ax in grid.axes}
        # # return cls(grid, data=grid.layout.make_mpo_ndim(mpo_dict))
        # new_mpo = grid.make_mpo_ndim(mpo_dict)
        # return cls(grid, new_mpo)

    @classmethod
    def get_select_elem_mps(cls, grid, inds, site_ind_id='i({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        # return cls(grid, grid.get_select_elems_mps(inds, site_ind_id, site_tag_id))
        return grid.get_select_elems_mps(inds, site_ind_id, site_tag_id)
        # axes = grid.axes
        # assert (len(axes) == len(inds)), 'number of inds needs to match number of dims'
        # mps_dict = {axes[x].axID: axes[x].get_select_elem_mps(inds[x], site_ind_id, site_tag_id)
        #             for x in range(len(axes))}
        # return cls(grid, data=grid.layout.make_mps_ndim(mps_dict))

    @classmethod
    def get_select_elem_mpo(cls, grid, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        # return cls(grid, grid.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id))
        return grid.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id)
        # axes = grid.axes
        # assert (len(axes) == len(inds)), 'number of inds needs to match number of dims'
        # mpo_dict = {axes[x].axID: axes[x].get_select_elem_mpo(inds[x], upper_ind_id, lower_ind_id, site_tag_id)
        #             for x in range(len(axes))}
        # # return cls(grid, data=grid.layout.make_mpo_ndim(mpo_dict))
        # new_mpo = grid.make_mpo_ndim(mpo_dict)
        # return cls(grid, new_mpo)

    @classmethod
    def from_dense_state(cls, data, grid, site_ind_id='i({})', site_tag_id='T({})', split_opts=None):
        mps = grid.map_state_to_mps(data, site_ind_id, site_tag_id, split_opts=split_opts)
        return cls(grid, data=mps)

    @classmethod
    def from_dense_operator(cls, data, grid, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            split_opts=None, **kwargs):
        mpo = grid.map_operator_to_mpo(data, upper_ind_id, lower_ind_id, site_tag_id, split_opts=split_opts)
        return cls(grid, data=mpo)


    def get_data(self, ax_order=None, ax_select: Optional[dict[int]] = None, pad_data=False):
        """ ax_select:  select element along each axis
        """

        if self.data is None:
            return None


        L = self.data.L
        if isinstance(self.data, qtn.MatrixProductState):
            shared, not_shared = self.data[0].filter_bonds(self.data[1])
            ancilla_left_inds = tuple([ix for ix in not_shared if ix != self.data.site_ind(0)])
            ancilla_left = tuple([self.data[0].ind_size(ix) for ix in ancilla_left_inds])

            shared, not_shared = self.data[-1].filter_bonds(self.data[-2])
            ancilla_right_inds = tuple([ix for ix in not_shared if ix != self.data.site_ind(L-1)])
            ancilla_right = tuple([self.data[-1].ind_size(ix) for ix in ancilla_right_inds])

            out = self.grid.map_mps_to_state(self, ax_select=ax_select,
                                              ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                              ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

            if ax_order is not None:
                axT = [np.nonzero(np.array(self.grid.axes,dtype=object) == ax)[0].item() for ax in ax_order]
                out.transpose(axT)

            return out

        elif isinstance(self.data, qtn.MatrixProductOperator):
            shared, not_shared = self.data[0].filter_bonds(self.data[1])
            ancilla_left_inds = tuple([ix for ix in not_shared if
                                       not (ix == self.data.upper_ind(0) or ix == self.data.lower_ind(0)) ])
            ancilla_left = tuple([self.data[0].ind_size(ix) for ix in ancilla_left_inds])

            shared, not_shared = self.data[-1].filter_bonds(self.data[-2])
            ancilla_right_inds = tuple([ix for ix in not_shared if
                                       not (ix == self.data.upper_ind(L-1) or ix == self.data.lower_ind(L-1)) ])
            ancilla_right = tuple([self.data[-1].ind_size(ix) for ix in ancilla_right_inds])

            out = self.grid.map_mpo_to_operator(self, ax_select=ax_select,
                                                 ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                                 ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

            if ax_order is not None:
                axTo = [np.nonzero(self.grid.axes == ax) for ax in ax_order]
                axTi = [len(axTo) + x for x in axTo]
                out.transpose(axTo+axTi)

            return out

        elif self.data is None:
            return None



    #######################
    ## MPS/MPX functions ##
    #######################

    def _parse_input_TN1D(self, other):
        if isinstance(other, type(self)):
            assert(other.grid == self.grid), 'self and other gridTN need to live on the same grids'
            other = other.data
        else:
            mpx_shape = tuple([other.phys_dim(i) for i in range(other.L)])
            assert(mpx_shape == self.grid.shape), 'self and mpo need to have the same shape'
        return other


    # @profile
    def apply(self, other, inplace=False, zipup=False, compress=False, compress_opts=None, add_cc=False, **kwargs) -> 'GridTN1D':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
        """
        # zipup = True
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if other is None:
            return None

        grid_mpx1 = self if inplace else self.copy()
        grid_mpx1.is_constant *= other.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in other.constant_axes]

        ## pad grid if necessary
        if grid_mpx1.grid != other.grid:
            if set(other.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                # print('mps1 > mps2')
                other = grid_mpx1.grid.pad_gtn_to_grid(other)
            elif set(grid_mpx1.grid.axes).issubset(set(other.grid.axes)):
                # print('mps1 < mps2')
                grid_mpx1 = other.grid.pad_gtn_to_grid(grid_mpx1)
                print('warning: elemental multiply not an inplace operation')
            else:
                # print('grid 1 axes', grid_mpx1.grid.axes, other.grid.axes)
                raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {other.grid}')

        # print('padded other', other)

        # other = self._parse_input_TN1D(other)
        other = other.data
        assert (isinstance(other, qtn.MatrixProductOperator)), 'other needs to be an MPO type'

        grid_data = grid_mpx1.data
        if add_cc:
            # if not(len(grid_data.shape) == len(other.shape) or len(grid_data.shape)*2 == len(other.shape)):
            grid_data = RealFourierBasis.convert_mps_to_full(grid_mpx1.grid, grid_data)

        # print('apply 1D self', grid_data)
        # print('apply 1D other', other)

        # print('1D apply, zipup?', zipup, compress)
        if zipup:  #  or compress:  ### and compress?
            form = compress_opts.get('form', 'right') if compress_opts is not None else 'right'

            # DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
            # compress_back = compress  # (DMAX is not None)
            # print('zipup', compress_back)

            # new_mpx = helper.apply_zipup(other, grid_mpx1.data, compress=compress, compress_opts=compress_opts)
            new_mpx = helper.apply_zipup(other, grid_data, compress=compress, compress_opts=compress_opts)
            grid_mpx1.canon_site = 0 if form=='right' else grid_mpx1.L-1
            # print(new_mpx.calc_current_orthog_center(), grid_mpx1.canon_site)

            # if compress_opts['max_bond'] is None:
            #     ## canonicalize but don't compress
            #     new_mpx = helper.apply_zipup(mpo, grid_mpx1.data, compress=False, compress_opts=compress_opts)
            # else:
            #     ## canonicalize and compress
            #     new_mpx = helper.apply_zipup(mpo, grid_mpx1.data, compress=True, compress_opts=compress_opts)
        else:
            ## no compression -- simple apply method
            # new_mpx = helper.apply(other, grid_mpx1.data)
            new_mpx = helper.apply(other, grid_data)
            if compress:
                helper.compress(new_mpx, compress_opts=compress_opts)

        # new_mpx = helper.apply(mpo, grid_mpx1.data) # , compress=compress, compress_opts=compress_opts)
        # if compress:
        #     helper.compress(new_mpx, compress_opts=compress_opts)

        ## not inplace update of qtn.TensorNetwork obj.
        grid_mpx1.data = new_mpx
        return grid_mpx1


    def add(self, gtn_mpx2, zipup=False, inplace=False, compress=False, compress_opts=None, **kwargs) -> 'GridTN1D':
        """ Add other grid_mpx (of the same type and on the same grid) to self
        """
        grid_mpx1 = self if inplace else self.copy(deep=True)

        if gtn_mpx2 is None:
            return grid_mpx1

        elif isinstance(gtn_mpx2, GridTN1D):
            if gtn_mpx2.data is None:
                return grid_mpx1

            if gtn_mpx2.grid != self.grid:
                if set(gtn_mpx2.grid.axes).issubset(self.grid.axes):
                    gtn_mpx2 = self.grid.pad_gtn_to_grid(gtn_mpx2)
                else:
                    raise ValueError(f'gtn_mpx2 on grid {gtn_mpx2.grid} not compatible with self on grid {self.grid}')

            if grid_mpx1.data is None:
                grid_mpx1.data = gtn_mpx2.data.copy()
                return grid_mpx1
                # if inplace:
                #     grid_mpx1.data = gtn_mpx2.data
                #     return grid_mpx1
                # else:
                #     return gtn_mpx2


        # print('gtn_mpx2', gtn_mpx2)
        mpx2 = self._parse_input_TN1D(gtn_mpx2)

        if compress_opts is None:
            compress_opts = {}

        if mpx2 is None:
            return grid_mpx1

        if isinstance(grid_mpx1.data, qtn.MatrixProductOperator):
            assert (isinstance(mpx2, qtn.MatrixProductOperator)), 'other needs to be an MPO type'
            helper.add_MPO(grid_mpx1.data, mpx2, inplace=True, zipup=zipup,
                           compress=compress, compress_opts=compress_opts)
        elif isinstance(grid_mpx1.data, qtn.MatrixProductState):
            assert (isinstance(mpx2, qtn.MatrixProductState)), 'other needs to be an MPS type'
            helper.add_MPS(grid_mpx1.data, mpx2, inplace=True, zipup=zipup,
                           compress=compress, compress_opts=compress_opts)
        elif grid_mpx1.data is None:
            grid_mpx1.data = mpx2.copy() if mpx2 is not None else None
        else:
            raise TypeError(f'self needs to be an MPO, MPS, or NoneType, not {type(self.data)}')

        if compress:
            if compress_opts.get('form', 'right') == 'right':
                self.canon_site = 0
            else:
                self.canon_site = self.L - 1
        else:
            self.canon_site = -1

        grid_mpx1.is_constant *= gtn_mpx2.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in gtn_mpx2.constant_axes]

        return grid_mpx1


    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTN':

        gtn = self if inplace else self.copy()
        if sub_gtn_mpx2 is None:
            return gtn

        gtn_mpx = gtn.data
        sub_mpx = sub_gtn_mpx2.data
        if sub_mpx is None:
            return gtn

        sub_inds = []
        for ax in sub_gtn_mpx2.grid.axes:
            sub_inds += self.grid.get_inds_in_axis(ax)

        min_ind = np.min(sub_inds)
        max_ind = np.max(sub_inds)

        gtn.data = helper.add_submpx(gtn_mpx, sub_mpx, (min_ind, max_ind+1), open_bc=open_bc, inplace=True, compress=compress,
                                    compress_opts=compress_opts)
        return gtn


    def scalar_multiply(self,scalar_const,inplace=False) -> 'GridTN1D':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.scalar_multiply(grid_mpx1.data,scalar_const,inplace=True)
        return grid_mpx1


    # def elemental_multiply(self, gtn_mps2, inplace=False, compress=False, compress_opts=None) -> 'GridTN1D':
    #
    #     if self.data is None:
    #         return self
    #
    #     grid_mpx1 = self if inplace else self.copy()
    #
    #     if isinstance(gtn_mps2, qtn.MatrixProductState):
    #         mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
    #         assert(self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
    #         gtn_mps2 = self.create_like()
    #         if self.data_type == DataType.MPS:   ## convert gtn_mps2 to MPO
    #             grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
    #         else:   ## use fact that self is MPO. apply method should take care of it
    #             grid_mpo2 = grid_mpx1
    #             grid_mpx1 = gtn_mps2
    #
    #     elif isinstance(gtn_mps2, qtn.MatrixProductOperator):
    #         mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
    #         assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
    #         grid_mpo2 = self.create_like()
    #         grid_mpo2.data = gtn_mps2
    #
    #     elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type==DataType.MPS:
    #         assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
    #         gtn_mps2 = gtn_mps2
    #         if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
    #             grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
    #         else:   ## use fact that self is MPO. apply method should take care of it
    #             grid_mpo2 = grid_mpx1
    #             grid_mpx1 = gtn_mps2
    #
    #     elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type==DataType.MPO:
    #         assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
    #         grid_mpo2 = gtn_mps2
    #
    #     else:
    #         raise TypeError(f'other must be GridTN with MPS or MPO data type, not {type(gtn_mps2)}')
    #
    #     grid_mpx1.apply(grid_mpo2, inplace=True, compress=compress, compress_opts=compress_opts)
    #     return grid_mpx1


    # def xmultiply(self, x_axes: Sequence['Axis'], x_power: int = 1, offsets: dict[Any, Numeric] = 0.0,
    #               scales: dict[Any, Numeric] = 1.0, inplace=False, compress=False, compress_opts=None) -> 'GridTN1D':
    #
    #     if self.data is None:
    #         return self
    #
    #     grid_mpx1 = self if inplace else self.copy()
    #     xmult_mpo = grid_mpx1.grid.get_xmultiply_mpo(x_axes, x_power=x_power, offsets=offsets, scales=scales)
    #
    #     # xmult_mpo_data = xmult_mpo.get_data()
    #     # plt.figure()
    #     # plt.imshow(xmult_mpo_data.reshape(64*16,64*16))
    #     # plt.show()
    #
    #     grid_mpx1.apply(xmult_mpo, inplace=True)  # , compress=compress, compress_opts=compress_opts)
    #
    #     # ax = x_axes[0]
    #     # xmult_mps = ax.map_state_to_mps(ax.xpts)
    #     # xmult_gtn = self.grid.make_mps_ndim({ax: xmult_mps})
    #     # print('xmult mps', xmult_mps.max_bond())
    #     # out_mps = helper.elemental_multiply(xmult_gtn.data, grid_mpx1.data, compress=False)
    #     # grid_mpx1.data = out_mps
    #
    #     if compress:
    #         grid_mpx1.compress(inplace=True, compress_opts=compress_opts)
    #     return grid_mpx1


    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTN1D':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if i is None:
            if form == 'right':
                i = 0
            elif form == 'left':
                i = self.L - 1

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.canonize(grid_mpx1.data, scale=scale, i=i, cur_orthog=cur_orthog)
        grid_mpx1.canon_site = i
        return grid_mpx1


    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True):
        canon_i = self.get_canon_site_from_axes(axes)
        return self.canonize(inplace=inplace, scale=scale, i=canon_i)


    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        """ site of mixed canonical form; included within inds in axes
        """
        ax_inds = np.sort([self.grid.axes.index(ax) for ax in axes])
        if len(ax_inds) == 1 or all(np.diff(ax_inds) == 1):
            if ax_inds[-1] != self.grid.ndim - 1:
                canon_i = sum([ax.L for ax in self.grid.axes[:1 + ax_inds[-1]]])
                canon_i -= 1
            else:   ## last (right-most) axis in grid
                canon_i = self.L - sum([ax.L for ax in axes]) # - 1
            print('canon i', axes, canon_i)
        else:
            raise ValueError('axes must either all be on left or right of MPS')
        return canon_i


    def compress(self, inplace=True, verbose=False, canonize=True, compress_opts=None, **kwargs) -> 'GridTN1D':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy(deep=True)
        grid_mpx1.data = helper.compress(grid_mpx1.data, canonize=canonize, compress_opts=compress_opts, verbose=verbose)

        if grid_mpx1.data is not None:
            if compress_opts is None:
                grid_mpx1.canon_site = grid_mpx1.L - 1
            else:
                if compress_opts.get('do_midpt', False):
                    grid_mpx1.canon_site = -1
                else:
                    if compress_opts.get('form', 'right') == 'right':
                        grid_mpx1.canon_site = 0
                    else:
                        grid_mpx1.canon_site = grid_mpx1.L-1

        return grid_mpx1


    # #####################
    # ## data processing ##
    # #####################
    #
    # def get_mpo_inverse(self,mpo,boundary_conditions=None,use_dmrg=False):
    #     """ mpo:  operator to take the inverse of
    #         boundary condition:  at the moment, an m x q**K**L array such that
    #                              for each m, vector.T * data = mth boundary condition
    #         expensive method right now -- take inverse directly
    #         optimization method:
    #     """
    #
    #     return self.grid.get_mpo_inverse(mpo, boundary_conditions, use_dmrg)


    # ################
    # ## processing ##
    # ################
    #
    # def interpolate_data(self, nsites, inplace=False):
    #     """ extend grid s.t. data is interpolated between existing grid points
    #     """
    #     raise NotImplementedError
    #
    #
    # def gaussian_smooth_data(self, inplace=False, sigma=0.3, proc_axes=None, mode='wrap', **compress_opts):
    #     """ process the data in some way
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     blur_mpo = self.grid.gaussian_smooth_mpo(sigma, proc_axes, mode)
    #
    #     if blur_mpo is not None:
    #         gtn.apply(blur_mpo, compress=False)  ## not a deep inplace operation
    #         gtn.compress(inplace=True, **compress_opts)
    #         # scalar_mps_field = helper.apply(blur_mpo, self.data, compress=compress,
    #         #                                 **self.grid.get_compress_opts(compress))
    #     # else:
    #         # scalar_mps_field = self.data if inplace else self.data.copy()
    #
    #     return gtn
    #
    #
    # def apply_absorbing_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     absorb_bc_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     if absorb_bc_mpo is not None:
    #         gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn
    #
    #
    # def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply reflecting_v bc to state.
    #         apply to derivative MPO?
    #     """
    #     gtn = self if inplace else self.copy()
    #     reflect_bc_mpo = self.grid.reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     if reflect_bc_mpo is not None:
    #         gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn


    ###############################
    ## application of operations ##
    ###############################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        assert (self.data_type == DataType.MPS), f'gtn needs to MPS type not {self.data_type}'

        gtn = self if inplace else self.copy(deep=False)

        mps: qtn.MatrixProductState = self.data if inplace else self.data.copy()
        if lower_ind_id == mps.site_ind_id or upper_ind_id == mps.site_ind_id:
            mps.site_ind_id = lower_ind_id + '_tmp_'
        site_ind_id = mps.site_ind_id
        site_tag_id = mps.site_tag_id

        # # L, q = self.L
        # out = qtn.TensorNetwork([])
        # for ax in self.grid.axes:
        #     idxs = self.grid.get_inds_in_axis(ax)
        #     q = ax.q
        #     for i in idxs:
        #         tens = mps[i]
        #         d_ijk = qtn.tensor_core.COPY_tensor(q, (site_ind_id.format(i), lower_ind_id.format(i),
        #                                                 upper_ind_id.format(i)),
        #                                             tags=(site_tag_id.format(i),))
        #         out.add_tensor( tens.contract(d_ijk) )
        # # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        # out = out.view_as(qtn.MatrixProductOperator, inplace=True, L=self.L, cyclic=False, site_tag_id=site_tag_id,
        #                                              upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)

        out = qtn.TensorNetwork([])
        for i in range(self.grid.L):
            q = self.grid.shape()[i]
            tens = mps[i]
            d_ijk = qtn.tensor_core.COPY_tensor(q, (site_ind_id.format(i), lower_ind_id.format(i),
                                                    upper_ind_id.format(i)),
                                                tags=(site_tag_id.format(i),))
            out.add_tensor(tens.contract(d_ijk))

        # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        out = out.view_as(qtn.MatrixProductOperator, inplace=True, L=self.L, cyclic=False, site_tag_id=site_tag_id,
                          upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        out.exponent = self.data.exponent

        gtn.data = out
        return gtn


    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', add_cc=False, compress=False,
                                    compress_opts=None):
        """ Use 3-leg elemental multiply TN, apply to self to turn into mpo
            for elemental multiplication (with some other, yet to be specified gmps)
            assumes indices of self goes from 0 to grid.L-1 (no missing indices)
            lower_ind_id, upper_ind_id = desired formats for output MPO
        """
        if self.data is None:   # zero
            return self.create_like(new_data=None)

        assert self.data_type==DataType.MPS, 'needs to be MPS to make MPO'
        new_mpo = qtn.TensorNetwork([])
        contract_ind = self.data.site_ind_id

        add_cc = self.grid.has_basis_k_real() or add_cc

        fix_tag = None
        if contract_ind == upper_ind_id:
            upper_ind_id += '_tmp'
            fix_tag = 'U'
        elif contract_ind == lower_ind_id:
            lower_ind_id += '_tmp'
            fix_tag = 'L'

        elem_mult_tn = self.grid.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=contract_ind,
                                                           out_ind_id=upper_ind_id, site_tag_id='_tmp{}')
        self_data = self.data
        elem_mult_tn_data = elem_mult_tn.data

        if add_cc:
            # conj_gtn = self.conj(inplace=False)
            # shift_inds = [1 if ax.is_real_k() else 0 for ax in self.grid.axes]
            # shift_mpo_gtn = self.grid.get_shift_mpo(shift_inds)
            # conj_gtn = conj_gtn.apply(shift_mpo_gtn)

            self_data = RealFourierBasis.convert_mps_to_full(self.grid, self_data, anc_ind=2)
                ### TODO:  need to deal with ancillas from comb branches (summing branches but not spine) move to Comb?
            # self_data = self_data.copy()
            #
            # for ax in self.grid.axes:
            #     if not ax.is_real_k():  continue
            #
            #     if ax.map.is_flipped():
            #         self_data[self.L - 1].new_ind(f'anc_elem_mult_1_{ax}')
            #     else:
            #         self_data[0].new_ind(f'anc_elem_mult_1_{ax}')
            #
            # conj_data = self_data.conj()  ## should be ok to take complex conj. of entire state
            # self_data = helper.add_MPS(self_data, conj_data, inplace=True, compress=False)

        for x in range(self.grid.L):
            tens_x1 = self_data[x]
            tens_x2 = elem_mult_tn_data[x]
            new_tens = tens_x1.contract(tens_x2)    ## includes ancilla inds
            new_tens.drop_tags(tags=tens_x2.tags)
            if fix_tag == 'L':
                new_tens.reindex({lower_ind_id.format(x): lower_ind_id[:-4].format(x)}, inplace=True)
            elif fix_tag == 'U':
                new_tens.reindex({upper_ind_id.format(x): upper_ind_id[:-4].format(x)}, inplace=True)

            new_mpo.add(new_tens)

        new_mpo.fuse_multibonds(inplace=True)

        if   fix_tag == 'U':  upper_ind_id = upper_ind_id[:-4]
        elif fix_tag == 'L':  lower_ind_id = lower_ind_id[:-4]
        new_mpo = new_mpo.view_as(qtn.MatrixProductOperator, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                  site_tag_id=self.data.site_tag_id, inplace=True, cyclic=False, L=self.data.L)
        new_mpo.exponent = self.data.exponent + elem_mult_tn.data.exponent

        gmpo = self.create_like(new_data=new_mpo)
        # gmpo.data = new_mpo

        if compress:
            gmpo.compress(compress_opts=compress_opts)

        return gmpo


    # def apply_xmultiply_mpo(self, x_axes=None, inplace=False, compress=True, **compress_opts):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...) or x*z*f(x,y,z...)
    #     """
    #     gtn = self if inplace else self.copy()
    #     x_mpo = self.grid.get_xmultiply_mpo(x_axes)
    #     gtn.apply(x_mpo, compress=compress, **compress_opts)
    #     return gtn
    #
    #
    # def take_firstderivative(self, ax, bc, deriv_params=None, recalc=False, inplace=False, compress=True,
    #                          **compress_opts):
    #     """ df/dx_i
    #         bc = boundary condition
    #     """
    #     gtn = self if inplace else self.copy()
    #     if deriv_params is None:     deriv_params = {}
    #     ddx_mpo = self.grid.get_firstderivative_mpo(ax, bc, **deriv_params)
    #     gtn.apply(ddx_mpo, inplace=True, compres=compress, **compress_opts)
    #     return gtn
    #
    #
    # def take_secondderivative(self, ax1, ax2, bc1, bc2, inplace=False, recalc=False, compress=True, **compress_opts):
    #     """ d^2f/dx^2 {x} + d^2f/dy^2 {y} + d^2f/dz^2 {z}
    #     """
    #     gtn = self if inplace else self.copy()
    #     d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, bc1, bc2)
    #     gtn.apply(d2_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn


    ###############
    ## integration/partial integration ##
    ################

    def apply_mps(self, mps, inplace=False, compress=True, compress_opts=None):
        """ Apply MPS on top of an MPS -> # or MPO -> MPS
        """
        if isinstance(mps, GridTN1D):
            mps = mps.data

        assert isinstance(mps, qtn.MatrixProductState), 'supplied TN must be MPS'

        if isinstance(self.data, qtn.MatrixProductOperator):
            gtn = self if inplace else self.copy()
            helper.mpo_transpose( gtn.data, inplace=True)
            out = helper.apply(gtn.data, mps, compress=compress, compress_opts=compress_opts)
            gtn.data = out
            return gtn
        elif isinstance(self.data, qtn.MatrixProductState):
            out = helper.ovlp(self.data, mps)
            return out
        else:
            raise NotImplementedError


    def meas_expec(self, obs_gtn: 'GridTN1D', integ_axes=None, is_sqrt=False, ancilla_reindex=None,
                   new_grid: 'Grid1D' = None, new_ax_deriv_configs=None, compress=False, compress_opts=None):
        """ integrate over specified dimensions. returns scalar if axes='all'
            otherwise, returns a Field object with reduced dimensionality
            inplace is dummy parameter.
            ancilla_reindex:  dict[old_anc: new_anc] for reindexing MPS when is_sqrt==True
                ancilla must be on the anchor tensor (indexed by 0)
                new and old ancilla are fused together -> old ancilla name
        """
        if integ_axes is None:
            integ_axes = self.grid.axes
            out_axes = []
        else:
            out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]

        if new_grid is None and len(out_axes) > 0:
            new_grid = self.grid.create_like(new_axes=out_axes)

        if new_ax_deriv_configs is None and new_grid is not None:
            new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}

        if self.data is None:
            if len(out_axes) > 0:
                return new_grid.make_empty_gridTN(ax_deriv_configs=new_ax_deriv_configs)
            else:
                return np.nan

        if obs_gtn is not None and obs_gtn.data_type == DataType.MPS:
            obs_gtn = obs_gtn.apply_elemental_multiply_op()

        assert self.data_type == DataType.MPS, 'integrate only defined for MPS GridTNs'

        ## no integration
        if len(integ_axes) == 0:
            if is_sqrt:
                gtn_cc = self.conj()
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)
                tmp = gtn_cc.apply_elemental_multiply_op()

                if obs_gtn is not None:
                    out = self.apply(obs_gtn, compress=False)
                else:
                    out = self
                out = out.apply(tmp, compress=compress, compress_opts=compress_opts)
                return out
            else:
                if obs_gtn is not None:
                    return self.apply(obs_gtn, compress=compress, compress_opts=compress_opts)
                else:
                    return self.copy()

        ## integrate all
        elif len(out_axes) == 0:
            integ = self.grid.get_integrals_mpx(axes=None, is_sqrt=False)

            if is_sqrt:
                # ## actually correct (don't need convolution bc of cc for k-space)
                # self_conj = self.conj(mangle_inner=True)
                # # dx = np.asscalar( np.prod([ax.get_integral_weight() for ax in integ_axes]) )
                # # helper.scalar_multiply(data_conj, dx, inplace=True)
                # integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
                # integ_weights = self.grid.make_mps_ndim(integ_weights)
                # if np.isscalar(integ_weights):
                #     self_conj.scalar_multiply(integ_weights, inplace=True)
                # else:
                #     integ_weights = integ_weights.mps_to_diag_mpo()
                #     self_conj.apply(integ_weights, inplace=True)
                #
                # data_conj = self_conj.data
                # if ancilla_reindex:  # for compatibility with comb
                #     data_conj.reindex(ancilla_reindex, inplace=True)
                # # print('self sqrt integ exp', self.data.exponent, data_conj.exponent)
                # result = helper.ovlp(self.data, data_conj)

                gtn_cc = self.conj()
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)

                integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
                integ_weights = self.grid.make_mps_ndim(integ_weights)
                # print('integ weights', integ_axes, integ_weights)

                if np.isscalar(integ_weights):
                    gtn_cc.scalar_multiply(integ_weights, inplace=True)
                else:
                    integ_weights = integ_weights.mps_to_diag_mpo()
                    gtn_cc.apply(integ_weights, inplace=True)

                if obs_gtn is not None:
                    result = helper.expectation_value(self.data, obs_gtn.data, bra=gtn_cc.data)
                else:
                    result = helper.ovlp(self.data, gtn_cc.data)
            else:

                if obs_gtn is not None:
                    result = helper.expectation_value(self.data, obs_gtn.data, bra=integ.data)
                else:
                    result = helper.ovlp(self.data, integ.data)

            return result  ## a scalar or Tensor (if there are ancilliary bonds)

        else:
            integ_gtn = self.grid.get_integrals_mpx(integ_axes) # .data
            remaining_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
            remaining_inds = []
            for ax in remaining_axes:
                remaining_inds += self.grid.get_inds_in_axis(ax)

            if new_grid is None:
                new_grid = self.grid.create_like(remaining_axes, self.grid.gridID + '_integ')

            if is_sqrt:
                # # tmp = helper.mps_to_diag_mpo(self.data.conj())
                # self_conj = self.copy()
                # self_conj.data.conj(inplace=True, mangle_inner=True)
                # if ancilla_reindex:
                #     self_conj.get_anchor_tens().reindex(ancilla_reindex, inplace=True)
                # tmp = self_conj.apply_elemental_multiply_op()
                # integ_mps = helper.apply(integ_mps, tmp.data)
                # if ancilla_reindex:
                #     fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
                #     integ_mps[self.get_anchor_ind()].fuse(fuse_anc, inplace=True)

                gtn_cc = self.conj()
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)
                tmp = gtn_cc.apply_elemental_multiply_op()
                integ_gtn = tmp.apply(integ_gtn, inplace=False)

                # if ancilla_reindex:
                #     fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
                #     integ_mps[self.get_anchor_ind()].fuse(fuse_anc, inplace=True)

                # plt.figure()
                # plt.imshow(gtn_cc.get_data()[:,:,0,0])
                # plt.colorbar()
                # plt.title('cc')
                #
                # plt.figure()
                # plt.imshow(self.get_data()[:,:,0,0])
                # plt.colorbar()
                # plt.title('self')
                # plt.show()

            integ_mps = integ_gtn.data #.copy()

            if obs_gtn is None:
                integ_mps.lower_ind_id = '_tmp{}_'
                integ_mps.upper_ind_id = self.data.site_ind_id
                comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace

                tags_list = [integ_mps.site_tag_id, self.data.site_tag_id]
                result = qtn.TensorNetwork((integ_mps, comp_mps),
                                           virtual=True, check_collisions=False)
                result.exponent = integ_mps.exponent + comp_mps.exponent
            else:
                integ_mps.lower_ind_id = '_tmpU{}_'
                integ_mps.upper_ind_id = self.data.site_ind_id
                # integ_mps.site_tag_id = integ_mps.site_tag_id + 'I'
                obs_mpo = obs_gtn.data.copy()
                obs_mpo.mangle_inner_()
                obs_mpo.lower_ind_id = '_tmp{}_'
                obs_mpo.upper_ind_id = '_tmpU{}_'
                # obs_mpo.site_tag_id = obs_mpo.site_tag_id + 'O'
                comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace

                tags_list = [integ_mps.site_tag_id, self.data.site_tag_id, obs_mpo.site_tag_id]
                result = qtn.TensorNetwork((integ_mps, obs_mpo, comp_mps),
                                           virtual=True, check_collisions=False)
                result.exponent = integ_mps.exponent + comp_mps.exponent + obs_mpo.exponent

            ## set contraction direction to minimize costs
            if self.grid.axes[-1] not in remaining_axes:
                site_range = range(integ_mps.L - 1, -1, -1)
                from_right = True
            else:
                site_range = range(integ_mps.L)
                from_right = False

            new_exponent = result.exponent
            for x in site_range:
                # tens_tags = [integ_mps.site_tag_id.format(x), self.data.site_tag_id.format(x)]
                tens_tags = [tag.format(x) for tag in tags_list]

                try:
                    if integ_mps.phys_dim(x, 'upper') == 1:
                        ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
                        if from_right:
                            # neighbor_tags = [integ_mps.site_tag_id.format(x - 1),
                            #                  self.data.site_tag_id.format(x - 1)]
                            neighbor_tags = [tag.format(x - 1) for tag in tags_list]
                        else:
                            # neighbor_tags = [integ_mps.site_tag_id.format(x + 1),
                            #                  self.data.site_tag_id.format(x + 1)]
                            neighbor_tags = [tag.format(x + 1) for tag in tags_list]
                        # print('select', result.select_tensors(tens_tags + neighbor_tags, which='any'))
                        result = result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
                        if isinstance(result, qtn.Tensor):
                            tens = result
                        else:
                            tens = result.select_tensors(tens_tags, which='any')[0]
                        tens.isel({integ_mps.upper_ind(x): 0}, inplace=True)
                        tens.drop_tags(tens_tags)
                        # print('tens', tens, x)
                    else:
                        raise TypeError

                except TypeError:   ## don't integrate over bonds; end of chain
                    result = result.contract(tens_tags, inplace=True, which='any')

                if isinstance(result, qtn.Tensor):
                    ## already caught integ_axes = all case, so will always need to be in TN form
                    result = qtn.TensorNetwork([result])
                    result.exponent = new_exponent

                result.fuse_multibonds(inplace=True)

            if self.data.site_tag_id != integ_mps.site_tag_id:
                result.drop_tags([integ_mps.site_tag_id.format(x) for x in site_range])

            ## renumber tensors
            if not self.grid.layout_type == LayoutType.PARALLEL_GROUP:
                remaining_inds = np.sort(remaining_inds)
                helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
                                    self.data.site_ind_id, inplace=True)

            ## convert to MPS
            result.view_like(self.data, L=new_grid.L, inplace=True)
            # result.exponent = integ_mps.exponent + comp_mps.exponent

            ## convert to GTN object
            gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)

            if compress:
                gtn_result.compress(compress_opts=compress_opts)

            return gtn_result


    def project(self, obs_gtn, proj_axes, new_grid=None, canonize=True, compress=False, compress_opts=None,
                ancilla_reindex = None, new_ax_deriv_configs=None, **kwargs) -> Optional['GridTN1D']:
        """ project obs_gtn onto manifold of self
        """
        if self.data is None:
            return None

        gtn = self.copy()
        if canonize:
            gtn = gtn.canonize_axes(proj_axes, inplace=True, scale=False)
            ## need projection tensors to actually be in canonical form (no coeffs)

        if new_grid is not None:
            # select_axes = [ax for ax in gtn.grid.axes if ax not in proj_axes]
            # print('grid select_axes', select_axes)
            new_grid = gtn.grid.get_subgrid(proj_axes, new_gridID=self.grid.gridID + '_proj')

        if new_ax_deriv_configs is None and new_grid is not None:
            new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}


        print('project check orthog')
        helper.check_orthog(gtn.data)
        print('gtn data', helper.norm(gtn.data))

        integ_axes = [ax for ax in self.grid.axes if ax not in proj_axes]
        integ_inds = []
        for ax in integ_axes:
            integ_inds += self.grid.get_inds_in_axis(ax)
        print('integ inds', integ_inds)

        remaining_axes = proj_axes
        remaining_inds = []
        for ax in remaining_axes:
            remaining_inds += self.grid.get_inds_in_axis(ax)

        gtn_cc = gtn.conj(mangle_inner=False, inplace=False)
        tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
        if ancilla_reindex is not None:
            tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)

        gtn.data.mangle_inner_(append='_')
        proj_mps = gtn_cc.data

        if obs_gtn is None:
            proj_mps.site_ind_id = gtn.data.site_ind_id
            comp_mps = gtn.data.reindex_sites('_tmp{}_')  ## not inplace
            proj_tens = [proj_mps[ind] for ind in integ_inds]

            tags_list = [proj_mps.site_tag_id, gtn.data.site_tag_id]
            result = qtn.TensorNetwork((proj_tens, comp_mps),
                                       virtual=True, check_collisions=False)
            # result.exponent = comp_mps.exponent
            new_exponent = comp_mps.exponent
        else:
            proj_mps.site_tag_id = gtn.data.site_tag_id
            obs_mpo = obs_gtn.data.copy()
            assert(isinstance(obs_mpo, qtn.MatrixProductOperator), 'obs_mpo must be MPO')
            obs_mpo.mangle_inner_()
            obs_mpo.lower_ind_id = '_tmp{}_'
            obs_mpo.upper_ind_id = gtn.data.site_ind_id
            # obs_mpo.site_tag_id = obs_mpo.site_tag_id + 'O'
            comp_mps = gtn.data.reindex_sites('_tmp{}_')  ## not inplace
            proj_tens = [proj_mps[ind] for ind in integ_inds]

            tags_list = [proj_mps.site_tag_id, gtn.data.site_tag_id, obs_mpo.site_tag_id]
            result = qtn.TensorNetwork((proj_tens, obs_mpo, comp_mps),
                                       virtual=True, check_collisions=False)
            # result.exponent = comp_mps.exponent + obs_mpo.exponent
            new_exponent = comp_mps.exponent + obs_mpo.exponent

        ## set contraction direction to minimize costs
        if self.grid.axes[-1] not in remaining_axes:
            site_range = range(proj_mps.L - 1, -1, -1)
            from_right = True
        else:
            site_range = range(proj_mps.L)
            from_right = False


        for x in site_range:
            tens_tags = [tag.format(x) for tag in tags_list]

            if x in integ_inds:
                ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
                if from_right:
                    neighbor_tags = [tag.format(x - 1) for tag in tags_list]
                else:
                    neighbor_tags = [tag.format(x + 1) for tag in tags_list]

                # print('contract', x, tens_tags, neighbor_tags)
                result = result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
                if isinstance(result, qtn.Tensor):
                    tens = result
                else:
                    tens = result.select_tensors(tens_tags, which='any')[0]
                tens.drop_tags(tens_tags)
                # print('tens', tens, x)
            else:
                result = result.contract(tens_tags, inplace=True, which='any')

            # print('result', result)

        if isinstance(result, qtn.Tensor):
            ## already caught integ_axes = all case, so will always need to be in TN form
            result = qtn.TensorNetwork([result])
            result.exponent = new_exponent
        else:
            result.fuse_multibonds(inplace=True)
            result.exponent = new_exponent

        print('result', helper.norm(result), result.exponent, new_exponent)

        if gtn.data.site_tag_id != proj_mps.site_tag_id:
            result.drop_tags([proj_mps.site_tag_id.format(x) for x in site_range])

        ## renumber tensors
        if not gtn.grid.layout_type == LayoutType.PARALLEL_GROUP:
            remaining_inds = np.sort(remaining_inds)
            helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
                                gtn.data.site_ind_id, inplace=True)

        ## convert to MPS
        result = result.view_like(self.data, L=new_grid.L, inplace=True)
        result.exponent = new_exponent
        # result.exponent = integ_mps.exponent + comp_mps.exponent

        ## convert to GTN object
        gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)

        # plt.figure()
        # plt.imshow(obs_gtn.get_data(ax_select = {obs_gtn.grid.axes[1]: (1,2)}))
        # plt.title('projected mpo (x)')
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(obs_gtn.get_data(ax_select={obs_gtn.grid.axes[0]: 32}))
        # plt.title('projected mpo (v)')
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(gtn_result.get_data(), aspect='auto')
        # plt.title('projected result')
        # plt.colorbar()
        # plt.show()
        #
        # from grid1D import Grid1D
        #
        # x_L = obs_gtn.grid.axes[0].L
        # grid_x = Grid1D('x', [obs_gtn.grid.axes[0]])
        # obs_x = obs_gtn.data[:x_L].copy()
        # obs_x._L = x_L
        # obs_x[-1].squeeze(inplace=True)
        # obs_x_gtn = grid_x.make_gridTN(obs_x)
        # psi_x = gtn.data[:x_L].copy()
        # psi_x._L = x_L
        # print('obs_x', obs_x)
        # print('psi_x', psi_x)
        # psi_x[-1].squeeze(inplace=True)
        # psi_x_gtn = grid_x.make_gridTN(psi_x)
        # Op_x = psi_x_gtn.apply(obs_x_gtn)
        # plt.figure()
        # plt.plot(psi_x_gtn.get_data(), label='f')
        # plt.plot(np.diag(obs_x_gtn.get_data()), label='diag op')
        # plt.plot(Op_x.get_data(), label='op*f')
        # print('expec x', helper.expectation_value(psi_x, obs_x))
        # plt.show()

        if compress:
            gtn_result.compress(compress_opts=compress_opts, inplace=True)

        return gtn_result


    def integrate(self, integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None,
                   compress=False, compress_opts=None, ancilla_reindex: dict[str, str] = None, **kwargs) \
            -> Union[Numeric, qtn.Tensor]:
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if self.data is None:
            return np.nan

        return self.meas_expec(None, is_sqrt=is_sqrt, integ_axes=integ_axes, new_grid=new_grid,
                                   new_ax_deriv_configs=new_ax_deriv_configs, ancilla_reindex=ancilla_reindex,
                                   compress=compress, compress_opts=compress_opts)

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
        out = MPS_USVT.from_MPS(self.data, canon_site=canon_site, cur_orthog=cur_orthog)
        if inplace:
            self.data = out
            return self
        else:
            return self.create_like(new_data = out)

    def convert_from_USVT(self, inplace=False, canon_site=None):
        new_gtn = self if inplace else self.copy()
        cur_orthog = self.canon_site
        assert(isinstance(new_gtn.data, MPS_USVT)), 'new_gtn data needs to be in MPS_USVT form'
        new_gtn.data.to_MPS(inplace=True)
        if canon_site is not None:
            new_gtn.canonize(inplace=True, scale=False, i=canon_site, cur_orthog=cur_orthog)
        return new_gtn

    def get_S_tensor(self, **kwargs):
        assert (isinstance(self.data, MPS_USVT)), 'new_gtn data needs to be in MPS_USVT form'
        return self.data.get_S_tensor()

    def project_bond(self, obs_gtn, bond_ind):
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=True)
        s_tens = self_data.select_tensors(self_data.s_tag)[0]
        s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))
        self_ind_L = next(iter(qtn.bonds(self_data[bond_ind], s_tens)))
        conj_ind_L = next(iter(qtn.bonds(conj_data[bond_ind], s_conj)))
        self_ind_R = next(iter(qtn.bonds(self_data[bond_ind + 1], s_tens)))
        conj_ind_R = next(iter(qtn.bonds(conj_data[bond_ind + 1], s_conj)))
        reindex_dict = {conj_ind_L: self_ind_L, conj_ind_R: self_ind_R}

        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id

        expec_tn = qtn.TensorNetwork((self_data, conj_data, obs_gtn.data))
        exponent = obs_gtn.exponent + self_data.exponent
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply = lambda x: x * 10**exponent)
        out.reindex(reindex_dict, inplace=True)
        return out


    def project_site(self, obs_gtn, site_ind, nsites=1) -> 'qtn.Tensor':
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=True)
        # print('project site', site_ind, 'nsites', nsites)
        # helper.check_orthog(self_data)
        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id
        exponent = self_data.exponent + obs_gtn.data.exponent

        reindex_dict = {}  # obs_gtn.data.upper_ind_id: obs_gtn.data.lower_ind_id}
        for ix in range(site_ind, site_ind + nsites):
            reindex_dict[obs_gtn.data.upper_ind_id.format(ix)] = obs_gtn.data.lower_ind_id.format(ix)
        if site_ind > 0:
            self_ind_L = self_data.bond(site_ind, site_ind - 1)
            conj_ind_L = conj_data.bond(site_ind, site_ind - 1)
            reindex_dict[conj_ind_L] = self_ind_L
        if site_ind < self.data.L - 1 - nsites:
            self_ind_R = self_data.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = conj_data.bond(site_ind + nsites - 1, site_ind + nsites)
            reindex_dict[conj_ind_R] = self_ind_R

        for s_ind in range(site_ind, site_ind + nsites):
            s_tens = self_data[s_ind]
            s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))  ## tossed

        expec_tn = qtn.TensorNetwork([self_data, conj_data, obs_gtn.data])
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply = lambda x: x * 10**exponent)
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        out.reindex(reindex_dict, inplace=True)
        return out


    def project_op_site(self, obs_gtn, site_ind, nsites=1) -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=False)
        conj_data.mangle_inner_(append='_')
        obs_gtn_data = obs_gtn.data.mangle_inner_()
        print('project site', site_ind, 'nsites', nsites)
        # helper.check_orthog(self_data)
        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id
        exponent = obs_gtn.data.exponent

        # reindex_dict = {}  # obs_gtn.data.upper_ind_id: obs_gtn.data.lower_ind_id}
        # for ix in range(site_ind, site_ind + nsites):
        #     reindex_dict[obs_gtn.data.upper_ind_id.format(ix)] = obs_gtn.data.lower_ind_id.format(ix)
        self_ind_L, conj_ind_L = None, None
        if site_ind > 0:
            self_ind_L = self_data.bond(site_ind, site_ind - 1)
            conj_ind_L = conj_data.bond(site_ind, site_ind - 1)
            # reindex_dict[conj_ind_L] = self_ind_L
        self_ind_R, conj_ind_R = None, None
        if site_ind < self.data.L - nsites:
            self_ind_R = self_data.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = conj_data.bond(site_ind + nsites - 1, site_ind + nsites)
            # reindex_dict[conj_ind_R] = self_ind_R

        for s_ind in range(site_ind, site_ind + nsites):
            s_tens = self_data[s_ind]
            s_tens = self_data._pop_tensor(next(iter(self_data._get_tids_from_tags(s_tens.tags))))  ## tossed
            s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))  ## tossed

        expec_tn = qtn.TensorNetwork([self_data, conj_data, obs_gtn_data])
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply = lambda x: x * 10**exponent)
        bonds_i = [obs_gtn.data.lower_ind_id.format(ix) for ix in range(site_ind, site_ind + nsites)]
        if self_ind_L is not None:
            bonds_i += [self_ind_L]
        if self_ind_R is not None:
            bonds_i += [self_ind_R]

        bonds_o = [obs_gtn.data.upper_ind_id.format(ix) for ix in range(site_ind, site_ind + nsites)]
        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        # out.reindex(reindex_dict, inplace=True)
        return out, bonds_i, bonds_o


    def project_op_bond(self, obs_gtn, bond_ind) -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """ project obs_gtn (mpo) onto LLL-(S)-RRR
        """
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=False)
        conj_data.mangle_inner_(append='_')
        exponent = obs_gtn.data.exponent

        s_tens = self_data.select_tensors(self_data.s_tag)[0]
        s_tens = self_data._pop_tensor(next(iter(self_data._get_tids_from_tags(s_tens.tags))))
        s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))

        print('s conj', s_tens, self_data[bond_ind], self_data[bond_ind + 1])
        print('s conj', s_conj, conj_data[bond_ind], conj_data[bond_ind + 1])

        self_ind_L = next(iter(qtn.bonds(self_data[bond_ind], s_tens)))
        conj_ind_L = next(iter(qtn.bonds(conj_data[bond_ind], s_conj)))
        self_ind_R = next(iter(qtn.bonds(self_data[bond_ind + 1], s_tens)))
        conj_ind_R = next(iter(qtn.bonds(conj_data[bond_ind + 1], s_conj)))
        # reindex_dict = {conj_ind_L: self_ind_L, conj_ind_R: self_ind_R}

        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id

        expec_tn = qtn.TensorNetwork((self_data, conj_data, obs_gtn.data))
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply=lambda x: x * 10 ** exponent)
        # out.reindex(reindex_dict, inplace=True)

        bonds_i = []
        if self_ind_L is not None:
            bonds_i += [self_ind_L]
        if self_ind_R is not None:
            bonds_i += [self_ind_R]

        bonds_o = []
        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        # out.reindex(reindex_dict, inplace=True)
        return out, bonds_i, bonds_o


    # def integrate(self, integ_axes=None, is_sqrt=False, ancilla_reindex=None, new_grid: 'Grid1D' = None,
    #               new_ax_deriv_configs=None, compress=False, compress_opts=None):
    #     """ integrate over specified dimensions. returns scalar if axes='all'
    #         otherwise, returns a Field object with reduced dimensionality
    #         inplace is dummy parameter.
    #         ancilla_reindex:  dict[old_anc: new_anc] for reindexing MPS when is_sqrt==True
    #             ancilla must be on the anchor tensor (indexed by 0)
    #             new and old ancilla are fused together -> old ancilla name
    #     """
    #     if integ_axes is None:
    #         integ_axes =self.grid.axes
    #         out_axes = []
    #     else:
    #         out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
    #
    #     if new_grid is None and len(out_axes)>0:
    #         new_grid = self.grid.create_like(new_axes=out_axes)
    #
    #     if new_ax_deriv_configs is None and new_grid is not None:
    #         new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}
    #
    #     if self.data is None:
    #         return new_grid.make_empty_gridTN(ax_deriv_configs=new_ax_deriv_configs)
    #
    #
    #     assert self.data_type == DataType.MPS, 'integrate only defined for MPS GridTNs'
    #
    #     # if integ_axes is None or set([ax.axID for ax in integ_axes])==set([ax.axID for ax in self.grid.axes]):
    #     if len(out_axes)==0:
    #         integ = self.grid.get_integrals_mpx(axes=None, is_sqrt=False)
    #
    #         if is_sqrt:
    #             ## spatial basis only
    #             # dxs = [ax.dx for ax in self.grid.axes]
    #             # result = helper.norm(self.data) ** 2 * np.prod(dxs)
    #
    #             ## actually correct (don't need convolution bc of cc for k-space)
    #             self_conj = self.conj(mangle_inner=True)
    #             # dx = np.asscalar( np.prod([ax.get_integral_weight() for ax in integ_axes]) )
    #             # helper.scalar_multiply(data_conj, dx, inplace=True)
    #             integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
    #             integ_weights = self.grid.make_mps_ndim(integ_weights)
    #             if np.isscalar(integ_weights):
    #                 self_conj.scalar_multiply(integ_weights, inplace=True)
    #             else:
    #                 integ_weights = integ_weights.mps_to_diag_mpo()
    #                 self_conj.apply(integ_weights, inplace=True)
    #
    #             data_conj = self_conj.data
    #             if ancilla_reindex:     # for compatibility with comb
    #                 data_conj.reindex(ancilla_reindex, inplace=True)
    #             # print('self sqrt integ exp', self.data.exponent, data_conj.exponent)
    #             result = helper.ovlp(self.data, data_conj)
    #         else:
    #             # integ_mps = self.grid.get_integrals_mpx(axes=None, is_sqrt=False).data
    #             result = helper.ovlp(self.data, integ.data)
    #
    #         return result       ## a scalar or Tensor (if there are ancilliary bonds)
    #
    #     else:
    #         integ_mps = self.grid.get_integrals_mpx(integ_axes).data
    #         remaining_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
    #         remaining_inds = []
    #         for ax in remaining_axes:
    #             remaining_inds += self.grid.get_inds_in_axis(ax)
    #
    #         if new_grid is None:
    #             new_grid = self.grid.create_like(remaining_axes, self.grid.gridID + '_integ')
    #
    #         if is_sqrt:
    #             # tmp = helper.mps_to_diag_mpo(self.data.conj())
    #             self_conj = self.copy()
    #             self_conj.data.conj(inplace=True, mangle_inner=True)
    #             if ancilla_reindex:
    #                 self_conj.get_anchor_tens().reindex(ancilla_reindex, inplace=True)
    #             tmp = self_conj.apply_elemental_multiply_op()
    #             integ_mps = helper.apply(integ_mps, tmp.data)
    #             if ancilla_reindex:
    #                 fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
    #                 integ_mps[self.get_anchor_ind()].fuse(fuse_anc,inplace=True)
    #
    #         integ_mps.lower_ind_id = '_tmp{}_'
    #         integ_mps.upper_ind_id = self.data.site_ind_id
    #         comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace
    #
    #         result = qtn.TensorNetwork((integ_mps, comp_mps),
    #                                      virtual=True, check_collisions=False)
    #
    #         ## set contraction direction to minimize costs
    #         if self.grid.axes[-1] not in remaining_axes:
    #             site_range = range(integ_mps.L - 1, -1, -1)
    #             from_right = True
    #         else:
    #             site_range = range(integ_mps.L)
    #             from_right = False
    #
    #         for x in site_range:
    #             tens_tags = [integ_mps.site_tag_id.format(x), self.data.site_tag_id.format(x)]
    #
    #             try:
    #                 if integ_mps.phys_dim(x, 'upper') == 1:
    #                     ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
    #                     if from_right:
    #                         neighbor_tags = [integ_mps.site_tag_id.format(x - 1),
    #                                          self.data.site_tag_id.format(x - 1)]
    #                     else:
    #                         neighbor_tags = [integ_mps.site_tag_id.format(x + 1),
    #                                          self.data.site_tag_id.format(x + 1)]
    #                     result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
    #                     tens = result.select_tensors(tens_tags, which='any')[0]
    #                     tens.isel({integ_mps.upper_ind(x): 0}, inplace=True)
    #                     tens.drop_tags(tens_tags)
    #                 else:
    #                     raise TypeError
    #             except TypeError:
    #                 result.contract(tens_tags, inplace=True, which='any')
    #
    #             result.fuse_multibonds(inplace=True)
    #
    #         if self.data.site_tag_id != integ_mps.site_tag_id:
    #             result.drop_tags([integ_mps.site_tag_id.format(x) for x in site_range])
    #
    #         ## renumber tensors
    #         if not self.grid.layout_type == LayoutType.PARALLEL_GROUP:
    #             remaining_inds = np.sort(remaining_inds)
    #             helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
    #                                 self.data.site_ind_id, inplace=True)
    #
    #         ## convert to MPS
    #         result.view_like(self.data, L=new_grid.L, inplace=True)
    #         result.exponent = integ_mps.exponent + comp_mps.exponent
    #
    #         ## convert to GTN object
    #         gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)
    #
    #         if compress:
    #             gtn_result.compress(compress_opts=compress_opts)
    #
    #         return gtn_result
    #
    #
    # def meas_expec(self, obs_gtn: 'GridTN', integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None,
    #                compress=False, compress_opts=None, ancilla_reindex: dict[str, str]=None, **kwargs) \
    #         -> Union[Numeric, qtn.Tensor]:
    #     """ integrate dx f(x) O(x) dx
    #         integrate dx g*(x) O(x) g(x) dx
    #     """
    #     if self.data is None:
    #         return np.nan
    #
    #     if obs_gtn is None or obs_gtn.data_type is None:
    #         expec = self.integrate(is_sqrt=is_sqrt, integ_axes=integ_axes, new_grid=new_grid,
    #                                new_ax_deriv_configs=new_ax_deriv_configs, ancilla_reindex=ancilla_reindex,
    #                                compress=compress, compress_opts=compress_opts)
    #         return expec
    #
    #     if integ_axes is None:
    #         integ_axes = self.grid.axes
    #
    #     if obs_gtn.data_type == DataType.MPS:
    #         obs_gtn = obs_gtn.apply_elemental_multiply_op()
    #
    #     if is_sqrt:
    #         # if integ_axes is not None or integ_axes != self.grid.axes:
    #         #     raise NotImplementedError('sqrt partial meas expec not implemented')
    #
    #         gtn_cc = self.conj()
    #         tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
    #         if ancilla_reindex is not None:
    #             tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)
    #
    #         integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
    #         integ_weights = self.grid.make_mps_ndim(integ_weights)
    #
    #         if len(integ_axes) == self.grid.ndim:
    #             if np.isscalar(integ_weights):
    #                 gtn_cc.scalar_multiply(integ_weights, inplace=True)
    #             else:
    #                 integ_weights = integ_weights.mps_to_diag_mpo()
    #                 gtn_cc.apply(integ_weights, inplace=True)
    #             expec = helper.expectation_value(self.data, obs_gtn.data, bra=gtn_cc.data)
    #
    #         else:
    #             out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
    #
    #
    #         return expec
    #     else:
    #         # if obs_gtn.data_type == DataType.MPS:
    #         #     expec = self.apply_mps(obs_gtn)
    #
    #         # integ = self.grid.get_integrals_mpx()
    #         # expec = helper.expectation_value(self.data, obs_gtn.data, bra=integ.data)
    #
    #         obs_state = self.apply(obs_gtn)
    #         expec = obs_state.integrate(integ_axes=integ_axes, new_grid=new_grid,
    #                                     new_ax_deriv_configs=new_ax_deriv_configs,
    #                                     compress=compress, compress_opts=compress_opts)
    #
    #         return expec
