import numpy as np

from setup.configs import *

import helper_quimb as helper
from gridTN import GridTN
from basis.basis_k import FourierBasis
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

    def __init__(self, grid: 'Grid1D', data: Optional[qtn.TensorNetwork]=None, ax_deriv_configs=None):
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
            # elif isinstance(data, dict):
            #     new_data = self.grid.make_mps_ndim(data)
            elif isinstance(data, qtn.TensorNetwork1D):
                new_data = data
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
            new_tn1D.data = self.data.copy() if deep and self.data is not None else self.data
        return new_tn1D

    # def norm(self, is_sqrt=False) -> Numeric:
    #     if self.data is not None:
    #         out = self.integrate(is_sqrt=is_sqrt)
    #         if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
    #             out = np.real(out)
    #         return out

    def max_bond(self) -> int:
        if self.data is not None:
            return self.data.max_bond()

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
    def from_dense_operator(cls, data, grid, upper_ind_id='o({})', lower_ind_id='o({})', site_tag_id='T({})',
                            split_opts=None):
        mpo = grid.map_operator_to_mpo(data, upper_ind_id, lower_ind_id, site_tag_id, split_opts=split_opts)
        return cls(grid, data=mpo)

    def get_data(self):

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

            return self.grid.map_mps_to_state(self, ancilla_right=ancilla_right,
                                              ancilla_right_inds=ancilla_right_inds, ancilla_left=ancilla_left,
                                              ancilla_left_inds=ancilla_left_inds)

        elif isinstance(self.data, qtn.MatrixProductOperator):
            shared, not_shared = self.data[0].filter_bonds(self.data[1])
            ancilla_left_inds = tuple([ix for ix in not_shared if
                                       not (ix == self.data.upper_ind(0) or ix == self.data.lower_ind(0)) ])
            ancilla_left = tuple([self.data[0].ind_size(ix) for ix in ancilla_left_inds])

            shared, not_shared = self.data[-1].filter_bonds(self.data[-2])
            ancilla_right_inds = tuple([ix for ix in not_shared if
                                       not (ix == self.data.upper_ind(L-1) or ix == self.data.lower_ind(L-1)) ])
            ancilla_right = tuple([self.data[-1].ind_size(ix) for ix in ancilla_right_inds])

            return self.grid.map_mpo_to_operator(self, ancilla_right=ancilla_right,
                                                 ancilla_right_inds=ancilla_right_inds, ancilla_left=ancilla_left,
                                                 ancilla_left_inds=ancilla_left_inds)
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


    def apply(self, mpo, inplace=False, compress=False, compress_opts=None) -> 'GridTN1D':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
        """
        if self.data is None:
            return self

        grid_mpx1 = self if inplace else self.copy()
        mpo = self._parse_input_TN1D(mpo)
        assert (isinstance(mpo,qtn.MatrixProductOperator)),'other needs to be an MPO type'

        new_mpx = helper.apply(mpo, grid_mpx1.data) # , compress=compress, compress_opts=compress_opts)
        if compress:
            helper.compress(new_mpx, compress_opts=compress_opts)
        ## not inplace update of qtn.TensorNetwork obj.
        grid_mpx1.data = new_mpx
        return grid_mpx1


    def add(self, gtn_mpx2, inplace=False, compress=False, compress_opts=None) -> 'GridTN1D':
        """ Add other grid_mpx (of the same type and on the same grid) to self
        """
        grid_mpx1 = self if inplace else self.copy(deep=True)
        mpx2 = self._parse_input_TN1D(gtn_mpx2)

        if mpx2 is None:
            return grid_mpx1

        if isinstance(grid_mpx1.data,qtn.MatrixProductOperator):
            assert (isinstance(mpx2,qtn.MatrixProductOperator)), 'other needs to be an MPO type'
            helper.add_MPO(grid_mpx1.data, mpx2, inplace=True, compress=compress, compress_opts=compress_opts)
        elif isinstance(grid_mpx1.data,qtn.MatrixProductState):
            assert (isinstance(mpx2,qtn.MatrixProductState)), 'other needs to be an MPS type'
            helper.add_MPS(grid_mpx1.data, mpx2, inplace=True, compress=compress, compress_opts=compress_opts)
        elif grid_mpx1.data is None:
            grid_mpx1.data = mpx2.copy() if mpx2 is not None else None
        else:
            raise TypeError(f'self needs to be an MPO, MPS, or NoneType, not {type(self.data)}')

        return grid_mpx1


    def scalar_multiply(self,scalar_const,inplace=False) -> 'GridTN1D':

        if self.data is None:
            return self

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


    def canonize(self, inplace=True, scale=True, form='right', i=None) -> 'GridTN1D':

        if self.data is None:
            return self

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.canonize(grid_mpx1.data, scale=scale, form=form, i=i)
        return grid_mpx1


    def compress(self, inplace=True, compress_opts=None, **kwargs) -> 'GridTN1D':

        if self.data is None:
            return self

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.compress(grid_mpx1.data, compress_opts=compress_opts)
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

    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})',
                                    compress=False, compress_opts=None):
        """ Use 3-leg elemental multiply TN, apply to self to turn into mpo
            for elemental multiplication (with some other, yet to be specified gmps)
            assumes indices of self goes from 0 to grid.L-1 (no missing indices)
            lower_ind_id, upper_ind_id = desired formats for output MPO
        """
        if self.data is None:   # zero
            return self

        assert self.data_type==DataType.MPS, 'needs to be MPS to make MPO'
        new_mpo = qtn.TensorNetwork([])
        contract_ind = self.data.site_ind_id

        fix_tag = None
        if contract_ind == upper_ind_id:
            upper_ind_id += '_tmp'
            fix_tag = 'U'
        elif contract_ind == lower_ind_id:
            lower_ind_id += '_tmp'
            fix_tag = 'L'

        elem_mult_tn = self.grid.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=contract_ind,
                                                           out_ind_id=upper_ind_id, site_tag_id='_tmp{}')

        for x in range(self.grid.L):
            tens_x1 = self.data[x]
            tens_x2 = elem_mult_tn.data[x]
            new_tens = tens_x1.contract(tens_x2)
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
        new_mpo.exponent = self.data.exponent

        gmpo = self.create_like()
        gmpo.data = new_mpo

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


    def integrate(self, integ_axes=None, is_sqrt=False, ancilla_reindex=None, new_grid: 'Grid1D' = None,
                  new_ax_deriv_configs=None, compress=False, compress_opts=None):
        """ integrate over specified dimensions. returns scalar if axes='all'
            otherwise, returns a Field object with reduced dimensionality
            inplace is dummy parameter.
            ancilla_reindex:  dict[old_anc: new_anc] for reindexing MPS when is_sqrt==True
                ancilla must be on the anchor tensor (indexed by 0)
                new and old ancilla are fused together -> old ancilla name
        """
        if self.data is None:
            return None

        assert self.data_type == DataType.MPS, 'integrate only defined for MPS GridTNs'

        if integ_axes is None:
            integ_axes =self.grid.axes
            out_axes = []
        else:
            out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]

        if new_grid is None and len(out_axes)>0:
            new_grid = self.grid.create_like(new_axes=out_axes)

        if new_ax_deriv_configs is None and new_grid is not None:
            new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}

        # if integ_axes is None or set([ax.axID for ax in integ_axes])==set([ax.axID for ax in self.grid.axes]):
        if len(out_axes)==0:
            integ = self.grid.get_integrals_mpx(axes=None, is_sqrt=False)

            if is_sqrt:
                ## spatial basis only
                # dxs = [ax.dx for ax in self.grid.axes]
                # result = helper.norm(self.data) ** 2 * np.prod(dxs)

                # ## this is maybe inefficient
                # self_conj = self.create_like(new_data=self.data.conj())
                # integ_f = integ_mps.elemental_multiply(self_conj)
                # result = helper.ovlp(self.data, integ_f.data)

                # ## maybe more efficient?
                # self_conj = self.create_like(new_data=self.data.conj(mangle_inner=True))
                # self_conj_mpo = self_conj.apply_elemental_multiply_op()
                #
                # # print('self conj', np.imag(self_conj.get_data()))
                # # plt.figure()
                # # plt.plot(np.imag(self_conj.get_data()))
                # # plt.show()
                #
                # # test = self_conj.apply(self_conj_mpo)
                # # print('Ek*Ek', np.real(test.get_data()))
                # # test1 = self_conj.get_data()
                # # print('test1', np.real(test1))
                # # print('np convolve', np.convolve(test1, test1, mode='same'))
                # # print('np convolve', np.convolve(test1, test1)[4:12])
                # # print(integ.get_data())
                #
                # result = helper.expectation_value(self.data, self_conj_mpo.data, integ.data)
                # # result = helper.ovlp(test.data, integ.data)
                # # print('result', np.dot(test.get_data(), integ.get_data()))

                ## actually correct (don't need convolution bc of cc for k-space)
                data_conj = self.data.conj(mangle_inner=True)
                if ancilla_reindex:
                    data_conj.reindex(ancilla_reindex, inplace=True)
                result = helper.ovlp(self.data, data_conj)
            else:
                # integ_mps = self.grid.get_integrals_mpx(axes=None, is_sqrt=False).data
                result = helper.ovlp(self.data, integ.data)

            return result       ## a scalar or Tensor (if there are ancilliary bonds)

        else:
            integ_mps = self.grid.get_integrals_mpx(integ_axes).data
            remaining_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
            remaining_inds = []
            for ax in remaining_axes:
                remaining_inds += self.grid.get_inds_in_axis(ax)

            if new_grid is None:
                new_grid = self.grid.create_like(remaining_axes, self.grid.gridID + '_integ')

            if is_sqrt:
                # tmp = helper.mps_to_diag_mpo(self.data.conj())
                self_conj = self.copy()
                if ancilla_reindex:
                    self_conj.get_anchor_tens().reindex(ancilla_reindex, inplace=True)
                tmp = self_conj.apply_elemental_multiply_op()
                integ_mps = helper.apply(integ_mps, tmp)
                if ancilla_reindex:
                    fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
                    integ_mps[self.get_anchor_ind()].fuse(fuse_anc,inplace=True)

            integ_mps.lower_ind_id = '_tmp{}_'
            integ_mps.upper_ind_id = self.data.site_ind_id
            comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace

            result = qtn.TensorNetwork((integ_mps, comp_mps),
                                         virtual=True, check_collisions=False)

            ## set contraction direction to minimize costs
            if self.grid.axes[-1] not in remaining_axes:
                site_range = range(integ_mps.L - 1, -1, -1)
                from_right = True
            else:
                site_range = range(integ_mps.L)
                from_right = False

            for x in site_range:
                tens_tags = [integ_mps.site_tag_id.format(x), self.data.site_tag_id.format(x)]

                try:
                    if integ_mps.phys_dim(x, 'upper') == 1:
                        ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
                        if from_right:
                            neighbor_tags = [integ_mps.site_tag_id.format(x - 1),
                                             self.data.site_tag_id.format(x - 1)]
                        else:
                            neighbor_tags = [integ_mps.site_tag_id.format(x + 1),
                                             self.data.site_tag_id.format(x + 1)]
                        result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
                        tens = result.select_tensors(tens_tags, which='any')[0]
                        tens.isel({integ_mps.upper_ind(x): 0}, inplace=True)
                        tens.drop_tags(tens_tags)
                    else:
                        raise TypeError
                except TypeError:
                    result.contract(tens_tags, inplace=True, which='any')

                result.fuse_multibonds(inplace=True)

            if self.data.site_tag_id != integ_mps.site_tag_id:
                result.drop_tags([integ_mps.site_tag_id.format(x) for x in site_range])

            ## renumber tensors
            helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
                                self.data.site_ind_id, inplace=True)

            ## convert to MPS
            result.view_like(self.data, L=new_grid.L, inplace=True)

            ## convert to GTN object
            gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)

            if compress:
                gtn_result.compress(compress_opts=compress_opts)

            return gtn_result


    def meas_expec(self, obs_gtn: 'GridTN', is_sqrt=False) -> Numeric:
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if obs_gtn.data_type == DataType.MPS:
            obs_gtn = obs_gtn.apply_elemental_multiply_op()

        if is_sqrt:
            expec = helper.expectation_value(self.data, obs_gtn.data)
            return expec
        else:
            # if obs_gtn.data_type == DataType.MPS:
            #     expec = self.apply_mps(obs_gtn)

            integ = self.grid.get_integrals_mpx()
            expec = helper.expectation_value(self.data, obs_gtn.data, bra=integ.data)
            return expec
