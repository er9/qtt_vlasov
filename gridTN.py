import pickle

import helper_dmrg
from setup.configs import *
import helper_quimb as helper
from helper_tdvp import TDVPSolver

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from coord.coord_sys import Coordinate
    from grid import Grid
    from grids_composite import CompositeGrid
    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']  # tuple[Grid]

""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
"""

########################################
## static methods to create GridTN1Ds ##
########################################




####################
## GridTN1D class ##
####################

class GridTN:

    def __init__(self, grid: 'GridType', data: list[dict['Axis','TNType']] = None,
                 ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None):
        """ grid:  grid on which the TN lives
            data:  data in compatible format for GridType
        """

        self.grid = grid
        self._data_type = None
        self.data = data     # list of dicts with grid.axIDs as keys
        # self.is_sqrt = False

        if ax_deriv_configs is None:
            self.ax_deriv_configs = {ax: DerivativeConfiguration() for ax in self.grid.axes}
        else:
            self.ax_deriv_configs = ax_deriv_configs

        self.canon_site = None  # None: not canonical; otherwise indexes site canonicalized around
        self.is_constant = False
        self.constant_axes = []


    def __repr__(self):
        return str(self.__class__) + ' ' + str(self.data)

    def __getitem__(self, item):
        raise NotImplementedError

    @property
    def exponent(self):
        return self.data.exponent

    def _get_data(self) -> 'TNType':
        return self._data

    def _set_data(self, data):
        self._data = data
        if data is not None:
            data_ = data[0]
            self.data_type = type( data_[next(iter(data_))] )
        self.constant_axes = []
        self.is_constant= False

    data = property(fget=_get_data,fset=_set_data)


    def _get_data_type(self):
        return self._data_type

    def _set_data_type(self, data_class):
        if data_class == qtn.MatrixProductState or issubclass(data_class, qtn.MatrixProductState):
            self._data_type = DataType.MPS
        elif data_class == qtn.MatrixProductOperator:   self._data_type = DataType.MPO
        elif data_class == qtn.TensorNetwork1D:         self._data_type = DataType.MPX
        elif data_class == MatrixProductTensor:         self._data_type = DataType.TN3
        else:
            raise TypeError('GridTN data must be MPS, MPO, TN1, or MPT')

    data_type = property(fget=_get_data_type, fset=_set_data_type)


    def update_deriv_params(self, ax, left_bc: Optional[BCType or str] = None,
                            right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
                            fd_type: Optional[FDType or str] = None, offset: Optional[int] = None):
        """ kwargs:  'order', 'fdtype', 'bc', 'upwind', 'v_ax'
        """
        dp = self.ax_deriv_configs[ax]
        dp.update(left_bc, right_bc, order, fd_type, offset) #, v_ax)


    def create_like(self, new_data=None) -> 'GridTN':
        raise NotImplementedError

    def copy(self, deep=True) -> 'GridTN':
        raise NotImplementedError

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN':
        raise NotImplementedError


    def save_data(self, fstr):
        pickle.dump(self.data, open(fstr + '.pkl','wb'))
        print('saved data', fstr + '.pkl')

    def reload_data(self, fstr, ax_deriv_configs=None):
        """ self.data set by loaded data
        """
        grid = self.grid
        return grid.load_gtn_data(fstr, self, ax_deriv_configs=ax_deriv_configs)

    @classmethod
    def load_data(cls, grid: 'Grid', fstr, ax_deriv_configs=None):
        """ generate gtn with loaded data
        """
        gtn = cls(grid, data=None, ax_deriv_configs=ax_deriv_configs)
        return grid.load_gtn_data(fstr, gtn)


    def norm(self, is_sqrt=False) -> Numeric:
        if self.data is not None:
            out = self.integrate(is_sqrt=is_sqrt)
            if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
                out = np.real(out)
            if is_sqrt:
                out = np.sqrt(out)
            return out

    def frobenius_norm(self) -> Numeric:
        if self.data is not None:
            # out = helper.ovlp(self.data, self.data.conj())
            out = self.ovlp( self.conj(mangle_inner=True, inplace=False) )
            # out = self.integrate(is_sqrt=True)
            # if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
            #     out = np.real(out)
            out = np.sqrt(out)
            # print('fro norm', out) # , self.data.norm() * 10**self.exponent)
            return out
        else:
            return 0.0

    def normalize(self, target_value, is_sqrt=False, inplace=False) -> 'GridTN':
        """ if basis is Spatial, then do scalar multiplication
            if spatial basis is Fourier, then only scale k=0 term
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)     # return None?

        is_spatial = [ax.basis.type == BasisType.SPATIAL for ax in self.grid.axes]
        is_fourier = [ax.basis.type == BasisType.FOURIER for ax in self.grid.axes]

        self_norm = self.norm(is_sqrt=is_sqrt)
        scale_val = target_value/np.conj(self_norm)
        if np.isnan(scale_val) and np.abs(self_norm) < 1.0e-12:
            scale_val = 0.0

        if True:  # np.all(is_spatial):
            out = self.scalar_multiply( scale_val, inplace=inplace )
        elif np.any(is_fourier):
            # only adjust k=0 term which may be normalized to 0
            axes_k = [self.grid.axes[i] for i in range(len(is_fourier)) if is_fourier[i]]
            select_dict = {ax: ax.get_select_elems_mpo([ax.zero_ind]) for ax in axes_k}
            select_zero_gtn = self.grid.make_mpo_ndim(select_dict)
            select_zero_gtn.scalar_multiply(scale_val - 1, inplace=True)
            iden_gtn = self.grid.get_iden_mpo()
            scale_gtn = iden_gtn.add(select_zero_gtn)

            out = self.apply(scale_gtn, inplace=inplace, compress=True)

        else:
            raise NotImplementedError

        return out

    def ovlp(self, other) -> Numeric:
        raise NotImplementedError

    def distance(self, other) -> Numeric:

        # self_conj = self.conj(mangle_inner=True, inplace=False)
        # norm2a = self.ovlp(self_conj)
        # norm2b = other.ovlp(other.conj(mangle_inner=True, inplace=False))
        # # ovlp_ab = self.ovlp(other.conj(inplace=False))
        # ovlp_ab = other.ovlp(self_conj)
        # dist = norm2a + norm2b - 2 * np.real(ovlp_ab)
        #
        # if dist < 0:
        #     print('neg dist', dist)
        # # if dist < 0 and np.abs(dist) < 1.0e-13:
        # #     print('small neg dist', dist)
        # #     dist = np.abs(dist)
        # dist1 = np.sqrt(dist)

        diff = self.add(other.scalar_multiply(-1, inplace=False))
        dist = diff.frobenius_norm()

        # print('distances', dist, dist1)
        return dist

    def max_bond(self) -> int:
        raise NotImplementedError

    def entanglement_entropy_all(self) -> list[Numeric]:
        raise NotImplementedError

    def get_anchor_ind(self) -> int:
        """ get "lead" tensor (for connecting to other GTNs in composite grid systems)
        """
        raise NotImplementedError

    def get_anchor_tens(self) -> qtn.Tensor:
        """ get "lead" tensor (for connecting to other GTNs in composite grid systems)
        """
        raise NotImplementedError

    def transpose(self, inplace=True, mangle_inner=False):
        """ take transpose of MPO
        """
        raise NotImplementedError


    #######################################
    ## class methods to build common TNs ##
    #######################################

    #### move to grid_* classes
    @classmethod
    def get_ones_mps(cls, grid: 'GridType', site_ind_id='i({})', site_tag_id='X({})') -> 'GriDTN':
        """ build ones vector mps on sepcified grid
        """
        raise NotImplementedError

    @classmethod
    def get_iden_mpo(cls, grid: 'GridType', upper_ind_id='i({})', lower_ind_id='o({})',
                     site_tag_id='X({})') -> 'GridTN':
        """ build identity mpo on specified grid
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mps(cls, grid: 'GridType', inds, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mpo(cls, grid: 'GridType', inds, upper_ind_id='i({})', lower_ind_id='o({})',
                            site_tag_id='X({})') -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        raise NotImplementedError

    @classmethod
    def from_dense_state(cls, data: np.ndarray, grid: 'GridType', site_ind_id='i({})',
                         site_tag_id='T({})', split_opts=None) -> 'GridTN':
        raise NotImplementedError

    @classmethod
    def from_dense_operator(cls, data: np.ndarray, grid: 'GridType', upper_ind_id='o({})',
                            lower_ind_id='i({})', site_tag_id='T({})', split_opts=None, **kwargs) -> 'GridTN':
        raise NotImplementedError

    ####

    def get_data(self, ax_order:Sequence['Axis']=None, ax_select:dict['Axis',int]=None, pad_data=False) -> np.ndarray:
        raise NotImplementedError

    def get_squeezed_data(self) -> tuple[np.ndarray, 'Grid']:
        data = self.get_data(ax_select={ax: 0 for ax in self.constant_axes})
        data_gr = self.grid.get_subgrid([ax for ax in self.grid.axes if ax not in self.constant_axes])
        return data, data_gr

    #######################
    ## MPS/MPX functions ##
    #######################

    def apply(self, other: 'GridTN', inplace=False, zipup=False, use_mg=False, compress=False, compress_opts=None,
              add_cc=False, **kwargs) -> 'GridTN':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
        """
        raise NotImplementedError

    def solve(self, operator: 'GridTN', compress_type: 'CompressType', inplace=False, compress_opts=None, is_H=True,
              **kwargs) -> 'GridTN':
        """ solves Ax=b using local
            self is b, operator is A, returns x
        """
        raise NotImplementedError

    def add(self, gtn_mpx2: 'GridTN', zipup=False, inplace=False, compress_type=CompressType.SVD,
            compress=False, compress_opts=None, **kwargs) -> 'GridTN':
        """ Add other grid_mpx (of the same type and on the same grid) to self
        """
        gtn = self if inplace else self.copy()
        if gtn.data is None:
            gtn.data = gtn_mpx2.data
        else:
            gtn.data += gtn_mpx2.data
        return gtn

    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTN':
        raise NotImplementedError

    def sum_apply(self, mpos: Sequence['GridTN'], inplace=False, zipup=False, compress=False, compress_opts=None):
        """ apply MPOs and then take the sum
        """
        new_gtn = self if inplace else self.copy()

        out = None
        for mpo in mpos:
            comp = self.apply(mpo, inplace=False, zipup=zipup, compress=False)
            out = comp if out is None else out.add(comp, inplace=True, compress=False)

        if compress:
            out.compress(inplace=True, compress_opts=compress_opts)

        new_gtn.data = out.data
        return new_gtn


    def scalar_multiply(self, scalar_const: Numeric, inplace=False) -> 'GridTN':
        raise NotImplementedError

    # def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, compress=False, compress_opts=None) -> 'GridTN':
    #     raise NotImplementedError

    # @profile
    def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, zipup=False, compress_type=CompressType.SVD,
                           compress=False, compress_opts=None, sub_compress_opts=None, add_cc=False) -> 'GridTN':
        ### changed add_cc default from True to False

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if gtn_mps2 is None or gtn_mps2.data is None:
            if inplace:
                self.data = None
                return self
            else:
                return self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy()

        if isinstance(gtn_mps2, qtn.MatrixProductState):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert(self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            gtn_mps2 = self.create_like(new_data=gtn_mps2)
            if self.data_type == DataType.MPS:   ## convert gtn_mps2 to MPO
                grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
            else:   ## use fact that self is MPO. apply method should take care of it
                grid_mpo2 = grid_mpx1
                grid_mpx1 = gtn_mps2

        elif isinstance(gtn_mps2, qtn.MatrixProductOperator):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            grid_mpo2 = self.create_like()
            grid_mpo2.data = gtn_mps2

        elif isinstance(gtn_mps2, GridTN):

            # print('gtn_mps2 is GTN', grid_mpx1.grid.axes, gtn_mps2.grid.axes)
            if grid_mpx1.grid != gtn_mps2.grid:
                if set(grid_mpx1.grid.axes).issubset(set(gtn_mps2.grid.axes)):
                    # print('mps1 < mps2')
                    grid_mpx1 = gtn_mps2.grid.pad_gtn_to_grid(grid_mpx1)
                    print('warning: elemental multiply not an inplace operation')
                elif set(gtn_mps2.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                    # print('mps1 > mps2')
                    gtn_mps2 = grid_mpx1.grid.pad_gtn_to_grid(gtn_mps2)
                else:
                    raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {gtn_mps2.grid}')

            # print('gtn_mps2 data type', gtn_mps2.data_type)
            # print('gtn', gtn_mps2)
            # print(gtn_mps2.__class__)
            if gtn_mps2.data_type==DataType.MPS:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'

                # if self.data_type == DataType.MPO:
                #     print('WARNING: ELEMENTAL MULTIPLY CHANGED FOR MPO * MPS')
                #
                # grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()

                gtn_mps2 = gtn_mps2
                if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                    grid_mpo2: 'GridTN' = gtn_mps2.apply_elemental_multiply_op()
                else:   ## use fact that self is MPO. apply method should take care of it
                    ## this does not accurately capture elemental multiply for arbitrary grid_mpx1 MPOs
                    raise NotImplementedError('maybe elem mult MPO * MPS needs to be redefined')
                    grid_mpo2 = grid_mpx1
                    grid_mpx1 = gtn_mps2

            elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type==DataType.MPO:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
                grid_mpo2 = gtn_mps2

        else:
            raise TypeError(f'other must be GridTN with MPS or MPO data type, not {type(gtn_mps2)}')

        grid_mpx1 = grid_mpx1.apply(grid_mpo2, inplace=True, zipup=zipup, compress_type=compress_type, compress=compress,
                                    compress_opts=compress_opts, add_cc=add_cc, sub_compress_opts=sub_compress_opts)
        # add_cc used to be True; probably for real Fourier basis??
        return grid_mpx1


    def xmultiply(self, x_axes: Sequence['Axis'], x_power: int = 1, offsets: dict[Any, Numeric] = 0.0,
                  scales: dict[Any, Numeric] = 1.0, inplace=False, zipup=False, compress_type=CompressType.SVD,
                  compress=False, compress_opts=None) -> 'GridTN1D':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy()
        xmult_mpo = grid_mpx1.grid.get_xmultiply_mpo(x_axes, x_power=x_power, offsets=offsets, scales=scales)

        # print('XMULT GTN', compress_type)
        grid_mpx1.apply(xmult_mpo, inplace=True, zipup=zipup, compress_type=compress_type,
                        compress=compress, compress_opts=compress_opts)
        # if compress:
        #     grid_mpx1 = grid_mpx1.compress(inplace=True, compress_opts=compress_opts)

        return grid_mpx1


    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTN':
        raise NotImplementedError

    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True) -> 'GridTN':
        raise NotImplementedError

    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        raise NotImplementedError

    def compress(self, inplace=True, verbose=False, canonize=True, compress_type=CompressType.SVD,
                 compress_opts: dict = None, sub_compress_opts: Optional[dict['SubCompressConfigType',dict]] = None,
                 **kwargs):
        raise NotImplementedError

    def contract(self):
        return self.data.contract()

    def scalar_add(self, scalar_val, inplace=False):
        """ add scalar values to gridTN
        """
        gtn = self if inplace else self.copy()
        ones = self.grid.get_ones_mps()
        ones.scalar_multiply(scalar_val, inplace=True)
        gtn = gtn.add(ones, inplace=True)
        return gtn

    # def apply_partial_mpx(self, mpo: MPOType or MPSType or 'GridTN', reset_grid=True, compress=True,
    #           **compress_opts) -> 'GridTN':
    #     raise NotImplementedError


    def apply_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)       ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)       ## probably not compressible?
            iqft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn_ = iqft_mpo.apply(gtn, zipup=True)
            gtn_.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
            gtn.data = gtn_.data
        else:
            raise NotImplementedError

        return gtn


    def apply_inverse_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            print('inverse qft mpo')
            qft_gtn = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            inv_qft_gtn = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?

            gtn_1 = qft_gtn.apply(gtn, zipup=True)
            gtn_1.apply(inv_qft_gtn, inplace=True, zipup=True, compress=True, compress_opts=compress_opts)
            gtn.data = gtn_1.data
        else:
            raise NotImplementedError

        return gtn

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
        raise NotImplementedError

    def convert_from_USVT(self, inplace=False, canon_site=None):
        raise NotImplementedError

    def get_S_tensor(self, **kwargs):
        raise NotImplementedError

    ################
    ## processing ##
    ################

    def interpolate_data(self, nsites, inplace=False):
        """ extend grid s.t. data is interpolated between existing grid points
        """
        raise NotImplementedError


    def gaussian_smooth_data(self,  sigma=0.3, proc_axes: Optional[Sequence['Axis']]=None, mode='wrap',
                             inplace=False, compress=True, compress_opts=None) -> 'GridTN':
        """ process the data in some way
        """
        gtn = self if inplace else self.copy(deep=False)
        blur_mpo = self.grid.gaussian_smooth_mpo(sigma=sigma, proc_axes=proc_axes, mode=mode, split_opts=compress_opts)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if blur_mpo is not None:
            gtn = gtn.apply(blur_mpo, inplace=True, compress=compress, compress_opts=compress_opts)  ## not deep inplace
        return gtn


    def apply_absorbing_bc(self, x_ax: 'Axis', v_ax: 'Axis', inplace=False, compress=True, compress_opts=None)\
            -> 'GridTN':
        """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
            (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
        """
        gtn = self if inplace else self.copy(deep=False)
        left_bc, right_bc = gtn.ax_deriv_configs[x_ax].bc

        if left_bc == BCType.ABSORBING or right_bc == BCType.ABSORBING:
            absorb_bc_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
            # print(absorb_bc_mpo, absorb_bc_mpo.data)
            # leave out compress_opts for default (most accurate) compression if compress=True

            # if absorb_bc_mpo is not None: # and absorb_bc_mpo.data is not None:
            gtn = gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        return gtn


    def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, compress_opts=None):
        """ apply reflecting_v bc to state.
            apply to derivative MPO?
        """
        gtn = self if inplace else self.copy()
        reflect_bc_mpo = self.grid.reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if reflect_bc_mpo is not None:
            if compress_opts is None:   compress_opts = {}
            gtn = gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn


    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':
        """ convert MPS into MPO
        """
        raise NotImplementedError


    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """ apply elemental multiply operator to self (e.g. convolution or d_ijk fct)
        """
        raise NotImplementedError


    # def apply_xmultiply_mpo(self, x_axes: list['Axis'] or None = None, offsets: dict['Axis', Numeric] = 0.0,
    #                         scales: dict['Axis', Numeric] = 1.0,inplace=False, compress=False, compress_opts=None):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...)
    #     """
    #     if self.data is None:
    #         return self
    #
    #     gtn = self if inplace else self.copy()
    #     x_mpo = self.grid.get_xmultiply_mpo(x_axes, offsets=offsets, scales=scales)
    #     if compress_opts is None:   compress_opts = {}
    #     gtn.apply(x_mpo, compress=compress, compress_opts=compress_opts)
    #     return gtn


    ## put bc into deriv_params to avoid carrying it around when calling curl, laplacian, etc.
    # @profile
    def take_firstderivative(self, ax: 'Axis', upwind_ax: Optional['Axis']=None,
                             deriv_config: 'DerivativeConfiguration' = None,
                             recalc=False, inplace=False, compress=False, compress_opts=None):
        """ df/dx_i
            bc = boundary condition
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config = gtn.ax_deriv_configs[ax] if deriv_config is None else deriv_config
        ddx_mpo = self.grid.get_firstderivative_mpo(ax, deriv_config=deriv_config, upwind_ax=upwind_ax, recalc=recalc)
        gtn.apply(ddx_mpo, inplace=True)

        ### TODO: selective compress (don't need to compress entire thing
        if compress:
            gtn = gtn.compress(inplace=True, compress_opts=compress_opts)
        return gtn


    def take_secondderivative(self, ax1: 'Axis', ax2: 'Axis' or None,
                              recalc=False, inplace=False, compress=False, compress_opts=None):
        """ self.order: finite difference expansion order """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax1]
        deriv_config2 = gtn.ax_deriv_configs[ax2] if ax2 is not None else None
        d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, deriv_config1=deriv_config1, deriv_config2=deriv_config2,
                                                    recalc=recalc)
        gtn = gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn


    def integrate(self, integ_axes: Sequence['Axis'] = None, is_sqrt=False, ancilla_reindex:dict[str,str]=None,
                  new_grid: 'Grid' = None, new_ax_deriv_configs: dict[Any,'DerivativeConfiguration'] = None,
                  compress=False, compress_opts=None):
        raise NotImplementedError


    def meas_expec(self, obs_gtn, integ_axes=None, is_sqrt=False, new_grid=None, exclude_axes=None,
                   new_ax_deriv_configs: dict[Any,'DerivativeConfiguration'] = None,
                   compress=False, compress_opts=None, **kwargs) -> Numeric:
        raise NotImplementedError


    def project(self, obs_gtn, proj_axes, new_grid=None, canonize=True, compress=False, compress_opts=None,
                new_ax_deriv_configs=None, **kwargs) -> 'GridTN':
        """ project obs_gtn onto manifold of self
        """
        raise NotImplementedError

    def project_bond(self, obs_gtn, bond_ind) -> 'qtn.Tensor':
        raise NotImplementedError

    def project_site(self, obs_gtn, site_ind, nsites=1) -> 'qtn.Tensor':
        raise NotImplementedError

    def project_op_bond(self, obs_gtn, bond_ind, left_env: 'qtn.Tensor' = None, right_env: 'qtn.Tensor' = None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        raise NotImplementedError

    def project_op_site(self, obs_gtn, site_ind, nsites=1, left_env: 'qtn.Tensor'=None, right_env:'qtn.Tensor'=None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        raise NotImplementedError

    def stagger_grid(self, stagger_vals: dict['Axis',int], inplace=False, compress=False, compress_opts=None):
        """
        stagger_vals: stagger the grid along each axis by specified value (in 1/2 grid point increments)
        """

        gtn = self if inplace else self.copy()

        stagger_mpos = {}
        for ax, val in stagger_vals.items():
            if val != 0:
                deriv_config = gtn.ax_deriv_configs[ax]
                shift = val//2
                avg = val%2

                stagger_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config)
                if avg:     ## 1/2 grid step discretization
                    stagger_mpo_2 = ax.get_shift_mpo(shift+1, boundary_conditions=deriv_config)
                    helper.add_MPO(stagger_mpo, stagger_mpo_2, inplace=True, compress=False)
                    helper.scalar_multiply(stagger_mpo, 0.5, inplace=True)

                if not (deriv_config.left_bc == BCType.PERIODIC or deriv_config.left_bc == BCType.ANTIPERIODIC
                        or deriv_config.left_bc != BCType.OPEN):
                    deriv_config.offset = deriv_config.offset + val

                stagger_mpos[ax] = stagger_mpo

        # orig_data = gtn.get_data()

        stagger_mpo = gtn.grid.make_mpo_ndim(stagger_mpos)
        gtn = gtn.apply(stagger_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        # new_data = gtn.get_data()

        # plt.figure()
        # plt.plot(orig_data)
        # plt.plot(new_data)
        # plt.show()

        return gtn


    def evolve_tdvp(self, dt, mpo_list, te_order=0, do_adapt=True, inplace=False,
                    compress_config=None):
        raise NotImplementedError


    ##########################
    ## sqrt field functions ##
    ##########################

    def get_complex_conj(self, inplace=False) -> 'GridTN':
        ## needs to be overwritten within each class
        # out = self if inplace else self.copy()
        raise NotImplementedError

    def mult_cc(self, inplace=False):
        cc = self.get_complex_conj()
        out = self.elemental_multiply(cc, inplace=inplace)
        return out


#########################
#### class functions ####
#########################

def add(gtn1: 'GridTN', gtn2: 'GridTN', inplace=False, **kwargs):
    return gtn1.add(gtn2, inplace=inplace, **kwargs)

def scalar_multiply(gtn1: 'GridTN', const: 'Numeric', inplace=False, **kwargs):
    return gtn1.scalar_multiply(const, inplace=inplace, **kwargs)

def euler(gtn1: 'GridTN', dt, deriv0=None, inplace=False, **kwargs):

    if deriv0 is None:
        return gtn1 if inplace else gtn1.copy()

    return gtn1.add( deriv0.scalar_multiply(dt), inplace=inplace, **kwargs)

def compute_derivative(gtn1: 'GridTN', derivative_mpos: Sequence['GridTN'], **kwargs):
    return gtn1.sum_apply(derivative_mpos, **kwargs)