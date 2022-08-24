import pickle
from setup.configs import *
import helper_quimb as helper

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

################
## common TNs ##
################

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

    data = property(fget=_get_data,fset=_set_data)


    def _get_data_type(self):
        return self._data_type

    def _set_data_type(self, data_class):
        if   data_class == qtn.MatrixProductState:      self._data_type = DataType.MPS
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
            out = self.integrate(is_sqrt=True)
            if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
                out = np.real(out)
            out = np.sqrt(out)
            return out
        else:
            return 0.0

    def normalize(self, target_value, inplace=False) -> 'GridTN':
        """ if basis is Spatial, then do scalar multiplication
            if spatial basis is Fourier, then only scale k=0 term
        """
        is_spatial = [ax.basis.type == BasisType.SPATIAL for ax in self.grid.axes]
        is_fourier = [ax.basis.type == BasisType.FOURIER for ax in self.grid.axes]

        self_norm = self.norm()
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
            scale_gtn = iden_gtn.add( select_zero_gtn )

            out = self.apply(scale_gtn, inplace=inplace, compress=True )

        else:
            raise NotImplementedError

        return out

    def ovlp(self, other) -> Numeric:
        raise NotImplementedError

    def distance(self, other) -> Numeric:
        norm_a = self.frobenius_norm()
        norm_b = other.frobenius_norm()
        ovlp_ab = self.ovlp(other)
        return np.sqrt(norm_a**2 + norm_b**2 - 2*np.real(ovlp_ab))

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

    def get_data(self, ax_select:dict['Axis',int]=None, pad_data=False) -> np.ndarray:
        raise NotImplementedError

    #######################
    ## MPS/MPX functions ##
    #######################

    def apply(self, other: 'GridTN', inplace=False, zipup=False, compress=False, compress_opts=None,
              add_cc=False) -> 'GridTN':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
        """
        raise NotImplementedError

    def add(self, gtn_mpx2: 'GridTN', inplace=False, compress=False, compress_opts=None) -> 'GridTN':
        """ Add other grid_mpx (of the same type and on the same grid) to self
        """
        gtn = self if inplace else self.copy()
        if gtn.data is None:
            gtn.data = gtn_mpx2.data
        else:
            gtn.data += gtn_mpx2.data
        return gtn

    def scalar_multiply(self, scalar_const: Numeric, inplace=False) -> 'GridTN':
        raise NotImplementedError

    # def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, compress=False, compress_opts=None) -> 'GridTN':
    #     raise NotImplementedError

    @profile
    def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, zipup=False, compress=False,
                           compress_opts=None) -> 'GridTN':

        if self.data is None:
            return self

        if gtn_mps2.data is None:
            return self.grid.make_empty_gridTN(ax_deriv_configs=self.ax_deriv_configs)

        grid_mpx1 = self if inplace else self.copy()

        if isinstance(gtn_mps2, qtn.MatrixProductState):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert(self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            gtn_mps2 = self.create_like()
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

            if grid_mpx1.grid != gtn_mps2.grid:
                if set(grid_mpx1.grid.axes).issubset(set(gtn_mps2.grid.axes)):
                    grid_mpx1 = gtn_mps2.grid.pad_gtn_to_grid(grid_mpx1)
                    print('warning: elemental multiply not an inplace operation')
                elif set(gtn_mps2.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                    gtn_mps2 = grid_mpx1.grid.pad_gtn_to_grid(gtn_mps2)
                else:
                    raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {gtn_mps2.grid}')

            if gtn_mps2.data_type==DataType.MPS:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
                gtn_mps2 = gtn_mps2
                if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                    grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
                else:   ## use fact that self is MPO. apply method should take care of it
                    grid_mpo2 = grid_mpx1
                    grid_mpx1 = gtn_mps2

            elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type==DataType.MPO:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
                grid_mpo2 = gtn_mps2

        else:
            raise TypeError(f'other must be GridTN with MPS or MPO data type, not {type(gtn_mps2)}')

        grid_mpx1.apply(grid_mpo2, inplace=True, zipup=zipup, compress=compress, compress_opts=compress_opts,
                        add_cc=True)
        return grid_mpx1

    # def xmultiply(self, x_axes: Sequence['Axis'], offsets: dict[Any,Numeric]=0.0, scales: dict[Any,Numeric]=1.0,
    #               inplace=False, compress=False, compress_opts=None) -> 'GridTN':
    #     raise NotImplementedError

    def xmultiply(self, x_axes: Sequence['Axis'], x_power: int = 1, offsets: dict[Any, Numeric] = 0.0,
                  scales: dict[Any, Numeric] = 1.0, inplace=False, compress=False, compress_opts=None) -> 'GridTN1D':

        if self.data is None:
            return self

        grid_mpx1 = self if inplace else self.copy()
        xmult_mpo = grid_mpx1.grid.get_xmultiply_mpo(x_axes, x_power=x_power, offsets=offsets, scales=scales)

        grid_mpx1.apply(xmult_mpo, inplace=True)  # , compress=compress, compress_opts=compress_opts)

        if compress:
            grid_mpx1.compress(inplace=True, compress_opts=compress_opts)

        return grid_mpx1


    def canonize(self, inplace=True, scale=True, form='right',i=None) -> 'GridTN':
        raise NotImplementedError

    def compress(self, inplace=True, compress_opts: dict = None,
                 sub_compress_opts: Optional[dict['SubCompressConfigType',dict]] = None, **kwargs):
        raise NotImplementedError

    def contract(self):
        return self.data.contract()

    def scalar_add(self, scalar_val, inplace=False):
        """ add scalar values to gridTN
        """
        gtn = self if inplace else self.copy()
        ones = self.grid.get_ones_mps()
        ones.scalar_multiply(scalar_val, inplace=True)
        gtn.add(ones, inplace=True)
        return gtn

    # def apply_partial_mpx(self, mpo: MPOType or MPSType or 'GridTN', reset_grid=True, compress=True,
    #           **compress_opts) -> 'GridTN':
    #     raise NotImplementedError


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
            gtn.apply(blur_mpo, inplace=True, compress=compress, compress_opts=compress_opts)  ## not deep inplace
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
            gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

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
            gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn


    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':
        """ convert MPS into MPO
        """
        raise NotImplementedError


    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', add_cc=False,
                                    compress=False, compress_opts=None):
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
    def take_firstderivative(self, ax: 'Axis', upwind_ax: Optional['Axis']=None, recalc=False, inplace=False,
                             compress=False, compress_opts=None):
        """ df/dx_i
            bc = boundary condition
        """
        if self.data is None:
            return self

        gtn = self if inplace else self.copy()
        deriv_config = gtn.ax_deriv_configs[ax]
        ddx_mpo = self.grid.get_firstderivative_mpo(ax, deriv_config=deriv_config, upwind_ax=upwind_ax, recalc=recalc)
        gtn.apply(ddx_mpo, inplace=True)

        if compress:   gtn.compress(inplace=True, compress_opts=compress_opts)
        return gtn


    def take_secondderivative(self, ax1: 'Axis', ax2: 'Axis' or None,
                              recalc=False, inplace=False, compress=False, compress_opts=None):
        """ self.order: finite difference expansion order """
        if self.data is None:
            return self

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax1]
        deriv_config2 = gtn.ax_deriv_configs[ax2] if ax2 is not None else None
        d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, deriv_config1=deriv_config1, deriv_config2=deriv_config2,
                                                    recalc=recalc)
        gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn


    def integrate(self, integ_axes: Sequence['Axis'] = None, is_sqrt=False, ancilla_reindex:dict[str,str]=None,
                  new_grid: 'Grid' = None, new_ax_deriv_configs: dict[Any,'DerivativeConfiguration'] = None,
                  compress=False, compress_opts=None):
        raise NotImplementedError


    def meas_expec(self, obs_gtn, is_sqrt=False) -> Numeric:
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
