import pickle

import helper_dmrg
from setup_.configs import *
import helper_quimb as helper
from helper_tdvp import TDVPSolver

from typing import TYPE_CHECKING
from gridTN_1D import *  # GridTN1D
import local_solvers.helper_dmrg_loc_sum as loc_sum

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from coord.coord_sys import Coordinate
    from grid import Grid
    from grids_composite import CompositeGrid

    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']  # tuple[Grid]

""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
    there's a primary MPS/MPO "data", and it also keeps track of "error" terms that arise from compression
    treat as a group of GridTN objects
"""


########################################
## static methods to create GridTN1Ds ##
########################################


####################
## GridTN1D_Sum class ##
####################

class GridTNSum:

    def __init__(self, data: Optional[GridTN] = None,
                 error_terms: Optional[Sequence[GridTN]] = None):
        """
        Attributes:
            data:  GridTN object
            error_terms: GridTNs that are corrections to the "data" MPS/MPO
            assume that the terms are orthogonal to each other
        """
        self.primary = data
        self.corrections = [] if error_terms is None else [et for et in error_terms]

    @property
    def terms(self) -> list['GridTN']:
        primary_term = [self.primary] if (self.primary is not None and self.primary.data is not None) else []
        return primary_term + self.corrections

    @property
    def num_terms(self):
        return len(self.corrections) + 1

    @property
    def grid(self) -> 'Grid':
        return self.primary.grid

    def create_like(self, new_data: GridTN =None, new_errors: Sequence[GridTN]=None) -> 'GridTNSum':
        new_tn1D = self.__class__(data=new_data, error_terms=new_errors)
        return new_tn1D

    def copy(self, deep=True) -> 'GridTNSum':
        new_tn1D = self.create_like()
        if self.primary is not None:
            new_tn1D.primary = self.primary.copy() if (deep and self.primary is not None) else self.primary

        new_tn1D.corrections = [err.copy() for err in self.corrections]
        return new_tn1D

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTNSum':
        new_tn1D = self if inplace else self.copy()
        new_tn1D.primary.conj(mangle_inner=mangle_inner, inplace=True)
        for err in self.corrections:
            err.conj(mangle_inner=mangle_inner, inplace=True)
        return new_tn1D

    def norm(self, is_sqrt=False) -> Numeric:
        out = self.primary.norm(is_sqrt=is_sqrt)
        for err in self.corrections:
            out += err.norm(is_sqrt=is_sqrt)
        if np.abs(np.imag(out)) < 1.0e-13:
            out = np.real(out)
        return out

    def ovlp(self, other: Union['GridTN', 'GridTNSum']) -> Numeric:

        if isinstance(other, GridTN):
            other_terms = other.data if other.data is not None else []
        else:
            other_terms = other.terms

        tot_ovlp = 0.0
        for t1 in self.terms:
            for t2 in other_terms:
                ovlp = t1.ovlp(t2)
                tot_ovlp += ovlp

        return tot_ovlp

    def get_value(self, compress=True, max_bond: int=None, cutoff: Numeric=CUTOFF) -> 'GridTN':
        out = self.primary.copy()
        for err in self.corrections:
            out.add(err, compress=False, inplace=True)
        if compress:
            print('compress', max_bond, cutoff)
            out.compress( inplace=True, compress_opts={'max_bond': max_bond, 'cutoff': cutoff})
        return out

    def distance(self, other) -> Numeric:

        if isinstance(other, GridTN):
            other_terms = other.data if other.data is not None else None
        else:
            other_terms = other.get_gtn()

        self_terms = self.get_value()
        return self_terms.distance(other_terms)


    def max_bonds(self) -> list[int]:
        return [term.max_bond() for term in self.terms]

    def mem_size(self) -> Numeric:
        """ returns in kilobytes
        """
        mem_sizes = [term.mem_size() for term in self.terms]
        return np.sum(mem_sizes) / 1000


    def check_orthog(self):
        for et in self.terms:
            et.check_orthog()

    def get_anchor_ind(self) -> int:
        return self.primary.get_anchor_ind()

    def transpose(self, inplace=True, mangle_inner=False):
        """ take transpose of MPO
        """
        gtn = self if inplace else self.copy()
        for et in gtn.terms:
            et.transpose(inplace=True, mangle_inner=mangle_inner)
        return gtn

    def get_like_iden(self):
        """ get identity MPO with appropriate shape
            note: allows MPO to be of different L from grid
        """
        return self.primary.get_like_iden()

    #######################################
    ## class methods to build common TNs ##
    #######################################

    @classmethod
    def get_ones_mps(cls, grid, site_ind_id='i({})', site_tag_id='X({})'):
        """ build ones vector mps on sepcified grid
        """
        return cls(data=grid.get_ones_mps(site_ind_id, site_tag_id))

    @classmethod
    def get_iden_mpo(cls, grid, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build identity mpo on specified grid
        """
        return cls(data=grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id))

    @classmethod
    def get_select_elem_mps(cls, grid, inds, site_ind_id='i({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        return cls(data=grid.get_select_elems_mps(inds, site_ind_id, site_tag_id))

    @classmethod
    def get_select_elem_mpo(cls, grid, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """ build MPO to select certain elements specified by inds
        """
        return cls(data=grid.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id))

    @classmethod
    def from_dense_state(cls, data, grid: 'Grid', site_ind_id='i({})', site_tag_id='T({})', split_opts=None, axes=None):
        mps = grid.map_state_to_mps(data, site_ind_id, site_tag_id, split_opts=split_opts)
        return cls(data=mps)

    @classmethod
    def from_dense_operator(cls, data, grid, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            split_opts=None, **kwargs):
        mpo = grid.map_operator_to_mpo(data, upper_ind_id, lower_ind_id, site_tag_id, split_opts=split_opts)
        return cls(data=mpo)

    def get_data(self, ax_order=None, ax_select: Optional[dict[int]] = None, pad_data=False):
        """ ax_select:  select element along each axis
        """
        tot_data = 0.0
        for et in self.terms:
            et_dat = et.get_data(ax_order=ax_order, ax_select=ax_select, pad_data=pad_data)
            if et_dat is not None:
                tot_data += et_dat

        return tot_data

    def check_ovlp(self):
        nt = self.num_terms
        terms = self.terms
        norm = self.norm(is_sqrt=True)   # np.linalg.norm(self.get_data())
        for i, j in np.ndindex(nt,nt):
            if i > j:
                continue
            print('i,j:', i, j, terms[i].ovlp(terms[j]) / norm**2)


    #######################
    ## MPS/MPX functions ##
    #######################

    # def _parse_input_TN1D(self, other):
    #     if isinstance(other, type(self)):
    #         assert (other.grid == self.grid), 'self and other gridTN need to live on the same grids'
    #         other = other.data
    #     else:
    #         mpx_shape = tuple([other.phys_dim(i) for i in range(other.L)])
    #         assert (mpx_shape == self.grid.shape), 'self and mpo need to have the same shape'
    #     return other

    @classmethod
    def add_list(cls, gtns_list: Sequence['GridTN'], compress_opts=None, cur_orthog: int=None) -> 'GridTNSum':

        max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
        cutoff = compress_opts.get('cutoff', CUTOFF) if compress_opts is not None else CUTOFF

        ref_gtn = next(iter(gtns_list))
        grid = ref_gtn.grid
        is_mps = ref_gtn.data_type == DataType.MPS

        tts_list = [grid.gtn_to_dmrg_format(ogtn) for ogtn in gtns_list]

        out, remainders = loc_sum.add_MPS_list_remainder(tts_list, max_bond=max_bond, cutoff=cutoff,
                                                         cur_orthog=cur_orthog)
        gtn = grid.dmrg_to_gtn_format(ref_gtn, out, is_mps=is_mps, inplace=False)

        gtn_remainders = [grid.dmrg_to_gtn_format(ref_gtn, rem, is_mps=is_mps, inplace=False) for rem in remainders]
        out = cls(data=gtn, error_terms=gtn_remainders)
        return out


    # @profile
    def apply(self, other: Union['GridTN', Sequence['GridTN'], 'GridTNSum'], compress_opts=None, ) -> 'GridTNSum':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
            inplace = False; not an inplace operation
        """

        if isinstance(other, (list,tuple)):
            other_terms = other
        elif isinstance(other, GridTN):
            other_terms = [other]
        elif isinstance(other, GridTNSum):
            other_terms = other.terms
        else:
            raise TypeError(f'{other.__class__} is not a valid input for GridTNSum "apply" operation')

        out_terms = []
        for t1 in self.terms:
            for t2 in other_terms:
                out = t1.apply(t2, compress=False, zipup=True, compress_opts={'form': 'right'})     # right canonical
                out_terms += [out]

        ## sum terms together via DMRG and then decompose into multiple sum terms
        out = self.add_list(out_terms, compress_opts=compress_opts, cur_orthog=0)
        return out


    def solve(self, operator: 'GridTN', compress_type: CompressType, inplace=False, use_A2=False, compress_opts=None,
              is_H=False, init_guess: Optional['GridTN'] = None, verbose_output=False, **kwargs
              ) -> Union[tuple['GridTN1D', float, bool], 'GridTN1D']:
        """ solve Ax=b using local optimization methods
        """
        raise NotImplementedError

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if operator is None:
            return self if inplace else self.copy()

        grid_mpx1 = self if inplace else self.copy()
        grid_mpx1.is_constant *= operator.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in operator.constant_axes]

        ## pad grid if necessary
        grid_mpx1, operator = grid_mpx1._match_grids(operator)
        # print('padded other', other)

        if compress_type is CompressType.DMRG:
            # print('solve compress opts', compress_opts)
            init_guess_data = init_guess.data if init_guess is not None else None
            max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
            if use_A2:
                out_data, err, is_conv = \
                    helper_dmrg_2.dmrg_solve_2(grid_mpx1.data, operator.data, init_guess=init_guess_data,
                                               opt_nsites=DEFAULT_OPT_NSITES, is_H=is_H,
                                               max_bond=max_bond, solve_type=DEFAULT_SOLVE_TYPE, **kwargs,
                                               )
            else:
                out_data, err, is_conv = \
                    helper_dmrg.dmrg_solve(grid_mpx1.data, operator.data, init_guess=init_guess_data,
                                           opt_nsites=DEFAULT_OPT_NSITES, is_H=is_H,
                                           max_bond=max_bond, solve_type=DEFAULT_SOLVE_TYPE, **kwargs,
                                           )

        elif compress_type is CompressType.MG:
            raise NotImplementedError('MG solver not yet implemented')
        else:
            raise ValueError('solver not compatible with', compress_type)

        grid_mpx1.data = out_data
        if verbose_output:
            return grid_mpx1, err, is_conv
        return grid_mpx1

    def add(self, gtn_mpx2: Union['GridTN', 'GridTNSum'], compress_opts=None) -> 'GridTNSum':
        """ Add other grid_mpx (of the same type and on the same grid) to self
            not inplace
        """
        other_terms = [gtn_mpx2] if isinstance(gtn_mpx2, GridTN) else gtn_mpx2.terms
        out = self.add_list(self.terms + other_terms, compress_opts=compress_opts)
        return out


    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTNSum':

        gtn = self.copy()
        gtn_mpx = gtn.primary.data
        sub_mpx = sub_gtn_mpx2.data

        if sub_gtn_mpx2 is None or sub_gtn_mpx2.data is None:
            return gtn

        sub_inds = []
        for ax in sub_gtn_mpx2.grid.axes:
            sub_inds += self.grid.get_inds_in_axis(ax)

        min_ind = np.min(sub_inds)
        max_ind = np.max(sub_inds)

        gtn.primary.data = helper.add_submpx(gtn_mpx, sub_mpx, (min_ind, max_ind + 1), open_bc=open_bc, inplace=True,
                                             compress=compress, compress_opts=compress_opts)
        return gtn


    def scalar_multiply(self, scalar_const, inplace=False) -> 'GridTNSum':
        gtn = self if inplace else self.copy()
        for term in gtn.terms:
            term.scalar_multiply(scalar_const, inplace=True)
        return gtn

    def evaluate_func(self, func: Callable, max_bond=None, inplace=False):
        # from local_solvers.local_cross_eval_old import local_cross_evaluator, Term_Cross
        raise NotImplementedError
        ## not trivial is function evaluation is not linear
        ## i guess using cross and targeting all of the same indices, it might be possible

    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTNSum':
        gtn = self if inplace else self.copy()
        for tm in self.terms:
            tm.canonize(inplace=True, scale=scale, form=form, i=i, cur_orthog=cur_orthog)
        return gtn

    def compress(self, inplace=True, compress_opts=None, norm_cutoff=None,
                 conservative=False, **kwargs) -> 'GridTNSum':

        gtn_list = [self.primary] + self.corrections
        out = self.add_list(gtn_list, compress_opts=compress_opts)

        if inplace:
            self.primary.data = out.primary.data
            self.corrections = out.corrections
            return self

        return out


    ###############################
    ## application of operations ##
    ###############################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTNSum':
        gtn = self if inplace else self.copy()
        for et in gtn.terms:
            et.mps_to_diag_mpo(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id, inplace=inplace)
        return gtn

    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """ Use 3-leg elemental multiply TN, apply to self to turn into mpo
            for elemental multiplication (with some other, yet to be specified gmps)
            assumes indices of self goes from 0 to grid.L-1 (no missing indices)
            lower_ind_id, upper_ind_id = desired formats for output MPO
        """
        new_primary = self.primary.apply_elemental_multiply_op(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id,
                                                               axes=axes, add_cc=add_cc, take_mps_cc=take_mps_cc,
                                                               compress=compress, compress_opts=compress_opts)
        new_corrections = [et.apply_elemental_multiply_op(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id,
                                                               axes=axes, add_cc=add_cc, take_mps_cc=take_mps_cc,
                                                               compress=compress, compress_opts=compress_opts)
                           for et in self.corrections]

        return self.__class__(new_primary, new_corrections)

    def take_firstderivative(self, ax: 'Axis', upwind_ax: Optional['Axis'] = None,
                             ax_deriv_config: dict['Axis', 'DerivativeConfiguration'] = None,
                             recalc=False, inplace=False, compress=False, compress_opts=None):
        """ df/dx_i
            bc = boundary condition
        """
        gtns = self if inplace else self.copy()

        gtns.primary.take_firstderivative(ax, upwind_ax=upwind_ax, ax_deriv_config=ax_deriv_config,
                                          recalc=recalc, inplace=True, compress=True)
        corrections = []
        for et in gtns.corrections:
            et = et.take_firstderivative(ax, upwind_ax=upwind_ax, ax_deriv_config=ax_deriv_config,
                                         recalc=recalc, inplace=True, compress=True)
            corrections += [et]

        gtns.corrections = corrections
        if compress:
            gtns.compress(compress_opts=compress_opts, inplace=True)

        return gtns



    ###############
    ## integration/partial integration ##
    ################

    def apply_mps(self, mps, inplace=False, compress=True, compress_opts=None):
        """ Apply MPS on top of an MPS -> # or MPO -> MPS
        """
        raise NotImplementedError
        # if isinstance(mps, GridTN1D):
        #     mps = mps.data
        #
        # assert isinstance(mps, qtn.MatrixProductState), 'supplied TN must be MPS'
        #
        # if isinstance(self.data, qtn.MatrixProductOperator):
        #     gtn = self if inplace else self.copy()
        #     helper.mpo_transpose(gtn.data, inplace=True)
        #     out = helper.apply(gtn.data, mps, compress=compress, compress_opts=compress_opts)
        #     gtn.data = out
        #     return gtn
        # elif isinstance(self.data, qtn.MatrixProductState):
        #     out = helper.ovlp(self.data, mps)
        #     return out
        # else:
        #     raise NotImplementedError

    # @profile
    def meas_expec(self, obs_gtn: 'GridTN1D', integ_axes=None, is_sqrt=False, ancilla_reindex=None,
                   new_grid: 'Grid1D' = None, exclude_axes=None, exclude_weights=False,
                   new_ax_deriv_configs=None, compress=False, compress_opts=None,
                   zipup=False, zipup_direction=-1, **kwargs):
        """ integrate over specified dimensions. returns scalar if axes='all'
            otherwise, returns a Field object with reduced dimensionality
            inplace is dummy parameter.
            ancilla_reindex:  dict[old_anc: new_anc] for reindexing MPS when is_sqrt==True
                ancilla must be on the anchor tensor (indexed by 0)
                new and old ancilla are fused together -> old ancilla name
        """
        kwargs = {'integ_axes': integ_axes, 'is_sqrt': is_sqrt, 'ancilla_reindex': ancilla_reindex,
                  'new_grid': new_grid, 'exclude_axes': exclude_axes, 'exclude_weights': exclude_weights,
                  'new_ax_deriv_config': new_ax_deriv_configs, 'compress': False}

        # ovlp = self.primary.meas_expec(obs_gtn, **kwargs)

        if is_sqrt:
            num_terms = len(self.terms)
            ## "exact"
            for i, j in np.ndindex(num_terms):
                t1, t2 = self.terms[i], self.terms[j]
                raise NotImplementedError

        else:
            expec_terms = []
            for et in self.terms:
                expec_terms += [et.meas_expec(obs_gtn, **kwargs)]

            if isinstance(expec_terms[0], GridTN):
                return self.add_list(expec_terms)
            else:
                return np.sum(expec_terms)


    def integrate(self, integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None, exclude_weights=False,
                  compress=False, compress_opts=None, ancilla_reindex: dict[str, str] = None, **kwargs) \
            -> Union[Numeric, 'GridTNSum']:
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if self.primary.data is None and len(self.corrections) == 0:
            return np.nan

        return self.meas_expec(None, is_sqrt=is_sqrt, integ_axes=integ_axes, new_grid=new_grid,
                               exclude_weights=exclude_weights, new_ax_deriv_configs=new_ax_deriv_configs,
                               ancilla_reindex=ancilla_reindex, compress=compress, compress_opts=compress_opts)

    #######################
    ###  Local TE fcts  ###
    #######################

    def evolve_tdvp(self, dt, mpo_list, te_order=0, do_adapt=False, inplace=False,
                    compress_config: CompressionConfiguration = None, expand_basis=None):
        raise NotImplementedError

    def evolve_tdmrg(self, dt, mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
        raise NotImplementedError

    def evolve_tdvp_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
                        compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
                        solver_type=LocalSolverType.TDDMRG, filter_bases=False, time=None, **kwargs):

        raise NotImplementedError

    def evolve_tdmrg_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
                         compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
                         solver_type=LocalSolverType.TDDMRG, filter_bases=False, time=None, **kwargs):

        raise NotImplementedError

