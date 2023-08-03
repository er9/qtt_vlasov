from setup.configs import *
import helper_quimb as helper
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridTN

from setup.quimb_TN1D import MatrixProductTensor, MatrixProductStateTN


class TN_State(qtn.TensorNetwork):
    """ TN of MPS * MPO
    """
    def __init__(self, data: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
                 outer_ind_id: str, L:int=None, inds=None, virtual=False, check_collisions=True):

        data = data if virtual else [t.copy() for t in data]
        super().__init__(data, virtual=True, check_collisions=check_collisions)
        self._outer_ind_id = outer_ind_id
        self.L = len(self.shape) if L is None else L
        self.inds = range(self.L) if inds is None else None

    @property
    def site_ind_id(self):
        return self._outer_ind_id

    @site_ind_id.setter
    def site_ind_id(self, new_ind_id):
        if new_ind_id != self.site_ind_id:
            ## check collisions
            clash_ix = self._inner_inds & tuple([new_ind_id.format(i) for i in self.inds])
            reind = {ix: qtn.rand_uuid() for ix in clash_ix}
            if len(reind) > 0:
                self.reindex(reind, inplace=True)
            self.reindex({self.site_ind_id.format(i): new_ind_id.format(i)
                          for i in self.inds}, inplace=True)
            self._outer_ind_id = new_ind_id


def tn_elemental_multiply(grid: 'Grid', tns_1: Sequence['TN_State', 'MatrixProductStateTN'],
                          tns_2: Sequence['TN_State', 'MatrixProductStateTN']
                          ) -> list['TN_State', 'MatrixProductStateTN']:
    """ sum(diag(MPS)) * sum(MPS TNs)
    """

    elem_mult_tn: MatrixProductTensor = grid.get_elemental_multiply_tn().get_TN()

    out_tns = []
    for tn1 in tns_1:
        for tn2 in tns_2:
            if tn1.site_ind_id == tn2.site_ind_id:
                tn2.site_ind_id = tn2.site_ind_id + '_1'
            elem_mult_tn.lower_ind_id = tn1.site_ind_id
            elem_mult_tn.reindex_extra_inds(0, tn2.site_ind_id)

            mult_tn = TN_State([tn1, tn2, elem_mult_tn], outer_ind_id=elem_mult_tn.upper_ind_id,
                               inds=range(tn1.L))
            out_tns += [mult_tn]
    return out_tns

def tn_multiply_sign(tns: Sequence['TN_State', 'MatrixProductStateTN'], sign, inplace=True
                     ) -> list['TN_State', 'MatrixProductStateTN']:
    out_tns = tns if inplace else [tn.copy() for tn in tns]
    for tn in out_tns:
        iterator = iter(tn.tensors)
        tens = next(iterator)
        while tens.norm() < 1.0e-10:
            tens = next(iterator)
        tens.modify(apply=lambda x: x * sign)
    return out_tns


def apply(tns_1: Sequence['TN_State', 'MatrixProductStateTN'], mpo: 'qtn.MatrixProductOperator'
          ) -> list['TN_State', 'MatrixProductStateTN']:
    """ MPO * sum(MPS TNs)
    """
    out_tns = []
    for tn1 in tns_1:
        tn1 = tn1.copy()
        tn1.site_ind_id = mpo.lower_ind_id
        tn1._site_ind_id = mpo.upper_ind_id
        tn1.add(mpo.tensors)
        out_tns += [tn1]
    return out_tns


class FieldTN(Field):
    """ Field object but where data are lists of TNs that are summed together
    """

    def __init__(self, name: str, grid: 'Grid',
                 data: dict['Coordinate', Sequence['qtn.TensorNetwork', 'MatrixProductStateTN']],
                 compress_config: Optional['CompressionConfiguration'] = None,
                 # deriv_configs: Optional[dict]=None,
                 is_sqrt=False):  # , init_split_opts: dict = None):  # , compress_opts: Optional[dict]=None):
        """
        name (str):  name of the field
        ncomp (int): maximum number of components (eg. 1 if scalar field, ndim if vector field)
        grid:  Grid object on which components live
        data:  dictionary containing MPSs indexed by integer denoting which component of the field
               the MPS represents (eg. 0 if MPS is representing a scalar field)
               keys:  ax.axID if a vector field OR None if a scalar field
        boundary_conditions: nested dict.
                first level: dict of boundary conditions for each field component.
                             define for all possible components
                second level: dictionary of boundary conditions along each axis
                              periodic, antiperiodic, open (default), reflecting, zero gradient
                              (elements are tuples to specify left end/right end of axis)
        coords: list of coordinate systems (predefined)
                coordinate systems take dict
                coords_ax:  dictionary in which the coordinate position (BCType or int) keys the desired axis

        """
        super().__init__(name, grid, data, compress_config=compress_config, is_sqrt=is_sqrt)


    def cross_product(self, other, coord_sys: 'CoordinateSystem', comps=None, other_comps=None,
                      check_elem_mult_anc=False, compress_level=0, inner_compress_level=0, zipup=False,
                      verbose_plot=False) -> 'FieldTN':

        if comps is None:           comps = coord_sys.coords
        if other_comps is None:     other_comps = comps

        out = {}
        for i in range(3):

            c0 = comps[i]
            c1, c2 = comps[(i + 1) % 3], other_comps[(i + 2) % 3]
            # print('out', c0, 'cross', c1, c2)

            try:
                out_comp_1 = tn_elemental_multiply(self.grid, self[c1], other[c2])

            except KeyError:
                out_comp_1 = None  ## vector_field[i] = 0

            try:
                out_comp_2 = tn_elemental_multiply(self.grid, self[c2], other[c1])
                out_comp_2 = tn_multiply_sign(out_comp_2, -1, inplace=True)


            except KeyError:
                out_comp_2 = None  ## vector_field[i] = 0

            if out_comp_1 is None and out_comp_2 is None:
                continue
            elif out_comp_1 is None and out_comp_2 is not None:
                out[c0] = out_comp_2
            elif out_comp_1 is not None and out_comp_2 is None:
                out[c0] = out_comp_1
            else:
                out[c0] = out_comp_1 + out_comp_2

        new_field = self.create_like(new_components=out)
        # if len(out) > 0:
        #     new_field.grid = out[next(iter(out))].grid

        return new_field
