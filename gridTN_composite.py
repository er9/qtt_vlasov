from setup.configs import *
from gridTN import GridTN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from grid import Grid
    from grids_composite import CompositeGrid
    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']


class GridTN_Composite(GridTN):

    def __init__(self, grid: 'GridType', data: list[dict['Axis','GridTN']] = None,
                 ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None):
        """ grid:  grid on which the TN lives
            data:  data in compatible format for GridType
        """
        super().__init__(grid, data, ax_deriv_configs)


    def _get_data(self):
        return self._data

    def _set_data(self, data):
        self._data = data
        if data is not None:
            data_ = data[0]
            self._data_type = data_.data_type # type( data_[next(iter(data_))] )

    data = property(fget=_get_data,fset=_set_data)


    def _get_data_type(self):
        return self._data_type

    data_type = property(fget=_get_data_type)

    def update_deriv_params(self, ax, left_bc: Optional[BCType or str] = None,
                            right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
                            fd_type: Optional[FDType or str] = None):
        """ kwargs:  'order', 'fdtype', 'bc', 'upwind', 'v_ax'
        """
        super().update_deriv_params(ax, left_bc, right_bc, order, fd_type)


    def create_like(self) -> 'GridTN':
        raise NotImplementedError

    def copy(self, deep=True) -> 'GridTN':
        raise NotImplementedError

    def norm(self, is_sqrt=False) -> Numeric:
        raise NotImplementedError

    def max_bond(self) -> int:
        raise NotImplementedError