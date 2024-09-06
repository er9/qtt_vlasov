from setup_.defaults import *
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup_.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper


if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN
    from field import Field, ScalarField


class EigSolver:

    def __init__(self,*fields: Union['Field','ScalarField'], field_names: Sequence[str] = None,
                 # field_compress_config: Optional[CompressionConfiguration] = None,
                 compress_levels = None, **kwargs
                 ):
        """ npts:  number of discretized points along each axis
            ndim:  dimension of system
            vec_fields:  list of strs to act as keys indexing vector fields
            scalar_fields:  list of strs to act as keys indexing scalar fields
            grid_type:  str describing grid_structure
        """

        if compress_levels is None:
            compress_levels = list(range(1,10))     # some arbitrary number; compress at all levels
        self._comp_levels = compress_levels

        self._fields: dict[Any, Optional[Field]] = {field.name: field for field in fields if field is not None}
        self.field_names = [field.name for field in fields if field is not None] \
                              if field_names is None else field_names

        ## pad system with 'None' fields as needed, prescribed by field_names
        for fn in self.field_names:
            if fn not in self._fields:  self._fields[fn] = None



    def __getitem__(self,field_name):
        """ return ith component of the field. for convenience
        """
        return self.get_field(field_name)   ## allow KeyError to be raised


    def __setitem__(self, field_name, new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        self.set_field(field_name, new_field)

    @property
    def fields(self):
        return {fn: self.get_field(fn) for fn in self.field_names}
        # return self._fields

    def get_field(self, field_name):
        """ return ith component of the field. for convenience
        """
        return self._fields[field_name]

    def set_field(self, field_name, new_field):
        """ set ith component of the field. for convenience
                    new_field is a Field object
        """
        assert (isinstance(new_field, Field) or new_field is None), 'new_field must be Field object or None'
        self._fields[field_name] = new_field


    def _get_compress_levels(self, compress_level, num_levels):

        if compress_level == 0:
            return [0] * num_levels

        comp_levels = self._comp_levels[compress_level-1:compress_level-1+num_levels]
        if len(comp_levels) < num_levels:
            comp_levels += [0]*(num_levels-len(comp_levels))
        return comp_levels



    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this without defining the fields
        """
        new_system = self.__class__(*new_fields, field_names=self.field_names,
                                    compress_levels=self._comp_levels)
                                    # field_compress_config=self.field_compress_config)
        return new_system


    def copy(self):
        """ copy self including the fields
        """
        copy_fields = [self.get_field(k).copy() if self.get_field(k) is not None else None
                       for k in self.field_names]
        new_system = self.create_like(*copy_fields, recalc=False)
        return new_system


