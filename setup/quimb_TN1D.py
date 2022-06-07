from typing import Union, Optional, Sequence
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork
from quimb.tensor.tensor_1d import TensorNetwork1D, TensorNetwork1DFlat, MatrixProductOperator



class MatrixProductTensor(MatrixProductOperator, TensorNetwork1DFlat, TensorNetwork1D, TensorNetwork):
    """ generalization of MPS, MPO to higher number of legs at each tensor
    """

    _EXTRA_PROPS = (
        '_L',
        'cyclic',
        '_site_tag_id',
        '_upper_ind_id',
        '_lower_ind_id',
        '_extra_ind_ids',   ## list of extra inds on each tensor
    )

    def __init__(self, tensors: Union[Sequence['qtn.Tensor'], TensorNetwork], upper_ind_id='o({})',
                 lower_ind_id='i({})', extra_ind_ids=()):

        if not isinstance(tensors, TensorNetwork):
            tensors = TensorNetwork(tensors)

        tensors.view_as(qtn.MatrixProductOperator, inplace=True, L=tensors.num_tensors, cyclic=False,
                        upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        super().__init__(tensors)

        self._extra_ind_ids = extra_ind_ids


    def reindex_extra_inds(self, extra_idx, new_ind):
        old_ind = self._extra_ind_ids[extra_idx]
        self.reindex({old_ind.format(i): new_ind.format(i) for i in range(self.L)}, inplace=True)
        self._extra_ind_ids = self._extra_ind_ids[:extra_idx] + (new_ind,) + self.extra_ind_ids[extra_idx+1:]

    @property
    def extra_ind_ids(self):
        return self._extra_ind_ids


    def left_canonize_site(self, i, bra=None):
        raise NotImplementedError

    def right_canonize_site(self, i, bra=None):
        raise NotImplementedError

    def left_compress_site(self, i, bra=None, **compress_opts):
        raise NotImplementedError

    def right_compress_site(self, i, bra=None, **compress_opts):
        raise NotImplementedError