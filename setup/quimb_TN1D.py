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

    def __init__(self, tensors: Union[Sequence['qtn.Tensor'], TensorNetwork], site_tag_id='T({})',
                 upper_ind_id='o({})',lower_ind_id='i({})', extra_ind_ids=(), **tn_opts):

        if not isinstance(tensors, TensorNetwork):
            tensors = TensorNetwork(tensors, **tn_opts)

        tensors.view_as(qtn.MatrixProductOperator, inplace=True, L=tensors.num_tensors, cyclic=False,
                        site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        super().__init__(tensors)

        self._extra_ind_ids = extra_ind_ids


    def copy(self, virtual=False, deep=False):
        """ TODO: copying returns an MPO object??
        """
        out = self.__class__(self, upper_ind_id=self.upper_ind_id, lower_ind_id=self.lower_ind_id,
                             extra_ind_ids=self.extra_ind_ids, site_tag_id=self.site_tag_id)
        return out


    # def view_like(self, like, inplace=False, **kwargs):
    #     raise NotImplementedError


    def reindex_extra_inds(self, extra_idx, new_ind):
        old_ind = self._extra_ind_ids[extra_idx]
        self.reindex({old_ind.format(i): new_ind.format(i) for i in range(self.L)}, inplace=True)
        self._extra_ind_ids = self._extra_ind_ids[:extra_idx] + (new_ind,) + self.extra_ind_ids[extra_idx+1:]

    @property
    def extra_ind_ids(self):
        return self._extra_ind_ids



class MatrixProductTN(TensorNetwork):
    """ generalization of MPS, MPO to higher number of legs at each tensor
    """
    def __init__(self, mpxs: Sequence[Union['qtn.MatrixProductState','qtn.MatrixProductOperator']], virtual=False):

        mpxs = mpxs if virtual else [mpx.copy() for mpx in mpxs]

        super().__init__(mpxs, virtual=True)
        self.data = self
        self.mpxs = mpxs
        self.L = mpxs[0].L
        # self.tensors = self.data.tensors

    def copy(self, virtual=False, deep=False):
        out = self.__class__(self.mpxs, virtual=virtual)
        return out

    # def conj(self, mangle_inner=False, inplace=False):
    #     out = self if inplace else self.copy()
    #     out.data.conj(mangle_inner=mangle_inner, inplace=True)
    #     return out

    def view_as(self, cls_name, inplace=False, **kwargs):
        out = self if inplace else self.copy()
        if cls_name is qtn.MatrixProductState or cls_name is qtn.MatrixProductOperator:
            for i in range(self.L):
                out = out.contract( out.site_tags(i), inplace=True)
            out.view_as(cls_name, inplace=True)
            return out
        elif cls_name is qtn.TensorNetwork:
            return out.data
        else:
            raise TypeError

    def get(self, i):
        """ get tensors associated with position i
        """
        return [mpx[i] for mpx in self.mpxs]

    def __getitem__(self, pos):
        return self.get(pos)

    def site_tags(self, i):
        tags = ()
        for mpx in self.mpxs:
            tags = tags + mpx.tags(i)
        return tags

    def bond(self, i, j):
        """ get associated bonds
        """
        return [mpx.bond(i, j) for mpx in self.mpxs]

    def apply(self, other, check_collisions=True):
        raise NotImplementedError
        # # out = self.copy()
        # other, out = qtn.tensor_network_align(other, self, inplace=False)
        # out.add(other, virtual=True, check_collisions=check_collisions)
        # out.upper_ind_id = other.upper_ind_id
        # return out

    # def add(self, other: Union['qtn.TensorNetwork','MatrixProductTN'], virtual=False, check_collisions=True):
    #     out = self.data.copy()
    #     other_tens = other.tensors if virtual else [t.copy() for t in other.tensors]
    #     out.add(other_tens, check_collisions=check_collisions)
    #     return out



class MatrixProductStateTN(MatrixProductTN):
    """ generalization of MPS, MPO to higher number of legs at each tensor
    """
    def __init__(self, mps: 'qtn.MatrixProductState', mpos: list['qtn.MatrixProductOperator']=None, virtual=False,
                 site_ind_id=None):

        mpos = [] if mpos is None else mpos
        super().__init__([mps] + mpos, virtual=virtual)
        self._operators = self.mpxs[1:]
        self._mps = self.mpxs[0]

        if len(self._operators) > 0:
            if site_ind_id is None:
                site_ind_id = self._mps.site_ind_id

            if len(self._operators) > 0:
                self._mps.site_ind_id = site_ind_id + '_tmp_'
                self._operators[0].lower_ind_id = self._mps.site_ind_id
                qtn.tensor_network_align(*self._operators[::-1], self._mps, inplace=True)
                self._operators[-1].upper_ind_id = site_ind_id

        # print('self.data', self.data)
        # self.data = qtn.TensorNetwork([self._mps] + self._operators, virtual=True)
        self._site_ind_id = site_ind_id

    @property
    def site_ind_id(self):
        return self._site_ind_id

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if len(self._operators) > 0:
            self._operators[-1].upper_ind_id = new_id
        else:
            self._mps.site_ind_id = new_id
        self._site_ind_id = new_id

    def copy(self, virtual=False, deep=False):
        out = self.__class__(self._mps, self._operators, virtual=virtual, site_ind_id=self.site_ind_id)
        return out

    def reindex_sites(self, new_site_ind_id, inplace=False):
        out = self if inplace else self.copy()
        if len(out._operators) > 0:
            out._operators[-1].upper_ind_id = new_site_ind_id
        else:
            out._mps.site_ind_id = new_site_ind_id
        return out

    def apply(self, other, check_collisions=True):
        out = self.copy()
        other = other.copy()
        if out.site_ind_id == other.upper_ind_id:
            other.lower_ind_id += '_tmp'
        other, out = qtn.tensor_network_align(other, self, inplace=True)
        out.add(other, virtual=True, check_collisions=check_collisions)
        out.upper_ind_id = other.upper_ind_id
        return out


class MatrixProductOperatorTN(MatrixProductTN):
    """ stacked MPOs: mpos[0] is applied first, e.g. An An-1 ... A1 A0 (times x)
    """
    def __init__(self, mpos: Sequence['qtn.MatrixProductOperator'], virtual=False,
                 lower_ind_id=None, upper_ind_id=None):

        super().__init__(mpos, virtual=virtual)
        self._operators = self.mpxs

        if len(self._operators) > 0:
            qtn.tensor_network_align(*self._operators[::-1], inplace=True)
            if lower_ind_id is not None:
                self._operators[0].lower_ind_id = lower_ind_id
            if upper_ind_id is not None:
                self._operators[-1].upper_ind_id = upper_ind_id

        # self.data = qtn.TensorNetwork(self._operators, virtual=True)
        self._upper_ind_id = self._operators[-1].upper_ind_id
        self._lower_ind_id = self._operators[0].lower_ind_id

    @property
    def upper_ind_id(self):
        return self._upper_ind_id

    @upper_ind_id.setter
    def upper_ind_id(self, new_id):
        self._operators[-1].upper_ind_id = new_id
        self._upper_ind_id = new_id

    @property
    def lower_ind_id(self):
        return self._lower_ind_id

    @lower_ind_id.setter
    def lower_ind_id(self, new_id):
        self._operators[0].lower_ind_id = new_id
        self._lower_ind_id = new_id

    def copy(self, virtual=False, deep=False):
        out = self.__class__(self._operators, virtual=virtual, upper_ind_id=self.upper_ind_id,
                             lower_ind_id=self.lower_ind_id)
        return out

    def apply(self, other, check_collisions=True):
        out = self.copy()
        other = other.copy()
        if out.upper_ind_id == other.upper_ind_id:
            other.lower_ind_id += '_tmp'
        other, out = qtn.tensor_network_align(other, self, inplace=True)
        out.add(other, virtual=True, check_collisions=check_collisions)
        out.upper_ind_id = other.upper_ind_id
        return out