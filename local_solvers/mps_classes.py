from setup.defaults import *

class MPS(qtn.MatrixProductState):

    cur_orthog: int = None
    select_inds: dict[int, Sequence[int]] = None

    @classmethod
    def from_quimb(cls, mps: 'qtn.MatrixProductState'):
        return MPS(mps)

    def copy(self, virtual=False, deep=True):
        out = MPS(self, virtual=virtual, deep=deep)
        out.cur_orthog = self.cur_orthog
        if self.select_inds is not None:
            out.select_inds = {k:v for k,v in self.select_inds.items()} if deep else self.select_inds
        return out

    def get_tens(self, i: int) -> Optional[qtn.Tensor]:
        try:
            return self[i]
        except IndexError:
            return None

    def bond(self, i:int, j:int):
        if (i == 0 and j == -1) or (i == -1 and j == 0):
            tens = self[0]
            bond_r = super().bond(0, 1)
            phys_i = self.site_ind(0)
            for ind in tens.inds:
                if ind != bond_r and ind != phys_i:
                    return ind
            return None
        elif (i == self.L and j == self.L-1) or (i == self.L-1 and j == self.L):
            tens = self[self.L-1]
            bond_l = super().bond(self.L-1, self.L-2)
            phys_i = self.site_ind(self.L-1)
            for ind in tens.inds:
                if ind != bond_l and ind != phys_i:
                    return ind
            return None
        else:
            return super().bond(i, j)

    def get_bra_tens(self, i: int, reindex_phys=False) -> Optional[qtn.Tensor]:
        tens = self.get_tens(i)
        if tens is None:
            return None
        bra_tens = self[i].conj()
        phys_ind = self.site_ind(i)
        bra_tens.modify(inds=[(ind if (ind == phys_ind and not reindex_phys) else ind + '_')
                              for ind in bra_tens.inds])
        return bra_tens

    def canonize(self, where, cur_orthog='calc', bra=None):
        # if self.cur_orthog is not None:
        #     cur_orthog = self.cur_orthog
        super().canonize(where, cur_orthog=cur_orthog, bra=bra)
        self.cur_orthog = where
        if bra is not None:
            bra.cur_orthog = where


MPO = qtn.MatrixProductOperator
