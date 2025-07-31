import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
from setup_.quimb_TN1D import MatrixProductTensor
from local_solvers.blocks import *
# from local_solvers_old.helper_tn import get_mps_matching_inds, sum_tens
# from helper_cross_v3 import canonize, canonize_tens_list, compress_tens_list
import local_solvers.helper_cross as helper_cross
from local_solvers.mps_classes import MPS, MPO
from setup_.quimb_TN1D import MatrixProductOperatorTN, MatrixProductGroup
import local_solvers.tensor_callables as tc
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.terms_3 import Term
from local_solvers.blocks_nlin import Block

# from setup_.TN_mpo import TNBlock_DMRG, TNBlock_Cross

from local_solvers.blocks_nlin import BlockNL, BlockNL_DMRG, BlockNL_Cross

MPOType = Union['qtn.MatrixProductOperator', 'MatrixProductOperatorTN', 'MatrixProductGroup']
SelectIndsType = dict[int, Sequence[int]]

class BlockType(Enum):
    DMRG = 'dmrg'
    CROSS = 'cross'

## todo: write wrapper to check if self.initialized
""" Generalization of "Term" class to allow for nonlinear blocks
    functionality essentially the same as standard Term
"""

class TermNL(ABC):

    def __init__(self,
                 kets: dict[Any, Union['MPS',qtn.MatrixProductState]],
                 bra: Optional[Union['MPS',qtn.MatrixProductState]],
                 operator: Optional[MPOType],
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 label: str = None,
                 ):

        self.initialized = False
        self.label = label

        self._kets = None
        self._bra = None
        # total_op = qtn.TensorNetwork([operator, *kets], virtual=True)
        # super().__init__(kets[0], bra, [], cur_orthog=cur_orthog, direction=direction, mps_coeff=mps_coeff)

        self.direction = direction
        target_orthog = 0 if direction > 0 else self.bra.L - 1
        self.canonize_func(bra, target_orthog, cur_orthog=cur_orthog)
        for ket in kets:
            self.canonize_func(ket, target_orthog, cur_orthog=cur_orthog)
        self._cur_orthog = target_orthog

        self.op_block: 'BlockNL' = self.block_class(kets, bra, operator, cur_orthog=target_orthog)

        # self.total_op = total_op
        self.operator = operator
        self.kets = kets
        self.bra = bra
        
        # self.vec_blocks = [TNBlock_DMRG([ket], None, ket, cur_orthog=cur_orthog) for ket in kets]
        ## just ensure that it's in canonical form...


    @property
    def type(self) -> LocalSolverType:
        raise NotImplementedError

    @property
    def block_class(self):
        raise NotImplementedError

    # @classmethod
    # def get_callable(cls, mpo_poly=1, mps_func=None):
    #     raise NotImplementedError

    # @property
    # def evaluated_site(self) -> 'qtn.Tensor':
    #     return self._out_site

    @classmethod
    def canonize_func(cls, mps, orthog, cur_orthog: int = None, **kwargs):
        raise NotImplementedError
        # kwargs.pop('scale')
        # return helper_quimb.canonize(mps, i=orthog, cur_orthog=cur_orthog, scale=False, **kwargs)
        # raise NotImplementedError

    @classmethod
    def check_func(cls, mps):
        return helper_quimb.check_orthog(mps)
        # raise NotImplementedError

    @property
    def L(self) -> int:
        return self.bra.L

    @property
    def kets(self) -> dict[Any,'MPS']:
        return self._kets

    @kets.setter
    def kets(self, kets: dict[Any,'MPS']):
        for k, ket in kets.items():
            if not isinstance(ket, MPS):
                ket.view_as(MPS, inplace=True)
        self._kets = kets

    def canonize(self, orthog, cur_orthog: int = None):

        canonize = self.canonize_func

        for ket in self.kets:
            canonize(ket, orthog, cur_orthog=cur_orthog)
        canonize(self.bra, orthog, cur_orthog=cur_orthog)
        
        return

    @property
    def cur_orthog(self) -> int:
        return self._cur_orthog

    def check_orthog(self):
        """
        """
        print('bra orthog')
        self.check_func(self.bra)
        for it, input in enumerate(self.kets):
            print(f'input {it} orthog')
            self.check_func(input)
        return


    # @property
    # def ket_ind_id(self):
    #     return self.ket.site_ind_id
    #
    @property
    def bra(self) -> Optional['MPS']:
        return self._bra
    
    @bra.setter
    def bra(self, bra: 'MPS'):
        if bra is not None:
            if not isinstance(bra, MPS):
                bra.view_as(MPS, inplace=True)
            self._bra = bra

    @property
    def bra_ind_id(self):
        return self.bra.site_ind_id + '_'

    @property
    def operator(self) -> MPOType:
        return self._operator

    @operator.setter
    def operator(self, op: Optional[MPOType]):
        if op is None:
            return

        op = op.copy()
        op.upper_ind_id = self.bra_ind_id
        # op.lower_ind_id = self.ket_ind_id
        ## requires kets/lower inds to already be aligned
        self._operator = op

    # @property
    # def direction(self) -> SweepDirection:
    #     return self._direction
    # 
    # @direction.setter
    # def direction(self, new_direction: SweepDirection):
    #     pass
    #     # self._direction = new_direction
    #     # if self.initialized:
    #     #     for it, blocks in self.op_blocks.items():
    #     #         for block in blocks:
    #     #             block.direction = self._direction
    #     #     self.vec_block.direction = self._direction


    # def get_intermediate_ket(self, tier: int):
    #     # it_ = min(tier, self.num_tiers - 1)
    #     if tier >= self.num_tiers - 1:
    #         return self.bra
    #     return self._intermediate_kets[tier]

    def copy(self, ket_copy=None, bra_copy=None, cur_orthog=None):

        if ket_copy is None:
            ket_copy = {k: ket.copy() for k, ket in self.kets.items()}
        if bra_copy is None:
            bra_copy = self.bra.copy()

        out = self.__class__(ket_copy, bra_copy, self.operator, cur_orthog=cur_orthog)
        return out


    def match_inds(self, other: 'TermNL'):
        for ket, oket in zip(self.kets, other.kets):
            ket.site_ind_id = oket.site_ind_id
            helper_quimb.match_inner_inds(ket, oket)
        
        helper_quimb.match_inner_inds(self.bra, other.bra)
        
        # try:
        #     ref_op = next(iter(other.operators))
        # except StopIteration:
        #     ref_op = None
        #
        # for op in self.operators:
        #     op.upper_ind_id = ref_op.upper_ind_id
        #     op.lower_ind_id = ref_op.lower_ind_id
        #     if ref_op is not None:
        #         helper_quimb.match_inner_inds(op, ref_op)
        #

    def initialize(self):
        if self.initialized:
            return

        # ## officially setting what was stored, does index matching
        # self.ket = self.ket
        # self.bra = self.bra
        # self.operators = self.operators

        print('is orthog?')
        self.check_func(self.bra)
        for ket in self.kets:
            self.check_func(ket)
        
        print('term init checked')
        # 
        # cur_orthog = self.cur_orthog
        # if cur_orthog is None:
        #     cur_orthog = 0 if self.direction == SweepDirection.RIGHT else self.L - 1
        #     print('term init', cur_orthog)
        # self.canonize(cur_orthog)
        # self._cur_orthog = cur_orthog

        self.initialize_blocks()

        self.initialized = True

    def initialize_blocks(self):
        assert self.op_block.cur_orthog == self.cur_orthog, \
            f'term and block center of orthogonalities are different ({self.op_block.cur_orthog} and {self.cur_orthog})'
        self.op_block.init_envs()
        
        ## assume kets are all in canonical form...
        # for vec_block in self.vec_blocks:
        #     vec_block.initialize_envs(self.cur_orthog)
        

    @property
    def blocks(self):
        return [self.op_block] #, self.vec_block

    # def get_bra_inds(self, left_site_pos: int, nsites: int):
    #     """ get inds corresponding to bra (output)
    #     """
    #     inds = [self.bra.site_ind_id.format(i) for i in range(left_site_pos, left_site_pos + nsites)]
    #     if self.operator is not None:
    #         inds = [ind + '_' for ind in inds]
    #     if left_site_pos > 0:
    #         lb = self.bra.bond(left_site_pos, left_site_pos - 1) + '_'
    #         inds += [lb]
    #     if left_site_pos + nsites < self.bra.L - 1:
    #         rb = self.bra.bond(left_site_pos + nsites - 1, left_site_pos + nsites) + '_'
    #         inds += [rb]
    #     return inds

    # def projected_bra_to_ket(self, left_site_pos: int, nsites: int, ket: 'qtn.MatrixProductState'):
    #     b_to_k = self.op_block.projected_bra_to_ket(left_site_pos, nsites, ket)
    #     return b_to_k


    def extend_env(self, i, direction): #, new_ket_site=None):
        self.op_block.extend_env(i, direction)
        self._cur_orthog = i + direction


    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tens_dict: dict[str,'qtn.Tensor']=None,
                           exclude: list[str]=None, return_combined=False):

        ## todo: check if exponent is included in site_tens_dict

        if not self.initialized:
            raise RuntimeError('term has not yet been initialized')

        # if site_tens_dict is None:
        #     site_tens_dict = {}
        #     for ket in self.kets:
        #         id = ket.site_tag_id
        #         if nsites > 1:
        #             tens = qtn.tensor_contract(*ket[left_site_pos: left_site_pos + nsites])
        #         else:
        #             tens = ket[left_site_pos].copy()
        #         tens.modify(apply=lambda x: x * 10**ket.exponent)
        #         site_tens_dict[id] = tens
                
                        
        # exclude = [] if exclude is None else exclude
        # site_tens = [v for k, v in site_tens_dict.items() if k not in exclude]
                
        # b2k_dict = self.projected_bra_to_ket(left_site_pos, nsites, self.bra)
        b2k_dict = {}
        for i in range(left_site_pos, left_site_pos + nsites):
            b2k = self.op_block.bra_orig_inds(i)
            b2k_dict.update(b2k)

        out_site = self.op_block.get_projected(left_site_pos, nsites, site_tens_dict)
        out_site.reindex(b2k_dict, inplace=True)

        return out_site



    def get_jacobian(self, ket_id: str, left_site_pos: int, nsites: int, return_combined = False,
                     transpose_bonds = None, site_tens_dict: dict[str, qtn.Tensor]=None
                     ) -> Union['qtn.TensorNetwork', 'qtn.Tensor']:

        eff_op = self.get_evaluated_site(left_site_pos, nsites, exclude=[ket_id],
                                         site_tens_dict=site_tens_dict,
                                         return_combined=return_combined)
        if return_combined:
            eff_op.transpose(transpose_bonds, inplace=True)

        return eff_op



class TermNL_DMRG(TermNL):

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.DMRG

    @classmethod
    def canonize_func(cls, mps, orthog, cur_orthog:int =None, **kwargs):

        i = orthog
        # print('CANONIZE CUR ORTHOG', mps.cur_orthog)
        # print('check', helper_quimb.check_orthog(mps))
        cur_orthog = None  # mps.cur_orthog
        tens_left, tens_right = [], []
        max_ind = mps.L - 1
        updated_inds = []
        for lx in range(mps.L):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds += [lx]
                # tens_left += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
                tens_left += [mps[lx]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds:
                    updated_inds += [lx]
                # tens_right += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
                tens_right += [mps[lx]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]

        # print([t.copy() for t in tens_left])
        # print([t.copy() for t in tens_right])
        # print([tens_left])
        # print([tens_right])

        helper.canonize_tens_list(*tens_left, inplace=True)
        helper.canonize_tens_list(*(tens_right[::-1]), inplace=True)

        mps._cur_orthog = orthog
        return mps

    @classmethod
    def check_func(cls, mps):
        return helper_quimb.check_orthog(mps)
        # raise NotImplementedError

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, full_matrices=False):
        tens_list = helper_quimb.canonize_tens_list(*tens_list, inplace=inplace, full_matrices=full_matrices)
        return tens_list

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        tens_list = helper_quimb.compress_tens_list(*tens_list, inplace=inplace, compress_opts={'max_bond': max_bond})
        return tens_list



class TermNL_Cross(TermNL):

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.Cross

    @classmethod
    def canonize_func(cls, mps: MPS, orthog: int, cur_orthog: int=None, select_inds=None):
        print('mps cur orthog', mps._cur_orthog, cur_orthog)
        select_inds = mps.select_inds if select_inds is None else select_inds
        if select_inds is None:
            select_inds = {}
        helper_cross.canonize(mps, orthog, cur_orthog=cur_orthog, select_inds=select_inds)
        mps._cur_orthog = orthog
        return mps

    @classmethod
    def check_func(cls, mps):
        return helper_cross.check_orthog(mps)

    @classmethod
    def canonize_tens_list(cls, tens_list: list['qtn.Tensor'], inplace=True, site_ind_ids: Sequence[str] = None,
                           site_inds: Sequence[int] = None):
        """ site_ind_ids, site_inds not actually optional
        """
        tens_list, sel_inds =  helper_cross.canonize_tens_list(tens_list, site_ind_ids, site_inds,
                                                               inplace=inplace)
        return tens_list, sel_inds

    @classmethod
    def compress_tens_list(cls, tens_list: list['qtn.Tensor'], inplace=True, max_bond=None,
                           site_ind_ids: Sequence[str] = None, site_inds: Sequence[int] = None):
        """ site_ind_ids, site_inds not actually optional
        """
        tens_list, sel_inds = helper_cross.compress_tens_list(tens_list, site_ind_ids, site_inds,
                                                              inplace=inplace, max_bond=max_bond)
        return tens_list, sel_inds



    # def canonize_ket_tens(self, left_site_pos: int, nsites: int, direction: SweepDirection, max_bond:int = None):
    #
    #     if direction == SweepDirection.RIGHT:
    #         assert(left_site_pos < self.L - 1), 'cannot left canonicalize last site'
    #         site_inds = range(left_site_pos, left_site_pos + nsites + 1)
    #     else:
    #         assert(left_site_pos > 0), 'cannot right canonicalize 0th site'
    #         site_inds = range(left_site_pos + nsites - 1, left_site_pos - 2, -1)
    #
    #     tens_list = [self.ket[ix] for ix in site_inds]
    #     _, select_inds = self.compress_tens_list(tens_list, site_ind_ids=[self.ket.site_ind_id],
    #                                              site_inds=list(site_inds), inplace=True,
    #                                              max_bond=max_bond)
    #     # print('select inds', select_inds)
    #     for ix, sel_inds in zip(site_inds, select_inds):
    #         # self.ket_select_inds[ix] = sel_inds
    #         self.ket.select_inds[ix] = sel_inds
    #
    #     # print('self.ket select inds', self.ket.select_inds)
    #     self.ket._cur_orthog = site_inds[-1]
    #     self.cur_orthog = site_inds[-1]
    #
    #     # print('canonnize ket tens check orthog')
    #     # print(helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))




