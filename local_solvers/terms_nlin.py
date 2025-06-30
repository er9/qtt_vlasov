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
import local_solvers.tensor_callables as tc
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.terms_3 import Term
from local_solvers.blocks_nlin import Block

from setup_.TN_mpo import TNBlock_DMRG, TNBlock_Cross

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
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
                 kets: list[Union['MPS',qtn.MatrixProductState]],
                 bra: Optional[Union['MPS',qtn.MatrixProductState]],
                 operator: MatrixProductTensor,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # mpo_poly: Union[Sequence[int], int] = 1,  ## int can also be -1, powers ordered from 0 to highest
                 # mps_power: int = 1,
                 mps_coeff: Numeric = 1.0,
                 label: str = None,
                 # mps_func: Callable = None,
                 # max_bond: int = None,
                 # num_tiers: int = 1,  ## number of intermediate kets in between operators = num_tiers - 1
                 # **canon_kwargs
                 ):

        self.initialized = False
        self.label = label

        self._kets = None
        self._bra = None
        total_op = qtn.TensorNetwork([operator, *kets], virtual=True)
        # super().__init__(kets[0], bra, [], cur_orthog=cur_orthog, direction=direction, mps_coeff=mps_coeff)

        self.op_block = TNBlock_DMRG(kets, bra, operator)

        self.direction = direction
        target_orthog = 0 if direction > 0 else self.bra.L - 1
        self.canonize_func(bra, target_orthog, cur_orthog=cur_orthog)
        for ket in kets:
            self.canonize_func(ket, target_orthog, cur_orthog=cur_orthog)
        self._cur_orthog = target_orthog

        self.total_op = total_op
        self.operator = operator
        self.kets = kets
        self.bra = bra
        
        # self.vec_blocks = [TNBlock_DMRG([ket], None, ket, cur_orthog=cur_orthog) for ket in kets]
        ## just ensure that it's in canonical form...


    @property
    def type(self) -> LocalSolverType:
        raise NotImplementedError


    @classmethod
    def get_callable(cls, mpo_poly=1, mps_func=None):

        raise NotImplementedError

    # @property
    # def proj_vec_targets(self) -> Sequence['qtn.Tensor']:
    #     return self._proj_vec_targets
    #
    # @property
    # def intermediate_sites(self) -> dict[int, Sequence['qtn.Tensor']]:
    #     return self._intermediate_sites

    @property
    def evaluated_site(self) -> 'qtn.Tensor':
        return self._out_site

    @classmethod
    def canonize_func(cls, mps, orthog, cur_orthog: int = None, **kwargs):
        kwargs.pop('scale')
        return helper_quimb.canonize(mps, i=orthog, cur_orthog=cur_orthog, scale=False, **kwargs)
        # raise NotImplementedError

    @classmethod
    def check_func(cls, mps):
        return helper_quimb.check_orthog(mps)
        # raise NotImplementedError

    @property
    def L(self) -> int:
        return self.bra.L

    @property
    def kets(self) -> 'MPS':
        return self._kets

    @kets.setter
    def kets(self, kets: 'MPS'):
        for ket in kets:
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

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize_func(self.bra, cur_orthog)
    #     return self.bra
    #
    @property
    def bra_ind_id(self):
        return self.bra.site_ind_id + '_'

    @property
    def operator(self) -> 'qtn.TensorNetwork':
        return self._operator

    @operator.setter
    def operator(self, op: Optional[MPType]):
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


    def copy_new(self, ket_copy=None, bra_copy=None):
        raise NotImplementedError

        # if ket_copy is None:
        #     ket_copy = self.ket.copy()
        #
        # if bra_copy is None:
        #     bra_copy = self.bra.copy() if self.bra is not None else None
        #
        # # out = self.__class__(ket, bra, )
        # out = self.__class__(ket_copy, bra_copy,
        #                      [],
        #                      cur_orthog=self.cur_orthog,
        #                      direction=self.direction,
        #                      num_tiers=self.num_tiers,
        #                      max_bond=self.max_bond)
        # return out

    def copy(self, ket_copy=None, bra_copy=None):
        raise NotImplementedError

        # out = self.copy_new(ket_copy) #, bra_copy)
        # ket_copy = out.ket
        #
        # if self.initialized:
        #     vec_block = self.vec_block.copy(ket_copy=ket_copy, bra_copy=out.get_intermediate_ket(0))
        #
        #     op_blocks_dict = {}
        #     for i in range(self.num_tiers - 1):
        #         op_blocks = []
        #         for op_block in self.op_blocks[i]:
        #             op_blocks += [op_block.copy(ket_copy=out.get_intermediate_ket(i),
        #                                         bra_copy=out.get_intermediate_ket(i+1))]
        #         op_blocks_dict[i] = op_blocks
        #
        #     out.op_blocks = op_blocks_dict
        #     out.vec_block = vec_block
        #
        # out.initialized = self.initialized
        # return out


    def match_inds(self, other: 'TermNL'):
        for ket, oket in zip(self.kets, other.kets):
            ket.site_ind_id = oket.site_ind_id
            helper_quimb.match_inner_inds(ket, oket)
        
        helper_quimb.match_inner_inds(self.bra, other.bra)
        
        try:
            ref_op = next(iter(other.operators))
        except StopIteration:
            ref_op = None

        for op in self.operators:
            op.upper_ind_id = ref_op.upper_ind_id
            op.lower_ind_id = ref_op.lower_ind_id
            if ref_op is not None:
                helper_quimb.match_inner_inds(op, ref_op)


    def initialize(self):
        # if self.initialized:
        #     return

        # ## officially setting what was stored, does index matching
        # self.ket = self.ket
        # self.bra = self.bra
        # self.operators = self.operators

        print('term init?')
        check_func = self.check_func
        check_func(self.bra)
        for ket in self.kets:
            check_func(ket)
        
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
        self.op_block.initialize_envs(self.cur_orthog)
        
        ## assume kets are all in canonical form...
        # for vec_block in self.vec_blocks:
        #     vec_block.initialize_envs(self.cur_orthog)
        

    @property
    def blocks(self):
        return [self.op_block] #, self.vec_block

    def get_bra_inds(self, left_site_pos: int, nsites: int):
        """ get inds corresponding to bra (output)
        """
        inds = [self.bra.site_ind_id.format(i) for i in range(left_site_pos, left_site_pos + nsites)]
        if self.operator is not None:
            inds = [ind + '_' for ind in inds]
        if left_site_pos > 0:
            lb = self.bra.bond(left_site_pos, left_site_pos - 1) + '_'
            inds += [lb]
        if left_site_pos + nsites < self.bra.L - 1:
            rb = self.bra.bond(left_site_pos + nsites - 1, left_site_pos + nsites) + '_'
            inds += [rb]
        return inds

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int, ket: 'qtn.MatrixProductState'):
        if self.op_block is None:
            b_to_k = {}
            if left_site_pos > 0:
                lb = ket.bond(left_site_pos, left_site_pos - 1)
                b_to_k[lb + '_'] = lb
            if left_site_pos + nsites < ket.L - 1:
                rb = ket.bond(left_site_pos + nsites - 1, left_site_pos + nsites)
                b_to_k[rb + '_'] = rb
        else:
            b_to_k = self.op_block.projected_bra_to_ket(left_site_pos, nsites, ket)
        return b_to_k


    def extend_env(self, i, direction): #, new_ket_site=None):

        for it, blocks in self.op_blocks.items():
            for block in blocks:
                block.extend_env(i, direction=direction)

        self._cur_orthog = i + direction


    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tens_dict: dict[str,'qtn.Tensor']=None,
                           exclude: list[str]=None, return_combined=False):

        if not self.initialized:
            raise RuntimeError('term has not yet been initialized')

        if site_tens_dict is None:
            site_tens_dict = {}
            for ket in self.kets:
                id = ket.site_tag_id
                if nsites > 1:
                    tens = qtn.tensor_contract(*ket[left_site_pos: left_site_pos + nsites])
                else:
                    tens = ket[left_site_pos].copy()
                tens.modify(apply=lambda x: x * 10**ket.exponent)
                site_tens_dict[id] = tens
                
                        
        exclude = [] if exclude is None else exclude
        site_tens = [v for k, v in site_tens_dict.items() if k not in exclude]
                
        b2k_dict = self.projected_bra_to_ket(left_site_pos, nsites, self.bra)
        eff_op = self.op_block.get_projected(left_site_pos, nsites,) # return_combined=False,)
        for st in site_tens:
            eff_op.add_tensor(st)

        if return_combined:
            out_site = eff_op.contract()  ## todo: always forget if eff_op exponent is included
            ## need to reindex to ket inds
            out_site.reindex(b2k_dict, inplace=True)
            # print('remove bra exponent')
            out_site.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))
            return out_site
        else:
            eff_op.exponent -= self.bra.exponent
            return eff_op


    def get_jacobian(self, ket_id: str, left_site_pos: int, nsites: int, return_combined = False,
                     transpose_bonds = None, site_tens_dict: dict[str, qtn.Tensor]=None
                     ) -> Union['qtn.TensorNetwork', 'qtn.Tensor']:

        eff_op = self.get_evaluated_site(left_site_pos, nsites, exclude=[ket_id],
                                         site_tens_dict=site_tens_dict,
                                         return_combined=return_combined)
        if return_combined:
            eff_op.transpose(transpose_bonds, inplace=True)

        return eff_op



class Term_DMRG(Term):

    def __init__(self,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState] = None,
                 operators: Optional[Sequence[qtn.MatrixProductOperator]] = None,
                 operator_k: Optional[qtn.MatrixProductOperator] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 mpo_poly: Union[Sequence[int], int] = 1,
                 mps_power: int = 1, mps_coeff: Numeric = 1,
                 max_bond: int = None,
                 num_tiers: int = 1
                 ):

        mps_func = None
        if mps_coeff != 1:
            mps_func = tc.get_element_wise_func(lambda x: x * mps_coeff)
        super().__init__(ket, bra, operators,
                         operator_k=operator_k, cur_orthog=cur_orthog, direction=direction,
                         mpo_poly=mpo_poly, mps_func=mps_func, max_bond=max_bond, num_tiers=num_tiers)
        self.mps_power = mps_power
        # if mps_coeff != 1:
        #     self.mps_func = tc.get_element_wise_func(lambda x: x * mps_coeff)
        # else:
        #     self.mps_func = None

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.DMRG

    # @classmethod
    # def canonize_func(cls, mps, orthog):
    #     helper_quimb.canonize(mps, i=orthog, scale=False, bra=None)
    #     mps.cur_orthog = orthog
    #     return mps

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

    # def canonize(self, cur_orthog):
    #     self.canonize_func(self.ket, cur_orthog)
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, cur_orthog)

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize_func(self.bra, cur_orthog)

    def copy_new(self, ket_copy=None, bra_copy=None):
        out = super().copy_new(ket_copy, bra_copy)
        out.mps_power = self.mps_power
        return out

    def copy(self, ket_copy=None, bra_copy=None):
        out = super().copy(ket_copy, bra_copy)
        out.mps_power = self.mps_power
        return out

    def initialize_blocks(self):

        intermediate_kets = {i: self.ket.copy() for i in range(self.num_tiers - 1)}

        print('init blocks cur orthog', self.cur_orthog)

        ## projection of self.ket
        bra = intermediate_kets.get(0, self.bra)
        if self.mps_power == 1:
            if self.operator_k is None:
                vec_block = BlockVector_DMRG(self.ket, bra, cur_orthog=self.cur_orthog)
            else:
                vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
                                                 cur_orthog=self.cur_orthog, power=self.mps_power)
                # vec_block = BlockOperator_DMRG(self.ket, bra, operator=self.operator_k,
                #                                cur_orthog=self.cur_orthog)
        else:
            if self.operator_k is None:
                vec_block = BlockPowerKet_DMRG(self.ket, bra, cur_orthog=self.cur_orthog, power=self.mps_power)
            else:
                vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
                                                 cur_orthog=self.cur_orthog, power=self.mps_power)

        ## projection of operators
        op_blocks_all = {}
        for it in range(self.num_tiers):
            op_blocks = []
            ket = intermediate_kets.get(it, self.bra)
            bra = intermediate_kets.get(it + 1, self.bra)

            for op in self.operators:
                if isinstance(op, qtn.MatrixProductOperator):
                    op_blocks += [BlockOperator_DMRG(ket, bra, operator=op.copy(),
                                                     anc_env_left=None, anc_env_right=None,
                                                     cur_orthog=self.cur_orthog)]
                elif isinstance(op, qtn.MatrixProductState):
                    op_blocks += [BlockDiagOperator_DMRG(ket, bra, operator=op.copy(),
                                                         anc_env_left=None, anc_env_right=None,
                                                         cur_orthog=self.cur_orthog)]
                else:
                    raise TypeError('not a valid type for operator')

            op_blocks_all[it] = op_blocks

        self._intermediate_kets = intermediate_kets
        self.op_blocks = op_blocks_all
        self.vec_block = vec_block
        # if vec_block.bra is not None and vec_block.bra is not self.bra:
        #     print('check orthog', helper.check_orthog(vec_block.bra))
        #     print('self.cur orthog', self.cur_orthog)
        #     exit()


    # def get_evaluated_rdm(self, left_site_pos: int, nsites: int, direction: SweepDirection,
    #                       transpose_inds: Sequence[str] = None, site_tens: 'qtn.Tensor' = None) -> 'qtn.Tensor':
    #
    #     if site_tens is None:
    #         site_tens = self.get_evaluated_site(left_site_pos, nsites)  ## has ket inds
    #
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #     assert (not at_end), 'should not need to evaluate rdm for targeting if at end'
    #
    #     ## horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
    #     if direction == SweepDirection.RIGHT:
    #         x_bond = self.ket.bond(left_site_pos + nsites, left_site_pos + nsites - 1)
    #     else:
    #         x_bond = self.ket.bond(left_site_pos, left_site_pos - 1)
    #
    #     ## bonds to not contract over
    #     b_to_k = self.projected_bra_to_ket(left_site_pos, nsites)
    #     k_to_b = {v: k for k, v in b_to_k.items()}
    #     ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
    #     new_bra_iso = [k_to_b[k] for k in ket_iso]
    #
    #     if transpose_inds is None:
    #         transpose_inds = [*ket_iso, *new_bra_iso]
    #
    #     ## evaluated rdm
    #     rdm = helper_dmrg.get_tens_rdm(site_tens, ket_iso, new_bra_iso, transpose_inds=transpose_inds)
    #     # site_conj = site_tens.conj()
    #     # site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     # rdm = qtn.tensor_contract(site_conj, site_tens, output_inds=transpose_inds)
    #     # rdm.modify(apply = lambda x: x / rdm.norm())
    #     print('rdm 1 norm', rdm.norm())
    #
    #     # ## self vec rdm
    #     # if isinstance(self.vec_block, BlockPowerKet_DMRG):
    #     #     vec_tens = qtn.tensor_contract(*[self.vec_block.ket[si] for si in range(left_site_pos, left_site_pos + nsites)])
    #     #     vec_tens.modify(apply = lambda x: x * 10**self.vec_block.ket.exponent)
    #     #     # vec_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #     #     # vec_tens.reindex({b: k for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     #
    #     #     vec_rdm = helper_dmrg.get_tens_rdm(vec_tens, ket_iso, new_bra_iso, transpose_inds=transpose_inds)
    #     #     # vec_conj = vec_tens.conj()
    #     #     # vec_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     #     # vec_rdm = qtn.tensor_contract(vec_conj, vec_tens, output_inds=transpose_inds)
    #     #
    #     #     # vec_rdm.modify(apply = lambda x: x / vec_rdm.norm())
    #     #     print('rdm 2 norm', vec_rdm.norm())
    #     #
    #     #     rdm = helper.add_tensors(rdm, vec_rdm)
    #
    #     return rdm


    # def extend_env(self, i, direction, new_ket_site=None):
    #     self.vec_block.decimate(i, direction=direction, new_ket_site=new_ket_site)
    #     self.vec_block.extend_env(i, direction=direction)
    #     for block in self.op_blocks:
    #         block.extend_env(i, direction=direction)



class Term_Cross(Term):

    def __init__(self,
                 ket: 'MPS',
                 bra: Optional['MPS'] = None,
                 operators: Optional[Sequence[qtn.MatrixProductOperator]] = None,
                 operator_k: Optional[qtn.MatrixProductOperator] = None,
                 # eval_func: Optional[Callable] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # ket_select_inds: dict[int, Sequence[int]] = None,
                 # bra_select_inds: dict[int, Sequence[int]] = None,
                 mpo_poly: Union[Sequence[int], int] = 1,
                 mps_power: int = 1, mps_coeff: Numeric = 1.0,
                 mps_func: Callable = None,
                 max_bond: int = None,
                 num_tiers: int = None,
                 ):

        if num_tiers is None:
            max_mpo_poly = len(mpo_poly) - 1 if isinstance(mpo_poly, (list, tuple)) else abs(mpo_poly)
            num_tiers = max_mpo_poly + 1

        super().__init__(ket, bra, operators,
                         operator_k=operator_k,
                         cur_orthog=cur_orthog, direction=direction,
                         mpo_poly=mpo_poly, mps_power=mps_power, mps_coeff=mps_coeff, mps_func=mps_func,
                         max_bond=max_bond, num_tiers=num_tiers)

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

    # def canonize(self, cur_orthog):
    #     self.canonize_func(self.ket, cur_orthog)
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, cur_orthog)

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize(self.bra, cur_orthog)

    def copy_new(self, ket_copy=None, bra_copy=None):

        if ket_copy is None:
            ket_copy = self.ket.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self.bra is not None else None

        # out = self.__class__(ket, bra, )
        out = self.__class__(ket_copy, bra_copy,
                             [op.copy() for op in self.operators],
                             cur_orthog=self.cur_orthog,
                             direction=self.direction,
                             num_tiers=self.num_tiers,
                             max_bond=self.max_bond,
                             mps_func=self.mps_func)
        out.mps_func = self.mps_func
        out.mpo_poly = self.mpo_poly
        out.eval_func = self.eval_func

        out._intermediate_kets = {k: v.copy() for k, v in self._intermediate_kets.items()}

        return out


    def create_like(self, new_ket):

        out = self.__class__(new_ket, None,
                             [op.copy() for op in self.operators],
                             cur_orthog=self.cur_orthog,
                             direction=self.direction,
                             num_tiers=self.num_tiers,
                             max_bond=self.max_bond,
                             mps_func=self.mps_func)
        out.mps_func = self.mps_func
        out.mpo_poly = self.mpo_poly
        out.eval_func = self.eval_func

        return out


    # def canonize(self, orthog, cur_orthog: int = None):
    #     ### assume self.ket, self.bra orthogonality doesn't matter,
    #     ### or is already of the appropriate canonicalization
    #     # self.canonize_func(self.ket, orthog, cur_orthog=cur_orthog)
    #     # self.canonize_func(self.bra, orthog, cur_orthog=cur_orthog)
    #
    #     # print('self.initialized', self.initialized)
    #     # if self.initialized:
    #     #     print('self.ket', self.vec_block.ket.select_inds)
    #     #     print('self.bra', self.vec_block.bra.select_inds)
    #
    #
    #     sel_inds = self.bra.select_inds
    #     # print('canonize ket', orthog, cur_orthog)
    #     # print('sel inds', sel_inds)
    #     self.ket = helper_cross.target_select_inds(self.ket, sel_inds, inplace=True, direction=SweepDirection.LEFT)
    #
    #     print('check orthog bra 2')
    #     self.check_func(self.bra)
    #     self.check_func(self.ket)
    #     print('canonize term 2')
    #
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, orthog, cur_orthog=cur_orthog)
    #
    #     # self.check_orthog()
    #     # print('done canonize term')
    #
    #     return self.ket



    def canonize_ket_tens(self, left_site_pos: int, nsites: int, direction: SweepDirection, max_bond:int = None):

        if direction == SweepDirection.RIGHT:
            assert(left_site_pos < self.L - 1), 'cannot left canonicalize last site'
            site_inds = range(left_site_pos, left_site_pos + nsites + 1)
        else:
            assert(left_site_pos > 0), 'cannot right canonicalize 0th site'
            site_inds = range(left_site_pos + nsites - 1, left_site_pos - 2, -1)

        tens_list = [self.ket[ix] for ix in site_inds]
        _, select_inds = self.compress_tens_list(tens_list, site_ind_ids=[self.ket.site_ind_id],
                                                 site_inds=list(site_inds), inplace=True,
                                                 max_bond=max_bond)
        # print('select inds', select_inds)
        for ix, sel_inds in zip(site_inds, select_inds):
            # self.ket_select_inds[ix] = sel_inds
            self.ket.select_inds[ix] = sel_inds

        # print('self.ket select inds', self.ket.select_inds)
        self.ket._cur_orthog = site_inds[-1]
        self.cur_orthog = site_inds[-1]

        # print('canonnize ket tens check orthog')
        # print(helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))


    def initialize_blocks(self):

        intermediate_kets = {i: self.bra.copy() for i in range(self.num_tiers - 1)}

        ## projection of self.ket  (have tier for self.ket, unless no operator is applied so tier is self.bra)
        bra = intermediate_kets.get(0, self.bra)
        vec_block = BlockVector_Cross(self.ket, bra, cur_orthog=self.cur_orthog)

        # ## projection of operators
        # op_blocks_all = {}
        # for it in range(self.num_tiers - 1):
        #     op_blocks = []
        #     ket = intermediate_kets.get(it, self.bra)
        #     bra = intermediate_kets.get(it + 1, self.bra)
        #
        #     for op in self.operators:
        #         if isinstance(op, qtn.MatrixProductOperator):
        #             op_blocks += [BlockOperator_Cross(ket, bra, operator=op,
        #                                              anc_env_left=None, anc_env_right=None,
        #                                              cur_orthog=self.cur_orthog)]
        #         elif isinstance(op, qtn.MatrixProductState):
        #             op_blocks += [BlockDiagOperator_Cross(ket, bra, operator=op,
        #                                                  anc_env_left=None, anc_env_right=None,
        #                                                  cur_orthog=self.cur_orthog)]
        #         else:
        #             raise TypeError('not a valid type for operator')
        #
        #     op_blocks_all[it] = op_blocks

        ## projection of operators
        op_blocks_all = {}
        for it, op in enumerate(self.operators):
            op_blocks = []
            ket = intermediate_kets.get(it, self.bra)
            bra = intermediate_kets.get(it + 1, self.bra)

            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_Cross(ket, bra, operator=op,
                                                  anc_env_left=None, anc_env_right=None,
                                                  cur_orthog=self.cur_orthog)]
            elif isinstance(op, qtn.MatrixProductState):
                op_blocks += [BlockDiagOperator_Cross(ket, bra, operator=op,
                                                      anc_env_left=None, anc_env_right=None,
                                                      cur_orthog=self.cur_orthog)]
            else:
                raise TypeError('not a valid type for operator')

            op_blocks_all[it] = op_blocks

        self._intermediate_kets = intermediate_kets
        self.op_blocks = op_blocks_all
        self.vec_block = vec_block



