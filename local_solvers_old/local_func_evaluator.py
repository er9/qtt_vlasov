import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from local_solvers_old.environments_v2 import Environment, EnvironmentSide
from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
from local_solvers_old.local_solver import SweepDirection
# from local_solvers_old.blocks import EnvironmentType, init_block, BlockOperator, BlockVector, BlockDiagKet_DMRG
# from local_solvers_old.blocks import BlockVector_DMRG, BlockOperator_DMRG, BlockOperator_Cross, BlockVector_Cross
from local_solvers_old.blocks import *
from local_solvers_old.helper_tn import get_mps_matching_inds, sum_tens
# from helper_cross_v3 import canonize, canonize_tens_list, compress_tens_list
import helper_cross_v3 as helper_cross

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
SelectIndsType = dict[int, Sequence[int]]

class BlockType(Enum):
    DMRG = 'dmrg'
    CROSS = 'cross'


class Term(ABC):

    def __init__(self, L, eval_func: Callable,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState],
                 operators: Optional[Sequence[qtn.MatrixProductOperator]],
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 **canon_kwargs
                 ):

        self.L = L
        self.block_type = None

        self._ket = None
        self._bra = None
        self._operators = []

        if cur_orthog is None:
            cur_orthog = 0 if direction == SweepDirection.RIGHT else L - 1
            ket = self.canonize(ket, cur_orthog, **canon_kwargs)
            if bra is not None:
                bra = self.canonize(bra, cur_orthog, **canon_kwargs)
        self.cur_orthog = cur_orthog

        self.ket = ket
        self.bra = bra
        self.operators = operators
        self.eval_func = eval_func

        self.op_blocks: Sequence[BlockOperator] = []
        self.vec_block: BlockVector = None


    @classmethod
    def canonize(cls, mps, orthog, **kwargs):
        raise NotImplementedError

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, **kwargs):
        raise NotImplementedError

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        raise NotImplementedError

    @property
    def ket(self) -> 'qtn.MatrixProductState':
        return self._ket

    @ket.setter
    def ket(self, ket: 'qtn.MatrixProductState'):
        self._ket = ket

    @property
    def ket_ind_id(self):
        return self.ket.site_ind_id

    @property
    def bra(self) -> Optional['qtn.MatrixProductState']:
        return self._bra

    @bra.setter
    def bra(self, bra: 'qtn.MatrixProductState'):
        if bra is not None:
            bra.site_ind_id = self.bra_ind_id
        self._bra = bra

    @property
    def bra_ind_id(self):
        return self.ket.site_ind_id + '_'

    @property
    def operators(self) -> Sequence['qtn.MatrixProductOperator']:
        return self._operators

    @operators.setter
    def operators(self, operators: Optional[Sequence[Union[Sequence[MPType], MPType]]]):
        if operators is None:
            operators = []

        operators = [(op.copy() if isinstance(op, qtn.MatrixProductOperator) else op) for op in operators]

        for op in operators:
            op.upper_ind_id = self.bra_ind_id
            op.lower_ind_id = self.ket_ind_id

        self._operators = operators

    def match_inds(self, other: 'Term'):
        self.ket.site_ind_id = other.ket.site_ind_id
        helper_quimb.match_inner_inds(self.ket, other.ket)
        helper_quimb.match_inner_inds(self.ket, other.ket)

        if self.bra is not None:
            self.bra.site_ind_id = other.bra_ind_id
            helper_quimb.match_inner_inds(self.bra, other.bra)
            helper_quimb.match_inner_inds(self.bra, other.bra)

        try:
            ref_op = next(iter(other.operators))
        except StopIteration:
            ref_op = None

        for op in self.operators:
            op.upper_ind_id = self.bra_ind_id
            op.lower_ind_id = self.ket_ind_id
            if ref_op is not None:
                helper_quimb.match_inner_inds(op, ref_op)

    def initialize_blocks(self):
        raise NotImplementedError

    def update_blocks(self):
        raise NotImplementedError

    @property
    def blocks(self):
        return self.op_blocks, self.vec_block

    def get_evaluated_site(self, left_site_pos: int, nsites: int):

        # site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # site_tens = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        # site_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        site_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)

        eff_ops = []
        for env in self.op_blocks:
            eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
            eff_ops += [eff_op]

        out_site = self.eval_func(site_tens, eff_ops)

        ## need to reindex to ket inds

        return out_site



class Term_DMRG(Term):

    def __init__(self, L, eval_func: Callable,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState],
                 operators: Optional[Sequence[qtn.MatrixProductOperator]],
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 ):
        super().__init__(L, eval_func, ket, bra, operators, cur_orthog = cur_orthog, direction = direction)

    @classmethod
    def canonize(cls, mps, orthog, **kwargs):
        helper_quimb.canonize(mps, i=orthog, scale=False, bra=None)
        return mps

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, full_matrices=False):
        tens_list = helper_quimb.canonize_tens_list(*tens_list, inplace=inplace, full_matrices=full_matrices)
        return tens_list

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        tens_list = helper_quimb.compress_tens_list(*tens_list, inplace=inplace, compress_opts={'max_bond': max_bond})
        return tens_list

    def initialize_blocks(self):
        op_blocks = []
        target_use_bra = True

        for op in self.operators:
            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_DMRG(op.L, self.ket, self.bra, operator=op,
                                                 anc_env_left=None, anc_env_right=None,
                                                 cur_orthog=self.cur_orthog)]
                target_use_bra = False

            elif op == self.ket or op == 'ket' or isinstance(op, qtn.MatrixProductState):
                if op == 'ket':  op = None
                op_blocks += [BlockDiagOperator_DMRG(self.L, self.ket, self.bra, operator=op,
                                                     anc_env_left=None, anc_env_right=None,
                                                     cur_orthog=self.cur_orthog)]

            else:
                raise TypeError('not a valid type for operator')

        vec_conj = self.bra if target_use_bra else None
        vec_block = BlockVector_DMRG(self.L, self.ket, vec_conj, cur_orthog=self.cur_orthog)

        op_blocks += (op_blocks, vec_block)

        self.op_blocks = op_blocks
        self.vec_block = vec_block

    def get_evaluated_rdm(self, left_site_pos: int, nsites: int, direction: SweepDirection,
                          transpose_inds: Sequence[str] = None) -> 'qtn.Tensor':

        out_site = self.get_evaluated_site(left_site_pos, nsites)  ## has ket inds

        at_end = (left_site_pos + nsites == self.L - 1) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
        assert (not at_end), 'should not need to evaluate rdm for targeting if at end'

        ## horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
        if direction == SweepDirection.RIGHT:
            x_bond = self.ket.bond(left_site_pos + nsites, left_site_pos + nsites - 1)
        else:
            x_bond = self.ket.bond(left_site_pos, left_site_pos - 1)

        ## bonds to not contract over
        b_to_k = self.op_blocks[0].projected_bra_to_ket(left_site_pos, nsites)
        k_to_b = {v: k for k, v in b_to_k.items()}
        ket_iso = [ind for ind in out_site.inds if ind != x_bond]
        new_bra_iso = [k_to_b[k] for k in ket_iso]

        site_conj = out_site.conj()
        site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)

        rdm = qtn.tensor_contract(site_conj, out_site, output_inds=transpose_inds)
        return rdm


class Term_Cross(Term):

    def __init__(self, L, eval_func: Callable,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState],
                 operators: Optional[Sequence[qtn.MatrixProductOperator]],
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 select_inds_ket: dict[int, Sequence[int]] = None,
                 select_inds_bra: dict[int, Sequence[int]] = None,
                 ):

        self.select_inds_ket = select_inds_ket
        self.select_inds_bra = select_inds_bra

        super().__init__(L, eval_func, ket, bra, operators, cur_orthog=cur_orthog, direction=direction)


    @classmethod
    def canonize(cls, mps: qtn.MatrixProductState, orthog: int, select_inds: SelectIndsType=None):
        return helper_cross.canonize(mps, orthog, select_inds=select_inds)

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

    def initialize_blocks(self):
        op_blocks = []
        target_use_bra = True

        for op in self.operators:
            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_Cross(op.L, self.ket, self.bra, operator=op,
                                                  anc_env_left=None, anc_env_right=None,
                                                  select_inds_bra=self.select_inds_bra,
                                                  select_inds_ket=self.select_inds_ket,
                                                  cur_orthog=self.cur_orthog)]
                target_use_bra = False

            elif isinstance(op, qtn.MatrixProductState):
                op_blocks += [BlockDiagOperator_Cross(self.L, self.ket, self.bra, operator=op,
                                                      anc_env_left=None, anc_env_right=None,
                                                      cur_orthog=self.cur_orthog,
                                                      select_inds_bra=self.select_inds_bra,
                                                      # select_inds_ket=self.select_inds_ket,
                                                      ## don't need select_inds_ket i think
                                                      )]

            else:
                raise TypeError('not a valid type for operator')

        vec_conj = self.bra if target_use_bra else None
        sel_inds = self.select_inds_bra if target_use_bra else self.select_inds_ket
        vec_block = BlockVector_Cross(self.L, self.ket, vec_conj, cur_orthog=self.cur_orthog,
                                      select_inds_bra=sel_inds)

        op_blocks += (op_blocks, vec_block)

        self.op_blocks = op_blocks
        self.vec_block = vec_block