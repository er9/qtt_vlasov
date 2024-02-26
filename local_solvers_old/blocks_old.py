import numpy as np
from abc import ABC

from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper
import local_solvers_old.helper_tn as helper_tn

# from local_solvers_old.environments_v2 import Environment, EnvironmentSide, EnvironmentType
# from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
# from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
from local_solvers_old.local_solver import SweepDirection

ProjType = Union[qtn.MatrixProductState, dict[int, tuple[int,...]]]

class EnvironmentType(IntEnum):
    DMRG = 0
    CROSS = 1

def init_block(env_type: EnvironmentType, ket, bra, operator=None,
               anc_env_left=None, anc_env_right=None,
               select_inds_ket=None, select_inds_bra=None,
               cur_orthog=False) -> 'Block':

    if env_type == EnvironmentType.DMRG:
        if operator is None:
            return BlockTarget_DMRG(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                    cur_orthog=cur_orthog)
        else:
            if operator in ['identity', 'I', 'iden', 'Iden', 'Identity']:
                return BlockOperator_DMRG(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                          cur_orthog=cur_orthog)
            else:
                return BlockOperator_DMRG(ket.L, ket, bra, operator=operator,
                                          anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                          cur_orthog=cur_orthog)
    elif env_type == EnvironmentType.CROSS:
        if operator is None:
            return BlockTarget_Cross(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                     select_inds_bra=select_inds_bra, cur_orthog=cur_orthog)
        else:
            if operator in ['identity', 'I', 'iden', 'Iden', 'Identity']:
                return BlockOperator_Cross(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                           select_inds_ket=select_inds_ket, select_inds_bra=select_inds_bra,
                                           cur_orthog=cur_orthog)
            else:
                return BlockOperator_Cross(ket.L, ket, bra, operator=operator,
                                           anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                           select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                                           cur_orthog=cur_orthog)

# class Block:
#
#     def __init__(self, L: int, env_type: EnvironmentType,
#                  ket: 'qtn.MatrixProductState',
#                  bra: 'qtn.MatrixProductState' = None,
#                  # operators_list: Sequence[qtn.MatrixProductOperator] = None,
#                  operator: 'qtn.MatrixProductOperator' = None,
#                  init_env: 'qtn.Tensor' = None,
#                  end_env: 'qtn.Tensor' = None,
#                  env_left: 'Environment' = None,
#                  env_right: 'Environment' = None,
#                  mps_inds: Sequence[int] = None,
#                  ket_horizontal_bonds: dict[int, str] = None,
#                  bra_horizontal_bonds: dict[int, str] = None
#                  ):
#
#         if env_left is not None and env_right is not None:
#             self.env_class = env_left.__class__
#             self.env_left = env_left
#             self.env_right = env_right
#         else:
#             if operator is None:
#                 self.env_class = TargetEnv_DMRG if env_type == EnvironmentType.DMRG else TargetEnv_Cross
#                 self.env_left = self.env_class(L, EnvironmentSide.LEFT, ket, bra,
#                                                init_env=init_env, end_env=end_env, mps_inds=mps_inds,
#                                                ket_horizontal_bonds=ket_horizontal_bonds,
#                                                bra_horizontal_bonds=bra_horizontal_bonds)
#                 self.env_right = self.env_class(L, EnvironmentSide.RIGHT, ket, bra,
#                                                 init_env=init_env, end_env=end_env, mps_inds=mps_inds,
#                                                 ket_horizontal_bonds=ket_horizontal_bonds,
#                                                 bra_horizontal_bonds=bra_horizontal_bonds)
#             else:
#                 self.env_class = OperatorEnv_DMRG if env_type == EnvironmentType.DMRG else OperatorEnv_Cross
#                 self.env_left = self.env_class(L, EnvironmentSide.LEFT, ket, bra, operator=operator,
#                                                init_env=init_env, end_env=end_env, mps_inds=mps_inds,
#                                                ket_horizontal_bonds=ket_horizontal_bonds,
#                                                bra_horizontal_bonds=bra_horizontal_bonds)
#                 self.env_right = self.env_class(L, EnvironmentSide.RIGHT, ket, bra, operator=operator,
#                                                 init_env=init_env, end_env=end_env, mps_inds=mps_inds,
#                                                 ket_horizontal_bonds=ket_horizontal_bonds,
#                                                 bra_horizontal_bonds=bra_horizontal_bonds)
#         self.ket = self.env_left.ket
#         self.bra = self.env_left.bra
#         self.operator = self.env_left.operator
#         self.L = self.env_left.L
#         self.mps_inds = self.env_left.mps_inds
#         self.ket_horizontal_bonds = self.env_left.ket_horizontal_bonds
#         self.bra_horizontal_bonds = self.env_left.bra_horizontal_bonds


class Block:

    def __init__(self, L: int, env_type: EnvironmentType,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 cur_orthog = None,
                 ):

        self.L = L

        ## left and right envs at site i
        # self.envs: dict[int, list['qtn.Tensor','qtn.Tensor']] = {i: [None, None] for i in range(L)}
        # if anc_env_left is not None:
        #     self.envs[0][0] = anc_env_left.copy()
        # if anc_env_right is not None:
        #     self.envs[L-1][1] = anc_env_right.copy()
        self.envs: dict[int, Optional['qtn.Tensor']] = {i: None for i in range(-1,L + 1)}
        if anc_env_left is not None:
            self.envs[-1] = anc_env_left.copy()
        if anc_env_right is not None:
            self.envs[L] = anc_env_right.copy()

        self.operator = operator  # .copy() if operator is not None else operator  ## why does this cause issues?
        self.ket = ket
        self.bra = bra

        self.mps_inds = list(range(L)) if mps_inds is None else mps_inds
        # self.bra_horizontal_bonds = {} if bra_horizontal_bonds is None else bra_horizontal_bonds
        # self.ket_horizontal_bonds = {} if ket_horizontal_bonds is None else ket_horizontal_bonds
        # if len(self.ket_horizontal_bonds) == 0 and len(self.bra_horizontal_bonds) == 0:
        #     ## those params are passed through self.copy()
        #     self.update_horizontal_bonds()
        self.update_horizontal_bonds()

        self.env_type = env_type
        self.cur_orthog = None
        self.init_envs(cur_orthog)


    def init_envs(self, cur_orthog):
        if cur_orthog is not None:
            for i in range(cur_orthog):
                self.extend_env(i, direction=SweepDirection.RIGHT)
            for i in range(self.L - 1, cur_orthog, -1):
                self.extend_env(i, direction=SweepDirection.LEFT)
        self.cur_orthog = cur_orthog


    def reinitialize(self):
        self.cur_orthog = None
        for k in self.envs:
            self.envs[k] = None

    def reinitialize_i(self, i: int):
        self.envs[i] = None

    def bra_horizontal_bond(self, i: int, step: Union[int, SweepDirection]):
        if self.bra is None:
            ind = self.ket_horizontal_bond(i, step)
            if ind is not None:
                ind = ind + '_'
            return ind

        if (i == 0 and step == -1) or (i == -1 and step == 1):
            anc_L = self.anc_env_L
            return None if anc_L is None else next(iter(self.bra[0].bonds(self.anc_env_L)))
        elif (i == self.L - 1 and step == 1) or (i == self.L and step == -1):
            anc_R = self.anc_env_R
            return None if anc_R is None else next(iter(self.bra[self.L - 1].bonds(self.anc_env_R)))
        else:
            return self.bra.bond(i, i + step)

    def ket_horizontal_bond(self, i: int, step: int):
        if (i == 0 and step == -1) or (i == -1 and step == 1):
            anc_L = self.anc_env_L
            return None if anc_L is None else next(iter(self.ket[0].bonds(self.anc_env_L)))
        elif (i == self.L - 1 and step == 1) or (i == self.L and step == -1):
            anc_R = self.anc_env_R
            return None if anc_R is None else next(iter(self.ket[self.L - 1].bonds(self.anc_env_R)))
        else:
            return self.ket.bond(i, i + step)

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
        for ix in range(left_site_pos, left_site_pos + nsites):
            if self.bra is None:
                if self.operator is not None:
                    inds_dict[self.operator.upper_ind(ix)] = self.ket.site_ind(ix)
            else:
                inds_dict[self.bra.site_ind(ix)] = self.ket.site_ind(ix)
        return inds_dict

    def update_horizontal_bonds(self):
        if self.bra is None:
            return

        same_bonds = False
        ind = 0
        while ind < self.L - 1 and not same_bonds:
            same_bonds = self.ket.bond(ind, ind + 1) == self.bra.bond(ind, ind + 1)
            ind += 1

        if same_bonds:
            self.bra.mangle_inner_(append='_')
        return

    def copy(self):

        out = self.__class__(self.L, self.env_type, self.ket.copy(),
                             self.bra.copy(), operator=self.operator, mps_inds=self.mps_inds,
                             )

        out.envs = {k: [op1.copy() if op1 is not None else None,
                        op2.copy() if op2 is not None else None] for k, (op1,op2) in self.envs.items()}
        return out

    @property
    def exponent(self):
        raise NotImplementedError
        # if self.operator is not None:
        #     return self.operator.exponent  ## <x|A|x>
        # else:
        #     return self.ket.exponent    ## <x|b>

    @property
    def anc_env_L(self):
        return self.envs[-1]

    @property
    def anc_env_R(self):
        return self.envs[self.L]

    def clear_envs(self):
        for k in range(self.L):
            self.envs[k] = None

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None):
        raise NotImplementedError

    def shift_left(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.LEFT)
        # self.cur_orthog += 1

    def shift_right(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.RIGHT)
        # self.cur_orthog -= 1

    def extend_env(self, i: int, direction: SweepDirection):
        raise NotImplementedError

    def norm(self):
        """ returns <bra|operator|ket>, without
        """
        if self.cur_orthog is None:
            raise NotImplementedError

        ix = self.cur_orthog
        env_L, env_R = self.envs[ix - 1], self.envs[ix + 1]
        tens_list = [self.ket[ix], self.bra[ix]]
        tens_list += [env_L] if env_L is not None else []
        tens_list += [env_R] if env_R is not None else []
        tens_list += [self.operator[ix]] if self.operator is not None else []

        ovlp = qtn.tensor_contract(*tens_list)

        tot_exponent = self.ket.exponent + self.bra.exponent
        if self.operator is not None:
            tot_exponent += self.operator.exponent

        return ovlp * 10 ** tot_exponent


class BlockDMRG(Block, ABC):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState' = None,
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            L: length of MPS
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operator: MPO corresponding to operator
            anc_env_left: env for ancillas on left side
            anc_env_right: env for ancillas on right side
            attribute envs:  dict[i, Tensor], left or right environemnt including tensors at site i
                i ranges from -1 to L
        """
        super().__init__(L, EnvironmentType.DMRG, ket, bra=bra, operator=operator,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         mps_inds=mps_inds, cur_orthog=cur_orthog)


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        if isinstance(self.bra, MatrixProductStateTN):
            bra_tens = self.bra[self.mps_inds[i]]
        else:
            bra_tens = [self.bra[self.mps_inds[i]]]

        if isinstance(self.ket, MatrixProductStateTN):
            ket_tens = self.ket[self.mps_inds[i]]
        else:
            ket_tens = [self.ket[self.mps_inds[i]]]

        if isinstance(self.operator, MatrixProductOperatorTN):
            op_tens = self.operator[self.mps_inds[i]]
        elif self.operator is None:
            op_tens = []
        else:
            op_tens = [self.operator[self.mps_inds[i]]]
            # op_tens = [mpo[pos] for mpo in self.operators]

        new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        # new_env_exp = new_env.exponent
        new_env = new_env.contract_tags(all)

        self.envs[i] = new_env
        self.cur_orthog = i + direction

        return new_env



class BlockCross(Block, ABC):

    def __init__(self, L: int,
                 ket: qtn.MatrixProductState,
                 bra: qtn.MatrixProductState,
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 select_inds_ket: dict[int, Optional[Sequence[int]]] = None,
                 select_inds_bra: dict[int, Optional[Sequence[int]]] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            L: length of MPS
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operator: MPO corresponding to operator
            anc_env_left: env for ancillas on left side
            anc_env_right: env for ancillas on right side
            select_inds_ket:  dict[int, Sequence[int]] for selecting inds at site i + left/right environment for ket
            select_inds_bra:  dict[int, Sequence[int]] for selecting inds at site i + left/right environment for bra

            TODO:  needs testing for when there are operators
        """
        self.select_inds_ket = {} if select_inds_ket is None else select_inds_ket
        self.select_inds_bra = {} if select_inds_bra is None else select_inds_bra

        super().__init__(L, EnvironmentType.CROSS, ket, bra=bra, operator=operator,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         mps_inds=mps_inds, cur_orthog=cur_orthog,
                         )


    def copy(self):
        out = super().copy()
        out.select_inds_ket = {k: v for k, v in self.select_inds_ket.items()}
        out.select_inds_bra = {k: v for k, v in self.select_inds_bra.items()}

####

class BlockTarget(Block, ABC):

    # def __init__(self, L: int, side: 'EnvironmentSide',
    #              ket: 'qtn.MatrixProductState',
    #              bra: 'qtn.MatrixProductState',
    #              init_env: 'qtn.Tensor' = None,
    #              end_env: 'qtn.Tensor' = None,
    #              mps_inds: Sequence[int] = None,
    #              ):
    #     """
    #     environment for <x|b>
    #     initialized assuming that all the indices are properly aligned
    #     ket: MPS for ket target wavefunction |b>
    #     bra: MPS for bra test wavefunction <x|
    #     """
    #     super().__init__(L, side, ket, bra=bra, init_env=init_env, end_env=end_env, mps_inds=mps_inds)

    @property
    def exponent(self):
        return self.ket.exponent

    def update_horizontal_bonds(self):
        super().update_horizontal_bonds()
        # self.ket.site_ind_id = self.bra.site_ind_id
        return


    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None):

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        b_left = self.envs[left_site_pos - 1]
        b_right = self.envs[left_site_pos + nsites]

        b_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):
            for si in site_inds:
                b_eff.add(self.ket[si])
        else:
            b_eff.add(self.ket[site_inds])

        if b_left is not None:
            b_eff.add(b_left)
        if b_right is not None:
            b_eff.add(b_right)

        b_eff.exponent += self.ket.exponent

        if return_combined:
            b_eff_tens = qtn.tensor_contract(*b_eff.tensors, preserve_tensor=True)
            b_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            if transpose_bonds is not None:
                b_eff_tens.transpose(*transpose_bonds, inplace=True)
            return b_eff_tens

        return b_eff


class BlockOperator(Block, ABC):

    # def __init__(self, L: int, side: 'EnvironmentSide',
    #              ket: 'qtn.MatrixProductState',
    #              bra: 'qtn.MatrixProductState',
    #              operator: 'qtn.MatrixProductOperator' = None,
    #              init_env: 'qtn.Tensor' = None,
    #              end_env: 'qtn.Tensor' = None,
    #              mps_inds: Sequence[int] = None,
    #              ):
    #     """
    #     environment for <bra| A | ket>, for computing P A P
    #     initialized assuming that all the indices are properly aligned
    #     side: which side of canonical site the environment corresponds to
    #     ket: MPS for ket test wavefunction |ket>
    #     bra: MPS for bra test wavefunction <bra|  (if None, defaults to conj(ket)
    #     operator: MPO. if None, uses the identity MPO
    #     """
    #     super().__init__(L, side, ket, bra=bra, operator=operator,
    #                      init_env=init_env, end_env=end_env, mps_inds=mps_inds,
    #                      )

    @property
    def exponent(self):
        return self.operator.exponent if self.operator is not None else 0.0

    def update_horizontal_bonds(self):
        super().update_horizontal_bonds()
        if self.bra is not None:
            self.bra.site_ind_id = self.operator.upper_ind_id
        self.ket.site_ind_id = self.operator.lower_ind_id
        return

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None):

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        assert(left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        A_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):
            for si in site_inds:
                A_eff.add(self.operator[si])
        else:
            A_eff.add(self.operator[site_inds])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.operator.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.operator.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            return A_eff_tens

        return A_eff

#####

class BlockTarget_DMRG(BlockDMRG, BlockTarget):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right, mps_inds=mps_inds,
                         cur_orthog=cur_orthog)


class BlockTarget_Cross(BlockCross, BlockTarget):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 select_inds_bra: dict[int, Optional[Sequence[int]]] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        self.select_inds_bra = {}
        super().__init__(L, ket, bra,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, mps_inds=mps_inds,
                         select_inds_bra=select_inds_bra, cur_orthog=cur_orthog)

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        ket_tens = self.ket[self.mps_inds[i]]
        site_ind = self.ket.site_ind_id.format(self.mps_inds[i])

        bond_old = self.bra_horizontal_bond(i - direction, direction)
        bond_new = self.bra_horizontal_bond(i, direction)
        bond_ket = self.ket_horizontal_bond(i, direction)

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        if direction == SweepDirection.RIGHT:
            if env_L is not None:
                ket_and_env = qtn.tensor_contract(ket_tens, env_L)
            else:
                ket_and_env = ket_tens.copy()
        else:
            if env_R is not None:
                ket_and_env = qtn.tensor_contract(ket_tens, env_R)
            else:
                ket_and_env = ket_tens.copy()

        if bond_old is not None:
            new_env = ket_and_env.fuse({bond_new: [site_ind, bond_old]})
            new_env.transpose(bond_new, bond_ket, inplace=True)
        else:
            new_env = ket_and_env.reindex({site_ind: bond_new})
            new_env.transpose(bond_new, bond_ket, inplace=True)

        select_inds = self.select_inds_bra[i]
        # print('select inds', select_inds)
        # select_inds = select_inds_l[i] if direction == SweepDirection.RIGHT else select_inds_r[i]
        if select_inds is None:
            # select_inds = maxvol_inds(new_env.data)  # n x r
            raise ValueError('need to compute/update select_inds to extend env')

        new_env.modify(apply=lambda x: x[select_inds, :])

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


class BlockOperator_DMRG(BlockDMRG, BlockOperator):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, ket, bra=bra, operator=operator, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         mps_inds=mps_inds, cur_orthog=cur_orthog)


class BlockDiagKet_DMRG(BlockDMRG, BlockOperator):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 # bra: 'qtn.MatrixProductState',
                 # operator: 'qtn.MatrixProductState',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        mpo = helper.mps_to_diag_mpo(ket, upper_ind_id=ket.site_ind_id + '_', lower_ind_id=ket.site_ind_id)
        # bra = ket.conj()
        # for tens in bra.tensors:
        #     tens.reindex({ind: ind + '_' for ind in tens.inds}, inplace=True)
        super().__init__(L, ket, None, operator=mpo, anc_env_left=anc_env_left,
                         anc_env_right=anc_env_right, mps_inds=mps_inds, cur_orthog=cur_orthog)

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        ### assuming ket is already in canonical form

        ket_tens = self.ket[self.mps_inds[i]]
        bra_tens = ket_tens.conj()
        bra_tens.reindex({ind: ind + '_' for ind in ket_tens.inds}, inplace=True)
        op_tens = ket_tens.reindex({ind: ind + '_o_' for ind in ket_tens.inds})
        site_ind = self.ket.site_ind_id.format(self.mps_inds[i])
        d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(i),
                                            (site_ind, site_ind + '_o_', site_ind + '_'),
                                            tags=(f'd_ijk({i})',))

        new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, d_ijk])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        new_env = new_env.contract_tags(all)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None):
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        A_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):
            for si in site_inds:

                phys_ind = self.ket.site_ind(si)
                d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
                                                    (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                    )
                op_tens = self.ket[si].copy()
                op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
                A_eff.add([op_tens, d_ijk])
        else:
            phys_ind = self.ket.site_ind(site_inds)
            d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(site_inds),
                                                (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                )
            op_tens = self.ket[site_inds].copy()
            A_eff.add([op_tens, d_ijk])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.ket.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            return A_eff_tens

        return A_eff

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
        for ix in range(left_site_pos, left_site_pos + nsites):
            inds_dict[self.ket.site_ind(ix) + '_'] = self.ket.site_ind(ix)
        return inds_dict


class BlockOperator_Cross(BlockCross, BlockOperator):

    def __init__(self, L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 select_inds_bra: dict[int, Optional[Sequence[int]]] = None,
                 select_inds_ket: dict[int, Optional[Sequence[int]]] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            select_inds_bra/ket: inds to select when moving canonical site from i to i +/- 1 (decimation)
                assumes select_inds are correctly set regardless of decimation direction
        """
        super().__init__(L, ket, bra, operator=operator,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, mps_inds=mps_inds,
                         select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                         cur_orthog=cur_orthog)

    # def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
    #     """ build left environment or right environment
    #         assumes bra, ket are properly canonicalized, bra already cc'ed if specified
    #         pos: position of env to extend
    #     """
    #     ## new env is ket_and_env[select_inds]
    #     op_tens = self.operator[self.mps_inds[i]]
    #     upper_ind = self.operator.upper_ind_id.format(self.mps_inds[i])
    #     lower_ind = self.operator.lower_ind_id.format(self.mps_inds[i])
    #
    #     bra_old = self.bra_horizontal_bond(i - direction, direction)
    #     ket_old = self.ket_horizontal_bond(i - direction, direction)
    #     op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
    #     bra_new = self.bra_horizontal_bond(i, direction)
    #     ket_new = self.ket_horizontal_bond(i, direction)
    #     op_new = self.operator.bond(i, i + direction)
    #
    #     envL, envR = self.envs[i - 1], self.envs[i + 1]
    #     env_tens = envL if direction == SweepDirection.RIGHT else envR
    #     if env_tens is not None:
    #         ket_and_env = qtn.tensor_contract(op_tens, env_tens)
    #     else:
    #         ket_and_env = op_tens.copy()
    #
    #     if bra_old is not None:
    #         new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],
    #                                     ket_new: [lower_ind, ket_old]})
    #         new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #     else:
    #         new_env = ket_and_env.reindex({upper_ind: bra_new,
    #                                        lower_ind: ket_new})
    #         new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #
    #     bra_select_inds = self.select_inds_bra[i]
    #     # bra_select_inds = self.select_inds_ket[i]
    #     if bra_select_inds is None:
    #         raise ValueError('need to compute/update bra (select_inds) to extend env')
    #
    #     ket_select_inds = self.select_inds_ket[i]
    #     if ket_select_inds is None:
    #         raise ValueError('need to compute/update ket (select_inds) to extend env')
    #
    #     # op_ind = next(iter(new_env.bonds(op_tens)))
    #     new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #     new_env.modify(apply=lambda x: (x[bra_select_inds, :, :])[:, ket_select_inds, :])
    #     # print('data', new_env.data.ndim)
    #     # new_data = new_env.data[bra_select_inds, :, :]
    #     # new_data = new_data[:, ket_select_inds, :]
    #     # new_env.modify(data=new_data)
    #
    #     self.envs[i] = new_env
    #     self.cur_orthog = i + direction
    #     return new_env


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        op_tens = self.operator[self.mps_inds[i]]
        upper_ind = self.operator.upper_ind_id.format(self.mps_inds[i])
        lower_ind = self.operator.lower_ind_id.format(self.mps_inds[i])

        bra_old = self.bra_horizontal_bond(i - direction, direction)
        ket_old = self.ket_horizontal_bond(i - direction, direction)
        op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
        bra_new = self.bra_horizontal_bond(i, direction)
        ket_new = self.ket_horizontal_bond(i, direction)
        op_new = self.operator.bond(i, i + direction)

        envL, envR = self.envs[i - 1], self.envs[i + 1]
        env_tens = envL if direction == SweepDirection.RIGHT else envR
        ket_tens = self.ket[i]   ## tensor with selected "columns" / "rows"
        if env_tens is not None:
            ket_and_env = qtn.tensor_contract(op_tens, env_tens, ket_tens)
        else:
            ket_and_env = qtn.tensor_contract(op_tens, ket_tens)

        if bra_old is not None:
            new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],
                                        })
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        else:
            new_env = ket_and_env.reindex({upper_ind: bra_new,
                                           })
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)

        bra_select_inds = self.select_inds_bra[i]
        # bra_select_inds = self.select_inds_ket[i]
        if bra_select_inds is None:
            raise ValueError('need to compute/update bra (select_inds) to extend env')

        # ket_select_inds = self.select_inds_ket[i]
        # if ket_select_inds is None:
        #     raise ValueError('need to compute/update ket (select_inds) to extend env')

        # op_ind = next(iter(new_env.bonds(op_tens)))
        new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        new_env.modify(apply=lambda x: x[bra_select_inds, :, :])
        # print('data', new_env.data.ndim)
        # new_data = new_env.data[bra_select_inds, :, :]
        # new_data = new_data[:, ket_select_inds, :]
        # new_env.modify(data=new_data)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env
