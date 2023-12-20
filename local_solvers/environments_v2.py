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
import local_solvers.helper_tn as helper_tn

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']
ProjType = Union[MPS_type, dict[int, tuple[int,...]]]


""" 
was trying to specify Ovlp/Operator x Cross/DMRG environments
i think it's possible but the kwargs would be annoying
"""

from enum import IntEnum

class EnvironmentSide(IntEnum):
    LEFT = 1  ## int denotes shift in position when extended environment
    RIGHT = -1



class Environment:

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'ProjType',
                 bra: 'ProjType' = None,
                 # operators_list: Sequence[qtn.MatrixProductOperator] = None,
                 operator: 'MPO_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 ket_horizontal_bonds: dict[int, str] = None,
                 bra_horizontal_bonds: dict[int, str] = None
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        """
        self.L = L  ## number of sites over which we'll need to make environments for
        self.side = side
        self.ket = ket  ## do not make copy -- want to update from outside
        self.bra = bra

        self.operator = operator  # .copy() if operator is not None else operator  ## why does this cause issues?

        if self.side == EnvironmentSide.RIGHT:
            self._envs = {L - 1: init_env.copy() if init_env is not None else None}
        else:
            self._envs = {0: init_env.copy() if init_env is not None else None}

        self.end_env = end_env
        self.mps_inds = list(range(self.ket.L)) if mps_inds is None else mps_inds
        self.bra_horizontal_bonds = {} if bra_horizontal_bonds is None else bra_horizontal_bonds
        self.ket_horizontal_bonds = {} if ket_horizontal_bonds is None else ket_horizontal_bonds
        # self.update_horizontal_bonds()

    def __getitem__(self, pos):
        return self._envs.get(pos, None)

    def __setitem__(self, pos, env: 'qtn.Tensor'):
        self._envs[pos] = env

    @property
    def exponent(self):
        raise NotImplementedError
        # if self.operator is not None:
        #     return self.operator.exponent  ## <x|A|x>
        # else:
        #     return self.ket.exponent    ## <x|b>

    def update_horizontal_bonds(self):
        raise NotImplementedError

    def get(self, pos):
        return self._envs.get(pos, None)

    def modify(self, ket=None, bra=None, operator=None, init_env=None, mps_inds=None):
        if ket is not None:     self.ket = ket
        if bra is not None:     self.bra = bra
        if operator is not None:    self.operator = operator
        if mps_inds is not None:    self.mps_inds = mps_inds
        self.reinitialize(init_env=init_env)
        self.update_horizontal_bonds()

    def create_like(self, ket=None, bra=None, operator=None, init_env=None, end_env=None, mps_inds=None):
        new_L = self.L if ket is None else ket.L
        self_init_env = self._envs[0] if self.side is EnvironmentSide.LEFT else \
            self._envs[self.L - 1]
        new_env = Environment(new_L, self.side,
                              ket=self.ket if ket is None else ket,
                              bra=self.bra if bra is None else bra,
                              operator=self.operator if operator is None else operator,
                              init_env=self_init_env if init_env is None else init_env,
                              end_env=self.end_env if end_env is None else end_env,
                              mps_inds=self.mps_inds if mps_inds is None else mps_inds)
        return new_env

    def copy(self):
        new_env = self.__class__(self.L, self.side, self.ket, self.bra,
                                 self.operator.copy() if self.operator is not None else self.operator,
                                 mps_inds=self.mps_inds,
                                 bra_horizontal_bonds=self.bra_horizontal_bonds,
                                 ket_horizontal_bonds=self.ket_horizontal_bonds,
                                 )
        new_env._envs = {k: env.copy() if isinstance(env, qtn.Tensor) else env for k, env in self._envs.items()}
        ## else None, Numeric
        return new_env

    def reinitialize(self, init_env=None):
        if self.side == EnvironmentSide.RIGHT:
            init_env = self._envs[self.L - 1] if init_env is None else init_env
            self._envs = {self.L - 1: init_env}
        else:
            init_env = self._envs[0] if init_env is None else init_env
            self._envs = {0: init_env}

    def reinitialize_i(self, i):
        """ reinitialize envs associate with MPS site position i
        """
        if self.side == EnvironmentSide.RIGHT:
            for ix in range(i):
                self._envs.pop(ix, None)
        else:
            for ix in range(i + 1, self.L):
                self._envs.pop(ix, None)

    def extend_env(self, pos: int) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        raise NotImplementedError

    def build_all_envs(self, end=None):
        """ compute all right envs for <Ax|b>
        """
        start = 0 if self.side is EnvironmentSide.LEFT else self.L - 1
        end = (-1 if self.side is EnvironmentSide.RIGHT else self.L) if end is None else end
        for i in range(start, end, self.side):
            self.extend_env(i)
        return

    def norm(self, bra_tens=None, ket_tens=None):
        """ returns <bra|operator|ket>, without
        """
        last_ind = self.L if self.side is EnvironmentSide.LEFT else -1
        last_env = self.get(last_ind - self.side)

        if bra_tens is None:
            bra_tens = self.bra[last_ind]
            assert(isinstance(bra_tens, qtn.Tensor)), 'need to define last bra_tens to compute norm'

        if ket_tens is None:
            ket_tens = self.ket[last_ind]
            assert (isinstance(ket_tens, qtn.Tensor)), 'need to define last ket_tens to compute norm'

        tens_list = [last_env, bra_tens, ket_tens]
        if self.operator is not None:
            tens_list += [self.operator[last_ind]]

        if self.end_env is not None:
            tens_list += [self.end_env]

        ovlp = qtn.tensor_contract(tens_list)

        tot_exponent = self.ket.exponent + self.bra.exponent
        if self.operator is not None:
            tot_exponent += self.operator.exponent

        return ovlp * 10 ** tot_exponent


class DMRGEnvironment(Environment, ABC):

    def __init__(self, L: int, side: 'EnvironmentSide',
                 ket: 'qtn.MatrixProductState',
                 bra: 'MPS_type' = None,
                 operator: 'MPO_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 **kwargs):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        """
        super().__init__(L, side, ket, bra=bra, operator=operator, init_env=init_env, end_env=end_env,
                         mps_inds=mps_inds, **kwargs)

        if len(self.ket_horizontal_bonds) == 0 and len(self.bra_horizontal_bonds) == 0:
            ## those params are passed through self.copy()
            self.update_horizontal_bonds()

    def extend_env(self, pos: int, **kwargs) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        env = self._envs[pos]
        if isinstance(self.bra, MatrixProductStateTN):
            bra_tens = self.bra[self.mps_inds[pos]]
        else:
            bra_tens = [self.bra[self.mps_inds[pos]]]

        if isinstance(self.ket, MatrixProductStateTN):
            ket_tens = self.ket[self.mps_inds[pos]]
        else:
            ket_tens = [self.ket[self.mps_inds[pos]]]

        if isinstance(self.operator, MatrixProductOperatorTN):
            op_tens = self.operator[self.mps_inds[pos]]
        elif self.operator is None:
            op_tens = []
        else:
            op_tens = [self.operator[self.mps_inds[pos]]]
            # op_tens = [mpo[pos] for mpo in self.operators]

        new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)
        if env is not None:
            new_env.add(env)

        new_env_exp = new_env.exponent
        new_env = new_env.contract_tags(all)
        if isinstance(new_env, qtn.Tensor):
            new_env.modify(apply=lambda x: x * 10 ** new_env_exp)
        else:
            new_env = new_env * 10 ** new_env_exp
        # print('new env', new_env, new_env.norm())
        # print('env exponent', new_env_exp)
        self._envs[pos + self.side.value] = new_env
        return new_env

    def build_all_envs(self, end=None):
        """ compute all right envs for <Ax|b>
        """
        start = 0 if self.side is EnvironmentSide.LEFT else self.L - 1
        end = (-1 if self.side is EnvironmentSide.RIGHT else self.L) if end is None else end
        for i in range(start, end, self.side):
            self.extend_env(i)
        return

    def update_horizontal_bonds(self):
        if self.side == EnvironmentSide.RIGHT:
            init_env = self._envs[self.L - 1]
            self.bra_horizontal_bonds = {i: self.bra.bond(self.mps_inds[i], self.mps_inds[i + 1])
                                         for i in range(0, self.L - 1)}
            self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i], self.mps_inds[i + 1])
                                         for i in range(0, self.L - 1)}
            # print('self. bra horizontal', self.bra_horizontal_bonds)
            # print('self. ket horizontal', self.ket_horizontal_bonds)

            self.bra_horizontal_bonds[self.L - 1] = None if init_env is None \
                else self.bra[self.mps_inds[-1]].bonds(init_env)
            self.ket_horizontal_bonds[self.L - 1] = None if init_env is None \
                else self.ket[self.mps_inds[-1]].bonds(init_env)

        else:
            init_env = self._envs[0]
            self.bra_horizontal_bonds = {i: self.bra.bond(self.mps_inds[i], self.mps_inds[i - 1])
                                         for i in range(1, self.L)}
            self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i], self.mps_inds[i - 1])
                                         for i in range(1, self.L)}

            self.bra_horizontal_bonds[0] = None if init_env is None \
                else self.bra[self.mps_inds[0]].bonds(init_env)
            self.ket_horizontal_bonds[0] = None if init_env is None \
                else self.ket[self.mps_inds[0]].bonds(init_env)

    def norm(self, **kwargs):
        """ returns <bra|operator|ket>, without
        """
        last_ind = self.L if self.side is EnvironmentSide.LEFT else -1
        ovlp = None  # self[last_ind]       ## this likely is not updated
        if ovlp is None:
            try:
                ovlp = self.extend_env(last_ind - self.side)
            except KeyError:
                self.build_all_envs()
                ovlp = self.extend_env(last_ind - self.side)

        if self.end_env is not None:
            ovlp = qtn.tensor_contract(ovlp, self.end_env)

        tot_exponent = self.ket.exponent + self.bra.exponent
        if self.operator is not None:
            tot_exponent += self.operator.exponent

        return ovlp * 10 ** tot_exponent



class CrossEnvironment(Environment, ABC):
    """ turns out to be equivalent to index selection
    """

    def __init__(self, L: int, side: 'EnvironmentSide',
                 ket: Union['qtn.MatrixProductState', dict[int, tuple[int,...]]],
                 bra: dict[int, tuple[int,...]],
                 operator: 'MPO_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 bra_horizontal_bonds: dict[int, str] = None,
                 ket_horizontal_bonds: dict[int, str] = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            operators: MPO corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)

            TODO:  needs testing for when there are operators
        """
        super().__init__(L, side, ket, bra=bra, operator=operator,
                         init_env=init_env, end_env=end_env, mps_inds=mps_inds,
                         bra_horizontal_bonds=bra_horizontal_bonds, ket_horizontal_bonds=ket_horizontal_bonds)

        # self.select_inds_dict = {} if select_inds_dict is None else select_inds_dict
        # self.bra_horizontal_bonds = {} if bra_horizontal_bonds is None else bra_horizontal_bonds
        # self.ket_horizontal_bonds = {}
        if len(self.ket_horizontal_bonds) == 0 and len(self.bra_horizontal_bonds) == 0:
            self.update_horizontal_bonds()

    def __getitem__(self, pos):
        return self._envs.get(pos, None)

    def __setitem__(self, pos, env: 'qtn.Tensor'):
        self._envs[pos] = env

    def update_horizontal_bonds(self):
        ## note: to extend to combs/other geometries, need to be able to index tensors and compute bonds between them
        if self.side == EnvironmentSide.RIGHT:
            init_env = self._envs[self.L - 1]

            if isinstance(self.ket, qtn.MatrixProductState):
                self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i], self.mps_inds[i + 1])
                                             for i in range(0, self.L - 1)}

            if len(self.bra_horizontal_bonds) == 0:
                self.bra_horizontal_bonds = {i: self.ket_horizontal_bonds[i] + '_'
                                             for i in range(0, self.L - 1)}

            if init_env is not None:
                env_inds = init_env.bonds(self.ket[self.mps_inds[-1]])
                self.ket_horizontal_bonds[self.L - 1] = next(iter(env_inds))
                if self.operator is not None:
                    env_inds = env_inds + init_env.bonds(self.operator[self.mps_inds[-1]])
                self.bra_horizontal_bonds[self.L - 1] = [ind for ind in init_env.inds if ind not in env_inds]
            else:
                self.bra_horizontal_bonds[self.L - 1] = None
                self.ket_horizontal_bonds[self.L - 1] = None

        else:
            init_env = self._envs[0]

            if isinstance(self.ket, qtn.MatrixProductState):
                self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i], self.mps_inds[i - 1])
                                             for i in range(1, self.L)}

            if len(self.bra_horizontal_bonds) == 0:
                self.bra_horizontal_bonds = {i: self.ket_horizontal_bonds[i] + '_'
                                             for i in range(1, self.L)}

            if init_env is not None:
                env_inds = init_env.bonds(self.ket[self.mps_inds[0]])
                self.ket_horizontal_bonds[0] = next(iter(env_inds))
                if self.operator is not None:
                    env_inds = env_inds + init_env.bonds(self.operator[self.mps_inds[0]])
                self.bra_horizontal_bonds[0] = [ind for ind in init_env.inds if ind not in env_inds]
            else:
                self.bra_horizontal_bonds[0] = None
                self.ket_horizontal_bonds[0] = None

    # def get(self, pos):
    #     return self._envs.get(pos, None)
    #
    # def modify(self, ket=None, operator=None, init_env=None, mps_inds=None):
    #     if ket is not None:     self.ket = ket
    #     if operator is not None:    self.operator = operator
    #     if mps_inds is not None:    self.mps_inds = mps_inds
    #     self.reinitialize(init_env=init_env)
    #     self.update_horizontal_bonds()
    #
    # def create_like(self, ket=None, operator=None, init_env=None, mps_inds=None):
    #     new_L = self.L if ket is None else ket.L
    #     self_init_env = self._envs[0] if self.side is EnvironmentSide.LEFT else self._envs[self.L - 1]
    #     new_env = CrossEnvironment(new_L, self.side,
    #                                ket=self.ket if ket is None else ket,
    #                                operator=self.operator if operator is None else operator,
    #                                init_env=self_init_env if init_env is None else init_env,
    #                                mps_inds=self.mps_inds if mps_inds is None else mps_inds)
    #     return new_env

    # def copy(self):
    #     # new_env = super().copy()
    #     new_env = self.__class__(self.L, self.side, self.ket, bra=self.bra,
    #                              operator=self.operator.copy() if self.operator is not None else self.operator,
    #                              mps_inds=self.mps_inds,
    #                              bra_horizontal_bonds = self.bra_horizontal_bonds,
    #                              ket_horizontal_bonds = self.ket_horizontal_bonds
    #                              )
    #     new_env._envs = {k: env.copy() if isinstance(env, qtn.Tensor) else env for k, env in self._envs.items()}
    #     return new_env

    # def reinitialize(self, init_env=None):
    #     super().reinitialize(init_env=init_env)
    #     if isinstance(self.ket, dict):
    #         self.ket.clear()
    #     if isinstance(self.bra, dict):
    #         self.bra.clear()
    #
    # def reinitialize_i(self, i):
    #     """ reinitialize envs associate with MPS site position i
    #     """
    #     super().reinitialize_i(i)
    #     # self.select_inds_dict.pop(i)
    #     if isinstance(self.ket, dict):
    #         self.ket.pop(i)
    #     if isinstance(self.bra, dict):
    #         self.bra.pop(i)

    # def extend_env(self, pos: int) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
    #     """ build left environment or right environment
    #         assumes bra, ket are properly canonicalized, bra already cc'ed if specified
    #         pos: position of env to extend
    #     """
    #     ## new env is ket_and_env[select_inds]
    #     if self.operator is not None:  ## select inds on input and output
    #         op_tens = self.operator[self.mps_inds[pos]]
    #         upper_ind = self.operator.upper_ind_id.format(self.mps_inds[pos])
    #         lower_ind = self.operator.lower_ind_id.format(self.mps_inds[pos])
    #
    #         bra_old = self.bra_horizontal_bonds[pos]
    #         ket_old = self.ket_horizontal_bonds[pos]
    #         bra_new = self.bra_horizontal_bonds[pos + self.side.value]
    #         ket_new = self.ket_horizontal_bonds[pos + self.side.value]
    #
    #         env_tens = self._envs[pos]
    #         if env_tens is not None:
    #             ket_and_env = qtn.tensor_contract(op_tens, env_tens)
    #         else:
    #             ket_and_env = op_tens.copy()
    #
    #         if bra_old is not None:
    #             new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],
    #                                         ket_new: [lower_ind, ket_old]})
    #             new_env.transpose(bra_new, ket_new, inplace=True)
    #         else:
    #             new_env = ket_and_env.reindex({upper_ind: bra_new,
    #                                            lower_ind: ket_new})
    #             new_env.transpose(bra_new, ket_new, inplace=True)
    #
    #         bra_select_inds = self.bra[pos]
    #         if bra_select_inds is None:
    #             raise ValueError('need to compute/update bra (select_inds) to extend env')
    #
    #         ket_select_inds = self.ket[pos]
    #         if ket_select_inds is None:
    #             raise ValueError('need to compute/update ket (select_inds) to extend env')
    #
    #         op_ind = next(iter(new_env.bonds(op_tens)))
    #         new_env.transpose(bra_new, ket_new, op_ind, inplace=True)
    #         new_env.modify(apply=lambda x: x[bra_select_inds, ket_select_inds, :])
    #
    #     else:
    #         ket_tens = self.ket[self.mps_inds[pos]]
    #         site_ind = self.ket.site_ind_id.format(self.mps_inds[pos])
    #
    #         bond_old = self.bra_horizontal_bonds[pos]
    #         bond_new = self.bra_horizontal_bonds[pos + self.side.value]
    #         bond_ket = self.ket_horizontal_bonds[pos + self.side.value]
    #
    #         env_tens = self._envs[pos]
    #         if env_tens is not None:
    #             ket_and_env = qtn.tensor_contract(ket_tens, env_tens)
    #         else:
    #             ket_and_env = ket_tens.copy()
    #
    #         if bond_old is not None:
    #             new_env = ket_and_env.fuse({bond_new: [site_ind, bond_old]})
    #             new_env.transpose(bond_new, bond_ket, inplace=True)
    #         else:
    #             new_env = ket_and_env.reindex({site_ind: bond_new})
    #             new_env.transpose(bond_new, bond_ket, inplace=True)
    #
    #         select_inds = self.bra[pos]
    #         # print('select inds', select_inds)
    #         if select_inds is None:
    #             # select_inds = maxvol_inds(new_env.data)  # n x r
    #             raise ValueError('need to compute/update select_inds to extend env')
    #
    #         new_env.modify(apply=lambda x: x[select_inds, :])
    #
    #     self._envs[pos + self.side.value] = new_env
    #     return new_env


    # def build_all_envs(self, end=None): # , select_inds_dict=None):
    #     """ compute all right envs for <Ax|b>
    #     """
    #     start = 0 if self.side is EnvironmentSide.LEFT else self.L - 1
    #     end = (-1 if self.side is EnvironmentSide.RIGHT else self.L) if end is None else end
    #     for i in range(start, end, self.side):
    #         # select_inds = None if select_inds_dict is None else select_inds_dict[i]
    #         # select_inds = self.select_inds.get(i, None)
    #         self.extend_env(i)   # , select_inds=select_inds)
    #     return



class TargetEnv(Environment, ABC):

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'qtn.MatrixProductState',
                 bra: 'MPS_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 bra_horizontal_bonds: dict[int, str] = None,
                 ket_horizontal_bonds: dict[int, str] = None
                 ):
        """
        environment for <x|b>
        initialized assuming that all the indices are properly aligned
        ket: MPS for ket target wavefunction |b>
        bra: MPS for bra test wavefunction <x|
        """
        super().__init__(L, side, ket, bra=bra, init_env=init_env, end_env=end_env, mps_inds=mps_inds,
                         bra_horizontal_bonds=bra_horizontal_bonds, ket_horizontal_bonds=ket_horizontal_bonds)

    @property
    def exponent(self):
        return self.ket.exponent


class OperatorEnv(Environment, ABC):

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'qtn.MatrixProductState',
                 operator: 'MPO_type' = None,
                 bra: 'MPS_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 bra_horizontal_bonds: dict[int, str] = None,
                 ket_horizontal_bonds: dict[int, str] = None
                 ):
        """
        environment for <bra| A | ket>, for computing P A P
        initialized assuming that all the indices are properly aligned
        side: which side of canonical site the environment corresponds to
        ket: MPS for ket test wavefunction |ket>
        bra: MPS for bra test wavefunction <bra|  (if None, defaults to conj(ket)
        operator: MPO. if None, uses the identity MPO
        """
        super().__init__(L, side, ket, bra=bra, operator=operator,
                         init_env=init_env, end_env=end_env, mps_inds=mps_inds,
                         bra_horizontal_bonds=bra_horizontal_bonds, ket_horizontal_bonds=ket_horizontal_bonds)

    @property
    def exponent(self):
        return self.operator.exponent if self.operator is not None else 0.0


#####

class TargetEnv_DMRG(DMRGEnvironment, TargetEnv):

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'qtn.MatrixProductState',
                 bra: 'MPS_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 **kwargs):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, side, ket, bra=bra, init_env=init_env, end_env=end_env, mps_inds=mps_inds, **kwargs)


class TargetEnv_Cross(CrossEnvironment, TargetEnv):

    def __init__(self, L: int, side: 'EnvironmentSide',
                 ket: 'qtn.MatrixProductState',
                 bra: dict[int, tuple[int]],
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 bra_horizontal_bonds: dict[int, str] = None,
                 **kwargs):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, side, ket, bra=bra, init_env=init_env, end_env=end_env, mps_inds=mps_inds,
                         bra_horizontal_bonds=bra_horizontal_bonds, **kwargs)

    def extend_env(self, pos: int) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        ket_tens = self.ket[self.mps_inds[pos]]
        site_ind = self.ket.site_ind_id.format(self.mps_inds[pos])

        bond_old = self.bra_horizontal_bonds[pos]
        bond_new = self.bra_horizontal_bonds[pos + self.side.value]
        bond_ket = self.ket_horizontal_bonds[pos + self.side.value]

        env_tens = self._envs[pos]
        if env_tens is not None:
            ket_and_env = qtn.tensor_contract(ket_tens, env_tens)
        else:
            ket_and_env = ket_tens.copy()

        if bond_old is not None:
            new_env = ket_and_env.fuse({bond_new: [site_ind, bond_old]})
            new_env.transpose(bond_new, bond_ket, inplace=True)
        else:
            new_env = ket_and_env.reindex({site_ind: bond_new})
            new_env.transpose(bond_new, bond_ket, inplace=True)

        select_inds = self.bra[pos]
        # print('select inds', select_inds)
        if select_inds is None:
            # select_inds = maxvol_inds(new_env.data)  # n x r
            raise ValueError('need to compute/update select_inds to extend env')

        new_env.modify(apply=lambda x: x[select_inds, :])

        self._envs[pos + self.side.value] = new_env
        return new_env


class OperatorEnv_DMRG(DMRGEnvironment, OperatorEnv):

    def __init__(self, L: int, side: 'EnvironmentSide',
                 ket: 'qtn.MatrixProductState',
                 operator: 'MPO_type' = None,
                 bra: 'MPS_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 **kwargs):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, side, ket, bra=bra, operator=operator, init_env=init_env, end_env=end_env,
                         mps_inds=mps_inds, **kwargs)


class OperatorEnv_Cross(CrossEnvironment, OperatorEnv):

    def __init__(self, L: int, side: 'EnvironmentSide',
                 ket: dict[int, tuple[int,...]],
                 operator: 'MPO_type' = None,
                 bra: dict[int, tuple[int,...]] = None,
                 init_env: 'qtn.Tensor' = None,
                 end_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None,
                 ket_horizontal_bonds: dict[int, str] = None,
                 bra_horizontal_bonds: dict[int, str] = None):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(L, side, ket, bra=bra, operator=operator,
                         init_env=init_env, end_env=end_env, mps_inds=mps_inds,
                         bra_horizontal_bonds=bra_horizontal_bonds, ket_horizontal_bonds=ket_horizontal_bonds)

    def extend_env(self, pos: int) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        op_tens = self.operator[self.mps_inds[pos]]
        upper_ind = self.operator.upper_ind_id.format(self.mps_inds[pos])
        lower_ind = self.operator.lower_ind_id.format(self.mps_inds[pos])

        bra_old = self.bra_horizontal_bonds[pos]
        ket_old = self.ket_horizontal_bonds[pos]
        bra_new = self.bra_horizontal_bonds[pos + self.side.value]
        ket_new = self.ket_horizontal_bonds[pos + self.side.value]

        env_tens = self._envs[pos]
        if env_tens is not None:
            ket_and_env = qtn.tensor_contract(op_tens, env_tens)
        else:
            ket_and_env = op_tens.copy()

        if bra_old is not None:
            new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],
                                        ket_new: [lower_ind, ket_old]})
            new_env.transpose(bra_new, ket_new, inplace=True)
        else:
            new_env = ket_and_env.reindex({upper_ind: bra_new,
                                           lower_ind: ket_new})
            new_env.transpose(bra_new, ket_new, inplace=True)

        bra_select_inds = self.bra[pos]
        if bra_select_inds is None:
            raise ValueError('need to compute/update bra (select_inds) to extend env')

        ket_select_inds = self.ket[pos]
        if ket_select_inds is None:
            raise ValueError('need to compute/update ket (select_inds) to extend env')

        op_ind = next(iter(new_env.bonds(op_tens)))
        new_env.transpose(bra_new, ket_new, op_ind, inplace=True)
        new_env.modify(apply=lambda x: x[bra_select_inds, ket_select_inds, :])

        self._envs[pos + self.side.value] = new_env
        return new_env
