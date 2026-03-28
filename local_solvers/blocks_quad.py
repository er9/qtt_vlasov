import numpy as np
from abc import ABC

from setup_.defaults import *
import helper_quimb
from local_solvers.mps_classes import MPS, MPO
from local_solvers.blocks import Block, SweepDirection, EnvironmentType
from grid1D import Grid1D

def init_quad_block(env_type: EnvironmentType, ket, bra, operator=None,
               anc_env_left=None, anc_env_right=None,
               select_inds_ket=None, select_inds_bra=None,
               cur_orthog=False) -> 'BlockQuad':

    raise NotImplementedError



class BlockQuad:

    def __init__(self,  # env_type: EnvironmentType,
                 ket1: Union['MPS', 'qtn.MatrixProductState'],
                 ket2: Union['MPS', 'qtn.MatrixProductState'],
                 bra: Optional[Union['MPS', 'MPO']],
                 operator: 'MatrixProductTensor',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 grid: 'Grid1D' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog=None,
                 conj_ket2=True,
                 ):

        L = ket1.L

        self._ket1: 'MPS' = None
        self._ket2: 'MPS' = None
        self._bra: Optional[Union['MPS', 'MPO']] = None
        self._operator: Optional['MatrixProductTensor'] = None
        self.grid = grid
        self.conj_ket2 = conj_ket2

        ## left and right envs at site i
        self.envs: dict[int, Optional['qtn.Tensor']] = {i: None for i in range(-1, L + 1)}
        if anc_env_left is not None:
            self.envs[-1] = anc_env_left.copy()
        if anc_env_right is not None:
            self.envs[L] = anc_env_right.copy()

        self.ket1 = ket1
        self.ket2 = ket2
        self.bra = bra
        if operator is None:
            operator = grid.get_elemental_multiply_tn(in1_ind_id=ket1.site_ind_id,
                                                      ind2_ind_id=ket2.site_ind_id + '_[2]',
                                                      out_ind_id=(bra.site_ind_id if isinstance(bra, MPS)
                                                                  else bra.lower_ind_id))
            operator = operator.data
        self.operator = operator  # .copy() if operator is not None else operator  ## why does this cause issues?

        # self.env_type = env_type
        self.cur_orthog = cur_orthog
        self.init_envs(cur_orthog)
        self._projected_site = None

    @property
    def L(self):
        return self.ket1.L

    @property
    def env_type(self):
        raise NotImplementedError

    @property
    def ket1(self):
        return self._ket1

    @ket1.setter
    def ket1(self, mps: qtn.MatrixProductState):
        mps.view_as(MPS, inplace=True)
        self._ket1 = mps

    @property
    def ket2(self):
        return self._ket2

    @ket2.setter
    def ket2(self, mps: qtn.MatrixProductState):
        mps.view_as(MPS, inplace=True)
        self._ket2 = mps

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, mps: Optional[Union[MPS, MPO]]):
        if mps is not None:
            if isinstance(mps, (qtn.MatrixProductState, MPS)):
                mps.view_as(MPS, inplace=True)
                self._bra = mps
                # if mps.site_ind_id != self.ket.site_ind_id:
                if self.operator is not None:
                    mps.site_ind_id = self.operator.upper_ind_id
                # self.update_horizontal_bonds()
            elif isinstance(mps, (qtn.MatrixProductOperator, MPO)):
                ## e.g., partial integrator
                mps.view_as(MPO, inplace=True)
                self._bra = mps
                try:
                    if self.operator is not None:
                        mps.lower_ind_id = self.operator.upper_ind_id
                except:
                    mps.upper_ind_id = mps.upper_ind_id + '_'
                    if self.operator is not None:
                        mps.lower_ind_id = self.operator.upper_ind_id

    @property
    def bra_site_ind(self):
        return self.operator.upper_ind_id

    @property
    def ket1_site_ind(self):
        return self.ket1.site_ind_id

    @property
    def ket2_site_ind(self):
        return self.ket2.site_ind_id

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, mpo: Optional[MatrixProductTensor]):
        self._operator = mpo
        mpo.upper_ind_id = (self.bra.site_ind_id if isinstance(self.bra, MPS) else self.bra.lower_ind_id) + '_'
        mpo.lower_ind_id = self.ket1.site_ind_id
        mpo.reindex_extra_inds(0, self.ket2.site_ind_id + '_[2]')
        mpo.mangle_inner_()
        return

    ######
    def init_envs(self, cur_orthog: int):
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

    def bra_site(self, i: int):
        # i = self.mps_inds[i]
        if isinstance(self.bra, MPS):
            tens = self.bra.get_bra_tens(i)
        else:
            tens = self.bra[i].conj()
        return tens.reindex({ind: ind + '_'for ind in tens.inds})

    def ket2_site(self, i: int):
        tens = self.ket2[i]
        out = tens.reindex({ind: ind + '_[2]' for ind in tens.inds})
        if self.conj_ket2:
            out = out.conj(inplace=True)
        return out

    def bra_horizontal_bond(self, i: int, step: Union[int, SweepDirection]):
        ind = self.bra.bond(i, i + step)
        if ind is not None:
            ind += '_'
        return ind

    def ket1_horizontal_bond(self, i: int, step: int):
        return self.ket1.bond(i, i + step)

    def ket2_horizontal_bond(self, i: int, step: int):
        return self.ket2.bond(i, i + step) + '_[2]'

    def get_ket2_inds(self, cur_orthog: int):
        k1_to_k2_inds = {self.ket1_horizontal_bond(cur_orthog, 1): self.ket2_horizontal_bond(cur_orthog, 1),
                         self.ket1_horizontal_bond(cur_orthog, -1): self.ket2_horizontal_bond(cur_orthog, -1),
                         self.ket1.site_ind(cur_orthog): self.ket2.site_ind(cur_orthog)
                         }
        return k1_to_k2_inds

    # def _projected_bra_to_ket_bond(self, left_site_pos: int):
    #     """ projected bra to ket when extracting a "bond"
    #     """
    #     bx_bond = self.bra_horizontal_bond(left_site_pos, 1)
    #     kx_bond = self.ket_horizontal_bond(left_site_pos, 1)
    #
    #     inds_dict = {bx_bond + 'L_': kx_bond + '_L', bx_bond + 'R_': kx_bond + '_R'}
    #     return inds_dict
    #
    # def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
    #     if nsites == 0:
    #         return self._projected_bra_to_ket_bond(left_site_pos)
    #
    #     inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
    #                      self.ket_horizontal_bond(left_site_pos, -1),
    #                  self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
    #                      self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
    #     for ix in range(left_site_pos, left_site_pos + nsites):
    #         if self.operator is not None:
    #             inds_dict[self.operator.upper_ind(ix)] = self.ket.site_ind(ix)
    #         else:
    #             if self.bra is not None:
    #                 inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
    #             # if self.bra is None:
    #             #     inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
    #             # else:
    #             #     inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
    #     return inds_dict

    # @property
    # def projected_site(self):
    #     return self._projected_site

    def copy(self, ket1_copy=None, ket2_copy=None, bra_copy=None):

        if ket1_copy is None:
            ket1_copy = self.ket1.copy()

        if ket2_copy is None:
            ket2_copy = self.ket2.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self.bra is not None else None

        out = self.__class__(ket1_copy, ket2_copy, bra_copy, self.operator, grid=self.grid, conj_ket2=self.conj_ket2)
        out.envs = {k: (env.copy() if env is not None else None) for k, env in self.envs.items()}
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

    # def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
    #                   return_intermediates=False, site_tens: 'qtn.Tensor'=None):
    #     raise NotImplementedError

    # def get_projected_Ax(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
    #                      site_tens: 'qtn.Tensor' = None):
    #     raise NotImplementedError

    # def get_projected_bond(self, left_site_pos:int, return_combined=False, transpose_bonds=None):
    #     """ get environment for bond between left_site_pos, left_site_pos + 1
    #         assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
    #         or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
    #     """
    #     raise NotImplementedError

    def shift_left(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.LEFT)
        # self.cur_orthog += 1

    def shift_right(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.RIGHT)
        # self.cur_orthog -= 1

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        bra_tens = [self.bra_site(i)]
        ket_tens = [self.ket1.get_tens(i), self.ket2_site(i)]
        op_tens = [self.operator[i]]  # [self.mps_inds[i]]

        # print('bra tens', bra_tens)
        # print('ket tens', ket_tens)
        # print('op tens', op_tens)

        new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)
        # print('bra tens', bra_tens[0])
        # print('ket tens', ket_tens[0])
        # print('op tens', op_tens[0])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        # new_env_exp = new_env.exponent
        new_env = new_env.contract_tags(all)
        # print('new env', new_env)
        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env

    def norm(self):
        """ returns <bra|operator|ket>, without
        """
        if self.cur_orthog is None:
            raise NotImplementedError

        ix = self.cur_orthog
        env_L, env_R = self.envs[ix - 1], self.envs[ix + 1]

        # b2k = self.projected_bra_to_ket(ix, 1)
        # k2b = {v: k for k, v in b2k.items()}
        bra_tens = self.bra_site(ix)
        tens_list = [self.ket1[ix], self.ket2_site(ix), self.operator[ix], bra_tens]
        tens_list += [env_L] if env_L is not None else []
        tens_list += [env_R] if env_R is not None else []

        # print('bra tens', bra_tens, bra_tens.norm())
        # print('ket1', self.ket1[ix], self.ket1[ix].norm())
        # print('ket2', self.ket2_site(ix), self.ket2_site(ix).norm())
        # print('self.operator', self.operator[ix], self.operator[ix].norm())
        # print('envL', env_L, env_L.norm() if env_L is not None else None)
        # print('envR', env_R, env_R.norm() if env_R is not None else None)
        # print('norm?')
        ovlp = qtn.tensor_contract(*tens_list)
        # if isinstance(ovlp, float):
        #     ovlp = qtn.Tensor(np.array([ovlp]), inds=('c0',))
        if isinstance(ovlp, qtn.Tensor):
            ovlp.drop_tags()
        # print('ovlp norm', ovlp.norm(), ovlp.data)

        tot_exponent = self.ket1.exponent + self.ket2.exponent + self.bra.exponent + self.operator.exponent
        # print('tot exponent', tot_exponent, self.operator.exponent)
        return ovlp * 10 ** tot_exponent

    def compute_expec_i(self, cur_orthog, ket1_site, ket2_site):

        # k1_to_k2_inds = {self.ket1_horizontal_bond(cur_orthog, 1): self.ket2_horizontal_bond(cur_orthog, 1),
        #                  self.ket1_horizontal_bond(cur_orthog, -1): self.ket2_horizontal_bond(cur_orthog, -1),
        #                  self.ket1.site_ind(cur_orthog): self.ket2.site_ind(cur_orthog)
        #                  }
        # ket2_site = ket2_site.reindex(k1_to_k2_inds)
        ket2_site = ket2_site.reindex({ind: ind + '_[2]' for ind in ket2_site.inds})
        # ket2_site = self.ket2_site(cur_orthog)

        if self.cur_orthog != cur_orthog:
            raise ValueError

        ix = self.cur_orthog
        env_L, env_R = self.envs[ix - 1], self.envs[ix + 1]

        # b2k = self.projected_bra_to_ket(ix, 1)
        # k2b = {v: k for k, v in b2k.items()}
        bra_tens = self.bra_site(ix)
        tens_list = [ ket2_site, ket1_site, self.operator[ix], bra_tens]
        tens_list += [env_L] if env_L is not None else []
        tens_list += [env_R] if env_R is not None else []

        # print('bra tens', bra_tens, bra_tens.norm())
        # print('ket1', ket1_site, ket1_site.norm())
        # print('ket2', ket2_site, ket2_site.norm())
        # print('self.operator', self.operator[ix], self.operator[ix].norm())
        # print('envL', env_L, env_L.norm() if env_L is not None else None)
        # print('envR', env_R, env_R.norm() if env_R is not None else None)
        # print('compute expec i?')

        # tot_exponent = self.ket1.exponent + self.bra.exponent + self.operator.exponent
        # proj_ket = qtn.tensor_contract(*tens_list[1:])
        # proj_ket = proj_ket * 10 ** tot_exponent
        # print('proj ket norm', proj_ket.norm(), tot_exponent)

        # print('tens list', tens_list)

        ovlp = qtn.tensor_contract(*tens_list)
        # print('ovlp', ovlp)
        # if isinstance(ovlp, float):
        #     ovlp = qtn.Tensor(np.array([ovlp]), inds=('c0',))
        if isinstance(ovlp, qtn.Tensor):
            ovlp.drop_tags()

        # print('ovlp', ovlp.data, ovlp.norm())

        tot_exponent = self.ket1.exponent + self.ket2.exponent + self.bra.exponent + self.operator.exponent
        return ovlp * 10 ** tot_exponent
