import numpy as np

from setup.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper
import local_solvers.helper_tn as helper_tn
from local_solvers.environments import EnvironmentSide, CrossEnvironment, DMRGEnvironment

""" 
Functions to perform DMRG/local optimizations for MPS, using quimb as backend
"""

from enum import IntEnum


class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

class SolveMethod(Enum):
    CGD = 'CGD'  # local CGD using quimb tensors
    CGDx = 'CGDx'  # local CGD using numpy
    LSQ = 'LSQ'  # least squares regression


DEFAULT_MAX_ITER = 20  # 10
DEFAULT_MAX_TOT_ITER = 50
DEFAULT_CONV_TOL = 1.0e-6
DEFAULT_SOLVE = SolveMethod.CGD
DEFAULT_MAX_WRONG_ITER = 10


MPO_type = Union['qtn.MatrixProductOperator', MatrixProductOperatorTN, IndexedMPO]
MPS_type = Union['qtn.MatrixProductState', MatrixProductStateTN, IndexedMPS]





#############################
###  minimize Ax=b error  ###
#############################

class LocalSolver:
    """ primary functions:
        - solve:
            - with least squares, conjugate gradient descent (CGD)
        - apply, add + compress
        - eigsolve:  (DMRG)
            minimizes <psi|H|psi> / <psi|psi>, equivalently minimizes <psi|H|psi> - lambda <psi|psi>
            sweep through sites i:
                H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
                N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
                --> minimize <psi|H|psi> - lambda <psi|psi>
                --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
                --> H_eff * T[i] - lambda * N_eff * T[i] = 0
                lambda will be the energy of the state
        operators: list of MPOs corresponding to operators being observed
                    <x| sum(A_i) |x>
        mps_ind_range:  in case only working on a portion of MPS (indices aren't 0 to L)
    """
    env_class: Type['Environment'] = Environment

    def __init__(self, ket_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence[Optional['MPO_type']]] = None,
                 operators_H: Optional[Sequence[Optional['MPO_type']]] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Optional[Sequence[int]] = None,
                 op_env0_Ls: Sequence['qtn.Tensor'] = None,
                 op_env0_Rs: Sequence['qtn.Tensor'] = None,
                 target_env0_Ls: Sequence['qtn.Tensor'] = None,
                 target_env0_Rs: Sequence['qtn.Tensor'] = None,
                 A_envsLs: Optional[Sequence['Environment']] = None,
                 A_envsRs: Optional[Sequence['Environment']] = None,
                 b_envsLs: Optional[Sequence['Environment']] = None,
                 b_envsRs: Optional[Sequence['Environment']] = None,
                 # combine_operators_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 combine_targets_func: Callable[[Sequence[qtn.Tensor]], qtn.Tensor] = None,
                 # left_envs: Optional[Sequence['Environment']] = None,
                 # right_envs: Optional[Sequence['Environment']] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 method = 'DMRG',
                 max_bond = None):

        self.virtual = virtual
        self.out_ind, self.in_ind = out_ind, in_ind
        self.env_class = Environment if method == 'DMRG' else CrossEnvironment

        self._ket = ket_state if virtual else ket_state.copy()

        self._targets = []
        self.targets = [t for t in targets if t is not None] if targets is not None else []
        self.operators = operators  # if virtual else [op.copy() for op in operators]
        self.operators_H = operators_H

        if bra_state is None:
            self._bra = None
            self.set_bra_from_ket()
        else:
            self.bra = bra_state

        # self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Ls[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]
        # self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
        #                              init_env=norm_env0_Rs[i]) if self.operators[i] is not None else None
        #                  for i in range(self.num_operators)]


        if A_envsLs is None:
            self.A_envsLs = [self.env_class(self.L, EnvironmentSide.LEFT, self.ket, bra=self.bra,
                                            operator=self.operators[ib],
                                            init_env=op_env0_Ls[ib] if op_env0_Ls is not None else None)
                             for ib in range(self.num_operators)]
        else:
            self.A_envsLs = list(A_envsLs)

        if A_envsRs is None:
            self.A_envsRs = [self.env_class(self.L, EnvironmentSide.RIGHT, self.ket, bra=self.bra,
                                            operator=self.operators[ib],
                                            init_env=op_env0_Rs[ib] if op_env0_Rs is not None else None)
                             for ib in range(self.num_operators)
                             ]

        if b_envsLs is None:
            self.b_envsLs = [self.env_class(self.L, EnvironmentSide.LEFT, self.targets[ib], bra=self.bra,
                                            init_env=target_env0_Ls[ib] if target_env0_Ls is not None else None, )
                             for ib in range(self.num_targets)]
        else:
            self.b_envsLs = list(b_envsLs)

        if b_envsRs is None:
            self.b_envsRs = [self.env_class(self.L, EnvironmentSide.RIGHT, self.targets[ib], bra=self.bra,
                                            init_env=target_env0_Rs[ib] if target_env0_Rs is not None else None, )
                             for ib in range(self.num_targets)]
        else:
            self.b_envsRs = list(b_envsRs)


        ## left index of MPS to perform dmrg over (eg. if only acting on region of MPS)
        self.mps_inds = list(range(0, self.ket.L)) if mps_inds is None else mps_inds
        self.out_ind, self.in_ind = out_ind, in_ind

        ## convergence params
        self.conv_tol = conv_tol
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_wrong_iter = max_wrong_iter
        self.max_bond = max_bond

        ## callables for combining operators, targets. defaults to summation
        self.targets_func = combine_targets_func
        # self.operators_func = combine_operators_func

        ## results
        self.err = np.inf  # self.check_err()
        self.is_conv = False  # np.abs(self.err) < conv_tol
        self.canon_direction = 0

    @property
    def L(self):
        return len(self.mps_inds)  # self.mps_inds[1] - self.mps_inds[0]

    @property
    def num_targets(self):
        return len(self._targets)

    @property
    def num_operators(self):
        return len(self._operators)

    @property
    def left_envs(self):
        return self.A_envsLs + self.b_envsLs

    @property
    def right_envs(self):
        return self.A_envsLs + self.b_envsLs

    @property
    def ket(self):
        return self._ket

    @ket.setter
    def ket(self, new_ket: qtn.MatrixProductState):
        self._ket = (new_ket if self.virtual else new_ket.copy()) if new_ket is not None else None
        if new_ket is not None:
            for target in self.targets:
                target.site_ind_id = new_ket.site_ind_id
            if self.num_operators > 0:
                for op in self.operators:
                    op.lower_ind_id = new_ket.site_ind_id
                # self.operators[0].lower_ind_id = new_ket.site_ind_id
        self.reinitialize_envs()

    @property
    def bra_site_ind(self):
        """ desired bra site index """
        return self.operators[-1].upper_ind_id

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, new_bra: qtn.MatrixProductState):
        self._bra = (new_bra if self.virtual else new_bra.copy()) if new_bra is not None else None
        if new_bra is not None:
            self.update_bra_inds()
        # if self.num_operators > 0:
        #     self._bra.site_ind_id = self.bra_site_ind  # self.operators[-1].upper_ind_id
        # else:
        #     self._bra.site_ind_id = self.ket.site_ind_id
        self.reinitialize_envs()

    def update_bra_inds(self):
        """ inplace """
        self._bra.mangle_inner_(append='_')
        if self.num_operators > 0:
            self._bra.site_ind_id = self.bra_site_ind  # self.operators[-1].upper_ind_id
        else:
            self._bra.site_ind_id = self.ket.site_ind_id

        ## reindex ancilla bonds -- this is jank/doesn't generally catch all cases but should work
        inds = self._bra[0].bonds(self.ket)
        reindex_map = {ind: ind + '_' for ind in inds if ind != self.ket.site_ind_id.format(0)}
        self._bra[0].reindex(reindex_map, inplace=True)

        inds = self._bra[self._bra.L - 1].bonds(self.ket)
        reindex_map = {ind: ind + '_' for ind in inds if ind != self.ket.site_ind_id.format(self._bra.L - 1)}
        self._bra[self._bra.L - 1].reindex(reindex_map, inplace=True)
        return

    def update_ket_tens(self, new_ket: Sequence['qtn.Tensor'], sites=None, reinit_envs=True,
                        new_exponent=None):
        """ inplace update of bra with specified sites
        """

        sites = range(self.L) if sites is None else sites
        # sites = self.mps_inds if sites is None else sites
        for ix, new_ket_tens in zip(sites, new_ket):
            i = self.mps_inds[ix]
            # ket_to_bra_inds = self.get_ket_to_bra_inds(ix)

            # new_ket_tens.transpose_like(self.ket[i], inplace=True)
            self._ket[i].modify(data=new_ket_tens.data,  # .conj(),
                                inds=tuple(new_ket_tens.inds))
            # inds=tuple([ket_to_bra_inds[ik] for ik in new_ket_tens.inds]))
            if reinit_envs:
                self.reinitialize_envs_i(ix)

        if new_exponent is not None:
            self._ket.exponent = new_exponent
        return

    def update_bra_tens(self, new_bra: Sequence['qtn.Tensor'], sites=None, reinit_envs=True,
                        new_exponent=None):
        """ inplace update of bra with specified sites
        """

        sites = range(self.L) if sites is None else sites
        # sites = self.mps_inds if sites is None else sites
        for ix, new_bra_tens in zip(sites, new_bra):
            i = self.mps_inds[ix]
            # ket_to_bra_inds = self.get_ket_to_bra_inds(ix)

            # print('self.bra[i]', i, self.bra[i])
            # print('new bra tens', new_bra_tens)
            # new_bra_tens.transpose_like(self.bra[i], inplace=True)
            self._bra[i].modify(data=new_bra_tens.data,  # .conj(),
                                inds=tuple(new_bra_tens.inds))
            # inds = tuple([ket_to_bra_inds[ik] for ik in new_bra_tens.inds]))
            if reinit_envs:
                self.reinitialize_envs_i(ix)

        if new_exponent is not None:
            self._bra.exponent = new_exponent
        return

    def set_bra_from_ket(self, sites=None, reinit_envs=True):

        if self._bra is None:
            self._bra = self.ket.conj(mangle_inner=False)  ## mangle inner if there are operators
            self.update_bra_inds()

            if reinit_envs:
                self.reinitialize_envs()

        else:  ## inplace update
            sites = range(self.L) if sites is None else sites
            ket_tens_list = []
            for ix in sites:
                ket_to_bra_inds = self.get_ket_to_bra_inds(ix)
                ket_tens = self.ket[self.mps_inds[ix]].conj()
                ket_tens = ket_tens.reindex(ket_to_bra_inds)
                ket_tens_list += [ket_tens]

            self.update_bra_tens(ket_tens_list, sites=sites, reinit_envs=reinit_envs)

        return

    @property
    def target_site_ind(self):
        """ desired site ind id for targets
        """
        return self.bra.site_ind_id

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets: Union['qtn.MatrixProductState', Sequence[qtn.MatrixProductState]]):
        if new_targets is None:
            self._targets = []
        elif isinstance(new_targets, qtn.TensorNetwork):
            self._targets = [new_targets if self.virtual else new_targets.copy()]
        else:
            self._targets = []
            for new_target in new_targets:
                new_target = new_target if self.virtual or new_target is None else new_target.copy()
                if new_target is not None:
                    new_target.site_ind_id = self.target_site_ind  # self.bra.site_ind_id
                    new_target.mangle_inner_()
                self._targets += [new_target]
        self.reinitialize_envs()  ## technically wouldn't have to update norm envs

        target_ovlps = 0.0
        for target in self._targets:
            if target is not None:
                target_ovlps += helper.ovlp(target, target.conj())
        for ti, tj in np.ndindex(len(self._targets), len(self._targets)):
            if ti > tj:
                if self._targets[ti] is not None and self._targets[tj] is not None:
                    out = helper.ovlp(self._targets[ti], self._targets[tj].conj())
                    target_ovlps += 2 * np.real(out)
        self.targets_norm = np.sqrt(target_ovlps)

    @property
    def operators(self) -> list['MPO_type']:
        return self._operators

    @property
    def operators_H(self) -> list['MPO_type']:
        if self._operators_H is None and len(self.operators) > 0:
            operators_H = [helper.mpo_conj_transpose(mpo) if mpo is not None else None for mpo in self.operators]
            self.operators_H = operators_H
        return self._operators_H

    @operators.setter
    def operators(self, new_operators: Sequence['MPO_type']):
        self._operators = [None] if new_operators is None else \
            [mpo if self.virtual or mpo is None else mpo.copy() for mpo in new_operators]

        upper_ind_id = None  # if self.bra is None else self.bra.site_ind_id
        for op in self._operators:
            if op is None:  continue
            # op.upper_ind_id = self.bra.site_ind_id
            if op.upper_ind_id == self.ket.site_ind_id:
                op.upper_ind_id = op.upper_ind_id + '_out'
            op.lower_ind_id = self.ket.site_ind_id
            if upper_ind_id is not None:
                op.upper_ind_id = upper_ind_id
            # op.mangle_inner_()
            upper_ind_id = op.upper_ind_id

        self.reinitialize_envs()

    @operators_H.setter
    def operators_H(self, ops_H: Sequence[MPO_type]):
        """ ops_H are already .H (conj + transpose)
        """
        if ops_H is not None:
            # assert (len(ops_H) == self.num_operators), 'num ops_H must equal num ops'
            self._operators_H = [mpo.copy() if mpo is not None else mpo for mpo in ops_H]
            for mpo_H in self._operators_H:
                if mpo_H is None:  continue
                ref_op = next(op for op in self.operators if op is not None)
                mpo_H.lower_ind_id = ref_op.upper_ind_id + '_tmp'
                mpo_H.upper_ind_id = ref_op.lower_ind_id + '_H'
                mpo_H.lower_ind_id = ref_op.upper_ind_id
            # self.reinitialize_envs()

    def create_like(self, copy=True, **kwargs):
        raise NotImplementedError

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)
        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        return new_solver

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.left_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            if env is not None:  env.reinitialize()

        for env in self.right_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            if env is not None:  env.reinitialize()

        self.err = np.inf
        self.is_conv = False

    def reinitialize_envs_i(self, i):
        """ if boundary envs are None, uses old values
        """
        for env in self.left_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            if i == 0:  continue
            if env is not None:  env.reinitialize_i(i)

        for env in self.right_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            if i == self.L - 1:  continue
            if env is not None:  env.reinitialize_i(i)

    def get_ket_to_bra_inds(self, i) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        ind1 = self.mps_inds[i]
        inds_dict = {self.ket.site_ind(ind1): self.bra.site_ind(ind1)}

        b1 = self.bra.select_tensors(self.bra.site_tag_id.format(ind1))[0]
        k1 = self.ket.select_tensors(self.ket.site_tag_id.format(ind1))[0]

        ## right bond
        if i < self.L - 1:
            ind2 = self.mps_inds[i + 1]
            b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]

            bbond = next(iter(b1.bonds(b2)))
            kbond = next(iter(k1.bonds(k2)))
            inds_dict[kbond] = bbond

        ## left bond
        if i > 0:
            ind2 = self.mps_inds[i - 1]
            b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]

            bbond = next(iter(b1.bonds(b2)))
            kbond = next(iter(k1.bonds(k2)))
            inds_dict[kbond] = bbond

        ## ancilla bonds (L-1)
        if i == self.L - 1 or i == 0:
            items = [item for k, item in inds_dict.items()]
            anc_b = [ind for ind in self.bra[ind1].inds if ind not in items]
            anc_k = [ind for ind in self.ket[ind1].inds if ind not in inds_dict.keys()]
            for (ab, ak) in zip(anc_b, anc_k):
                inds_dict[ak] = ab

        return inds_dict

    def get_bra_to_ket_inds(self, i) -> dict[str, str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket

    def canonize(self, i, cur_orthog=None):
        ## difficulties with mps inds not continuous
        ## instead use canonize from list

        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        i = self.get_mps_ind(i)
        updated_inds = []
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds += [lx]
                tens_left += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds:
                    updated_inds += [lx]
                tens_right += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]
        helper.canonize_tens_list(*tens_left, inplace=True)
        helper.canonize_tens_list(*(tens_right[::-1]), inplace=True)

        # tens_i = tens_left[-1]
        # tens_i_norm = tens_i.norm()
        # tens_i.modify(apply=lambda x: x / tens_i_norm)
        # self.ket.exponent = self.ket.exponent + np.log10(tens_i_norm)

        # print('updated inds', updated_inds)
        self.set_bra_from_ket(sites=updated_inds)

    def compress(self, i, cur_orthog=None, compress_opts=None):
        ## difficulties with mps inds not continuous
        ## instead use canonize from list

        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                tens_left += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                tens_right += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]
        helper.compress_tens_list(*tens_left, inplace=True, compress_opts=compress_opts)
        helper.compress_tens_list(*(tens_right[::-1]), inplace=True, compress_opts=compress_opts)
        self.set_bra_from_ket()

    def left_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i + 1)
        helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
        self.set_bra_from_ket(sites=[i, i + 1])

    def right_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i - 1)
        helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
        self.set_bra_from_ket(sites=[i - 1, i])

    def add_rand(self, canon_direction, strength=0.01):
        """ canon direction: desired end canon direction
                             sweep direction left/right -> right/left canon
        """
        # rand_mps = qtn.MPS_rand_state(self.L, self.max_bond, self.ket.phys_dim(1))
        # rand_mps[0].modify(apply=lambda x: x * strength)
        # helper.add_MPS(self.ket, rand_mps, compress=False, inplace=True)
        ### above: seems to just get stuck at some other worse local minimum.

        helper.add_rand_noise(self.ket, strength=strength, inplace=True)

        canon_end = 0 if canon_direction == SweepDirection.RIGHT else self.L - 1
        self.canonize(canon_end)
        canon_end = self.L - 1 if canon_direction == SweepDirection.RIGHT else 0
        self.compress(canon_end, compress_opts={'max_bond': self.max_bond})
        # helper.compress(self.ket, scale=False, compress_opts={'max_bond': self.max_bond})
        # helper.add_rand_noise(self.ket, strength=strength, inplace=True)
        # self.set_bra_from_ket()

    def get_mps_ind(self, i):
        return self.mps_inds[i]

    #########################

    def get_b_eff(self, left_site_pos:int, nsites:int, return_combined=True, transpose_bonds=None
                 ) -> Optional[Union[qtn.Tensor, Sequence[qtn.TensorNetwork]]]:
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]

        b_effs = []
        for i in range(self.num_targets):
            b_left = self.b_envsLs[i][left_site_pos]
            b_right = self.b_envsRs[i][left_site_pos + nsites - 1]

            b_eff = qtn.TensorNetwork([])
            b = self.targets[i]
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff.add(b[si])
            else:
                b_eff.add(b[site_inds])

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            b_eff.exponent += b.exponent
            b_effs += [b_eff]

        if return_combined:
            if self.targets_func is None:
                return helper_tn.sum_eff_TNs(b_effs, transpose_bonds=transpose_bonds)
            else:
                out = self.targets_func(b_effs)
                out.transpose(*transpose_bonds, inplace=True)
                return out

        return b_effs

    def get_A_effs(self, left_site_pos:int, nsites:int, return_combined=False, transpose_bonds=None) \
            -> Optional[Union[qtn.Tensor, Sequence[qtn.TensorNetwork]]]:

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]

        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][left_site_pos]
            A_right = self.A_envsRs[i][left_site_pos + nsites - 1]

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    A_eff.add(A[si])
            else:
                A_eff.add(A[site_inds])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            A_eff.exponent += A.exponent
            A_effs += [A_eff]

        if return_combined:
            return helper_tn.sum_eff_TNs(A_effs, transpose_bonds=transpose_bonds)

        return A_effs


    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', new_bra_tens=None,
                      new_bra_exponent=None):
        """ update ket, bra with new_site
            i: int of mps site
        """
        if direction == SweepDirection.RIGHT:
            ## update ket
            ind1 = self.mps_inds[i]
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            ## canonicalize ket to next site
            if i < self.L - 1:
                ind2 = self.mps_inds[i + 1]
                helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
                ## update bra
                if new_bra_tens is not None:
                    helper.canonize_tens_list(*new_bra_tens, inplace=True)
                    self.update_bra_tens(new_bra_tens, sites=[i, i + 1], new_exponent=new_bra_exponent)
                else:
                    self.set_bra_from_ket(sites=[i, i + 1])
            else:
                ## update bra
                if new_bra_tens is not None:
                    self.update_bra_tens(new_bra_tens, sites=[i])
                else:
                    self.set_bra_from_ket(sites=[i])

        else:
            ## update k
            ind1 = self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = self.mps_inds[i - 1]
                helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
                ## update bra
                self.set_bra_from_ket(sites=[i, i - 1])
            else:
                ## update bra
                self.set_bra_from_ket(sites=[i])

        return

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i + 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site1.data, inds=site1.inds)
            self.ket[ind2].modify(data=site2.data, inds=site2.inds)

            ## update bra
            self.set_bra_from_ket(sites=[i, i + 1])

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            ## update bra
            self.set_bra_from_ket(sites=[i, i - 1])

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        end = nsites - 1 if end is None else end

        if canonize:
            self.canonize(end)

        for env in self.right_envs:
            if env is not None:
                for i in range(self.L - 1, end, -1):
                    env.extend_env(i)

        self.canon_direction = EnvironmentSide.RIGHT
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        end = self.L - nsites if end is None else end

        if canonize:
            self.canonize(end)

        for env in self.left_envs:
            if env is not None:
                for i in range(end):
                    env.extend_env(i)

        self.canon_direction = EnvironmentSide.LEFT
        return

    def _update_envs_left(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos + 1)
            # helper.canonize(self.ket, i=pos + 1)
            # self.set_bra_from_ket()
        for env in self.left_envs:
            if env is not None:  env.extend_env(pos)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos - 1)
            # helper.canonize(self.ket, i=pos - 1)
            # self.set_bra_from_ket()
        for env in self.right_envs:
            if env is not None:  env.extend_env(pos)
        return

    def solve(self, nsites, *args, init_direction=SweepDirection.RIGHT, canonize=False, verbose=False, **kwargs):

        if init_direction == SweepDirection.RIGHT:
            if canonize:
                self.canonize(0)  ## does not scale canonical tensors

            ## ovlp <Ax|b>, <Ax|x> init envs
            self._build_all_envs_right(1, canonize=False)
            self.sweep_l2r(nsites, *args, verbose=verbose, **kwargs)
            self.sweep_r2l(nsites, *args, verbose=verbose, **kwargs)

        else:
            if canonize:
                self.canonize(self.L - 1)  ## does not scale canonical tensors

            ## ovlp <Ax|b>, <Ax|x> init envs
            self._build_all_envs_left(1, canonize=False)
            self.sweep_r2l(nsites, *args, verbose=verbose, **kwargs)
            self.sweep_l2r(nsites, *args, verbose=verbose, **kwargs)

        print('done sweep')
        # exit()
        return self.ket

    def sweep_l2r(self, nsites, *args, verbose=False, **kwargs):
        raise NotImplementedError

    def sweep_r2l(self, nsites, *args, verbose=False, **kwargs):
        raise NotImplementedError

    def _site_solve(self, left_site_pos: int, nsites: int, **kwargs) -> tuple[qtn.Tensor, Numeric]:
        raise NotImplementedError


    ###

    def check_err(self, *args, **kwargs) -> 'float':
        """ also updates self.err, self.is_conv
        """
        raise NotImplementedError

    def _get_xAAx_(self) -> float:
        """ compoute <Ax|Ax> """
        old_site_ind_id = self.bra.site_ind_id
        if self.num_operators > 0:
            self.bra.site_ind_id = self.operators_H[-1].upper_ind_id

        if self.num_operators > 0:
            exp_AA_tot = 0.0
            ket_copy = self.ket.copy()
            bra_copy = self.bra.copy()
            for i, j in np.ndindex(self.num_operators, self.num_operators):
                # norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
                #                             [mpo for mpo in self.operators_H], virtual=True)
                if self.operators[i] is not None and self.operators_H[j] is not None:
                    norm_Ax = qtn.TensorNetwork([ket_copy, bra_copy, self.operators[i], self.operators_H[j]],
                                                virtual=False)  ## can mangle indices
                    A_eff_exponent = norm_Ax.exponent
                    # print('norm Ax', norm_Ax)
                    exp_AA = norm_Ax.contract_tags(all)
                    exp_AA *= 10 ** A_eff_exponent
                    exp_AA_tot += exp_AA
            norm_Ax = exp_AA_tot
        else:
            norm_Ax = helper.norm(self.ket) ** 2

        self.bra.site_ind_id = old_site_ind_id
        return norm_Ax

    def _get_xAb_(self) -> float:
        """ compute <x_o A_oi.H|b_i>
            (should be old but it's ok...)
        """
        ovlp_b_tot = 0.
        if self.num_operators > 0:
            bra_copy = self.bra.copy()
            # bra_copy = helper.scalar_multiply(bra_copy, -1)
            for i, ib in np.ndindex(self.num_operators, self.num_targets):
                if self.operators_H[i] is not None:
                    bra_copy.site_ind_id = self.operators_H[i].upper_ind_id
                    ovlp_b = qtn.TensorNetwork([self.targets[ib], bra_copy, self.operators_H[i]], virtual=False)
                    b_eff_exponent = self.operators_H[i].exponent + bra_copy.exponent + self.targets[ib].exponent
                    # ovlp_b = qtn.TensorNetwork([self.targets[ib].conj(), self.ket, self.operators[i]], virtual=False)
                    # b_eff_exponent = self.operators[i].exponent + self.ket.exponent + self.targets[ib].exponent
                    ovlp_b = ovlp_b.contract_tags(all)
                    ovlp_b_tot = ovlp_b_tot + ovlp_b * 10 ** b_eff_exponent
        else:
            for ib in range(self.num_targets):
                ovlp_b = qtn.TensorNetwork([self.targets[ib], self.bra], virtual=False)
                b_eff_exponent = self.bra.exponent + self.targets[ib].exponent
                ovlp_b = ovlp_b.contract_tags(all)
                ovlp_b_tot = ovlp_b_tot + ovlp_b * 10 ** b_eff_exponent

        return ovlp_b_tot
