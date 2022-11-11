from setup.defaults import *
import time
from scipy import linalg
import quimb.tensor as qtn
import helper_quimb as helper

from enum import IntEnum
class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = 1
    RIGHT = -1

class EnvironmentSide(IntEnum):
    LEFT = 1    ## int denotes shift in position when extended environment
    RIGHT = -1

class EnvironmentType(IntEnum):
    EXPEC = 1
    OVLP = 0
    NORM = 2

DEFAULT_MAX_ITER = 100
DEFAULT_CONV_TOL = 1.0e-6

""" basic functions 
"""
def dmrg_add_mps(mps_list: Sequence[qtn.MatrixProductState], init_guess: Optional[qtn.MatrixProductState]=None,
                 weights=None, normalize=True, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER):
    """ obtain approximation of
    """
    if weights is None:   weights = [1]*len(mps_list)

    if init_guess is None:
        init_guess = helper.scalar_multiply(mps_list[0], weights[0], inplace=False)
    else:
        init_guess = init_guess.copy()
    init_guess_conj = init_guess.conj()

    total_norm = np.sqrt( np.sum( [np.abs(w)**2 * mps.norm()**2 for (w, mps) in zip(weights, mps_list)] ))
    if total_norm < np.sqrt(CUTOFF):
        return None

    L = init_guess.L
    helper.canonize(init_guess, i=0)
    old_ind_ids = []
    ovlps = []
    for mps in mps_list:
        old_ind_ids += [mps.site_ind_id]
        mps.site_ind_id = init_guess_conj.site_ind_id
        ovlps += [ qtn.TensorNetwork( [mps, init_guess_conj], virtual=True) ]

    return new_mps

def dmrg_compress(target_mps: qtn.MatrixProductState, init_guess: Optional[qtn.MatrixProductState], normalize=True,
                  **dmrg_opt_kwargs):
    """ compress state
    """
    return dmrg_add([1.0], [target_mps], normalize=normalize, init_guess=init_guess, **dmrg_opt_kwargs)


def conjugate_gradient_descent(A: qtn.MatrixProductOperator, b: qtn.MatrixProductState,
                               init_guess: Optional[qtn.MatrixProductState]=None, conv_tol=DEFAULT_CONV_TOL,
                               dmrg_nsites: int=1):
    """ solve Ax = b via conjugate gradient descent
        with compressions via dmrg. can use any compression scheme though?
    """
    return new_mps

""" 
Functions to perform DMRG/local optimizations for MPS, using quimb as backend
"""
class Environment:

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState' = None,
                 operators_list: Sequence[qtn.MatrixProductOperator] = None,
                 init_env: 'qtn.Tensor' = None):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        """
        self.L = L        ## number of sites over which we'll need to make environments for
        self.side = side
        self.ket = ket    ## do not make copy -- want to update from outside
        self.bra = bra
        self.operators = operators_list if operators_list is not None else []
            ## ordered from ket to bra
        self.exponent = self.ket.exponent + self.bra.exponent \
                        + np.sum([mpo.exponent for mpo in self.operators])

        if self.side == EnvironmentSide.RIGHT:
            self._envs = {L - 1: init_env}
        else:
            self._envs = {0: init_env}

    def __getitem__(self, pos):
        return self._envs.get(pos, None)

    def __setitem__(self, pos, env:'qtn.Tensor'):
        self._envs[pos] = env

    def get(self, pos):
        return self._envs.get(pos, None)

    def reinitialize(self):
        if self.side == EnvironmentSide.RIGHT:
            self._envs = {self.L - 1: self._envs[self.L - 1]}
        else:
            self._envs = {0: self._envs[0]}

    def extend_env(self, pos: int) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        env = self._envs[pos]
        new_env = qtn.TensorNetwork([self.bra[pos], self.ket[pos], ] + [mpo[pos] for mpo in self.operators])
        if env is not None:
            new_env.add(env)
        new_env = new_env.contract_tags('all')
        self._envs[pos + self.side.value] = new_env
        return new_env



#############################
###  minimize Ax=b error  ###
#############################

class LocalSolver:
    """ primary functions:
        - solve:
            - with least squares, conjugate gradient descent (CGD)
        - eigsolve:  (DMRG)
            minimizes <psi|H|psi> / <psi|psi>, equivalently minimizes <psi|H|psi> - lambda <psi|psi>
            sweep through sites i:
                H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
                N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
                --> minimize <psi|H|psi> - lambda <psi|psi>
                --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
                --> H_eff * T[i] - lambda * N_eff * T[i] = 0
                lambda will be the energy of the state
    """
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['qtn.MatrixProductState']] = None,
                 operators: Optional[Sequence['qtn.MatrixProductOperator']] = None,
                 mps_opt_inds: Optional[Sequence[int]] = None,
                 left_envs: Optional[Sequence['Environment']] = None,
                 right_envs: Optional[Sequence['Environment']] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

        self._ket = trial_state.copy()
        self.operators = operators
        self.bra = trial_state.conj(mangle_inner=(self.num_operators > 0))  # mangle if there are operators
        self.targets = targets
        self.left_envs = [] if left_envs is None else left_envs
        self.right_envs = [] if right_envs is None else right_envs

        ## indices of MPS to perform dmrg over (eg. if only acting on region of MPS)
        self.mps_opt_inds = list(range(self._ket.L)) if mps_opt_inds is None else mps_opt_inds

        self.conv_tol = conv_tol
        self.max_iter = max_iter
        self.max_bond = max_bond

    @property
    def L(self):
        return len(self.mps_opt_inds)

    @property
    def num_targets(self):
        return len(self._targets)

    @property
    def num_operators(self):
        return len(self._operators)

    @property
    def ket(self):
        return self._ket

    @ket.setter
    def ket(self, new_ket: qtn.MatrixProductState):
        self._ket = new_ket.copy()
        for target in self.targets:
            target.site_ind_id = new_ket.site_ind_id
        if self.num_operators > 0:
            self.operators[0].lower_ind_id = new_ket.site_ind_id

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, new_bra: qtn.MatrixProductState):
        self._bra = new_bra
        self._bra.site_ind_id = self.ket.site_ind_id + '_H'

    def _set_bra_from_ket(self):
        self._bra = self.ket.conj(mangle_inner=(self.num_operators > 0))    ## mangle inner if there are operators
        if self.num_operators > 0:
            self._bra.site_ind_id = self.operators[-1].upper_ind_id
        else:
            self._bra.site_ind_id = self.ket.site_ind_id

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets: Union['qtn.MatrixProductState', Sequence[qtn.MatrixProductState]]):
        if new_targets is None:
            self._targets = []
        elif isinstance(new_targets, qtn.TensorNetwork):
            self._targets = [new_targets]
        else:
            self._targets = []
            for new_target in new_targets:
                new_target = new_target.copy()
                new_target.site_ind_id = self.ket.site_ind_id if self.num_operators==0 \
                                            else self.operators[-1].lower_ind_id
                self._targets += [new_target]


    @property
    def operators(self) -> list['qtn.MatrixProductOperator']:
        return self._operators

    @operators.setter
    def operators(self, new_operators: Sequence[qtn.MatrixProductOperator]):
        self._operators = [] if new_operators is None else [mpo.copy() for mpo in new_operators]

        if len(self._operators) > 0:
            qtn.tensor_network_align(*self.operators[::-1], inplace=True)
            self._operators[0].lower_ind_id = self.ket.site_ind_id
            self._operators[-1].upper_ind_id = self.bra.site_ind_id

        # self._operators_H = [helper.mpo_transpose(mpo.conj(), inplace=True) for mpo in self._operators[::-1]]
        # for i in range(self.num_operators):
        #     mpo = self._operators_H[i]
        #     if i != 0:
        #         mpo.lower_ind_id = mpo.lower_ind_id + '_H'
        #     mpo.upper_ind_id = mpo.upper_ind_id + '_H'

    @property
    def operator_exponent(self):
        exps = [mpo.exponent for mpo in self.operators]
        return np.sum(exps)


    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.left_envs:
            env.reinitialize()

        for env in self.right_envs:
            env.reinitialize()


    #########################

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        if direction == SweepDirection.RIGHT:
            ## update ket
            self.ket[i].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            helper.canonize_tens_list(self.ket[i], self.ket[i + 1], inplace=True)

            ## update bra
            bra_to_ket_dict = {self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i): self.bra.site_ind(i)}
            bra_inds_0 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i].inds]
            self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)

            bra_to_ket_dict = {self.ket.bond(i + 1, i + 2): self.bra.bond(i + 1, i + 2),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i + 1): self.bra.site_ind(i + 1)}
            bra_inds_1 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i + 1].inds]
            self.bra[i + 1].modify(data=self.ket[i + 1].data.conj(), inds=bra_inds_1)

        else:
            ## update k
            self.ket[i].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            helper.canonize_tens_list(self.ket[i], self.ket[i + 1], inplace=True)

            ## update bra
            bra_to_ket_dict = {self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i): self.bra.site_ind(i)}
            bra_inds_0 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i].inds]
            self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)

            bra_to_ket_dict = {self.ket.bond(i - 1, i - 2): self.bra.bond(i - 1, i - 2),
                               self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.site_ind(i - 1): self.bra.site_ind(i - 1)}
            bra_inds_1 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i - 1].inds]
            self.bra[i - 1].modify(data=self.ket[i - 1].data.conj(), inds=bra_inds_1)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:

            _, left_inds = self.ket[i].filter_bonds(self.ket[i + 1])
            site1, site2 = site_i.split(left_inds, absorb='right', max_bond=self.max_bond)

            ## update ket
            self.ket[i].modify(data=site1.data, inds=site1.inds)
            self.ket[i + 1].modify(data=site2.data, inds=site2.inds)

            ## update bra
            bra_to_ket_dict = {self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i): self.bra.site_ind(i)}
            bra_inds_0 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i].inds]
            self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)

            bra_to_ket_dict = {self.ket.bond(i + 1, i + 2): self.bra.bond(i + 1, i + 2),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i + 1): self.bra.site_ind(i + 1)}
            bra_inds_1 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i + 1].inds]
            self.bra[i + 1].modify(data=self.ket[i + 1].data.conj(), inds=bra_inds_1)

        else:

            _, left_inds = self.ket[i].filter_bonds(self.ket[i - 1])    # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', max_bond=self.max_bond)

            ## update ket
            self.ket[i].modify(data=site2.data, inds=site2.inds)
            self.ket[i - 1].modify(data=site1.data, inds=site1.inds)

            ## update bra
            bra_to_ket_dict = {self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.bond(i, i + 1): self.bra.bond(i, i + 1),
                               self.ket.site_ind(i): self.bra.site_ind(i)}
            bra_inds_0 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i].inds]
            self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)

            bra_to_ket_dict = {self.ket.bond(i - 1, i - 2): self.bra.bond(i - 1, i - 2),
                               self.ket.bond(i, i - 1): self.bra.bond(i, i - 1),
                               self.ket.site_ind(i - 1): self.bra.site_ind(i - 1)}
            bra_inds_1 = [bra_to_ket_dict[k_ind] for k_ind in self.ket[i - 1].inds]
            self.bra[i - 1].modify(data=self.ket[i - 1].data.conj(), inds=bra_inds_1)

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        if canonize:
            helper.canonize(self.ket, i=0)

        for env in self.right_envs:
            for i in range(self.L - 1, nsites - 1, -1):
                env.extend_env(i)
        return

    def _build_all_envs_left(self, nsites: int, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        if canonize:
            helper.canonize(self.ket, i=self.L - 1)
        for env in self.left_envs:
            for i in range(self.L - nsites):
                env.extend_env(i)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        if canonize:
            helper.canonize(self.ket, i=pos + 1)
        for env in self.left_envs:
            env.extend_env(pos)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        if canonize:
            helper.canonize(self.ket, i=pos - 1)
        for env in self.right_envs:
            env.extend_env(pos)
        return

    # def _update_ovlp_envs_right(self, i:int, canonize=False):
    #     """ compute all right envs for <Ax|b> at position i - 1
    #     """
    #     ix = self.mps_opt_inds[i]
    #     if canonize:
    #         helper.canonize(self.ket, i=ix)
    #     for bi in range(self.num_targets):
    #         b_envsR = self.b_envsRs[bi]
    #         b_envR = b_envsR.get(i, None)
    #         if b_envR is None:
    #             tens_R_list = [self.bra[ix], self.targets[bi][ix]] + [mpo[ix] for mpo in self.operators_H]
    #             b_envR = qtn.tensor_contract(tens_R_list)
    #         else:
    #             b_envR = extend_env(ix, b_envR, self.targets[bi], operators=self.operators_H, bra=self.bra)
    #         b_envsR[i - 1] = b_envR
    #     return [b_envsR[i - 1] for b_envsR in self.b_envsRs]
    #
    #
    # def _update_ovlp_envs_left(self, i:int, canonize=False):
    #     """ compute left env for <Ax|b> at position i + 1
    #     """
    #     ix = self.mps_opt_inds[i]
    #     if canonize:
    #         helper.canonize(self.ket, i=self.L - 1)
    #     for bi in range(self.num_targets):
    #         b_envsL = self.b_envsLs[bi]
    #         b_envL = b_envsL.get(i, None)
    #         if b_envL is None:
    #             tens_L_list = [self.bra[ix], self.targets[bi][ix]] + [mpo[ix] for mpo in self.operators_H]
    #             b_envL = qtn.tensor_contract(tens_L_list)
    #         else:
    #             b_envL = extend_env(ix, b_envL, self.targets[bi], operators=self.operators_H, bra=self.bra)
    #         b_envsL[i + 1] = b_envL
    #     return [b_envsL[i + 1] for b_envsL in self.b_envsLs]
    #
    #
    # def _update_norm_envs_right(self, i, canonize=False):
    #     """ compute right env for <Ax|Ax> at position i - 1
    #     """
    #     if canonize:
    #         helper.canonize(self.ket, i=i - 1)
    #
    #     if self.num_operators:
    #         A_envR = self.A_envsR.get(i, None)
    #         ix = self.mps_opt_inds[i]
    #         if A_envR is None:
    #             tens_R_list = [self.bra[ix], self.ket[ix]] + [mpo[ix] for mpo in self.operators_H] + \
    #                           [mpo[ix] for mpo in self.operators]
    #             A_envR = qtn.tensor_contract(tens_R_list)
    #         else:
    #             A_envR = extend_env(ix, A_envR, self.ket, operators=self.operators_H + self.operators, bra=self.bra)
    #         self.A_envsR[i - 1] = A_envR
    #         return A_envR
    #     else:
    #         return None
    #
    #
    # def _update_norm_envs_left(self, i, canonize=False):
    #     """ compute all left env for <Ax|Ax> at position i + 1
    #     """
    #     if canonize:
    #         helper.canonize(self.ket, i=i + 1)
    #
    #     if self.num_operators > 0:
    #         A_envL = self.A_envsL.get(i, None)
    #         ix = self.mps_opt_inds[i]
    #         if A_envL is None:
    #             tens_L_list = [self.bra[ix], self.ket[ix]] + [mpo[ix] for mpo in self.operators_H] + \
    #                           [mpo[ix] for mpo in self.operators]
    #             A_envL = qtn.tensor_contract(tens_L_list)
    #         else:
    #             A_envL = extend_env(ix, A_envL, self.ket, operators=self.operators_H + self.operators, bra=self.bra)
    #         self.A_envsL[i + 1] = A_envL
    #         return A_envL
    #     else:
    #         return None
    #
    #
    # def _update_expec_envs_right(self, i, canonize=False):
    #     """ compute right env for <Ax|Ax> at position i - 1
    #     """
    #     if canonize:
    #         helper.canonize(self.ket, i=i - 1)
    #
    #     if self.num_operators:
    #         A_envR = self.A_envsR.get(i, None)
    #         ix = self.mps_opt_inds[i]
    #         if A_envR is None:
    #             tens_R_list = [self.bra[ix], self.ket[ix]] + [mpo[ix] for mpo in self.operators]
    #             A_envR = qtn.tensor_contract(tens_R_list)
    #         else:
    #             A_envR = extend_env(ix, A_envR, self.ket, operators=self.operators_H + self.operators, bra=self.bra)
    #         self.A_envsR[i - 1] = A_envR
    #         return A_envR
    #     else:
    #         return None
    #
    #
    # def _update_expec_envs_left(self, i, canonize=False):
    #     """ compute all left env for <Ax|Ax> at position i + 1
    #     """
    #     if canonize:
    #         helper.canonize(self.ket, i=i + 1)
    #
    #     if self.num_operators > 0:
    #         A_envL = self.A_envsL.get(i, None)
    #         ix = self.mps_opt_inds[i]
    #         if A_envL is None:
    #             tens_L_list = [self.bra[ix], self.ket[ix]] + [mpo[ix] for mpo in self.operators]
    #             A_envL = qtn.tensor_contract(tens_L_list)
    #         else:
    #             A_envL = extend_env(ix, A_envL, self.ket, operators=self.operators_H + self.operators, bra=self.bra)
    #         self.A_envsL[i + 1] = A_envL
    #         return A_envL
    #     else:
    #         return None





class DMRGSolver(LocalSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['qtn.MatrixProductState']] = None,
                 operators: Optional[Sequence['qtn.MatrixProductOperator']] = None,
                 mps_opt_inds: Optional[Sequence[int]] = None,
                 expec_env0_L: 'qtn.Tensor' = None,
                 expec_env0_R: 'qtn.Tensor' = None,
                 norm_env0_L: 'qtn.Tensor' = None,
                 norm_env0_R: 'qtn.Tensor' = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

        super().__init__(trial_state, operators = operators, mps_opt_inds=mps_opt_inds,
                         expec_env0_L=expec_env0_L, expec_env0_R=expec_env0_R,
                         conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)


    def eigsolve(self, nsites: int, init_direction=SweepDirection.RIGHT, target_eig=0):
        return

    def eigsolve_1site(self, **eigsolve_kwargs):
        return self.eigsolve(1, **eigsolve_kwargs)

    def eigsolve_2site(self, **eigsolve_kwargs):
        return self.eigsolve(2, **eigsolve_kwargs)

    #################################################
    ### DMRG, solve for eigenvalues, eigenvectors ###
    #################################################

    def dmrg_optimize_site(self, input_inds: Sequence[str], output_inds: Sequence[str], left: 'qtn.Tensor',
                           right: 'qtn.Tensor',
                           H: Union[qtn.Tensor, qtn.TensorNetwork], eig_target: int = 0, is_H: bool = True) -> \
            tuple[float, qtn.Tensor]:
        """ site optimization for eigensolver. ie: minimizes <psi|H|psi> / <psi|psi>
            H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
            N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
            --> minimize <psi|H|psi> - lambda <psi|psi>
            --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
            --> H_eff * T[i] - lambda * N_eff * T[i] = 0
            lambda will be the energy of the state

            sites: qtn.Tensor, to be optimized (of the desired shape)
            left: qtn.Tensor, left environment <psi|H|psi>
            right: qtn.Tensor, right environment <psi|H|psi>
            H: Sequence[qtn.Tensor] or qtn.TensorNetwork, corresponding MPO site of operator H
        """
        ### L * H * R = d/dT*[i] d/dT[i] <psi|H|psi>
        H_eff = qtn.TensorNetwork([left, right])
        H_eff.add(H)
        H_eff = H_eff.contract_tags('all')
        H_eff.transpose(*output_inds, *input_inds, inplace=True)

        ### H_eff[i] * T[i] = lambda T[i]  (assumes in canonical form)
        shape_out = H_eff.shape[:len(output_inds)]
        shape_in = H_eff.shape[len(output_inds):]  ## shape_in = shape_out
        H_eff_data = H_eff.data.reshape(np.prod(shape_out), -1)
        if is_H:
            eigvals, eigvecs = linalg.eigh(H_eff_data)
        else:
            eigvals, eigvecs = linalg.eig(H_eff_data)

        eigval = eigvals[eig_target]
        eigvec = eigvals[:, eig_target]

        eigvec = eigvec.reshape(shape_in)
        eigvec_tens = qtn.Tensor(data=eigvec, inds=tuple(input_inds))
        return eigval, eigvec_tens

    def dmrg_1site():
        """ dmrg sweep with 1-site optimization
        """
        raise NotImplementedError

    def dmrg_2site():
        """ dmrg sweep with 2-site optimization
        """
        raise NotImplementedError


class LinearSolver(LocalSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['qtn.MatrixProductState']] = None,
                 operators: Optional[Sequence['qtn.MatrixProductOperator']] = None,
                 mps_opt_inds: Optional[Sequence[int]] = None,
                 norm_env0_L: 'qtn.Tensor' = None,
                 norm_env0_R: 'qtn.Tensor' = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

        operators = [] if operators is None else list(operators)
        operators_H = [helper.mpo_transpose(mpo.conj(), inplace=True) for mpo in self._operators[::-1]]
        for i in range(self.num_operators):
            mpo = operators_H[i]
            if i != 0:
                mpo.lower_ind_id = mpo.lower_ind_id + '_H'
            mpo.upper_ind_id = mpo.upper_ind_id + '_H'

        super().__init__(trial_state, operators = operators + operators_H, targets = targets,
                         mps_opt_inds=mps_opt_inds, conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)

        self.A_envsL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_L)
        self.A_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_R)

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, self.operators_H,
                                 init_env = ovlp_env0_Ls[ib])
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, self.operators_H,
                                 init_env=ovlp_env0_Rs[ib])
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        self.left_envs  = [self.A_envsL,] + self.b_envsLs
        self.right_envs = [self.A_envsR,] + self.b_envsRs


    @property
    def operators_H(self):
        return self._operators[self.num_operators//2:]

    @property
    def operators_X(self):
        return self._operators[:self.num_operators//2]


    def _site_solve(self, sites: Union[int,slice],) -> qtn.Tensor:
        """
        bra_to_ket_inds:  dictionary mapping bra inds (T*[i]) to ket inds (T[i])
            b: site i of target MPS
            b_left, b_right: left and right of overlap <Ax|b>

            sites: int or slice(start, stop, step)
        """
        site_inds = self.mps_opt_inds[sites]

        ### AL * A * AR = d/dT*[i] d/dT[i] <Ax|Ax>
        A_left = self.A_envsL[site_inds[0]]
        A_right = self.A_envsR[site_inds[-1]]
        if self.num_operators > 0:
            A_eff = qtn.TensorNetwork([])
            for A in self.operators:
                A_eff.add(A[sites])
            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)
            A_eff = A_eff.contract_tags('all')
        else:
            A_eff = None
        A_eff_exponent = 0 if A_eff is None else self.A_envsL.exponent

        ### BL * B * BR = d/dT*[i] <Ax|b>
        b_eff: Optional['qtn.Tensor'] = None
        for ti in range(self.num_targets):
            b_left = self.b_envsLs[ti][site_inds[0]]
            b_right = self.b_envsRs[ti][site_inds[-1]]
            b_eff_t = qtn.TensorNetwork([b_left, b_right, self.targets[ti][sites]])
            # b_eff_t.add(self.targets[ti][sites])
            for A in self.operators_H:
                b_eff_t.add(A[sites])
            b_eff_t = b_eff_t.contract_tags('all')
            b_eff_exponent = self.b_envsLs[ti].exponent
            b_eff_t.modify(apply=lambda x: 10**b_eff_exponent * x)

            if b_eff is None:
                b_eff = b_eff_t
            else:
                b_eff_t.transpose_like(b_eff, inplace=True)
                b_eff.modify(apply=lambda x: x + b_eff_t.data)
        bra_inds = b_eff.inds if b_eff is not None else None


        ## change indices from bra (missing T*[i]) to ket
        bra_to_ket_inds = {}
        if A_left is not None:
            left_bra_bond = A_left.bonds(self.bra[site_inds[0]])
            left_ket_bond = A_left.bonds(self.ket[site_inds[0]])
            bra_to_ket_inds[left_bra_bond] = left_ket_bond

        if A_right is not None:
            right_bra_bond = A_right.bonds(self.bra[site_inds[-1]])
            right_ket_bond = A_right.bonds(self.ket[site_inds[-1]])
            bra_to_ket_inds[right_bra_bond] = right_ket_bond

        for site in site_inds:
            bra_to_ket_inds[self.bra.site_ind(site)] = self.ket.site_ind(site)

        ## solve for T[i]
        if A_eff is None:
            x_eff = b_eff.reindex(bra_to_ket_inds)
            x_eff.modify(apply= lambda x : x * 10**(-self.ket.exponent), inplace=True)
            ## b_eff_exponent already in b_eff tensor
        else:
            ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]
            A_eff.transpose(*bra_inds, *ket_inds, inplace=True)
            out_shape = A_eff.shape[:len(bra_inds)]
            in_shape = A_eff.shape[len(bra_inds):]
            A_eff_data = A_eff.data.reshape(np.prod(out_shape),-1)
            if b_eff is not None:
                b_eff_data = b_eff.data.reshape(-1)
            else:
                b_eff_data = np.zeros((A_eff_data.shape[0],))
            x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
            x_eff_data = x_eff_data.reshape(in_shape)
            x_eff_data *= 10 ** (-A_eff_exponent - self.ket.exponent)   ## b_eff_exponent already in b_eff
            x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

        return x_eff


    def site_solve_CGD(self, sites: Union[int,slice], init_guess=None) -> qtn.Tensor:
        raise NotImplementedError


    def solve(self, nsites: int, init_direction=SweepDirection.RIGHT):
        """ iterative solver for entire MPS
            minimize || Ax-b ||_2 = <Ax|Ax> + <b|b> - <Ax|b> - <b|Ax>
            sweep through sites i:
                d/dT*[i] () = d/dT*[i] <Ax|Ax> - <Ax|b> = 0
                    A_eff = d/dT*[i] d/dT[i] <Ax|Ax>
                    b_eff = d/dT*[i] <Ax|b>
                --> A_eff T[i] = b_eff

            A can be the identity (A is None) --> optimizing || x - b ||_2
            with proper canonicalization of x, A_left, A_right should be identity

            A: self.operators:
                can be composed of a series of MPOs (A1, ..., Am) order from ket to bra
            b: self.targets:
                can be written as a sum of MPSs (scaling included in the MPS; not kept track of separately)
            x: self.ket (updated in place)
        """
        L = self.ket.L
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L-1
        helper.canonize(self.ket, i=canon_site)

        ## ovlp <Ax|b>, <Ax|Ax> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(nsites, canonize=False)
        else:
            self._build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        err = 100
        direction = init_direction
        while err > self.conv_tol and it < self.max_iter:
            prev_err = err
            it += 1

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites):
                    ix = self.mps_opt_inds[i]

                    inds = slice(i,i+nsites)
                    site_i = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site(ix, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(ix, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)

                err = self.check_err(L-nsites,nsites)

            else:

                ## left sweep
                for i in range(L - 1, nsites - 1, -1):
                    ix = self.mps_opt_inds[i]

                    inds = slice(i - nsites + 1, i + 1)
                    site_i = self._site_solve(inds)

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(ix, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(ix, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)

                err = self.check_err(0, nsites)

            print('err', it, err)

            if err > prev_err:
                print('Warning: error went up')

            direction *= -1     ## swaps sweep direction

        return self.ket


    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)


    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)


    def check_err(self, i, nsites) -> float:
        """ calculates <Ax|Ax> - <Ax|b>
        """
        ix = self.mps_opt_inds[i]

        ## <Ax|Ax>
        norm_Ax = qtn.TensorNetwork(self.ket[ix:ix+nsites] + self.bra[ix:ix+nsites] +
                                    [mpo[ix:ix+nsites] for mpo in self.operators] +
                                    [mpo[ix:ix+nsites] for mpo in self.operators_H])
        A_envL = self.A_envsL[i]
        if A_envL is not None:
            norm_Ax.add(A_envL)

        A_envR = self.A_envsL[i + nsites - 1]
        if A_envR is not None:
            norm_Ax.add(A_envR)

        norm_Ax = norm_Ax.contract_tags('all')
        A_eff_exponent = (self.operator_exponent + self.ket.exponent) * 2
        norm_Ax *= 10**A_eff_exponent

        ## <Ax|b>
        ovlp_b_tot = 0.
        for ib in range(self.num_targets):
            ovlp_b = qtn.TensorNetwork(self.targets[ib][ix:ix+nsites] + self.bra[ix:ix+nsites]
                                       + [mpo[ix:ix+nsites] for mpo in self.operators_H])
            b_envL = self.b_envsLs[ib].get(i, None)
            if b_envL is not None:
                ovlp_b.add(b_envL)

            b_envR = self.b_envsRs[ib].get(i + nsites - 1, None)
            if b_envR is not None:
                ovlp_b.add(b_envR)

            ovlp_b = ovlp_b.contract_tags('all')
            b_eff_exponent = self.operator_exponent + self.bra.exponent + self.targets[ib].exponent
            ovlp_b_tot += ovlp_b * 10**b_eff_exponent

        return norm_Ax - ovlp_b_tot



#############################
###  minimize Ax=b error  ###
###   multigrid method    ###
#############################


# def extend_env(pos: int, env: qtn.Tensor, ket: qtn.MatrixProductState,
#                operators: Sequence[qtn.MatrixProductOperator], bra: Optional[qtn.MatrixProductState] = None,
#                direction: Optional['SweepDirection'] = None) \
#         -> qtn.Tensor:
#     """ build left environment or right environment
#         assumes bra, ket are properly canonicalized, bra already cc'ed if specified
#         bra, H, ket are tensors corresponding to the ith site of the MPS
#
#         ket: MPS for ket test wavefunction
#         bra: MPS for bra test wavefunction (if not specified, is cc of ket)
#         operators: list of MPOs corresponding to operators being observed
#             <x|An ... A2 A1 |x> (ordered from ket to bra)
#         env: environment tensor to extend.
#         direction: sweep direction (builds opposite env). needs to be defined if bra is None
#     """
#     if bra is None:
#         bra = ket[pos].conj(inplace=False)
#         assert(direction is not None), 'need to specify sweep direction if bra is None'
#
#         mpo_ind = ket.site_ind(pos)
#         if direction == SweepDirection.RIGHT:    ## grow left env
#             env_ind = ket.bond(pos - 1, pos)
#             anc_ind = ket.bond(pos, pos + 1)
#         else:   ## grow right env
#             anc_ind = ket.bond(pos - 1, pos)
#             env_ind = ket.bond(pos, pos + 1)
#
#         new_env = qtn.TensorNetwork([env, ket[pos], ] + [mpo[pos] for mpo in operators])
#         new_env_inds = new_env.outer_inds()
#         unconctracted_inds = [ind for ind in env.inds if ind in new_env_inds]
#         ## this should correspond to bond to contract with bra[site]
#         new_env_ind = unconctracted_inds[0]
#
#         new_mpo_ind = mpo_ind if len(operators) == 0 else operators[-1].upper_ind_id
#         new_anc_ind = anc_ind + '_cc'
#
#         bra.reindex({mpo_ind: new_mpo_ind, env_ind: new_env_ind, anc_ind: new_anc_ind}, inplace=True)
#         new_env.add(bra)
#
#     else:
#         new_env = qtn.TensorNetwork([env, bra[pos], ket[pos], ] + [mpo[pos] for mpo in operators])
#
#     new_env = new_env.contract_tags('all')
#     return new_env
