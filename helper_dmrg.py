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

DEFAULT_MAX_ITER = 10 # 100
DEFAULT_CONV_TOL = 1.0e-6


################################
### numpy optimzation method ###
################################

def multigrid_optimization(A, b, x=None, max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL):
    raise NotImplementedError


def conjugate_gradient_descent(A, b, x=None, max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL):
    """ solve Ax = b via conjugate gradient descent
        using normal matrices and vectors
    """
    norm_b = np.linalg.norm(b)
    if x is None:
        x, r = 0, b
    else:
        r = b - np.dot(A, x)
    d = r

    it = 0
    err = np.linalg.norm(r)
    while it < max_iter and err > conv_tol * norm_b:
        alpha = np.linalg.norm(r)**2 / np.vdot(d.conj(), np.dot(A, d))

        x_ = x + alpha * d
        r_ = r - alpha * np.dot(A, d)
        beta_ = np.linalg.norm(r_)**2 / np.linalg.norm(r)**2
        d_ = r_ + beta_ * d

        x, r, d, = x_, r_, d_
        err = np.linalg.norm(r)
        print('it', it, err)
        it += 1

    if it < max_iter:
        print(f'cgd: error {err} in {it} iterations')
    else:
        print(f'cgd did not converge: error {err} in {it} iterations')

    return x



########################
### helper functions ###
########################

def qtn_conjugate_gradient_descent_1site(A: qtn.TensorNetwork, b: qtn.Tensor, x: qtn.Tensor,
                                         output_to_input_inds: dict[str, str],
                                         max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL) \
        -> tuple[qtn.Tensor, float]:
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
        A: matrix, b: target, x: solution
        r: residual = b - Ax
        d: search direction
    """
    norm_b = b.norm()
    input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.reindex(input_to_output_inds, inplace=False)
    if x is not None:
        Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        Ax_tens = Ax.contract_tags(all)
        Ax_tens.transpose_like(r, inplace=True)
        r.modify(apply = lambda data: data - Ax_tens.data)

    d = r.copy()       ## d: search r
    norm_r = r.norm()

    it = 0
    err = r.norm()
    while it < max_iter and err > conv_tol * norm_b:

        ## alpha = np.linalg.norm(r) ** 2 / np.vdot(d.conj(), np.dot(A, d))
        dT = d.conj()
        d.reindex(output_to_input_inds, inplace=True)
        proj = qtn.TensorNetwork([dT, A, d], virtual=True, check_collisions=False)
        alpha = err ** 2 / proj.contract_tags(all)

        ## update x:  x_ = x + alpha * d
        if x is None:
            x = d.copy()
            x.modify(apply = lambda data: data * alpha)
            # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        else:
            x.modify(apply = lambda data: data + alpha * d.data)

        ## update residual r:  r_ = r - alpha * np.dot(A, d)
        r_proj = qtn.tensor_contract(*A.tensors, d)
        r_proj.transpose_like(r, inplace=True)
        r.modify(apply = lambda data: data - alpha * r_proj.data)
        new_norm_r = r.norm()

        ## update search direction d:
        ## beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        ## d_ = r_ + beta_ * d
        beta_ = new_norm_r ** 2 / norm_r ** 2
        d.reindex(input_to_output_inds, inplace=True)
        d.transpose_like(r, inplace=True)
        d.modify(apply = lambda data: data * beta_ + r.data )

        err = new_norm_r
        norm_r = new_norm_r
        it += 1

        print('it', it, err)

    if it < max_iter:
        print(f'cgd: error {err} in {it} iterations')
    else:
        print(f'cgd did not converge: error {err} in {it} iterations')

    return x, err


def mps_coarsen(mps: qtn.MatrixProductState, integ_inds: Sequence[int], coarsen_method='select',
                inplace=False):
    """ integ_inds: inds in mps corresponding to the finest grid mesh that we want to select/integrate over
    """

    mps = mps if inplace else mps.copy()

    if coarsen_method == 'select':
        for i in integ_inds:
            mps[i].isel({mps.site_ind_id(i): 0}, inplace=True)
    elif coarsen_method == 'average':
        integ_mps = qtn.TensorNetwork([])
        for i in integ_inds:
            integ_mps.add( qtn.Tensor(np.array([0.5, 0.5]), inds=(mps.site_ind(i),), tags=('IX',)) )
        mps_site_tags = [mps.site_tag(i) for i in integ_inds]
        mps.add(integ_mps)
        mps.contract_tags(('IX',) + tuple(mps_site_tags), which='any', inplace=True)
    elif coarsen_method == '':
        ### 1 2 1 weighting
        integ_mps = qtn.TensorNetwork([])
        raise NotImplementedError
    elif coarsen_method == 'Galerkin':
        raise NotImplementedError
    else:
        raise ValueError

    return mps


#########################
### optimization fcts ###
#########################

def dmrg_linear_combine(mps_list: Sequence[qtn.MatrixProductState],
                        operator_list: Sequence[Optional['qtn.MatrixProductOperator']],
                        weights: Sequence['float'] = None,
                        init_guess: Optional[qtn.MatrixProductState] = None,
                        normalize=True, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

    ### would need to update LinearSolver to add operators applied to targets.
    ### need to double check the difference in contraction computational cost?

    ### horizontal contraction: O( (Dw^2 d) d (D^2 d) + (D (dD) (Dw D) )
    ### vertical contraction: O( (Dd) D (Dw D) * 2 + Dw (d^2 Dw) D^2 )
    ###                    or O( (Dd) D (Dw D) + (d Dw) (d Dw) (D^2) + D (d D) Dw D )
    ###
    ### costs are comparable ~ O (Dw^2 D^2 + D^3 Dw)
    raise NotImplementedError


def dmrg_add_mps(mps_list: Sequence[qtn.MatrixProductState], init_guess: Optional[qtn.MatrixProductState]=None,
                 weights=None, normalize=True, opt_nsites=1, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_bond=None):
    """ obtain approximation of x = sum(b)
    """
    if weights is not None:
        mps_list = [helper.scalar_multiply(mps, weight) for mps, weight in zip(mps_list, weights)]

    init_guess = mps_list[0] if init_guess is None else init_guess

    # approx_total_norm = np.sqrt( np.sum( [np.abs(w)**2 * mps.norm()**2 for (w, mps) in zip(weights, mps_list)] ))
    # if approx_total_norm < np.sqrt(CUTOFF):
    #     return None

    solver = LinearSolver(init_guess, targets=mps_list, conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)
    new_mps = solver.solve(opt_nsites)
    return new_mps


def dmrg_compress(target_mps: qtn.MatrixProductState, init_guess: Optional[qtn.MatrixProductState] = None, normalize=True,
                  **dmrg_opt_kwargs):
    """ compress state
    """
    return dmrg_add_mps([target_mps], init_guess=init_guess, normalize=normalize, **dmrg_opt_kwargs)



def mps_conjugate_gradient_descent(A: 'qtn.MatrixProductOperator', b: Optional['qtn.MatrixProductState'] = None,
                                   x: Optional['qtn.MatrixProductState'] = None,
                                   conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER):
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
            A: matrix, b: target, x: solution
            r: residual = b - Ax
            d: search direction
    """
    norm_b = helper.norm(b)
    # input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.copy()
    r.site_ind_id = A.upper_ind_id
    # r = b.reindex(input_to_output_inds, inplace=False)
    if x is not None:
        Ax = helper.apply(A, x)     ## dmrg apply? dmrg compress?
        r = dmrg_add_mps([r, Ax], weights=[1, -1], conv_tol=conv_tol, max_iter=max_iter)

    d = r.copy()  ## d: search r
    norm_r = helper.norm(r)

    it = 0
    err = r.norm()
    while it < max_iter and err > conv_tol * norm_b:

        ## alpha = np.linalg.norm(r) ** 2 / np.vdot(d.conj(), np.dot(A, d))
        d.site_ind_id = A.lower_ind_id
        proj = helper.expectation_value(d, A)
        alpha = err ** 2 / proj

        ## update x:  x_ = x + alpha * d
        if x is None:
            x = d.copy()
            helper.scalar_multiply(x, alpha, inplace=True)
        else:
            x = dmrg_add_mps([x, d], weights = [1, alpha])
            # x.modify(apply=lambda data: data + alpha * d.data)

        ## update residual r:  r_ = r - alpha * np.dot(A, d)
        r_proj = helper.apply(A, d)
        r = dmrg_add_mps([r, r_proj], weights=[1, -alpha])
        new_norm_r = helper.norm(r)

        ## update search direction d:
        ## beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        ## d_ = r_ + beta_ * d
        beta_ = new_norm_r ** 2 / norm_r ** 2
        d.site_ind_id = A.upper_ind_id
        d = dmrg_add_mps([d, r], weights = [beta_, 1])

        err = new_norm_r
        norm_r = new_norm_r
        it += 1

        print('it', it, err)

    if it < max_iter:
        print(f'cgd: error {err} in {it} iterations')
    else:
        print(f'cgd did not converge: error {err} in {it} iterations')

    return x, err



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

            ## get bonds with neighbors/envs on the right
            self.bra_horizontal_bonds = {i: self.bra.bond(i,i + 1) for i in range(0, L - 1)}
            self.ket_horizontal_bonds = {i: self.ket.bond(i,i + 1) for i in range(0, L - 1)}

            self.bra_horizontal_bonds[L - 1] = None if init_env is None else self.bra[L-1].bonds(init_env)
            self.bra_horizontal_bonds[L - 1] = None if init_env is None else self.bra[L - 1].bonds(init_env)

        else:
            self._envs = {0: init_env}

            ## get bonds with neighbors/envs on the left
            self.bra_horizontal_bonds = {i: self.bra.bond(i, i - 1) for i in range(1, L)}
            self.ket_horizontal_bonds = {i: self.ket.bond(i, i - 1) for i in range(1, L)}

            self.bra_horizontal_bonds[0] = None if init_env is None else self.bra[0].bonds(init_env)
            self.bra_horizontal_bonds[0] = None if init_env is None else self.ket[0].bonds(init_env)


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
        new_env = new_env.contract_tags(all)
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
                 # bra_state: Optional['qtn.MatrixProductState'] = None,
                 mps_ind_range: Optional[tuple[int]] = None,
                 left_envs: Optional[Sequence['Environment']] = None,
                 right_envs: Optional[Sequence['Environment']] = None,
                 virtual: bool = False,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

        self.virtual = virtual
        self.left_envs = [] if left_envs is None else left_envs
        self.right_envs = [] if right_envs is None else right_envs

        self._bra = None
        self._ket = trial_state if virtual else trial_state.copy()
        self.operators = operators
        self.set_bra_from_ket()
        self.targets = targets

        ## left index of MPS to perform dmrg over (eg. if only acting on region of MPS)
        self.mps_ind_range = (0, self.ket.L) if mps_ind_range is None else mps_ind_range

        self.conv_tol = conv_tol
        self.max_iter = max_iter
        self.max_bond = max_bond

    @property
    def L(self):
        return self.mps_ind_range[1] - self.mps_ind_range[0]

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
        self._ket = new_ket if self.virtual else new_ket.copy()
        for target in self.targets:
            target.site_ind_id = new_ket.site_ind_id
        if self.num_operators > 0:
            self.operators[0].lower_ind_id = new_ket.site_ind_id
        self.reinitialize_envs()

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, new_bra: qtn.MatrixProductState):
        self._bra = new_bra if self.virtual else new_bra.copy()
        if self.num_operators > 0:
            self._bra.site_ind_id = self.operators[-1].upper_ind_id
        else:
            self._bra.site_ind_id = self.ket.site_ind_id
        self.reinitialize_envs()

    def set_bra_from_ket(self, sites=None):
        if self._bra is None:
            self._bra = self.ket.conj(mangle_inner=True)    ## mangle inner if there are operators
            if self.num_operators > 0:
                self._bra.site_ind_id = self.operators[-1].upper_ind_id
            else:
                self._bra.site_ind_id = self.ket.site_ind_id
        else:   ## inplace update
            sites = range(self.L) if sites is None else sites
            for i in sites:
                ket_to_bra_inds = self.get_ket_to_bra_inds(i)
                # bra_temp = self.ket[i].reindex(ket_to_bra_inds)
                # bra_temp.transpose_like(self._bra[i], inplace=True)
                # self._bra[i].modify(data=bra_temp.data.conj())
                self._bra[i].modify(data=self.ket[i].data.conj(),
                                    inds=tuple([ket_to_bra_inds[ik] for ik in self.ket[i].inds]))
            self._bra.exponent = self.ket.exponent

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
                new_target = new_target if self.virtual else new_target.copy()
                new_target.site_ind_id = self.bra.site_ind_id
                self._targets += [new_target]
        self.reinitialize_envs()    ## technically wouldn't have to update norm envs

        # self.targets_norm = np.linalg.norm( [target.norm() * 10**target.exponent for target in new_targets] )
        target_ovlps = 0.0
        for target in new_targets:
            target_ovlps += helper.ovlp(target, target.conj())
        for ti, tj in np.ndindex(len(new_targets), len(new_targets)):
            if ti > tj:
                out = helper.ovlp(new_targets[ti], new_targets[tj].conj())
                target_ovlps += 2*np.real(out)
        self.targets_norm = np.sqrt(target_ovlps)

        print('targets norm', self.targets_norm)

    @property
    def operators(self) -> list['qtn.MatrixProductOperator']:
        return self._operators

    @property
    def operators_H(self) -> list['qtn.MatrixProductOperator']:
        return self._operators_H

    @operators.setter
    def operators(self, new_operators: Sequence[qtn.MatrixProductOperator]):
        self._operators = [] if new_operators is None else \
                            [mpo if self.virtual else mpo.copy() for mpo in new_operators]

        if len(self._operators) > 0:
            qtn.tensor_network_align(*self.operators[::-1], inplace=True)
            self._operators[0].lower_ind_id = self.ket.site_ind_id
            # self._operators[-1].upper_ind_id = self.bra.site_ind_id

        ## Hermitian conjugate of operators
        operators_H = [helper.mpo_transpose(mpo.conj(), inplace=True) for mpo in new_operators]
        operators_H = operators_H[::-1]
        for i in range(len(operators_H)):
            mpo = operators_H[i]
            if i != 0:
                mpo.lower_ind_id = mpo.lower_ind_id + '_H'
            mpo.upper_ind_id = mpo.upper_ind_id + '_H'
        self._operators_H = operators_H

        self.reinitialize_envs()

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


    def get_ket_to_bra_inds(self, i) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        inds_dict = {self.ket.site_ind(i): self.bra.site_ind(i)}

        ## right bond
        if i < self.L-1:
            inds_dict[self.ket.bond(i,i+1)] = self.bra.bond(i,i+1)

        ## left bond
        if i > 0:
            inds_dict[self.ket.bond(i,i-1)] = self.bra.bond(i,i-1)

        ## ancilla bonds (L-1)
        if i == self.L-1 or i == 0:
            items = [item for k, item in inds_dict.items()]
            anc_b = [ind for ind in self.bra[i].inds if ind not in items]
            anc_k = [ind for ind in self.ket[i].inds if ind not in inds_dict.keys()]
            for (ab, ak) in zip(anc_b, anc_k):
                inds_dict[ak] = ab

        return inds_dict


    def get_bra_to_ket_inds(self, i) -> dict[str,str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket


    def canonize(self, i):
        helper.canonize(self.ket, i=i)
        self.set_bra_from_ket()


    def get_mps_ind(self, i):
        return self.mps_ind_range[0] + i


    #########################

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        if direction == SweepDirection.RIGHT:
            ## update ket
            site_i.transpose_like(self.ket[i], inplace=True)
            self.ket[i].modify(data=site_i.data)

            ## canonicalize ket to next site
            helper.canonize_tens_list(self.ket[i], self.ket[i + 1], inplace=True)

            ## update bra
            self.set_bra_from_ket(sites=[i,i+1])

        else:
            ## update k
            self.ket[i].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            helper.canonize_tens_list(self.ket[i], self.ket[i - 1], inplace=True)

            ## update bra
            self.set_bra_from_ket(sites=[i,i-1])

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
            self.set_bra_from_ket(sites=[i, i + 1])

            # ## update bra
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i)
            # bra_inds_0 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i].inds]
            # self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)
            #
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i + 1)
            # bra_inds_1 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i + 1].inds]
            # self.bra[i + 1].modify(data=self.ket[i + 1].data.conj(), inds=bra_inds_1)

        else:

            _, left_inds = self.ket[i].filter_bonds(self.ket[i - 1])    # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', max_bond=self.max_bond)

            ## update ket
            self.ket[i].modify(data=site2.data, inds=site2.inds)
            self.ket[i - 1].modify(data=site1.data, inds=site1.inds)

            ## update bra
            self.set_bra_from_ket(sites=[i, i - 1])

            # ## update bra
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i)
            # bra_inds_0 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i].inds]
            # self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)
            #
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i - 1)
            # bra_inds_1 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i - 1].inds]
            # self.bra[i - 1].modify(data=self.ket[i - 1].data.conj(), inds=bra_inds_1)

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        end = nsites - 1 if None else end

        if canonize:
            self.canonize(end)

        for env in self.right_envs:
            for i in range(self.L - 1, end, -1):
                env.extend_env(i)
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        end = self.L - nsites

        if canonize:
            self.canonize(end)

        for env in self.left_envs:
            for i in range(end):
                env.extend_env(i)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos + 1)
            # helper.canonize(self.ket, i=pos + 1)
            # self.set_bra_from_ket()
        for env in self.left_envs:
            env.extend_env(pos)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos - 1)
            # helper.canonize(self.ket, i=pos - 1)
            # self.set_bra_from_ket()
        for env in self.right_envs:
            env.extend_env(pos)
        return




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
        H_eff = H_eff.contract_tags(all)
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
                 mps_ind_range: tuple[int,int] = None,
                 norm_env0_L: 'qtn.Tensor' = None,
                 norm_env0_R: 'qtn.Tensor' = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None, solve_type:str=None):

        operators = [] if operators is None else list(operators)

        super().__init__(trial_state, operators = operators, targets = targets,
                         mps_ind_range=mps_ind_range, conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)

        self.A_envsL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_L)
        self.A_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_R)

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, [],
                                 init_env = ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None )
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, [],
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None )
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        self.left_envs  = [self.A_envsL,] + self.b_envsLs
        self.right_envs = [self.A_envsR,] + self.b_envsRs

        self.solve_type = solve_type


    # def _get_solve_method(self):
    #     if self.solve_type == 'CGD':
    #         return self._site_solve_CGD
    #     else:   ## defaults to LSQ
    #         return self._site_solve


    def _site_solve(self, sites: Union[int,slice],) -> qtn.Tensor:
        """
        sites: int or slice(start, stop, step)
        """
        site_pos = list(range(self.L))[sites]
        site_ind_0 = self.mps_ind_range[0] + site_pos[0]
        site_inds = slice( site_ind_0, site_ind_0 + len(site_pos) )

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_left = self.A_envsL[site_pos[0]]
        A_right = self.A_envsR[site_pos[-1]]
        if self.num_operators > 0:
            A_eff = qtn.TensorNetwork([])
            for A in self.operators:
                A_eff.add(A[site_inds])
            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)
            # print('A eff exponent', A_eff.exponent, self.A_envsL.exponent)
            A_eff.exponent = self.A_envsL.exponent
        else:
            A_eff = None


        ### BL * B * BR = d/dT*[i] <x|b>
        b_eff: Optional['qtn.Tensor'] = None
        for ti in range(self.num_targets):
            b_left = self.b_envsLs[ti][site_pos[0]]
            b_right = self.b_envsRs[ti][site_pos[-1]]

            b_eff_t = qtn.TensorNetwork([])
            if b_left is not None:
                b_eff_t.add(b_left)
            if b_right is not None:
                b_eff_t.add(b_right)

            b_eff_t.add(self.targets[ti][site_inds])

            # for A in self.operators_H:
            #     b_eff_t.add(A[sites])

            b_eff_t = b_eff_t.contract_tags(all)
            b_eff_exponent = self.b_envsLs[ti].exponent
            # print('b eff exponent', b_eff_exponent)
            b_eff_t.modify(apply=lambda x: 10**b_eff_exponent * x)

            if b_eff is None:
                b_eff = b_eff_t
            else:
                b_eff_t.transpose_like(b_eff, inplace=True)
                b_eff.modify(apply=lambda x: x + b_eff_t.data)

        ## change indices from bra (missing T*[i]) to ket
        bra_to_ket_inds = {} # self.get_bra_to_ket_inds(site_inds[0])
        for site_ind in range(site_inds.start, site_inds.stop, site_inds.step if site_inds.step is not None else 1):
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_ind))

        bra_inds = b_eff.inds if b_eff is not None else None
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        ## solve for T[i]
        if A_eff is None:
            x_eff = b_eff.reindex(bra_to_ket_inds)
            x_eff.modify(apply= lambda x : x * 10**(-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type == 'CGD':
                A_eff_exponent = A_eff.exponent
                x_tens = qtn.tensor_contract(*self.ket[site_inds])
                x_tens.transpose(*ket_inds, inplace=True)
                x_tens.modify(apply=lambda x: x * 0.)
                x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, x_tens, bra_to_ket_inds)
                x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))

            elif self.solve_type == 'CGDx':

                x_tens = qtn.tensor_contract(*self.ket[site_inds])
                x_tens.transpose(*ket_inds, inplace=True)

                A_eff_exponent = A_eff.exponent
                A_eff = A_eff.contract_tags(all)
                A_eff.transpose(*bra_inds, *ket_inds, inplace=True)
                out_shape = A_eff.shape[:len(bra_inds)]
                in_shape = A_eff.shape[len(bra_inds):]
                A_eff_data = A_eff.data.reshape(np.prod(out_shape), -1) # * 10**A_eff_exponent

                if b_eff is not None:
                    b_eff_data = b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                x_eff_data = conjugate_gradient_descent(A_eff_data, b_eff_data,) # x=x_tens.data.reshape(-1))
                # x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
                x_eff_data = x_eff_data.reshape(in_shape)
                x_eff_data *= 10 ** (-A_eff_exponent)
                ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)
            else:
                A_eff_exponent = A_eff.exponent
                A_eff = A_eff.contract_tags(all)
                ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]
                # print('bra inds', bra_inds, 'ket inds', ket_inds)
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
                x_eff_data *= 10 ** (-A_eff_exponent)
                ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

        return x_eff


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
        self.canonize(canon_site)

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
                    ix = self.get_mps_ind(i)

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

                err = self.check_err()

            else:

                ## left sweep
                for i in range(L - 1, nsites - 1, -1):
                    ix = self.get_mps_ind(i)

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

                err = self.check_err()

            print('err', it, err)

            if err > prev_err:
                print('Warning: error went up')

            direction *= -1     ## swaps sweep direction

        return self.ket


    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)


    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)


    def check_err(self) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
        """
        old_site_ind_id = self.bra.site_ind_id
        if self.num_operators > 0:
            self.bra.site_ind_id = self.operators_H[-1].upper_ind_id

        ## <xA|Ax>
        norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
                                    [mpo for mpo in self.operators_H], virtual=True)

        A_eff_exponent = norm_Ax.exponent
        norm_Ax = norm_Ax.contract_tags(all)
        norm_Ax *= 10**A_eff_exponent

        ## <xA|b>
        ovlp_b_tot = 0.
        for ib in range(self.num_targets):
            ovlp_b = qtn.TensorNetwork([self.targets[ib], self.bra] +
                                        [mpo for mpo in self.operators_H], virtual=True)

            ovlp_b = ovlp_b.contract_tags(all)
            b_eff_exponent = self.operator_exponent + self.bra.exponent + self.targets[ib].exponent

            ovlp_b_tot = ovlp_b_tot + ovlp_b * 10**b_eff_exponent

        self.bra.site_ind_id = old_site_ind_id

        # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
        err2 = (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2
        if err2 < 0 and np.abs(err2) < 1.0e-12:
            err2 = np.abs(err2)

        return np.sqrt(err2) / self.targets_norm



#############################
###  minimize Ax=b error  ###
###   multigrid method    ###
#############################

class MultigridLinearSolver(LinearSolver):

    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['qtn.MatrixProductState']] = None,
                 operators: Optional[Sequence['qtn.MatrixProductOperator']] = None,
                 mps_ind_range: tuple[int,int] = None,
                 coarseness_mps_inds: dict[int, Sequence[int]] = None,
                 coarseness_level: int = 0,
                 norm_env0_L: 'qtn.Tensor' = None,
                 norm_env0_R: 'qtn.Tensor' = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 solve_type: str = None, coarsen_type: str = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):
        """ solves Ax=b using a multigrid optimization scheme


            mps_coarseness:  dict where key indicates coarseness level (0 is finest),
                and maps to inds in mps corresponding to that coarseness level
                default: mps from left to right is increasingly fine
        """
        operators = [] if operators is None else list(operators)

        super().__init__(trial_state, operators = operators, targets = targets,
                         mps_ind_range=mps_ind_range,
                         norm_env0_L = norm_env0_L, norm_env0_R = norm_env0_R,
                         ovlp_env0_Ls = ovlp_env0_Ls, ovlp_env0_Rs = ovlp_env0_Rs,
                         conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)

        self.A_envsL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_L)
        self.A_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators,
                                   init_env = norm_env0_R)

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, [],
                                 init_env = ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None )
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, [],
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None )
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        self.left_envs  = [self.A_envsL,] + self.b_envsLs
        self.right_envs = [self.A_envsR,] + self.b_envsRs

        self.solve_type = solve_type
        self.coarsen_type = coarsen_type
            ## select point: select
            ## average over points: (S+ + S-)/2 and select. stencil for higher dimensions?
        self.coarseness_level = coarseness_level

        if coarseness_mps_inds is None:
            self.coarseness_mps_inds = {i: self.L - 1 - i for i in range(0, self.L)}
            ## as one goes along the MPS grid becomes finer
        else:
            self.coarseness_mps_inds = coarseness_mps_inds


    def create_like(self, trial_state: 'qtn.MatrixProductState',
                    targets: Optional[Sequence['qtn.MatrixProductState']] = None,
                    operators: Optional[Sequence['qtn.MatrixProductOperator']] = None,
                    coarseness_level: int = None):

        new_solver = MultigridLinearSolver(
                        self.ket if trial_state is None else trial_state,
                        targets = self.targets if targets is None else targets,
                        operators = self.operators if operators is None else operators,
                        coarseness_level = self.coarseness_level if coarseness_level is None else coarseness_level,
                        mps_ind_range = self.mps_ind_range, coarseness_mps_inds = self.coarseness_mps_inds,
                        solve_type = self.solve_type, coarsen_type = self.coarsen_type,
                        conv_tol=self.conv_tol, max_iter=self.max_iter, max_bond=self.max_bond)

        new_solver.A_envsL = self.A_envsL
        new_solver.A_envsR = self.A_envsR
        new_solver.b_envsLs = self.b_envsLs
        new_solver.b_envsLs = self.b_envsLs
        return new_solver


    def coarsen(self):

        grid_level = self.coarseness_level
        integ_inds = self.coarseness_mps_inds[grid_level]

        if self.coarsen_type == 'select':

            ket_ = self.ket.copy()
            for i in integ_inds:
                ket_[i].isel({self.ket.site_ind(i): 0}, inplace=True)

            operators_ = [mpo for mpo in self.operators[:-1]]
            mpo_last = self.operators[-1].copy()
            for i in integ_inds:
                mpo_last[i].isel({mpo_last.upper_ind(i): 0}, inplace=True)
            operators_ += [mpo_last]

            targets_ = []
            for target in self.targets:
                target_ = target.copy()
                for i in integ_inds:
                    target_[i].isel({target.site_ind(i): 0}, inplace=True)

            new_sys = self.create_like(ket_, targets=targets_, operators=operators_,
                                       coarseness_level=self.coarseness_level+1)
            return new_sys


        elif self.coarsen_type == 'midpt':
            raise NotImplementedError
            ## would be easier if I made all these objects GridTNs...


    def _solve_coarse(self, max_coarseness):
        if self.coarseness_level < max_coarseness:
            coarse_sys = self.coarsen()
            coarse_sys._solve_coarse(max_coarseness)


    def _optimize_from_coarse(self, coarse_x):
        """ Ax = b, where x is obtained from coarse guess
        """
        mps_inds = self.coarseness_mps_inds[self.coarseness_level]

        ## obtain initial guesses by assuming aall other indices are integrated/coarsened over
        for ind in mps_inds:
            other_inds = [i for i in mps_inds if i != ind]

            target =



    def solve(self, nsites: int, max_coarseness=None):
        """ multigrid iterative solver for entire MPS

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

        if max_coarseness is None:
            if self.max_bond is None:
                max_coarseness = 0
            else:
                max_coarseness = np.ceil(np.log2(self.max_bond)) // 2

        integ_inds = self.coarseness_mps_inds[0]
        canon_site = integ_inds[0]
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|Ax> init envs
        self._build_all_envs_right(nsites, end=canon_site, canonize=False)
        self._build_all_envs_left(nsites, end=canon_site, canonize=False)




        ## recursive solver
        coarse_level = 0
        while coarse_level <= max_coarseness:

            coarse_sys = self.coarsen()

        it = 0
        err = 100
        direction = init_direction
        while err > self.conv_tol and it < self.max_iter:
            prev_err = err
            it += 1

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites):
                    ix = self.get_mps_ind(i)

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

                err = self.check_err()

            else:

                ## left sweep
                for i in range(L - 1, nsites - 1, -1):
                    ix = self.get_mps_ind(i)

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

                err = self.check_err()

            print('err', it, err)

            if err > prev_err:
                print('Warning: error went up')

            direction *= -1     ## swaps sweep direction

        return self.ket