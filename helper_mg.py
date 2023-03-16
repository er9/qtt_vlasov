from setup.defaults import *
import time
from scipy import linalg
import quimb.tensor as qtn
import helper_quimb as helper
import setup.quimb_TN1D as q1D
from helper_dmrg import *

class CoarsenMethod(IntEnum):
    SELECT = 1
    SUM = 2
    AVERAGE = 3
    GALERKIN = -1

DEFAULT_COARSEN = CoarsenMethod.AVERAGE

################################
### numpy optimzation method ###
################################

def multigrid_optimization(A, b, x=None, max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL):
    raise NotImplementedError


#########################
### optimization fcts ###
#########################

def mg_linear_combine(mps_list: Sequence[qtn.MatrixProductState],
                      operator_list: Sequence[Optional['MPO_type']],
                      weights: Sequence['float'] = None,
                      init_guess: Optional[qtn.MatrixProductState] = None,
                      opt_nsites=1, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None,
                      **solver_kwargs):

    """ finds approximation of x that solves: prod(A) x = sum(b * weights)
        b = mps_list, A = operator list
    """
    if weights is not None:
        mps_list = [helper.scalar_multiply(mps, weight) for mps, weight in zip(mps_list, weights)]

    init_guess = mps_list[0] if init_guess is None else init_guess

    solver = MultigridLinearSolver(init_guess, targets=mps_list, operators=operator_list, conv_tol=conv_tol,
                                   max_iter=max_iter, max_bond=max_bond, **solver_kwargs)
    return solver.solve(opt_nsites)


def mg_add_mps(mps_list: Sequence[qtn.MatrixProductState],
               init_guess: Optional[qtn.MatrixProductState]=None,
               weights: Sequence['float'] = None,
               opt_nsites=1, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None,
               **solver_kwargs):
    """ obtain approximation of x = sum(b), weights applied to mps_list
    """
    if weights is not None:
        mps_list = [helper.scalar_multiply(mps, weight) for mps, weight in zip(mps_list, weights)]

    init_guess = mps_list[0].copy() if init_guess is None else init_guess

    solver = MultigridLinearSolver(init_guess, targets=mps_list, conv_tol=conv_tol, max_iter=max_iter,
                                   max_bond=max_bond, **solver_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def mg_apply(init_mps: qtn.MatrixProductState, mpo: qtn.MatrixProductOperator,
             init_guess: qtn.MatrixProductState = None,
             opt_nsites=2, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None,
             **solver_kwargs):

    init_guess = init_mps if init_guess is None else init_guess
    target = MatrixProductStateTN(init_mps, [mpo])
    solver = MultigridLinearSolver(init_guess, targets=[target],
                                   conv_tol=conv_tol, max_iter=max_iter,
                                   max_bond=max_bond, **solver_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def mg_solve(soln_mps: qtn.MatrixProductState, operator: qtn.MatrixProductOperator,
             init_guess: qtn.MatrixProductState = None,
             opt_nsites=2, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None, **solve_kwargs):
    """ solve Ax = b
    """
    init_guess = helper.scalar_multiply(soln_mps, 0.) if soln_mps is None else init_guess   ## should be zero, probably?
    solver = MultigridLinearSolver(init_guess, targets=[soln_mps], operators=[operator],
                                   conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond,
                                   **solve_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def mg_compress(target_mps: qtn.MatrixProductState, init_guess: Optional[qtn.MatrixProductState] = None,
                max_iter=50, **mg_opt_kwargs):
    """ compress state
        init guess should have the desired bond dimension... unless 2 site optimization is used to adapt D
    """
    mg_opt_kwargs.setdefault('opt_nsites',2)
    cutoff = mg_opt_kwargs.pop('cutoff', CUTOFF)
    cutoff_mode = mg_opt_kwargs.pop('cutoff_mode', CUTOFF_MODE)
    return mg_add_mps([target_mps], init_guess=init_guess, max_iter=max_iter, **mg_opt_kwargs)


########################
### helper functions ###
########################

def mps_coarsen(mps: qtn.MatrixProductState, coarsen_inds: Sequence[int], integ_inds: Sequence[int],
                coarsen_method: CoarsenMethod=DEFAULT_COARSEN, inplace=False):
    """ integ_inds: inds in mps corresponding to the finest grid mesh that we want to select/integrate over
        coarsen_inds dict:
            key:  fine index
            value:  coarse index
        coarse inds (multiple for multiple dimensions)
    """
    mps = mps if inplace else mps.copy()
    coarsen_inds = np.array(coarsen_inds)
    integ_inds = np.array(integ_inds)
    extra_contract_tags = ()

    ## apply integration operations
    if coarsen_method > 0:
        for ind in integ_inds:
            if coarsen_method is CoarsenMethod.SELECT:
                site_op = np.array([1.0,0.0])
                # mps[ind].isel({mps.site_ind_id.format(ind): 0}, inplace=True)
            elif coarsen_method is CoarsenMethod.AVERAGE:
                site_op = np.array([1.0, 1.0]) * 1./np.sqrt(2)
            elif coarsen_method is CoarsenMethod.SUM:
                site_op = np.array([1.0, 1.0])
            else:
                raise NotImplementedError
            integ_tens = qtn.Tensor(site_op, inds=(mps.site_ind_id.format(ind),),
                                    tags=(mps.site_tag_id.format(ind), 'IX'))
            mps.add(integ_tens)
            # extra_contract_tags = ('IX',)
    else:
        if coarsen_method is CoarsenMethod.GALERKIN:
            ### 1 2 1 weighting
            ### easier to just work with a n-dimensional matrix and deal with all tensors at once
            integ_mps = qtn.TensorNetwork([])
            raise NotImplementedError
        elif coarsen_method == 'Galerkin':
            raise NotImplementedError
        else:
            raise ValueError

    ### contract mps as needed...
    drop_tags = ()
    for ind in integ_inds:

        adj_idx = np.flatnonzero( np.abs(coarsen_inds - ind) == 1 )
        try:
            adj_ind = coarsen_inds[adj_idx[0]]
        except IndexError:
            adj_idx = np.flatnonzero( np.abs(integ_inds - ind) == 1 )
            adj_ind = integ_inds[adj_idx[0]]

        site_tag = mps.site_tag_id.format(ind)
        if isinstance(site_tag, str):
            integ_site_tag = (site_tag,)
        else:   ## MPSTN object
            integ_site_tag = tuple(iter(site_tag))
        drop_tags = drop_tags + integ_site_tag

        site_tag = mps.site_tag_id.format(adj_ind)
        if isinstance(site_tag, str):
            adj_site_tag = (site_tag,)
        else:
            adj_site_tag = tuple(iter(site_tag))

        # print('mps to contract', mps)
        # print('contract tags', extra_contract_tags + integ_site_tag + adj_site_tag)
        mps.contract_tags(extra_contract_tags + integ_site_tag + adj_site_tag, which='any', inplace=True)
        # print('contracted mps', mps)

    mps.drop_tags(drop_tags + ('IX',))
    mps._L = mps.L - len(integ_inds)
    # print('mps.L', mps.L)
    return mps


def mpo_coarsen(mpo: qtn.MatrixProductOperator, coarsen_inds: Sequence[int], integ_inds: Sequence[int],
                coarsen_method: CoarsenMethod=DEFAULT_COARSEN, inplace=False):
    """ integ_inds: inds in mps corresponding to the finest grid mesh that we want to select/integrate over
    """

    mpo = mpo if inplace else mpo.copy()

    if coarsen_method is CoarsenMethod.SELECT:
        for i in integ_inds:
            mpo[i].isel({mpo.site_ind_id(i): 0}, inplace=True)
    elif coarsen_method is CoarsenMethod.SUM:
        integ_mps = qtn.TensorNetwork([])
        for i in integ_inds:
            integ_mps.add(qtn.Tensor(np.array([1.0, 1.0]), inds=(mpo.upper_ind(i),), tags=('IX',)))
            integ_mps.add(qtn.Tensor(np.array([1.0, 1.0]), inds=(mpo.lower_ind(i),), tags=('IX',)))
        mpo_site_tags = [mpo.site_tag(i) for i in integ_inds]
        mpo.add(integ_mps)
        mpo.contract_tags(('IX',) + tuple(mpo_site_tags), which='any', inplace=True)
    elif coarsen_method is CoarsenMethod.GALERKIN:
        ### 1 2 1 weighting
        integ_mps = qtn.TensorNetwork([])
        raise NotImplementedError
    elif coarsen_method == 'Galerkin':
        raise NotImplementedError
    else:
        raise ValueError

    return mpo


#############################
###  minimize Ax=b error  ###
###   multigrid method    ###
#############################

class MultigridLinearSolver(LinearSolver):

    def __init__(self, trial_state: Union['qtn.MatrixProductState','IndexedMPS'],
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 mps_inds: Sequence[int] = None,
                 coarseness_mps_inds: dict[int, set[int]] = None,
                 coarseness_level: int = 0, max_coarseness: int = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 solve_type: SolveMethod = DEFAULT_SOLVE, coarsen_type: CoarsenMethod = DEFAULT_COARSEN,
                 minimal_opt: bool = True,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):
        """ solves Ax=b using a multigrid optimization scheme

            mps_coarseness:  dict where key indicates coarseness level (0 is finest),
                and maps to inds in mps corresponding to that coarseness level
                default: mps from left to right is increasingly fine

            operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)

            norm_env0: envs for <x|A|x>
            ovlp_env0: envs for <x|b>   (or is it <b|x>?)
            assumes envs for <x|x> = identity
        """
        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        trial_state = q1D.view_as_indexed(trial_state, site_positions=mps_inds)
        operators = [q1D.view_as_indexed(mpo, site_positions=mps_inds) for mpo in operators] \
                            if operators is not None else None
        targets = [q1D.view_as_indexed(target, site_positions=mps_inds) for target in targets] \
                            if targets is not None else None

        super().__init__(trial_state, operators = operators, targets = targets,
                         mps_inds = mps_inds,
                         norm_env0_Ls = norm_env0_Ls, norm_env0_Rs = norm_env0_Rs,
                         ovlp_env0_Ls = ovlp_env0_Ls, ovlp_env0_Rs = ovlp_env0_Rs,
                         conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)

        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Ls[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Rs[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, None,
                                 mps_inds = self.mps_inds,
                                 init_env = ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None )
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, None,
                                 mps_inds=self.mps_inds,
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None )
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        # self.left_envs  = self.A_envsLs + self.b_envsLs
        # self.right_envs = self.A_envsRs + self.b_envsRs

        self.solve_type = solve_type
        self.coarsen_type = coarsen_type
        self.minimal_opt = minimal_opt
        ## select point: select
        ## average over points: (S+ + S-)/2 and select. stencil for higher dimensions?

        self.coarseness_level = coarseness_level
        if coarseness_mps_inds is None:
            self.coarseness_mps_inds = {i: {self.L - 1 - i} for i in range(0, self.L)}
            ## as one goes along the MPS grid becomes finer
            ## increasing coarseness level --> coarser grid (finer grids are integrated out)
        else:
            self.coarseness_mps_inds = coarseness_mps_inds
        # print('self.coarseness mps inds', self.coarseness_mps_inds)

        self.max_coarseness = max_coarseness
        if max_coarseness is None:
            if self.max_bond is None:
                self.max_coarseness = 0
            else:
                indL, indR = self.mps_inds[0], self.mps_inds[-1]
                tensL, tensR = self.ket[indL], self.ket[indR]
                ancL_size, ancR_size = 1, 1

                if tensL.ndim > 2:
                    shared, lone = tensL.filter_bonds(self.ket[indL+1])
                    for lb in iter(lone):
                        ancL_size *= tensL.ind_size(lb)

                if tensR.ndim > 2:
                    shared, lone = tensR.filter_bonds(self.ket[indR-1])
                    for lb in iter(lone):
                        ancR_size *= tensR.ind_size(lb)

                coarsened_L = self.L
                for cl in sorted(self.coarseness_mps_inds):

                    coarsened_L -= len(self.coarseness_mps_inds[cl])
                    max_bond_size = 2**(coarsened_L // 2) * ancL_size * ancR_size
                    # max_bond_loc = coarsened_L // 2 - int(np.log2(ancL_size / ancR_size))

                    # max_bond_size = 0
                    # for ind in self.coarseness_mps_inds[cl]:
                    #     print('integ ind', ind, self.ket.phys_dim())
                    #     max_bond_size_L = ancL_size * self.ket.phys_dim() ** ind
                    #     max_bond_size_R = ancR_size * self.ket.phys_dim() ** (self.L - ind - 1)
                    #     print('max bond size', max_bond_size_L, max_bond_size_R)
                    #     max_bond_size = max(max_bond_size, (min(max_bond_size_L, max_bond_size_R)))

                    # print('MAX BOND SIZE', max_bond_size, coarsened_L, self.max_bond)
                    if max_bond_size > self.max_bond:
                        continue
                    else:
                        max_coarseness = cl
                        break

                self.max_coarseness = max_coarseness
                ## negative if L allowed by max_bond < actual ket length
                ## depends on if there any ancillas or not...

            # print('max coarseness', self.max_coarseness, self.max_bond)


    def create_like(self, copy=True, new_ket: 'qtn.MatrixProductState' = None, **kwargs):

        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L - 1] for A_envR in self.A_envsRs]
        ovlp_env0_Ls = [b_envL[0] for b_envL in self.b_envsLs]
        ovlp_env0_Rs = [b_envR[self.L - 1] for b_envR in self.b_envsRs]

        # if new_ket is not None:
        #     print('create like ket', new_ket.site_positions)
        # t = kwargs.get('targets', None)
        # print('create like target', t[0])
        # print('t data', t[0].data)
        # print('copy t', t[0].copy())
        new_solver = MultigridLinearSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                           targets=kwargs.get('targets',
                                                              [t.copy() if copy else t for t in self.targets]),
                                           operators=kwargs.get('operators',
                                                                [o.copy() if copy else o for o in self.operators]),
                                           mps_inds=kwargs.get('mps_inds',self.mps_inds),
                                           coarseness_mps_inds=kwargs.get('coarseness_mps_inds',
                                                                          self.coarseness_mps_inds),
                                           coarseness_level=kwargs.get('coarseness_level', self.coarseness_level),
                                           max_coarseness=kwargs.get('max_coarsness', self.max_coarseness),
                                           norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                           norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                           ovlp_env0_Ls=kwargs.get('ovlp_env0_Ls', ovlp_env0_Ls),
                                           ovlp_env0_Rs=kwargs.get('ovlp_env0_Rs', ovlp_env0_Rs),
                                           solve_type=kwargs.get('solve_type', self.solve_type),
                                           coarsen_type=kwargs.get('coarsen_type', self.coarsen_type),
                                           conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                           max_iter=kwargs.get('max_iter', self.max_iter),
                                           max_bond=kwargs.get('max_bond', self.max_bond),
                                           minimal_opt=kwargs.get('minimal_opt', self.minimal_opt))
        # print('new solver', new_solver.ket.site_positions)
        return new_solver


    def copy(self, deep=True):

        new_solver = self.create_like(copy=deep)

        new_solver.A_envsLs = [env.copy() for env in self.A_envsLs]
        new_solver.A_envsRs = [env.copy() for env in self.A_envsRs]
        new_solver.b_envsLs = [env.copy() for env in self.b_envsLs]
        new_solver.b_envsLs = [env.copy() for env in self.b_envsLs]

        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        return new_solver


    # def get_init_guess(self):
    #     """ set self.ket as initial ket, coarsen until get desired bond dimension
    #     """
    #     if np.prod(self.ket.shape) <= self.max_bond**2:
    #         ### solve directly
    #         b = self.targets[0].copy()
    #         for target in self.targets[1:]:
    #             b = helper.add_MPS(b, target, inplace=True, compress=True)
    #
    #         if self.num_operators > 0:
    #             A = self.operators[0].copy()
    #             for op in self.operators[1:]:
    #                 A = helper.apply(op, A, compress=True)
    #             init_guess = np.linalg.solve(A, b)
    #         else:
    #             init_guess = b
    #
    #         init_guess.site_ind_id = self.ket.site_ind_id
    #         self.reinitialize_envs()
    #         self.ket = init_guess
    #         return init_guess
    #
    #     else:
    #         coarse_sys = self.coarsen()
    #         return coarse_sys.get_init_guess()


    def coarsen(self):
        """ obtain mps at next caorseness level
        """
        integ_inds = sorted(self.coarseness_mps_inds[self.coarseness_level])
        coarsen_inds = sorted(self.coarseness_mps_inds[self.coarseness_level + 1])
        new_sys = self._coarsen_inds(integ_inds, coarsen_inds)
        # new_sys.coarseness_level += 1
        return new_sys


    def _coarsen_inds(self, integ_inds, coarsen_inds):
        """ integrate over specified inds (note: doesn't change coarseness level)
            should not be in place
        """
        mps_inds = self.ket.site_positions   # self.mps_inds
        # print('mps inds', mps_inds, integ_inds, coarsen_inds)
        mps_integ_inds = integ_inds # [mps_inds[ind] for ind in integ_inds]
        mps_coarsen_inds = coarsen_inds # [mps_inds[ind] for ind in coarsen_inds]
        new_mps_inds = [ind for ind in mps_inds if ind not in integ_inds]
        # print('new mps inds', new_mps_inds)

        ket_ = mps_coarsen(self.ket, integ_inds=mps_integ_inds, coarsen_inds=mps_coarsen_inds,
                           coarsen_method=self.coarsen_type)
        ket_._site_positions = new_mps_inds
        # print('ket positions', ket_.site_positions)

        if self.coarsen_type is CoarsenMethod.SUM:
            mpo_coarsen_type = CoarsenMethod.AVERAGE
        elif self.coarsen_type is CoarsenMethod.AVERAGE:
            mpo_coarsen_type = CoarsenMethod.SUM
        else:
            mpo_coarsen_type = self.coarsen_type

        operators_ = []
        for mpo in self.operators:
            mpo_ = mpo_coarsen(mpo, integ_inds=mps_integ_inds, coarsen_inds=mps_coarsen_inds,
                                     coarsen_method=mpo_coarsen_type)
            if isinstance(mpo_, MatrixProductOperatorTN):
                for sub_mp in mpo_.mpxs:
                    sub_mp._site_positions = new_mps_inds
            else:
                mpo_._site_positions = new_mps_inds
            operators_ += [mpo_]

        # operators_ = [mpo_coarsen(mpo, integ_inds=mps_integ_inds, coarsen_inds=mps_coarsen_inds,
        #                           coarsen_method=mpo_coarsen_type)
        #               for mpo in self.operators]

        targets_ = []
        for target in self.targets:
            target_ = mps_coarsen(target, integ_inds=mps_integ_inds, coarsen_inds=mps_coarsen_inds,
                                  coarsen_method=self.coarsen_type)
            if isinstance(target_, MatrixProductOperatorTN):
                for sub_mp in target_.mpxs:
                    sub_mp._site_positions = new_mps_inds
            else:
                target_._site_positions = new_mps_inds
            # print('coarsened target', target_)
            targets_ += [target_]

        # print('create like')
        # print('ket_', ket_)
        # print('target', targets_[0])
        new_sys = self.create_like(new_ket=ket_, targets=targets_, operators=operators_,
                                   mps_inds=new_mps_inds,
                                   coarseness_level=self.coarseness_level + 1)
        return new_sys


    def _prolongate_from_coarse(self, coarse_sys: 'MultigridLinearSolver',
                                inplace=False, conv_tol=0, max_iter=None):

        # ## test against mps_coarse function
        # test_sys = coarse_sys.copy()
        # test_target = self.targets[0].copy()
        # test_ket = coarse_sys.ket.copy()

        new_ket = coarse_sys.ket.copy()  # self.ket if inplace else self.ket.copy()
        new_sys = self if inplace else self.copy()

        coarse_mps_inds = sorted( coarse_sys.coarseness_mps_inds[coarse_sys.coarseness_level] )
        pad_mps_inds = sorted( self.coarseness_mps_inds[self.coarseness_level] )
        # print('pad mps inds', pad_mps_inds)

        site_ind_id = new_sys.ket.site_ind_id
        site_tag_id = new_sys.ket.site_tag_id

        # print('self coarsen type', self.coarsen_type)
        # print('targets', self.num_targets, self.targets_norm)
        # print('self', helper.norm(self.ket))
        if self.coarsen_type is CoarsenMethod.SELECT:
            integ_vec = np.array([1.0, 0.0])
        elif self.coarsen_type is CoarsenMethod.SUM:
            integ_vec = np.array([1.0, 1.0])
        elif self.coarsen_type is CoarsenMethod.AVERAGE:
            integ_vec = np.array([0.9, 1.1]) * 1./np.sqrt(2)
        elif self.coarsen_type is CoarsenMethod.GALERKIN:
            raise NotImplementedError
        else:
            raise ValueError('not valid coarsen method')


        for i in range(len(coarse_mps_inds)):
            ind1 = coarse_mps_inds[i]
            try:
                ind2 = coarse_mps_inds[i + 1]
            except IndexError:
                ind2 = None

            tag1 = site_tag_id.format(ind1)
            tens1 = next(iter(new_ket.select_tensors(tag1)))

            if ind2 is None:
                ind1_ = ind1
                for ix in range(ind1 + 1, max(pad_mps_inds) + 1):
                    # print('ix',ix, pad_mps_inds)
                    if ix in pad_mps_inds:  ## add padding tensor
                        tag_ix = site_tag_id.format(ix)
                        # integ_vec = np.array([0.5] * new_sys.ket.phys_dim())
                        tens_ix = qtn.Tensor(integ_vec, inds=(site_ind_id.format(ix),), tags=(tag_ix,))

                        # ### test
                        # test_target.add(tens_ix)

                        ## get bond name to match ket from fine grid
                        fine_b1 = self.ket.bond(ind1_, ix)
                        # print('add new bond', fine_b1)
                        qtn.new_bond(tens1, tens_ix, name=fine_b1)
                        # print('t', tens1, tens_ix)

                        new_ket.add(tens_ix, virtual=True)
                        tens1 = tens_ix
                        ind1_ = ix

            else:
                tag2 = site_tag_id.format(ind2)
                tens2 = next(iter(new_ket.select_tensors(tag2)))
                bond_size = tens1.shared_bond_size(tens2)
                ind1_ = ind1

                for ix in range(ind1+1, ind2):
                    if ix in pad_mps_inds:      ## add padding tensor
                        tag_ix = site_tag_id.format(ix)
                        new_ket.insert_operator(np.eye(bond_size), tens1, tens2, tags=(tag_ix,), inplace=True)

                        # ### test
                        # tens_ix = qtn.Tensor(integ_vec, inds=(site_ind_id.format(ix),))
                        # test_target.add(tens_ix)

                        tens_ix = next(iter(new_ket.select_tensors(tag_ix)))
                        tmp_out = tens_ix.contract(qtn.Tensor(integ_vec, inds=(site_ind_id.format(ix),)))
                        tens_ix.modify(data=tmp_out.data, inds=tmp_out.inds)
                        # print('tens ix', tens_ix, site_ind_id.format(ix))
                        # exit()

                        ## reindex tens1 to match ket from fine grid
                        b1 = next(iter(tens_ix.bonds(tens1)))
                        b2 = next(iter(tens_ix.bonds(tens2)))
                        b1_dict, b2_dict = {}, {}
                        try:
                            b1_dict[b1] = self.ket.bond(ind1_,ix)
                            tens1.reindex(b1_dict, inplace=True)
                        except ValueError:
                            pass
                        try:
                            b2_dict[b2] = self.ket.bond(ix, ind2)
                            tens2.reindex(b2_dict, inplace=True)
                        except ValueError:
                            pass
                        tens_ix.reindex({**b1_dict, **b2_dict}, inplace=True)

                        tens1 = tens_ix
                        ind1_ = ix

        # ##### test coarse
        # # test_target.contract_tags(tags=('T(5)','T(4)'), which='any', inplace=True)
        # # test_target.drop_tags(tags=('T(5)',))
        # test_target._L = test_target._L - len(pad_mps_inds)
        # test_sys2 = test_sys.create_like(new_ket=test_ket, targets=[test_target])
        # print('test err 2', test_sys2.check_err())
        # print('test err', test_sys.check_err())

        ## 1 or 2 site solve
        new_ket._L = new_ket._L + len(pad_mps_inds)
        new_ket._site_positions = self.ket.site_positions    ## fine grid site positions
        # print('new ket', new_ket.L, self.L, new_ket)
        # print('new ket', new_ket._site_positions)


        # test_new = LinearSolver(new_ket, targets=new_sys.targets,)
        # print('test new err', test_new.check_err())

        print('PROLONGATE')
        new_sys.ket = new_ket
        new_sys.set_bra_from_ket()
        # print('new sys err', new_sys.check_err())
        # exit()
        # new_sys.reinitialize_envs()
        # print('new ket', new_ket)
        # print('new target', new_sys.targets[0])
        # print('new env ket', new_sys.b_envsLs[0].ket)
        # print('new env bra', new_sys.b_envsLs[0].bra)
        opt_inds = self.coarseness_mps_inds[self.coarseness_level] if self.minimal_opt else None
        new_sys.linear_solve(2, conv_tol=conv_tol, max_iter=max_iter, opt_inds=opt_inds)
        # print('prologated solved', helper.norm(new_sys.ket))

        # exit()
        return new_sys

        # solver = new_sys.create_like(new_ket=new_ket) #, targets=target_fine)
        # return solver.linear_solve(2, conv_tol=conv_tol, max_iter=max_iter, opt_inds=opt_inds)
        # solver = super(new_sys.__class__, new_sys).create_like(new_ket=new_ket, targets=target_fine)
        # return solver.solve(2, conv_tol=conv_tol, max_iter=max_iter, opt_inds=opt_inds)     ## return system


    def linear_solve(self, nsites: int, **solve_kwargs):
        return super().solve(nsites, **solve_kwargs)

    def solve(self, nsites: int, max_coarseness=None, conv_tol=0, **kwargs):
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
        # print('HERE', self.coarseness_level, self.max_coarseness)
        ### if at maximum coarseness level ###
        if self.coarseness_level > self.max_coarseness:
            print('LINEAR SOLVE')
            self.linear_solve(nsites, conv_tol=conv_tol)
            return self.ket, self.err

        ### if don't have to "solve" (just compress) and target.max_bond < self.max_bond already
        elif self.num_operators == 0:
            target_max_bond = np.sum([t.max_bond() for t in self.targets])
            # print('target max bond', target_max_bond, self.max_bond)
            if self.max_bond is None or target_max_bond < self.max_bond:
                # print('target max bond < self.max bond')
                super().solve(nsites, conv_tol=conv_tol)
                return self.ket, self.err

        ### else coarsen and prolongate ###
        # print('coarsening', self.coarseness_level)
        integ_inds = self.coarseness_mps_inds[self.coarseness_level]
        left_canon_site = min(integ_inds)
        right_canon_site = max(integ_inds)
        self.canonize(left_canon_site)

        ## ovlp <Ax|b>, <Ax|Ax> init envs
        self._build_all_envs_right(nsites, end=right_canon_site, canonize=False)
        self._build_all_envs_left(nsites, end=left_canon_site, canonize=False)

        ## recursive solver
        coarse_sys = self.coarsen()
        print('COARSE SOLVE')
        coarse_sys.solve(nsites, max_coarseness=max_coarseness)
        print('PROLONGATE')
        # print('prolong target', [helper.norm(t) for t in self.targets])
        new_sys = self._prolongate_from_coarse(coarse_sys, inplace=True)    ## return system
            ## how would inplace work? i think it should be ok?
        err = self.check_err()

        # if err > 1.5:
        #     print('exit mg')
        #     exit()

        return new_sys.ket, err
