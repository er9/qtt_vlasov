import numpy as np

from setup.defaults import *
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2

from helper_dmrg import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gridTN import GridTN

""" implicit solver for Ax = b, where x, b are composed of multiple components
"""
class BaseSolver(Enum):
    LOCAL = LocalSolver
    DMRG = DMRGSolver
    LINEAR = LinearSolver
    LINEAR2 = helper_dmrg_2.LinearSolver



def implicit_solver(ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                    b_vecs: dict[int, Sequence['MPS_type']],
                    init_guess: dict[int, Sequence['MPS_type']] = None, nsites=2, verbose_output=False,
                    **solver_kwargs):

    solver = BlockedLinearSolver(ncomps, init_guess, b_vecs, A_mats, **solver_kwargs)
    solver.solve(nsites)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets



class BlockedLocalSolver:
    """ DMRG solver with multiple components at once
    """
    def __init__(self, ncomps, trial_state: dict[int,'qtn.MatrixProductState'],
                 targets: Optional[dict[int,Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int],Sequence['MPO_type']]] = None,
                 base_solver = BaseSolver.LOCAL,
                 mps_inds: Optional[Sequence[int]] = None,
                 # left_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 # right_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL,
                 max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solvers=None,
                 **env_kwargs):

        self.num_comps = ncomps
        if solvers is not None:     ## for copying
            pass

        else:
            solvers: dict[tuple[int,int], LocalSolver] = {}
            if trial_state is None:
                if targets is not None:
                    trial_state = {k: bs[0].copy() for k, bs in targets.items()}
                else:
                    raise NotImplementedError

            ref_init_x = None
            ref_op = None
            for (oo,ii), As in operators.items():

                bs = targets.get(oo, None)

                if bs is None and ii not in trial_state:
                    continue

                init_x = trial_state.get(ii, None)
                if init_x is None:  init_x = bs[0].copy()
                # init_x.expand_bond_dimension(max_bond_x, inplace=True)
                # print('max bond x', max_bond_x)
                # print('init_x', init_x.max_bond(), init_x)

                #### !!! match inner inds, bond dimensions of init_x !!! ####
                if ref_init_x is None:
                    ref_init_x = init_x
                else:
                    helper.match_inner_inds(init_x, ref_init_x, inplace=True)
                    init_x.site_ind_id = ref_init_x.site_ind_id

                if ref_op is None:
                    ref_op = As[0]
                else:
                    As[0].upper_ind_id = ref_op.upper_ind_id + '_tmp'
                    As[0].lower_ind_id = ref_op.lower_ind_id
                    As[0].upper_ind_id = ref_op.upper_ind_id

                select_envs = {kw: (env.get(oo, None) if env is not None else None) for kw, env in env_kwargs.items()}

                ## set up solver
                solver = base_solver.value(init_x, bs, As, # virtual=virtual,
                                           mps_inds=mps_inds, **select_envs,
                                           conv_tol=conv_tol, max_iter=max_iter,
                                           max_tot_iter=max_tot_iter, max_bond=max_bond)
                print('solver', solver)
                solvers[(oo,ii)] = solver

        self.solvers = solvers
        self.err = np.inf
        self.is_conv = False

        self.conv_tol = conv_tol if conv_tol is not None else DEFAULT_CONV_TOL
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_bond = max_bond

    @property
    def L(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k].mps_inds)  # self.mps_inds[1] - self.mps_inds[0]

    # @property
    # def max_bond(self):
    #     return np.max([solver.max_bond for k, solver in self.solvers.items()])

    @property
    def num_targets(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k]._targets)

    @property
    def num_operators(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k]._operators)

    @property
    def left_envs(self):
        all_left_envs = {}
        for k, solver in self.solvers.items():
            all_left_envs[k] = solver.left_envs
        return all_left_envs

    @property
    def right_envs(self):
        all_right_envs = {}
        for k, solver in self.solvers.items():
            all_right_envs[k] = solver.right_envs
        return all_right_envs

    @property
    def kets(self):
        all_kets = {}
        for (oo,ii), solver in self.solvers.items():
            all_kets[ii] = solver.ket
        return all_kets

    @property
    def ket(self):
        solver = self.solvers[next(iter(self.solvers))]
        return solver.ket

    @property
    def bra(self):
        solver = self.solvers[next(iter(self.solvers))]
        return solver.bra

    @property
    def component_ind(self):
        return '_comp_ind_'

    def create_like(self, copy=True, **kwargs):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.create_like(copy=copy, **kwargs)
        return self.__class__(self.num_comps, None, solvers=new_solvers)

    def copy(self, deep=True):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.copy(deep=deep)
        new_block_solver = self.__class__(self.num_comps, None, solvers=new_solvers)
        new_block_solver.err = self.err
        new_block_solver.is_conv = self.is_conv
        return new_block_solver

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for k, solver in self.solvers.items():
            solver.reinitialize_envs()

        self.err = np.inf
        self.is_conv = False

    def reinitialize_envs_i(self, i):
        """ if boundary envs are None, uses old values
        """
        for k, solver in self.solvers.items():
            solver.reinitialize_envs_i(i)

    def get_ket_to_bra_inds(self, i) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        solver = self.solvers[next(iter(self.solvers))]
        return solver.get_ket_to_bra_inds(i)

    def get_bra_to_ket_inds(self, i) -> dict[str, str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket

    def canonize(self, i, cur_orthog=None):
        for k, solver in self.solvers.items():
            solver.canonize(i, cur_orthog=cur_orthog)

    def compress(self, i, cur_orthog=None, compress_opts=None):
        for k, solver in self.solvers.items():
            solver.compress(i, cur_orthog=cur_orthog, compress_opts=compress_opts)

    def left_canonize_site(self, i):
        for k, solver in self.solvers.items():
            solver.left_canonize_site(i)

    def right_canonize_site(self, i):
        for k, solver in self.solvers.items():
            solver.right_canonize_site(i)

    def get_mps_ind(self, i):
        solver = self.solvers[next(iter(self.solvers))]
        return solver.mps_inds[i]

    #########################

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', old_x_ind_sizes=None):
        """ update ket
            i: int of mps site
            site_i:  has an extra ind indexing the component
        """
        for key, solver in self.solvers.items():
            out_ind, in_ind = key
            site_i_k = site_i.isel({self.component_ind: in_ind})
            if old_x_ind_sizes is not None:
                ind_size_dict = old_x_ind_sizes[in_ind]
                helper.tensor_adjust_shape(site_i_k, ind_size_dict)
            solver._update_1site(i, site_i_k, direction=direction)
        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', old_x_ind_sizes=None):
        """ update ket, bra with new_site
            i: mps_site
            site_i:  has an extra ind indexing the component
        """
        for key, solver in self.solvers.items():
            out_ind, in_ind = key
            site_i_k = site_i.isel({self.component_ind: in_ind})
            if old_x_ind_sizes is not None:
                ind_size_dict = old_x_ind_sizes[in_ind]
                helper.tensor_adjust_shape(site_i_k, ind_size_dict)
            solver._update_2site(i, site_i_k, direction=direction)
        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            solver._build_all_envs_right(nsites, end=end, canonize=canonize)
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            solver._build_all_envs_left(nsites, end=end, canonize=canonize)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            solver._update_envs_left(pos, canonize=canonize)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            solver._update_envs_right(pos, canonize=canonize)
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def check_err(self, *args, **kwargs) -> 'float':
        """ also updates self.err, self.is_conv
        """
        raise NotImplementedError



class BlockedLinearSolver(BlockedLocalSolver):
    def __init__(self, ncomps, trial_state: dict[int,'qtn.MatrixProductState'],
                 targets: Optional[dict[int,Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int,int],Sequence['MPO_type']]] = None,
                 is_H = True,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 norm_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Ls: dict[tuple[int,int], Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Rs: dict[tuple[int,int], Sequence['qtn.Tensor']] = None,
                 err_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 err_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solve_type:SolveMethod=DEFAULT_SOLVE,
                 solvers=None,
                 ):


        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        init_state = trial_state
        super().__init__(ncomps, init_state, targets=targets, operators=operators, mps_inds=mps_inds, conv_tol=conv_tol,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter,
                         base_solver=BaseSolver.LINEAR,
                         norm_env0_Rs=norm_env0_Rs, norm_env0_Ls=norm_env0_Ls,
                         ovlp_env0_Rs=ovlp_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
                         err_env0_Rs=err_env0_Rs, err_env0_Ls=err_env0_Ls,
                         solvers=solvers)

        for k, solver in self.solvers.items():
            solver.is_H = is_H
            solver.solve_type = solve_type

        self.solve_type = solve_type
        self.is_H = is_H


    # @profile
    def _site_solve(self, sites: Union[int,slice],) -> tuple[qtn.Tensor,Sequence]:
        """
        sites: int or slice(start, stop, step)
        """
        # print("SITE SOLVE", self.solve_type, sites)
        site_pos = list(range(self.L))[sites]
        site_inds = self.get_mps_ind(sites)
        # site_inds = sites if isinstance(sites, slice) else slice(sites)


        A_effs_dict, b_eff_dict = {}, {}
        for key, solver in self.solvers.items():
            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs_dict[key] = solver._get_A_effs(sites, return_sum=True)
            ### BL * B * BR = d/dT*[i] <x|b>
            b_eff_dict[key] = solver._get_b_eff(sites)


        def combine_Aeffs_TN(As_dict: dict[tuple[int,int],Sequence[qtn.TensorNetwork]], outer_inds):
            """ get combined A_eff (TN) for each operator in self.operators
            """
            A_effs_list = []
            for i in range(self.num_operators):
                sum_A_eff = None
                for k, A_effs in As_dict.items():
                    A_eff = A_effs[i].copy()
                    index_array = np.zeros((self.num_comps, self.num_comps))
                    index_array[k[0], k[1]] = 1
                    index_tens = qtn.Tensor(index_array,
                                            tags='comp', inds=(self.component_ind + '_', self.component_ind))
                    A_eff.add(index_tens)

                    if sum_A_eff is None:
                        sum_A_eff = A_eff
                    else:
                        for t in sum_A_eff.tensors:
                            t1 = sum_A_eff.select_tensors(t.tags)[0]
                            sum_inds = tuple([o_ind for o_ind in outer_inds if o_ind in t1.inds]) + \
                                       (self.component_ind, self.component_ind + '_')
                            t.direct_product(t1, sum_inds= sum_inds, inplace=True)

                A_effs_list += [sum_A_eff]

            return A_effs_list


        def combine_Aeffs_tens(As_dict: dict[tuple[int, int], qtn.Tensor], bra_inds, ket_inds):
            """ get combined A_eff (tens) for each operator in self.operators
                d/dx_o,k d/dx_i,k <x_o |A_oi| x_i>
            """
            sum_A_eff = None
            sum_inds = [self.component_ind + '_'] + bra_inds + [self.component_ind] + ket_inds
            for (oo,ii), A_eff in As_dict.items():
                index_array = np.zeros((self.num_comps, self.num_comps))
                index_array[oo, ii] = 1
                index_tens = qtn.Tensor(index_array,
                                        tags='comp', inds=(self.component_ind + '_', self.component_ind))
                A_eff = A_eff.contract(index_tens)
                A_eff.transpose(*sum_inds, inplace=True)

                if sum_A_eff is None:
                    sum_A_eff = A_eff
                else:
                    # sum_A_eff.direct_product(A_eff, sum_inds=sum_inds, inplace=True)
                    helper.tensor_direct_product(sum_A_eff, A_eff, sum_inds=sum_inds, inplace=True,
                                                 force_match=True)

            return sum_A_eff


        def combine_beffs_tens(bs_dict: dict[[int,int], qtn.Tensor], outer_inds):
            """ get combined b_eff (tens),
                b_effs are already summed over targets in self.target
                sum_i d/dx_o,k <x_o |A_oi| b_i>
            """
            sum_b_eff = None
            sum_inds = [self.component_ind + '_',] + outer_inds
            for (oo,ii), b_eff in bs_dict.items():
                index_array = np.zeros((self.num_comps,))
                index_array[oo] = 1
                index_tens = qtn.Tensor(index_array,
                                        tags='comp', inds=(self.component_ind + '_',))
                b_eff = b_eff.contract(index_tens)
                b_eff.transpose(*sum_inds, inplace=True)

                if sum_b_eff is None:
                    sum_b_eff = b_eff
                else:
                    # sum_b_eff.direct_product(b_eff, sum_inds=sum_inds, inplace=True)
                    helper.tensor_direct_product(sum_b_eff, b_eff, sum_inds=sum_inds, inplace=True,
                                                 force_match=True)

            return sum_b_eff


        def combine_xeff_tens(xs_dict: dict[int, qtn.Tensor], outer_inds):
            """ get combined b_eff (tens),
                b_effs are already summed over targets in self.target
            """
            sum_x_eff = None
            sum_inds = [self.component_ind,] + outer_inds
            for ii, x_eff in xs_dict.items():
                index_array = np.zeros((self.num_comps,))
                index_array[ii] = 1
                index_tens = qtn.Tensor(index_array,
                                        tags='comp', inds=(self.component_ind,))
                x_eff = x_eff.contract(index_tens)
                x_eff.transpose(*sum_inds, inplace=True)

                if sum_x_eff is None:
                    sum_x_eff = x_eff
                else:
                    helper.tensor_direct_product(sum_x_eff, x_eff, sum_inds=sum_inds, inplace=True,
                                                 force_match=True)
            return sum_x_eff


        ## combine As, bs
        start = np.min(site_pos) if isinstance(site_pos, list) else site_pos
        end = np.max(site_pos) if isinstance(site_pos, list) else site_pos

        bra_inds = []  # [self.component_ind + '_']
        if start > 0:
            bra_inds += [self.bra.bond(start - 1, start)]
        if end < self.L - 1:
            bra_inds += [self.bra.bond(end, end + 1)]
        bra_inds += [self.bra.site_ind_id.format(i) for i in site_inds]

        bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        old_x_shape = {}
        for ki, ket in self.kets.items():
            ind_size_dict = {}
            if start > 0:
                ind_size_dict[ket.bond(start - 1, start)] = ket.bond_size(start-1, start)
            if end < self.L - 1:
                ind_size_dict[ket.bond(end, end + 1)] = ket.bond_size(end, end + 1)
            old_x_shape[ki] = ind_size_dict

        combined_A_eff = combine_Aeffs_tens(A_effs_dict, bra_inds, ket_inds)
        combined_b_eff = combine_beffs_tens(b_eff_dict, bra_inds)


        ## solve for T[i]
        if combined_A_eff is None:
            x_eff = combined_b_eff.reindex(bra_to_ket_inds)
            x_eff.modify(apply= lambda x : x * 10**(-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type is SolveMethod.CGD:
                # A_eff_exponent = A_eff.exponent
                x_dict = {}
                for k, ket in self.kets.items():
                    if isinstance(site_inds, (list, tuple)):
                        ket_tensors = [ket[si] for si in site_inds]
                    else:
                        ket_tensors = ket[site_inds]
                    x_tens = qtn.tensor_contract(*ket_tensors)
                    x_dict[k] = x_tens
                x_tens = combine_xeff_tens(x_dict, ket_inds)
                # x_tens.transpose(self.component_ind, *ket_inds, inplace=True)

                # x_tens.modify(apply=lambda x: x * 0.)
                if self.is_H:
                    x_eff, error = qtn_conjugate_gradient_descent_1site([combined_A_eff], combined_b_eff, x_tens,
                                                                        bra_to_ket_inds)
                else:
                    x_eff, error = qtn_conjugate_gradient_squared_1site([combined_A_eff], combined_b_eff, x_tens,
                                                                        bra_to_ket_inds)
                # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))

            elif self.solve_type is SolveMethod.CGDx:

                x_dict = {}
                for k, ket in self.kets.items():
                    if isinstance(site_inds, (list, tuple)):
                        ket_tensors = [ket[si] for si in site_inds]
                    else:
                        ket_tensors = ket[site_inds]
                    x_tens = qtn.tensor_contract(*ket_tensors)
                    x_dict[k] = x_tens
                x_tens = combine_xeff_tens(x_dict, ket_inds)

                # A_eff_exponent = A_eff.exponent
                out_shape = combined_A_eff.shape[:len(bra_inds) + 1]    ## include component ind
                in_shape = combined_A_eff.shape[len(bra_inds) + 1:]
                A_eff_data = combined_A_eff.data.reshape(np.prod(out_shape), -1)

                if combined_b_eff is not None:
                    b_eff_data = combined_b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                print('A eff', A_eff_data.shape, np.linalg.norm(A_eff_data))
                print('beff', np.linalg.norm(b_eff_data))
                print('xtens', x_tens.norm())

                if False:  # self.is_H:
                    x_eff_data, error = conjugate_gradient_descent(A_eff_data, b_eff_data, x=x_tens.data.reshape(-1))
                else:
                    x_eff_data, error = conjugate_gradient_squared(A_eff_data, b_eff_data, x=x_tens.data.reshape(-1))

                x_eff_data = x_eff_data.reshape(in_shape)
                x_eff = qtn.Tensor(data=x_eff_data, inds=tuple([self.component_ind] + ket_inds))

            else:   ## ALS optimzation
                out_shape = combined_A_eff.shape[:len(bra_inds)]
                in_shape = combined_A_eff.shape[len(bra_inds):]
                A_eff_data = combined_A_eff.data.reshape(np.prod(out_shape),-1)

                if combined_b_eff is not None:
                    b_eff_data = combined_b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
                x_eff_data = x_eff_data.reshape(in_shape)
                # x_eff_data *= 10 ** (-A_eff_exponent)
                ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

        return x_eff, old_x_shape


    # @profile
    def solve(self, nsites: int, init_direction=SweepDirection.RIGHT, conv_tol=0,
              skip_inds: set = None, opt_inds: set = None, verbose=False, **kwargs):
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
        # verbose = True
        conv_tol = self.conv_tol if conv_tol == 0 else conv_tol
        L = self.ket.L
        # print('L', self.L, self.ket.L)
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L-1
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(nsites, canonize=False)
        else:
            self._build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        print('dmrg self.max_bond', self.max_bond, self.ket.max_bond())
        if self.max_bond is None or self.ket.max_bond() < self.max_bond:
            print('getting error')
            err = self.check_err(dense=True)        ## true is actually faster for large MPO bond dim?
            print('init error dmrg', err, self.err)
        else:
            err = self.err

        conv_it, prev_err = 0, err

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = init_direction
        while err > conv_tol and conv_it < self.max_iter and it < self.max_tot_iter:

            it += 1

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites + 1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i,i+nsites)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({ix + iix for iix in range(nsites)}):
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds, self.L)
                        # print(opt_inds.intersection({ix + iix for iix in range(nsites)}))
                        if not opt_inds.intersection({self.get_mps_ind(i + iix) for iix in range(nsites)}):
                            # print('opt', i, ix, {self.get_mps_ind(i + iix) for iix in range(nsites)}, opt_inds)
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    # print('update L', inds)
                    site_i, old_x_ind_sizes = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site(i, site_i, direction, old_x_ind_sizes)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction, old_x_ind_sizes)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)

                self.canon_direction = EnvironmentSide.LEFT
                err = self.check_err(dense=True)

            else:
                ## left sweep
                for i in range(L - 1, nsites - 2, -1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i - nsites + 1, i + 1)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({self.get_mps_ind(i - iix) for iix in range(nsites)}):
                            self.right_canonize_site(i)     # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds)
                        # print(opt_inds.intersection({ix - iix for iix in range(nsites)}))
                        if not opt_inds.intersection({ix - iix for iix in range(nsites)}):
                            self.right_canonize_site(i)     # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    # print('update R', inds)
                    site_i, old_x_ind_sizes = self._site_solve(inds)

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(i, site_i, direction, old_x_ind_sizes)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction, old_x_ind_sizes)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)

                self.canon_direction = EnvironmentSide.RIGHT
                err = self.check_err(dense=True)

            if verbose:
                print('err', it, err)
                # print('err np', self.check_err_np())
            direction *= -1  ## swaps sweep direction

            if np.abs((prev_err - err)/err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err:
                min_solver = self.copy(deep=True)
                # min_ket = self.ket.copy()
                min_err = err
            else:
                print('Warning: solve error went up', err, min_err)
                # helper.replace_mps(self._ket, old_solver.ket)
                # helper.replace_mps(self._bra, old_solver.bra)
                # # self.ket = old_solver.ket
                # # self.reinitialize_envs()
                # self.A_envsL = old_solver.A_envsL
                # self.A_envsR = old_solver.A_envsR
                # self.b_envsRs = old_solver.b_envsRs
                # self.b_envsLs = old_solver.b_envsLs
                # self.left_envs = old_solver.left_envs
                # self.right_envs = old_solver.right_envs
                #
                # print('back to err', self.check_err())
                # print('numpy err', self.check_err_np())
                # print('fail direction', direction*-1)
                # # return old_solver.ket
                # break
                ## if not breaking, would need to recanonicalize, recalc envs

        # conv = min_err < conv_tol

        ## revert to optimal results
        if err > min_err:
            self._ket = min_solver.ket
            self._bra = min_solver.bra
            self.A_envsRs = min_solver.A_envsRs
            self.A_envsLs = min_solver.A_envsLs
            self.b_envsLs = min_solver.b_envsLs
            self.b_envsRs = min_solver.b_envsRs
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            self.canon_direction = min_solver.canon_direction

        return self.ket, self.err, self.is_conv


    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)


    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)


    def check_err_np(self):
        """
        """
        return


    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
            for each output component:
                0 = \sum_i A_oi |x_i> - |b_o>
                \sum_i <A_oi x_i | A_oi x_i> - 2Re[<b_o|A_oi|x_i>] + <b_o|b_o>
        """
        dense = False
        if dense:
            raise NotImplementedError

        else:
            normAAs = {}   # key o: list [<A_oi x_i | A_oi x_i>]
            ovlps = {}     # key o: list [<b_o|A_oi|x_i>]
            b2_norms = {}  # key o

            for (oo,ii), solver in self.solvers.items():
                print('solver', oo, ii)

                norm_Ax = solver._get_xAAx_()
                ovlp_b_tot = solver._get_xAb_()
                b_norm = solver.targets_norm

                if oo not in normAAs:
                    normAAs[oo] = norm_Ax
                else:
                    normAAs[oo] += norm_Ax

                if oo not in ovlps:
                    ovlps[oo] = 2 * np.real(ovlp_b_tot)
                else:
                    ovlps[oo] += 2 * np.real(ovlp_b_tot)

                if oo not in b2_norms:
                    b2_norms[oo] = b_norm**2

            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err_o = {oo: (normAAs[oo] + b2_norms[oo] - ovlps[oo])/b2_norms[oo] for oo in b2_norms.keys()}
            print('errs_o', err_o, b2_norms.keys())
            err2 = np.sum([np.abs(val) for oo, val in err_o.items()])
            err = np.sqrt(np.abs(err2))

        print('err', err, self.conv_tol)
        self.err = err
        self.is_conv = err < self.conv_tol

        return err







