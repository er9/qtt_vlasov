import numpy as np
import itertools
import quimb

from setup.defaults import *
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2

from helper_tdvp import *
from helper_tdvp import _sum_eff_TNs
from helper_dmrg import *
from helper_tdvp_v2 import *

class BaseSolver(Enum):
    LOCAL = LocalSolver
    DMRG = DMRGSolver
    TDDMRG = TDDMRGSolver_v2
    LINEAR = LinearSolver
    LINEAR2 = helper_dmrg_2.LinearSolver


def block_tddmrg(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                 x_vecs: dict[int, 'qtn.MatrixProductState'],
                 b_vecs: dict[int, Sequence['MPS_type']] = None,
                 verbose_output=False,
                 init_direction=SweepDirection.RIGHT,
                 te_order=4, **solver_kwargs):
    print('BLOCK TD-DMRG 2')
    solver = BlockTDDMRGSolver(ncomps, x_vecs, b_vecs, A_mats, te_order=te_order, **solver_kwargs)
    solver.take_time_step(dt, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets



class BlockTDDMRGSolver:
    """ TD-DMRG solver with multiple components at once
    """

    def __init__(self, ncomps, x_state: dict[int, 'qtn.MatrixProductState'],
                 sources: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 operators_H: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 base_solver=BaseSolver.TDDMRG,
                 mps_inds: Optional[Sequence[int]] = None,
                 # left_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 # right_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL,
                 max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER,
                 te_order=4,
                 max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None, solvers=None,
                 **env_kwargs):
        """ each solver indexed by output ind, input ind (j, i)
                computes b_eff:  eg. d/dT*[s] <x_j | b_i>
                computes A_eff:  eg. d/dT*[s] d/dT[s] <x_j | O_ji | x_i>
                where x_j, operators_H don't have to be conj of x_i, operators
        """

        self.num_comps = ncomps
        if solvers is not None:  ## for copying
            pass

        else:

            solvers: dict[tuple[int, int], LocalSolver] = {}
            if x_state is None:     ## shouldn't be none
                x_state = {}

            ref_init_x = x_state[next(iter(x_state))]

            bra_states = {}
            for k, ket in x_state.items():
                helper.match_inner_inds(ket, ref_init_x, inplace=True)
                ket.site_ind_id = ref_init_x.site_ind_id

                bra = ket.conj()
                # bra.mangle_inner_(append='_')
                ## solver adds an extra '_' if this is here. move until after solvers are defined
                bra_states[k] = bra

            # for k, mpo in operators.items():
            #     print(k, mpo)

            # ## jank fix for MPS obtained from comb
            # check_phys_inds = False
            # if len(ref_init_x.shape) != ref_init_x.L:
            #     check_phys_inds = True

            # print('ref init x', ref_init_x)
            ref_op = operators[next(iter(operators))][0]
            for (oo, ii), As in operators.items():

                init_x = x_state.get(ii, None)
                init_bra = bra_states.get(oo, None)

                if ref_op is None:
                    try:
                        ref_op = next(op for op in As if op is not None)
                    except StopIteration:
                        raise StopIteration
                    # ref_op = As[0]
                else:
                    for i in range(len(As)):
                        if As[i] is None:  continue
                        As[i].upper_ind_id = ref_op.upper_ind_id + '_tmp'
                        As[i].lower_ind_id = ref_op.lower_ind_id
                        As[i].upper_ind_id = ref_op.upper_ind_id

                select_envs = {kw: (env.get(oo, None) if env is not None else None) for kw, env in env_kwargs.items()}
                ## input (ket) is the same if ii is the same, so envs only depend on oo

                ## set up solver
                # As_H = None if oo == ii else operators_H[(oo, ii)]
                # As_H = operators_H.get((oo, ii), None)
                print('o,i', oo, ii, [A is not None for A in As])
                print('ket is None?', init_x is None)
                print('bra is None?', init_bra is None)
                # if As_H is not None:  print([A is not None for A in As_H])
                if init_x is not None and init_bra is not None:

                    # if ii not in ref_ket_inds:
                    #     ref_ket_inds[ii] = (oo, ii)
                    # if oo not in ref_bra_inds:
                    #     ref_bra_inds[oo] = (oo, ii)

                    print('base solver', base_solver.value)
                    solver = base_solver.value(init_x, None, As, operators_H=None, bra_state=init_bra,
                                               in_ind=ii, out_ind=oo, # virtual=True,
                                               mps_inds=mps_inds, **select_envs,
                                               conv_tol=conv_tol, max_iter=max_iter,
                                               max_tot_iter=max_tot_iter, max_bond=max_bond)

                    print('oo', 'ii', oo, ii, init_bra.exponent, init_x.exponent)
                    print('oo', 'ii', oo, ii, solver.bra.exponent, solver.ket.exponent)

                    solvers[(oo, ii)] = solver

            self.current_state = x_state
            self.current_bras = bra_states
            for k, bra in self.current_bras.items():
                bra.mangle_inner_(append='_')

            self.b_envsLs, self.b_envsRs = {}, {}
            if sources is not None:
                # raise NotImplementedError
                for oo, bs in sources.items():

                    bLs, bRs = [], []
                    init_bra = bra_states.get(oo, None)
                    if init_bra is not None:

                        for ib in range(len(bs)):
                            b_envL = Environment(self.L, EnvironmentSide.LEFT, bs[ib], init_bra, None,)
                                                 # mps_inds=self.mps_inds,
                                                 # init_env=ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None)
                            b_envR = Environment(self.L, EnvironmentSide.RIGHT, bs[ib], init_bra, None,)
                                                 # mps_inds=self.mps_inds,
                                                 # init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None)
                            bLs += [b_envL]
                            bRs += [b_envR]

                    self.b_envsLs[oo] = bLs
                    self.b_envsRs[oo] = bRs

        self.te_order = te_order
        self.sources = sources
        self.solvers = solvers
        self.err = np.inf
        self.is_conv = False

        self.conv_tol = conv_tol if conv_tol is not None else DEFAULT_CONV_TOL
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_wrong_iter = max_wrong_iter
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
        all_left_envs.update(self.b_envsLs)
        return all_left_envs

    @property
    def right_envs(self):
        all_right_envs = {}
        for k, solver in self.solvers.items():
            all_right_envs[k] = solver.right_envs
        all_right_envs.update(self.b_envsRs)
        return all_right_envs

    @property
    def kets(self):
        all_kets = {}
        for (oo, ii), solver in self.solvers.items():
            all_kets[ii] = solver.ket
        return all_kets

    @property
    def ket(self):
        tmp = iter(self.current_state)
        while True:
            ket = self.current_state[next(tmp)]
            if ket is not None:
                break
        # solver = self.solvers[next(iter(self.solvers))]
        return ket

    @property
    def bra(self):
        ### do it tihs way for the correct index labels
        tmp = iter(self.solvers)
        while True:
            solver = self.solvers[next(tmp)]
            if solver.bra is not None:
                break
        # solver = self.solvers[next(iter(self.solvers))]
        return solver.bra

    @property
    def component_ind(self):
        return self.ket.site_ind_id

    def create_like(self, copy=True, **kwargs):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.create_like(copy=copy, **kwargs)
        new_solver = self.__class__(self.num_comps, self.current_state, solvers=new_solvers)
        new_solver.current_state = self.current_state
        new_solver.current_bras = self.current_bras
        new_solver.te_order = self.te_order
        new_solver.sources = self.sources
        return new_solver

    def copy(self, deep=True):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.copy(deep=deep)
        new_block_solver = self.__class__(self.num_comps, self.current_state, solvers=new_solvers)
        new_block_solver.err = self.err
        new_block_solver.is_conv = self.is_conv
        new_block_solver.current_state = self.current_state
        new_block_solver.current_bras = self.current_bras
        new_block_solver.te_order = self.te_order
        new_block_solver.sources = self.sources
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
        tmp = iter(self.solvers)
        while True:
            solver = self.solvers[next(tmp)]
            if solver.ket is not None and solver.bra is not None:
                break
        return solver.get_ket_to_bra_inds(i)

    def get_bra_to_ket_inds(self, i) -> dict[str, str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket

    def get_mps_ind(self, i: Union[int, Sequence[int], slice]):
        solver = self.solvers[next(iter(self.solvers))]
        if isinstance(i, (int, slice)):
            return solver.mps_inds[i]
        else:
            return [solver.mps_inds[ii] for ii in i]

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            if solver.ket is not None and solver.bra is not None:
                solver._build_all_envs_right(nsites, end=end, canonize=canonize)
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            if solver.ket is not None and solver.bra is not None:
                solver._build_all_envs_left(nsites, end=end, canonize=canonize)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            if solver.ket is not None and solver.bra is not None:
                solver._update_envs_left(pos, canonize=canonize)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            if solver.ket is not None and solver.bra is not None:
                solver._update_envs_right(pos, canonize=canonize)
        return

    ######

    def canonize(self, i, cur_orthog=None):
        ### relies on fact that canonicalization is done in the td-DMRG site solve

        for ik in self.current_state:
            out_mps = helper.canonize(self.current_state[ik], i=i, cur_orthog=cur_orthog, scale=False,
                                      bra=self.current_bras[ik])

            if out_mps is None:     ## o.w. is inplace
                self.current_state[ik] = None
                self.current_bras[ik] = None

        for k, solver in self.solvers.items():
            oo, ii = k
            bra_o = self.current_bras[oo]
            ket_i = self.current_state[ii]

            if bra_o is None:
                solver.bra = None
            else:
                for s in range(solver.L):
                    solver.bra[s].transpose_like(bra_o[s], inplace=True)
                    solver.bra[s].modify(data=bra_o[s].data)

            if ket_i is None:
                solver.ket = None
            else:
                for s in range(solver.L):
                    solver.ket[s].transpose_like(ket_i[s], inplace=True)
                    solver.ket[s].modify(data=ket_i[s].data)

        #     if solver.ket is not None:
        #         helper.check_orthog(solver.ket)
        #     if solver.bra is not None:
        #         helper.check_orthog(solver.bra)
        #
        # exit()

        return

    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor']):

        ket_to_bra_inds = self.get_ket_to_bra_inds(i)

        ## update self.current_state
        for ii, ket in self.current_state.items():
            try:
                new_ket_tens = site_i_dict[ii].copy()
                old_ket_tens = ket[i]
                new_ket_tens.transpose_like(old_ket_tens, inplace=True)
                old_ket_tens.modify(data = new_ket_tens.data)
            except KeyError:
                pass

        for ii, bra in self.current_bras.items():
            try:
                new_bra_tens = site_i_dict[ii].conj().reindex(ket_to_bra_inds)
                old_bra_tens = bra[i]
                new_bra_tens.transpose_like(old_bra_tens, inplace=True)
                old_bra_tens.modify(data=new_bra_tens.data)
            except KeyError:
                pass

        for (oo,ii), solver in self.solvers.items():
            try:
                new_ket_tens = site_i_dict[ii].copy()
                solver.update_ket_tens([new_ket_tens], sites=[i])  # , new_exponent=ref_k_solver.ket.exponent)
            except KeyError:
                pass

            try:
                new_bra_tens = site_i_dict[oo].conj().reindex(ket_to_bra_inds)
                solver.update_bra_tens([new_bra_tens], sites=[i])  # , new_exponent=ref_b_solver.bra.exponent)
            except KeyError:
                pass

            solver.reinitialize_envs_i(i)

        return

    def _update_2site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        raise NotImplementedError
        sites_1, sites_2 = {}, {}
        i2 = i + 1 if direction == SweepDirection.RIGHT else i - 1

        for k, tens in site_i_dict.items():
            ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i2)
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])  # shared, not shared
            site1, site2 = tens.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            sites_1[k] = site1
            sites_2[k] = site2

        self._update_1site(i, sites_1)
        self._update_1site(i2, sites_2)

        return

    ####

    def _get_b_eff_dict_(self, sites, bra_inds):
        """
        BL * B * BR = d/dT*[s] <x_j|b_i> d_ij
        dict indexes xj
        """
        b_eff_dict = {}
        ### d/dT*[s] <x_j| A_ji.T | b_i>
        for key, solver in self.solvers.items():
            if key[0] == key[1]:
                out = solver._get_b_eff(sites)
                if out is not None:
                    out.transpose(*bra_inds, inplace=True)
                    b_eff_dict[key[0]] = out
                else:
                    shape = [solver.bra.ind_size(b_ind) for b_ind in bra_inds]
                    # print('shape?', shape)
                    # exit()
                    b_eff_dict[key[0]] = qtn.Tensor(np.zeros(tuple(shape)), inds=bra_inds)
                # print('b eff size', key[0], b_eff_dict[key[0]].shape)
        return b_eff_dict

    def get_time_evolved_site(self, dt, sites, xeff_dict, Aeff_dict, beff_dict, bonds_i, bonds_o,
                              te_order=4) -> dict[int, 'qtn.Tensor']:
        """ dx/dt = Ax + b
        """
        if dt == 0:
            return xeff_dict

        def deriv_func(xeff_dict_, dt=1, **kwargs) -> dict[int,'qtn.Tensor']:
            eff_Ax_dict = {}
            for (oo, ii), A_effs in Aeff_dict.items():
                # state_exponent1 = self.solvers[(oo,ii)].ket.exponent - self.solvers[(oo,ii)].bra.exponent
                state_exponent = self.current_state[ii].exponent - self.current_bras[oo].exponent
                eff_Ax = []
                xeff = xeff_dict_.get(ii,None)
                if xeff is not None:
                    for deriv_tn in A_effs:
                        Ax_tn = qtn.TensorNetwork([deriv_tn, xeff])
                        ## only has exponent from A. need to include exponent from xeff ([ii])
                        ## and subtract exponent from output (oo) where xeff will fit into
                        # print('Ax_tn exponent', Ax_tn.exponent)
                        # print('state exponent', state_exponent, state_exponent1)
                        Ax_tn.exponent = Ax_tn.exponent + state_exponent
                        eff_Ax += [Ax_tn]
                        # print('eff Ax', oo, ii, eff_Ax[-1])
                        # print('self.ket', self.kets[ii])
                        # print('self.bra', self.kets[oo])
                        ## dist_submpx exponent should equal 0

                beff = beff_dict.get(oo, None)
                if beff is not None:
                    eff_Ax += [qtn.TensorNetwork([beff])]

                if oo not in eff_Ax_dict:
                    eff_Ax_dict[oo] = eff_Ax
                else:
                    eff_Ax_dict[oo] += eff_Ax

            out_dict = {}
            for oo, eff_Axs in eff_Ax_dict.items():
                # print('eff Axs', [(eff.exponent if eff is not None else None) for eff in eff_Axs])
                out = _sum_eff_TNs(eff_Axs, transpose_bonds=bonds_o)
                if out is not None:
                    # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
                    out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
                    out.modify(apply=lambda x: x * dt)
                    out_dict[oo] = out
            return out_dict

        def add_func(tens1_dict, tens2_dict, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens1_dict if inplace else {k: tens.copy() for k, tens in tens1_dict.items()}
            tot_keys = set(tens1_dict.keys()).union(tens2_dict.keys())
            for k in tot_keys:
                try:
                    tens2 = tens2_dict[k]
                    try:
                        tens1 = out_dict[k]
                        out_dict[k] = helper.add_tensors(tens1, tens2, inplace=True)
                    except IndexError:
                        out_dict[k] = tens2.copy()
                except IndexError:
                    pass
            return out_dict

        def scale_func(tens_dict, scale_val, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens_dict if inplace else {k: tens.copy() for k, tens in tens_dict.items()}
            for k, tens in out_dict.items():
                tens.modify(apply=lambda x: x * scale_val)
            return out_dict

        def euler_func(state, dt, deriv0, **kwargs) -> dict[int,'qtn.Tensor']:
            return add_func(state, scale_func(deriv0, dt))

        ### implicit funcs
        def scale_tn_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.TensorNetwork':
            tens = helper.scalar_multiply(tens, scale_val, inplace=inplace)
            return tens

        def solve_func_cgd(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()}

            out = self.cgd(sites, init_guess, deriv_ops, explicit_contribution,
                           max_iter=max_iter, conv_tol=conv_tol, **kwargs)
            # print('solve err', err)
            return out

        def solve_func_lsq(deriv_ops: dict[int,Sequence['qtn.TensorNetwork']],
                           explicit_contribution: dict[int,'qtn.Tensor'],
                           init_guess=None, max_iter=None, conv_tol=None, **kwargs):

            # shape_ = explicit_contribution.shape
            # size_ = int(np.prod(shape_))

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            deriv_ops_ = {k: _sum_eff_TNs(d_ops, [*bonds_o, *bonds_i]) for k, d_ops in deriv_ops.items()}
            # explicit_contribution = explicit_contribution.transpose(*bonds_i)
            # print('A eff tens', A_eff_tens, explicit_contribution)

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()}

            out = self.lsq(sites, init_guess, deriv_ops_, explicit_contribution,
                           max_iter=max_iter, conv_tol=conv_tol, **kwargs)
            return out

        def identity_func(xeff_dict: dict[int,Union['qtn.Tensor', 'qtn.TensorNetwork']]):
            iden_dict = {}
            for i, state in xeff_dict.items():
                shape_ = state.shape
                size_ = int(np.prod(shape_))
                iden = qtn.Tensor( np.eye(size_).reshape(*shape_, *shape_),
                                   inds=(*bonds_o,*bonds_i) )
                iden_dict[i] = qtn.TensorNetwork([iden])
            return iden_dict

        ### perform actual TE
        if te_order == 4:
            out = helper_TE.rk4(xeff_dict, dt, euler_func, deriv_func, add_func, scale_func)
        elif te_order == 22:
            out = helper_TE.solve_crank_nicolson(xeff_dict, euler_func, deriv_func, add_func, scale_tn_func,
                                                 Aeff_dict, identity_func, solve_func_lsq, dt)
        else:
            raise NotImplementedError

        return out


    def get_site_total_rdm(self, dt, left_site_pos, x_bond, nsites=1, **kwargs) -> dict[int,'qtn.Tensor']:
        """
        """
        site_inds = self.get_mps_ind(slice(left_site_pos,left_site_pos + nsites,1))
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
        for key, solver in self.solvers.items():
            ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
            if solver.ket is not None and solver.bra is not None:
                A_effs_dict[key] = solver._get_A_effs(left_site_pos, nsites)
                # print('A effs', key, A_effs_dict[key], [eff.exponent for eff in A_effs_dict[key]])
                # print('solver ket', key[1], solver.ket.exponent)
                # print('solver bra', key[0], solver.bra.exponent)
                # print('ops', [op.exponent for op in solver.operators])

        b_eff_dict = self._get_b_eff_dict_(site_inds, bonds_o)

        x_dict = {}
        for k, ket in self.kets.items():
            if ket is not None:
                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [ket[si] for si in site_inds]
                else:
                    ket_tensors = ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*bonds_i, inplace=True)
                x_dict[k] = x_tens

        ## bond to contract over
        def compute_denmat(ket_dict: dict[int,'qtn.Tensor']):
            denmat_dict = {}
            for k_, ket_ in ket_dict.items():
                bra_ = ket_.conj()
                bra_.reindex({kk: b for kk, b in ket_to_bra_inds.items() if kk != x_bond}, inplace=True)
                denmat_dict[k_] = qtn.tensor_contract(bra_, ket_)
            return denmat_dict

        rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
        rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]

        target_times = [0, dt/3, 2*dt/3, dt]
        weights = [1./3, 1./6, 1./6, 1./3]
        # target_times = [0, dt/3, 2*dt/3, 4*dt/3, dt]
        # weights = [1./3, 1./9, 1./9., 1./9, 1./3]
        # target_times = [0, (1 - np.sqrt(3)/2) * dt , dt, (1 + np.sqrt(3)/2) * dt ]
        # weights = [1./4, 1./4, 1./4, 1./4]
        # target_times = [dt]
        # weights = [1.]

        ## for accumulated time steps
        sort_inds = np.argsort(target_times)
        target_times = [0] + [target_times[i + 1] - target_times[i] for i in sort_inds[:-1]]
        weights = [weights[i] for i in sort_inds]

        tot_denmat = {}
        for w, tt in zip(weights, target_times):
            new_xeff_dict = self.get_time_evolved_site(tt, site_inds, x_dict, A_effs_dict, b_eff_dict,
                                                       bonds_i, bonds_o)
            x_dict = new_xeff_dict       ## for accumulated time steps
            # for k, val in new_xeff_dict.items():
            #     print('x eff', k, val)
            #     print(self.current_state[k])

            rho0_dict = compute_denmat(x_dict)
            for k, rho0 in rho0_dict.items():
                rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|
                rho0.modify(apply=lambda x: x * w)

            for k in rho0_dict.keys():
                if k not in tot_denmat:
                    tot_denmat[k] = rho0_dict[k].copy()
                else:
                    tot_denmat[k].modify(apply=lambda x: x + rho0_dict[k].data)

        return tot_denmat


    # @profile
    def _site_time_evolution(self, dt,left_site_pos, nsites=1,
                             compress_direction=CompressDirection.RIGHT, ):
        """
        sites: int or slice(start, stop, step)
        solve local A' x = b',  (1/2?) <x|Ax> - <x|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A|x> = d/dT[i] d/dT[i]^* \sum_ij <x_j|A_ji|x_i>
            where b' = d/dT[i]^* <x|b> = d/dT[i] \sum_ij <x_j|b_i>
        """
        # print("SITE SOLVE", self.solve_type, sites)
        sites = list( range(left_site_pos, left_site_pos + nsites) )
        site_pos = sites
        site_inds = self.get_mps_ind(sites)

        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        if compress_direction == CompressDirection.RIGHT:
            sweep_dir = SweepDirection.RIGHT
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            sweep_dir = SweepDirection.LEFT
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        # print('self ket', self.ket)
        # print('self bra', self.bra)

        bra_inds = [self.bra.site_ind_id.format(i) for i in site_inds]
        min_ind, max_ind = min(site_inds), max(site_inds)
        if min_ind > 0:
            bra_inds += [self.bra.bond(min_ind - 1, min_ind)]
        if max_ind < self.L - 1:
            bra_inds += [self.bra.bond(max_ind, max_ind + 1)]

        bra_to_ket_inds = {} # self.bra.site_ind_id.format(i): self.ket.site_ind_id.format(i) for i in site_inds}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        # print('left site pos', left_site_pos, nsites, compress_direction)
        # print('ind1', ind1, ind2, at_end)
        # print('bra inds', bra_inds)
        # print('bra-to-ket', bra_to_ket_inds)

        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        bonds_o = bra_inds
        bonds_i = ket_inds

        ## targeted time steps
        if not at_end:

            ## bond to contract over
            if compress_direction == CompressDirection.RIGHT:
                x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
            else:
                x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

            ## solve for T[i]
            tot_denmat_dict = self.get_site_total_rdm(dt, left_site_pos, x_bond, nsites=nsites)

            updated_ket_1 = {}
            updated_ket_2 = {}
            for k, tot_denmat in tot_denmat_dict.items():

                rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
                rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
                nb = len(rho_bonds_o)

                # print('rho bonds i', rho_bonds_i)
                # print('rho bonds o', rho_bonds_o)
                # print('tot denmat')
                # print(k, tot_denmat)

                tot_denmat.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)

                denmat_shape = tot_denmat.shape
                tens_shape = denmat_shape[:nb]
                sq_shape = np.prod(denmat_shape[:nb])

                eigval, eigvec = np.linalg.eigh(tot_denmat.data.reshape(sq_shape, sq_shape))

                sort_inds = np.argsort(np.abs(eigval))[::-1]
                ev_max = eigval[sort_inds[0]]
                # eigval_inds = np.where(np.abs(eigval) > 1.0e-12 * ev_max)
                new_sort_inds = []
                for si in sort_inds:
                    if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
                        new_sort_inds += [si]
                    else:
                        break
                sort_inds = new_sort_inds

                if self.max_bond is not None:
                    if self.max_bond // nsites > 0:
                        eigvec = eigvec[:, sort_inds[:self.max_bond // nsites]]

                eigvec = eigvec.reshape(*tens_shape, -1)
                eigvec = qtn.Tensor(data=eigvec, inds=tuple(rho_bonds_i) + (x_bond + '_',))

                ket = self.current_state[k]
                if nsites == 1:
                    eigvec.transpose_like(ket[ind1], inplace=True)
                    # print('ket1', ket[ind1])
                    # print('ket2', ket[ind2])
                    # print('eigvec', eigvec)
                    next_site = qtn.tensor_contract(eigvec.conj(), ket[ind1], ket[ind2])
                    next_site.transpose_like(ket[ind2], inplace=True)
                    # ket[ind1].modify(data=eigvec.data)
                    # ket[ind2].modify(data=next_site.data)

                    eigvec.modify(inds = ket[ind1].inds)
                    next_site.modify(inds = ket[ind2].inds)
                    updated_ket_1[k] = eigvec  # ket[ind1]
                    updated_ket_2[k] = next_site  # ket[ind2]

                elif nsites == 2:
                    ## decompose eigvec into two sites, tensor contract ind1 site
                    bond_ms, left_inds = ket[ind1].filter_bonds(ket[ind2])
                    bond_m = next(iter(bond_ms))
                    ##
                    ev1, ev2 = qtn.tensor_split(eigvec, left_inds, absorb='right', bond_ind=bond_m + '_',
                                                max_bond=self.max_bond)
                    next_site = qtn.tensor_contract(ev1.conj(), ket[ind1], ket[ind2])
                    ev1.transpose_like(ket[ind1], inplace=True)
                    next_site.transpose_like(ket[ind2], inplace=True)
                    ev1.modify(inds = ket[ind1].inds)
                    next_site.modify(inds=ket[ind2].inds)
                    # ket[ind1].modify(data=ev1.data)
                    # ket[ind2].modify(data=next_site.data)
                    # print('ind1', ind1, ind2)
                    # helper.check_orthog(self.ket)

                    updated_ket_1[k] = ev1  # ket[ind1]
                    updated_ket_2[k] = next_site  # ket[ind2]
                else:
                    raise NotImplementedError

            ## i think this is already done in canonize
            self._update_1site(ind1, updated_ket_1)
            self._update_1site(ind2, updated_ket_2)

        else:

            A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
            for key, solver in self.solvers.items():
                ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
                A_effs_dict[key] = solver._get_A_effs(left_site_pos, nsites)
                # if self.solve_type is not SolveMethod.CGD:
                #     A_effs_dict[key].transpose(*bonds_o, *bonds_i, inplace=True)

            b_eff_dict = self._get_b_eff_dict_(site_inds, bonds_o)

            x_dict = {}
            for k, ket in self.kets.items():
                if ket is not None:
                    if isinstance(site_inds, (list, tuple)):
                        ket_tensors = [ket[si] for si in site_inds]
                    else:
                        ket_tensors = ket[site_inds]
                    x_tens = qtn.tensor_contract(*ket_tensors)
                    x_tens.transpose(*bonds_i, inplace=True)
                    x_dict[k] = x_tens

            new_xeff_dict = self.get_time_evolved_site(dt, site_inds, x_dict, A_effs_dict, b_eff_dict,
                                                       bonds_i, bonds_o, te_order=self.te_order)

            # i think this is already done in canonize
            if nsites == 1:
                self._update_1site(left_site_pos, new_xeff_dict)
            elif nsites == 2:
                self._update_2site(left_site_pos, new_xeff_dict, direction=sweep_dir)
            else:
                raise NotImplementedError


        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            ## canonicalization is not needed here, but is used by subclass TDDMRGSolver_v3 for ket0, bra0
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2):
                self._update_envs_left(i, canonize=False)

            # for k, solver in self.solvers.items():
            #     print('solver (comp R)', k, 'ind2', ind2, self.ket.L)
            #     if solver.ket is not None and solver.bra is not None:
            #         helper.check_orthog(solver.ket)
            #         helper.check_orthog(solver.bra)

        else:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2, -1):
                self._update_envs_right(i, canonize=False)

            # for k, solver in self.solvers.items():
            #     print('solver (comp L)', k, 'ind2', ind2, self.ket.L)
            #     if solver.ket is not None and solver.bra is not None:
            #         helper.check_orthog(solver.ket)
            #         helper.check_orthog(solver.bra)

        return


    # @profile
    def take_time_step(self, dt, grid=None, do_adapt=False, init_direction=SweepDirection.RIGHT, **kwargs):
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
        L = self.ket.L
        # print('L', self.L, self.ket.L)
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(1, canonize=False)

            ### left to right sweep
            self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### right to left sweep
            self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        else:
            self._build_all_envs_left(1, canonize=False)

            ### right to left sweep
            self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### left to right sweep
            self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done sweep')

        return self.kets

    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r block tddmrg', dt, 'do adapt', do_adapt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = CompressDirection.RIGHT
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                if verbose:  print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = max_bond  # min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                max_bond_ = min(max_bond, 2**(site_ind+1), 2**(L-site_ind-1))
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind += 1
                    continue

            nsites = 2 if adapt else 1

            ## forward propagation
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
            self._site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.L
        # do_adapt = False
        max_bond = self.max_bond
        print('r2l tdvp', dt, 'do adapt', do_adapt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = CompressDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                if verbose:   print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                 max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                 adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

            if not adapt:  # one site update of site_ind
                if nsites == 2:  ## previously updated this site
                    nsites = 1
                    site_ind -= 1
                    continue

            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
            self._site_time_evolution(dt, cur_orthog, nsites=nsites, compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites

        print('done right to left sweep')
        return self.ket


