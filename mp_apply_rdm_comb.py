import os
import quimb
from setup.configs import *
import helper_quimb as helper
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from helper_dmrg import Environment, EnvironmentSide
from grid import Grid
from grid1D import Grid1D
from gridTN import GridTN
from gridTN_composite import GridTN_Composite
from gridTN_1D import GridTN1D
from grid_comb import GridsComb
from gridTN_1Dcomb import GridTN1DComb

from multiprocessing import Process, Queue, Manager, Pool, cpu_count

if USE_MP:
    # import concurrent.futures
    import mpi4py.futures



def apply_rdm_1D(nb, b_self: 'GridTN1D', b_other, *args, **kwargs):
    return nb, *b_self.apply_rdm(b_other, *args, **kwargs)

def build_branch_envs(nb, b_env: 'Environment'):
    b_env.build_all_envs()
    return nb, b_env[-1]


# @profile
def apply_rdm(comb_gtn, other,
              direction=1, inplace=False, compress_opts=None, verbose=False) -> 'GridTN1DComb':
    """ Apply grid_mpo to self, assuming they exist on the same grid
        contract (and compress) using rdm method
    """
    gtnc = comb_gtn if inplace else comb_gtn.copy(deep=False)  # original data will be replaced
    other = other.copy()
    if compress_opts is None:
        compress_opts = {}
    cutoff = compress_opts.get('cutoff', CUTOFF)
    max_bond = compress_opts.get('max_bond', None)

    if comb_gtn.data is None:
        return gtnc

    use_mp_1 = False
    use_mp   = USE_MP

    # if use_mp_1 or use_mp:
    #     # after importing numpy, reset the CPU affinity of the parent process so
    #     # that it will use all cores
    #     os.system("taskset -p 0xff %d" % os.getpid())


    if verbose:
        print('self', gtnc)
        print('other', other)

    gtnc.collect_exponents()
    other.collect_exponents()

    if not gtnc.grid == other.grid:
        # branch ordering is determined by self and then other if not included already
        pad_grids = [gl for gl in other.grid.grids if gl not in gtnc.grid.grids]
        if len(pad_grids) > 0:
            grids_list = gtnc.grid.grids + tuple(pad_grids)
            cgrid = GridsComb(gtnc.grid.gridID + other.grid.gridID, grids_list)

            new_branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
        else:
            cgrid = gtnc.grid
            new_branch_inds1 = None  # range(len(cgrid.cgrids))

        grids_list = cgrid.grids
        new_branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
    else:
        cgrid = gtnc.grid
        new_branch_inds1 = new_branch_inds2 = None  # range(len(cgrid.grids))

    # print('branch inds', branch_inds1, branch_inds2)
    nbranches = len(cgrid.grids)

    ### pad gtn spines
    spine1 = gtnc.spine
    spine1.mangle_inner_()
    spine2 = other.spine

    spine_ind_id = spine1.site_ind_id
    spine1 = helper.pad_mps(spine1, new_branch_inds1, nbranches)
    if new_branch_inds1 is not None:
        branch_old_to_new1 = {spine_ind_id.format(i): spine_ind_id.format(new_branch_inds1[i])
                              for i in range(gtnc.grid.ngrids)}
    else:
        branch_old_to_new1 = {}

    if spine_ind_id == other.spine_ind_id:
        other.spine_ind_id = other.spine_ind_id + '_o_'
    spine2 = helper.pad_mps(spine2, new_branch_inds2, nbranches)
    if new_branch_inds2 is not None:
        branch_old_to_new2 = {other.spine_ind_id.format(i): other.spine_ind_id.format(new_branch_inds2[i])
                              for i in range(other.grid.ngrids)}
    else:
        branch_old_to_new2 = {}

    idx0 = 0 if direction > 0 else nbranches - 1
    idx1 = nbranches - 1 if direction > 0 else 0



    ### compute envs <mps1 | mpo2.T | mpo2 | mps1>
    #### get branch envs
    new_branches = {}
    branch_envs = {}
    remove_branches = []

    if use_mp or use_mp_1:
        # executor = concurrent.futures.ProcessPoolExecutor(nbranches)
        executor = mpi4py.futures.MPIPoolExecutor(nbranches)

    if use_mp_1:
        futures = []

    for nb in range(nbranches):

        if verbose:  print('get branch env', nb)

        subgrid = cgrid.grids[nb]
        b_self = gtnc.get_branch(subgrid)
        b_other = other.get_branch(subgrid)

        if b_self is not None:
            b_self.get_anchor_tens().reindex(branch_old_to_new1, inplace=True)
        if b_other is not None:
            b_other.get_anchor_tens().reindex(branch_old_to_new2, inplace=True)

        ## calculate envs
        if b_self is None and b_other is None:
            new_branches[nb] = 'empty'
            remove_branches += [nb]

        elif b_self is None and b_other is not None:
            raise NotImplementedError

        elif b_self is not None and b_other is None:  ## don't update this branch
            ## canonize branch into spine. avoid changing exponent
            b_self.canonize(i=b_self.get_anchor_ind(), inplace=True, scale=False)
            tens0 = b_self.get_anchor_tens()
            helper.canonize_tens_list(tens0, spine1[nb], inplace=True)

            anc_ind = spine1.site_ind_id.format(nb)
            branch_envs[nb] = b_self.integrate(is_sqrt=True, exclude_weights=True,
                                               ancilla_reindex={anc_ind: anc_ind + '_'})
            new_branches[nb] = b_self
            # print('use old b_self', nb)

        else:
            # b_other.data.lower_ind_id = b_self.data.site_ind_id
            b_self.data.site_ind_id = b_other.data.lower_ind_id
            b_other.data.mangle_inner_()

            b_self_bra = b_self.conj()
            for t in b_self_bra.data:
                t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
            b_self_bra.data._site_ind_id = b_self_bra.data.site_ind_id + '_'

            b_other_bra = b_other.conj()
            b_other_bra.transpose()
            for t in b_other_bra.data:
                t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
            b_other_bra.data._upper_ind_id = b_other_bra.data.upper_ind_id + '_'
            b_other_bra.data._lower_ind_id = b_other_bra.data.lower_ind_id + '_'
            # b_other_bra.data.lower_ind_id = b_other.data.upper_ind_id

            if b_self.get_anchor_ind() == 0:
                AAT = MatrixProductOperatorTN([b_other.data, b_other_bra.data])
                b_env = Environment(b_self.data.L, EnvironmentSide.RIGHT, b_self.data, b_self_bra.data, AAT)
                if use_mp_1:
                    futures += [executor.submit(build_branch_envs, *(nb, b_env))]
                    # for x in futures:
                    #     print('running (1)?', nb, x.running(), 'done?', x.done(), cpu_count())
                else:
                    b_env.build_all_envs()
                    branch_envs[nb] = b_env[-1]
                # print('b env norm', b_env[-1].inds, b_env[-1].norm())
            else:
                raise NotImplementedError

    if use_mp_1:
        # concurrent.futures.wait(futures)
        mpi4py.futures.wait(futures)
        for x in futures:
            nb, branch_env = x.result()
            branch_envs[nb] = branch_env
        # executor.shutdown()


    ### compute spine envs
    spine1_bra = spine1.conj()
    for t in spine1_bra:
        t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
    spine1_bra._site_ind_id = spine1_bra.site_ind_id + '_'

    spine2_bra = spine2.conj()
    for t in spine2_bra:
        t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
    spine2_bra._site_ind_id = spine2_bra.site_ind_id + '_'

    ## spine envs
    def build_spine_envs(env_direction):
        spine_envs = {}

        branch_inds = range(nbranches - 1) if env_direction > 0 else range(nbranches - 1, 0, -1)
        for nb in branch_inds:
            prev_nb = nb - np.sign(env_direction)
            prev_spine_tens = spine_envs.get(prev_nb, None)

            contract_spine_list = [spine1[nb], spine2[nb], spine2_bra[nb], spine1_bra[nb]]
            if prev_spine_tens is not None:
                contract_spine_list += [prev_spine_tens]

            b_env_tens = branch_envs.get(nb, None)
            if b_env_tens is not None:
                contract_spine_list += [b_env_tens]

            spine_envs[nb] = qtn.tensor_contract(*contract_spine_list)
        return spine_envs

    left_spine_envs = build_spine_envs(1)
    right_spine_envs = build_spine_envs(-1)


    ### apply_rdm for each branch (that needs it)
    # branch_Cs = manager.dict() if use_mp else {}
    branch_Cs = {}
    if use_mp:
        futures = []

    for nb in range(nbranches):

        subgrid = cgrid.grids[nb]
        b_self = gtnc.get_branch(subgrid)
        b_other = other.get_branch(subgrid)

        if b_self is not None and b_other is not None:
            left_env = left_spine_envs.get(nb - 1, None)
            right_env = right_spine_envs.get(nb + 1, None)

            env_list = [spine1[nb], spine2[nb], spine2_bra[nb], spine1_bra[nb]]
            if left_env is not None:   env_list += [left_env]
            if right_env is not None:  env_list += [right_env]
            branch_env = qtn.TensorNetwork(env_list)

            if use_mp:
                futures += [executor.submit( apply_rdm_1D, *(nb, b_self, b_other),
                                             **{'direction': -1, 'left_env': branch_env,
                                                'compress': False, 'compress_opts': compress_opts,
                                                'open_end': True, 'verbose': verbose,},  )]
                for x in futures:
                    print('running?', nb, x.running(), 'done?', x.done(), cpu_count())
            else:
                new_branch, C_tens = b_self.apply_rdm(b_other,  # bra_self=b_self_bra, bra_other=b_other_bra,
                                                      direction=-1, left_env=branch_env,
                                                      compress=False, compress_opts=compress_opts,
                                                      open_end=True, verbose=verbose,)

                new_branches[nb] = new_branch
                branch_Cs[nb] = C_tens


    # actually evaluate stuff?
    if use_mp:
        for x in futures:
            print('running?', nb, x.running(), 'done?', x.done(), cpu_count())

        # concurrent.futures.wait(futures)
        mpi4py.futures.wait(futures)

        for x in futures:
            # print('running?', x.running(), 'done?', x.done())
            out = x.result()
            nb, new_branch, C_tens = out
            new_branches[nb] = new_branch
            branch_Cs[nb] = C_tens

    if use_mp or use_mp_1:
        executor.shutdown()

    # print('move on')


    # ### apply_rdm update to spine
    # Us = {}
    # spine_C_tens = None
    # for nb in range(idx0, idx1, np.sign(direction)):    # l2r if direction > 0
    #
    #     # print('spine rdm', nb, direction)
    #     contract_tens_list = [spine1[nb], spine2[nb], spine2_bra[nb], spine1_bra[nb]]
    #     ket_inds = []
    #
    #     env = left_spine_envs.get(nb - 1, None) if direction < 0 else \
    #           right_spine_envs.get(nb + 1, None)
    #     if env is not None:
    #         contract_tens_list += [env]
    #
    #     branch_C_tens = branch_Cs.get(nb, None)
    #     if branch_C_tens is not None:
    #         branch_C_tens_conj = branch_C_tens.conj(inplace=False)
    #         branch_C_tens_conj.reindex({ind: ind + '_' for ind in branch_C_tens.inds}, inplace=True)
    #         contract_tens_list += [branch_C_tens, branch_C_tens_conj]
    #         ket_inds += [ind for ind in branch_C_tens.inds if ind not in spine1[nb].inds and ind not in spine2[nb].inds]
    #         # print('branch C tens', nb, branch_C_tens.norm())
    #         # print('branch C', branch_C_tens)
    #     else:
    #         ## branch_C_tens is identity bc we've already canonicalized the branch
    #         ket_inds += [spine1.site_ind_id.format(nb)]
    #         # print(branch_envs[nb].data)
    #
    #     if spine_C_tens is not None:
    #         # print('spine C tens', spine_C_tens.norm())
    #         # print('spine tens', spine_C_tens)
    #         spine_C_tens_conj = spine_C_tens.conj(inplace=False)
    #         spine_C_tens_conj.reindex({ind: ind + '_' for ind in spine_C_tens.inds}, inplace=True)
    #         contract_tens_list += [spine_C_tens, spine_C_tens_conj]
    #         ket_inds += [ind for ind in spine_C_tens.inds if ind not in spine1[nb].inds and ind not in spine2[nb].inds]
    #
    #
    #     rdm = qtn.tensor_contract(*contract_tens_list)
    #     if True:  print('rdm', nb, rdm.shape, rdm.inds)
    #
    #     bra_inds = [ind + '_' for ind in ket_inds]
    #     rdm.transpose(*bra_inds, *ket_inds, inplace=True)
    #     sq_shape = rdm.shape[:len(ket_inds)]
    #
    #     rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
    #     # print('rdm is H?', np.linalg.norm(rdm_mat - rdm_mat.T.conj()) / np.linalg.norm(rdm_mat))
    #     evals, evecs = np.linalg.eigh(rdm_mat)
    #
    #     ### quimb version
    #     rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
    #     evals, evecs = quimb.eigh(rdm_mat,
    #                               k=(max_bond if max_bond is not None else -1), tol=cutoff, which='LM',
    #                               sort=True, return_vecs=True, )
    #     ## evals assorted in ascending order; targetting largest magnitude eigvals
    #     evecs = evecs[:, ::-1].conj()  ## qarray object
    #
    #     # ### numpy version
    #     # # print('rdm is H?', np.linalg.norm(rdm_mat - rdm_mat.T.conj()) / np.linalg.norm(rdm_mat))
    #     # evals, evecs = np.linalg.eigh(rdm_mat)
    #     #
    #     # if verbose:  print('cutoff', cutoff, 'max_bond', max_bond)
    #     # sort_inds = np.argsort(evals)[::-1]
    #     #
    #     # if max_bond is not None:
    #     #     sort_inds = sort_inds[:max_bond]
    #     #
    #     # evals = evals[sort_inds]
    #     # if evals[-1] / evals[0] < cutoff:
    #     #     cut_ind = np.argmax(evals / evals[0] < cutoff)
    #     #     sort_inds = sort_inds[:cut_ind]
    #     #
    #     # # print('spine rdm evals', evals)
    #     # evecs = evecs[:, sort_inds].conj()  ## input inds, sort_inds (horizontal bond)
    #     #####
    #
    #     evecs_tens = qtn.Tensor(data=evecs.reshape(*sq_shape, -1), inds=ket_inds + [f's_{nb + np.sign(direction)}'],
    #                             tags=spine1.site_tag(nb))
    #     Us[nb] = evecs_tens.conj(inplace=False)
    #
    #     contract_tens_list = [spine2[nb], spine1[nb], evecs_tens]
    #     if branch_C_tens is not None:
    #         contract_tens_list += [branch_C_tens]
    #     if spine_C_tens is not None:
    #         contract_tens_list += [spine_C_tens]
    #     spine_C_tens = qtn.tensor_contract(*contract_tens_list)
    #
    #
    # ## last spine tens
    # # print('last spine tens', idx1)
    # contract_tens_list = [spine1[idx1], spine2[idx1]]
    # if spine_C_tens is not None:
    #     contract_tens_list += [spine_C_tens]
    # branch_C_tens = branch_Cs.get(idx1, None)
    # if branch_C_tens is not None:
    #     contract_tens_list += [branch_C_tens]
    # tens1 = qtn.tensor_contract(*contract_tens_list)
    # # print('last spine tens', tens1.norm())
    #
    # branch = new_branches[idx1]
    # if branch is not None:
    #     h_ind = f'h_{branch.get_anchor_ind()}'
    #     tens1.reindex({h_ind: spine1.site_ind_id.format(idx1)}, inplace=True)
    # tens1.drop_tags()
    # tens1.add_tag(spine1.site_tag(idx1))
    # Us[idx1] = tens1

    ###################

    # #### inefficient spine method: just contract spine together
    # Us = {}
    # for nb in range(nbranches):
    #     contract_tens_list = [spine1[nb], spine2[nb]]
    #     branch_C_tens = branch_Cs.get(nb, None)
    #     if branch_C_tens is not None:
    #         contract_tens_list += [branch_C_tens]
    #     tens1 = qtn.tensor_contract(*contract_tens_list)
    #     print('last spine tens', tens1.norm())
    #
    #     branch = new_branches[nb]
    #     if branch is not None:
    #         h_ind = f'h_{branch.get_anchor_ind()}'
    #         tens1.reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
    #     tens1.drop_tags()
    #     tens1.add_tag(spine1.site_tag(nb))
    #     Us[nb] = tens1

    #### spine using zipup method
    Us = {}
    prev_spine_tens = None
    for nb in range(idx0, idx1, np.sign(direction)):
        next_nb = nb + np.sign(direction)

        contract_tens_list = [spine1[nb], spine2[nb]]
        branch_C_tens = branch_Cs.get(nb, None)
        if branch_C_tens is not None:
            contract_tens_list += [branch_C_tens]
        if prev_spine_tens is not None:
            contract_tens_list += [prev_spine_tens]
        tens_spine = qtn.tensor_contract(*contract_tens_list)

        right_inds = [spine1.bond(nb, next_nb), spine2.bond(nb, next_nb)]
        left_inds = [ind for ind in tens_spine.inds if ind not in right_inds]
        tens1, tens2 = qtn.tensor_split(tens_spine, left_inds, absorb='right',
                                        cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

        branch = new_branches[nb]
        if branch is not None:
            h_ind = f'h_{branch.get_anchor_ind()}'
            tens1.reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
        tens1.drop_tags()
        tens1.add_tag(spine1.site_tag(nb))
        Us[nb] = tens1
        prev_spine_tens = tens2

    contract_tens_list = [spine1[idx1], spine2[idx1]]
    branch_C_tens = branch_Cs.get(idx1, None)
    if branch_C_tens is not None:
        contract_tens_list += [branch_C_tens]
    if prev_spine_tens is not None:
        contract_tens_list += [prev_spine_tens]
    tens1 = qtn.tensor_contract(*contract_tens_list)
    tens1.drop_tags()
    tens1.add_tag(spine1.site_tag(idx1))
    Us[idx1] = tens1

    ##############

    ## so that bonds to the branches will match site_ind_id
    new_branches_copy = {}
    for nb in range(nbranches):
        if nb in remove_branches:  continue
        branch = new_branches[nb]
        h_ind = f'h_{branch.get_anchor_ind()}'
        # print('Us', nb, Us[nb])
        Us[nb].reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
        # print('branch', nb, branch)
        branch.get_anchor_tens().reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
        # tens1 = branch.get_anchor_tens()
        # print('tens1', tens1)
        # tens1.reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
        # print('tens1', tens1, h_ind, spine1.site_ind_id.format(nb))
        # print('branch', nb, branch)
        new_branches_copy[nb] = branch.copy()

    ## string Us together to make new spine
    # print('spine norms', [Us[i].norm() for i in range(nbranches)])
    # print('branch norms', [helper.norm(b.data) for k, b in new_branches.items()])
    new_spine = qtn.TensorNetwork([Us[i] for i in range(nbranches)])
    new_spine.fuse_multibonds(inplace=True)
    new_spine.view_like(spine1, inplace=True)
    # print('new spine', new_spine)
    # new_spine.site_ind_id = spine1.site_ind_id

    # print('new branches', new_branches_copy)

    for nb in remove_branches:
        new_branches.pop(nb)

    gtnc.spine = new_spine
    gtnc.branches = new_branches_copy
    gtnc._exponent += other.exponent
    gtnc._sign *= other.sign

    gtnc.canon_site = idx1

    # print('spine orthog', helper.check_orthog(new_spine), direction, idx1)
    # form = 'right' if idx0 < idx1 else 'left'
    # compress_opts = compress_opts.copy()
    # compress_opts['form'] = form
    # helper.compress(gtnc.spine, canonize=False, scale=False, compress_opts=compress_opts)
    # gtnc.canon_site = idx0

    # for gr, b in gtnc.branches.items():
    #     print('gr', gr, helper.check_orthog(b.data))
    # print('spine orthog', helper.check_orthog(gtnc.spine), gtnc.canon_site)

    # print('gtnc', gtnc)
    # print('gtnc', gtnc.norm(is_sqrt=True))
    # if gtnc.norm(is_sqrt=True) > 10:
    #     exit()

    # print('pre comp gtnc', gtnc.max_bond(), gtnc)
    comp_dir = 0 if idx1 != 0 else 1
    gtnc.compress(inplace=True, compress_opts=compress_opts, direction=comp_dir)
    # print('post comp gtnc', gtnc.max_bond(), gtnc)

    return gtnc


if __name__ == '__main__':
    pass
