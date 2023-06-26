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



def compress_rdm_1D(nb, b_self: 'GridTN1D', *args, **kwargs):
    return nb, *b_self.compress_rdm(*args, **kwargs)

def build_branch_envs(nb, b_env: 'Environment'):
    b_env.build_all_envs()
    return nb, b_env[-1]


# @profile
def compress_rdm(comb_gtn, back_compress=True,
                 direction=1, inplace=False, compress_opts=None, verbose=False) -> 'GridTN1DComb':
    """ Apply grid_mpo to self, assuming they exist on the same grid
        contract (and compress) using rdm method
    """
    gtnc = comb_gtn if inplace else comb_gtn.copy(deep=False)  # original data will be replaced
    if compress_opts is None:
        compress_opts = {}
    cutoff = compress_opts.get('cutoff', CUTOFF)
    max_bond = compress_opts.get('max_bond', None)

    if comb_gtn.data is None:
        return gtnc

    use_mp_1 = False
    use_mp   = USE_MP

    if verbose:
        print('self', gtnc)

    gtnc.collect_exponents()
    # gtnc.distribute_sign()
    # gtnc.distribute_exponents()

    cgrid = gtnc.grid
    new_branch_inds1 = None  # range(len(cgrid.grids))

    # print('branch inds', branch_inds1, branch_inds2)
    nbranches = len(cgrid.grids)

    ### pad gtn spines
    spine1 = gtnc.spine
    spine1.mangle_inner_()

    spine_ind_id = spine1.site_ind_id
    # spine1 = helper.pad_mps(spine1, new_branch_inds1, nbranches)
    # branch_old_to_new1 = {}

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

        ## calculate envs
        if b_self is None:
            new_branches[nb] = 'empty'
            remove_branches += [nb]

        else:
            anc_ind = spine1.site_ind_id.format(nb)
            branch_envs[nb] = b_self.integrate(is_sqrt=True, exclude_weights=True,
                                               ancilla_reindex={anc_ind: anc_ind + '_'})
            new_branches[nb] = b_self
            # print('use old b_self', nb)


    ### compute spine envs
    spine1_bra = spine1.conj(inplace=False)
    for t in spine1_bra:
        t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
    spine1_bra._site_ind_id = spine1_bra.site_ind_id + '_'


    ## spine envs
    def build_spine_envs(env_direction):
        spine_envs = {}

        branch_inds = range(nbranches - 1) if env_direction > 0 else range(nbranches - 1, 0, -1)
        for nb in branch_inds:
            prev_nb = nb - np.sign(env_direction)
            prev_spine_tens = spine_envs.get(prev_nb, None)

            contract_spine_list = [spine1[nb], spine1_bra[nb]]
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
        b_self = gtnc.get_branch(subgrid).copy()

        left_env = left_spine_envs.get(nb - 1, None)
        right_env = right_spine_envs.get(nb + 1, None)

        env_list = [spine1[nb], spine1_bra[nb]]
        if left_env is not None:   env_list += [left_env]
        if right_env is not None:  env_list += [right_env]
        branch_env = qtn.TensorNetwork(env_list)

        if use_mp:
            futures += [executor.submit( compress_rdm_1D, *(nb, b_self),
                                         **{'direction': -1, 'left_env': branch_env,
                                            'compress': False, 'compress_opts': compress_opts,
                                            'open_end': True, 'verbose': verbose,},  )]
            for x in futures:
                print('running?', nb, x.running(), 'done?', x.done(), cpu_count())
        else:
            new_branch, C_tens = b_self.compress_rdm(direction=-1, left_env=branch_env,
                                                     back_compress=False, compress_opts=compress_opts,
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


    ### apply_rdm update to spine
    Us = {}
    spine_C_tens = None
    for nb in range(idx0, idx1, np.sign(direction)):    # l2r if direction > 0

        # print('spine rdm', nb, direction)
        contract_tens_list = [spine1[nb], spine1_bra[nb]]
        ket_inds = []

        env = left_spine_envs.get(nb - 1, None) if direction < 0 else \
              right_spine_envs.get(nb + 1, None)
        if env is not None:
            contract_tens_list += [env]

        branch_C_tens = branch_Cs.get(nb, None)
        if branch_C_tens is not None:
            branch_C_tens_conj = branch_C_tens.conj(inplace=False)
            branch_C_tens_conj.reindex({ind: ind + '_' for ind in branch_C_tens.inds}, inplace=True)
            contract_tens_list += [branch_C_tens, branch_C_tens_conj]
            ket_inds += [ind for ind in branch_C_tens.inds if ind not in spine1[nb].inds]
            # print('branch C tens', nb, branch_C_tens.norm())
            # print('branch C', branch_C_tens)

        if spine_C_tens is not None:
            # print('spine C tens', spine_C_tens.norm())
            # print('spine tens', spine_C_tens)
            spine_C_tens_conj = spine_C_tens.conj(inplace=False)
            spine_C_tens_conj.reindex({ind: ind + '_' for ind in spine_C_tens.inds}, inplace=True)
            contract_tens_list += [spine_C_tens, spine_C_tens_conj]
            ket_inds += [ind for ind in spine_C_tens.inds if ind not in spine1[nb].inds]


        # rdm = qtn.tensor_contract(*contract_tens_list)
        rdm = qtn.TensorNetwork([contract_tens_list])
        rdm = rdm.contract()
        if True:  print('rdm', nb, rdm.shape, rdm.inds)

        bra_inds = [ind + '_' for ind in ket_inds]
        rdm.transpose(*ket_inds, *bra_inds, inplace=True)
        sq_shape = rdm.shape[:len(ket_inds)]

        try:
            # raise np.linalg.LinAlgError
            ### quimb version
            rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
            # print('rdm is H?', np.linalg.norm(rdm_mat - rdm_mat.T.conj()) / np.linalg.norm(rdm_mat))
            if max_bond is None:
                evals, evecs = quimb.eigh(rdm_mat, k=-1,
                                          sort=True, return_vecs=True, )
            else:
                evals, evecs = quimb.eigh(rdm_mat,
                                          k=(max_bond if max_bond is not None else -1), tol=cutoff, which='LM',
                                          sort=True, return_vecs=True, )
            ## evals assorted in ascending order; targetting largest magnitude eigvals
            evecs = evecs.conj()  ## qarray object

            tmp = np.einsum('ai,aj', evecs, evecs.conj())
            error = np.linalg.norm(tmp - np.eye(tmp.shape[0]))
            if error > 1.0e-12:
                print('spine scipy eigh error', nb, error)
                raise np.linalg.LinAlgError('scipy eigh gave nonorthonormal bases')

        except np.linalg.LinAlgError:

            ### numpy version -- has some bug??
            rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
            # print('rdm is H?', np.linalg.norm(rdm_mat - rdm_mat.T.conj()) / np.linalg.norm(rdm_mat))
            evals, evecs = np.linalg.eigh(rdm_mat)

            if verbose:  print('cutoff', cutoff, 'max_bond', max_bond)
            sort_inds = np.argsort(evals)[::-1]

            if max_bond is not None:
                sort_inds = sort_inds[:max_bond]

            # old_evals = evals
            evals = evals[sort_inds]
            # if max_bond is not None:
            #     print('eigvals > max_bond', np.linalg.norm(evals[max_bond:]))

            if (evals[-1] / evals[0]) < cutoff:
                cut_ind = np.argmax((evals / evals[0]) < cutoff)
                # print('cut_ind', cut_ind, evals[cut_ind])
                sort_inds = sort_inds[:cut_ind]

            # fig1, ax1 = plt.subplots()
            # ax1.semilogy(evals, label='all')
            # ax1.semilogy(old_evals[sort_inds],label='cut')
            # ax1.legend()
            # plt.show()

            # print('spine rdm evals', evals)
            evecs = evecs[:, sort_inds].conj()  ## input inds, sort_inds (horizontal bond)
            print('used np sort inds')

        ######

        # tmp = np.einsum('ai,aj', evecs, evecs.conj())
        # error = np.linalg.norm(tmp - np.eye(tmp.shape[0]))
        # if error > 1.0e-12:
        #     print('spine scipy eigh error', nb, error)
        #     raise np.linalg.LinAlgError('scipy eigh gave nonorthonormal bases')

        evecs_tens = qtn.Tensor(data=evecs.reshape(*sq_shape, -1), inds=ket_inds + [f's_{nb + np.sign(direction)}'],
                                tags=spine1.site_tag(nb))
        Us[nb] = evecs_tens.conj(inplace=False)

        contract_tens_list = [spine1[nb], evecs_tens]
        if branch_C_tens is not None:
            contract_tens_list += [branch_C_tens]
        if spine_C_tens is not None:
            contract_tens_list += [spine_C_tens]
        spine_C_tens = qtn.tensor_contract(*contract_tens_list)


    ## last spine tens
    # print('last spine tens', idx1)
    contract_tens_list = [spine1[idx1]]
    if spine_C_tens is not None:
        contract_tens_list += [spine_C_tens]
    branch_C_tens = branch_Cs.get(idx1, None)
    if branch_C_tens is not None:
        contract_tens_list += [branch_C_tens]
    tens1 = qtn.tensor_contract(*contract_tens_list)
    # print('last spine tens', tens1.norm())

    branch = new_branches[idx1]
    if branch is not None:
        h_ind = f'h_{branch.get_anchor_ind()}'
        tens1.reindex({h_ind: spine1.site_ind_id.format(idx1)}, inplace=True)
    tens1.drop_tags()
    tens1.add_tag(spine1.site_tag(idx1))
    Us[idx1] = tens1

    # ##################
    #
    # #### spine using normal compress
    # Us = {}
    # prev_spine_tens = None
    # for nb in range(idx0, idx1, np.sign(direction)):
    #     next_nb = nb + np.sign(direction)
    #
    #     contract_tens_list = [spine1[nb]]
    #     branch_C_tens = branch_Cs.get(nb, None)
    #     if branch_C_tens is not None:
    #         contract_tens_list += [branch_C_tens]
    #     if prev_spine_tens is not None:
    #         contract_tens_list += [prev_spine_tens]
    #     tens_spine = qtn.tensor_contract(*contract_tens_list)
    #
    #     right_inds = [spine1.bond(nb, next_nb)]
    #     left_inds = [ind for ind in tens_spine.inds if ind not in right_inds]
    #     tens1, tens2 = qtn.tensor_split(tens_spine, left_inds, absorb='right',
    #                                     cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
    #
    #     branch = new_branches[nb]
    #     if branch is not None:
    #         h_ind = f'h_{branch.get_anchor_ind()}'
    #         tens1.reindex({h_ind: spine1.site_ind_id.format(nb)}, inplace=True)
    #     tens1.drop_tags()
    #     tens1.add_tag(spine1.site_tag(nb))
    #     Us[nb] = tens1
    #     prev_spine_tens = tens2
    #
    # contract_tens_list = [spine1[idx1]]
    # branch_C_tens = branch_Cs.get(idx1, None)
    # if branch_C_tens is not None:
    #     contract_tens_list += [branch_C_tens]
    # if prev_spine_tens is not None:
    #     contract_tens_list += [prev_spine_tens]
    # tens1 = qtn.tensor_contract(*contract_tens_list)
    # tens1.drop_tags()
    # tens1.add_tag(spine1.site_tag(idx1))
    # Us[idx1] = tens1
    #
    # ##############

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
    new_spine = qtn.TensorNetwork([Us[i] for i in range(nbranches)])
    new_spine.fuse_multibonds(inplace=True)
    new_spine.view_like(spine1, inplace=True)

    # print('new branches', new_branches_copy)

    for nb in remove_branches:
        new_branches.pop(nb)

    gtnc.spine = new_spine
    gtnc.branches = new_branches_copy
    gtnc.canon_site = idx1

    # print('pre comp gtnc')
    # gtnc.check_orthog()

    # print('pre comp gtnc', gtnc.max_bond(), gtnc)
    if back_compress:
        comp_dir = 0 if idx1 != 0 else 1
        gtnc.compress(inplace=True, compress_opts=compress_opts, direction=comp_dir)
        # print('post comp gtnc', gtnc.max_bond(), gtnc)

    return gtnc


if __name__ == '__main__':
    pass
