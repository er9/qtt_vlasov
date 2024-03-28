import helper_quimb
from setup.configs import *
import time
import scipy.linalg as linalg
import quimb.tensor as qtn
# import helper_quimb as helper
from local_solvers.defaults import *
import local_solvers.helper_tn as helper_tn
from local_solvers.mps_classes import MPS

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']


def get_tens_rdm(tens: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                 transpose_inds: Optional[Sequence[str]]=None):
    """ tens.conj() * tens, contracted over inds not specified in bra to ket inds
    """
    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    if transpose_inds is None:
        transpose_inds = [*ket_iso, *bra_iso]

    tens_conj = tens.conj()
    tens_conj.reindex({k: b for k, b in zip(ket_iso, bra_iso)}, inplace=True)

    rdm = qtn.tensor_contract(tens_conj, tens, output_inds=transpose_inds)

    return rdm


def get_tot_rdm(tensors: Sequence['qtn.Tensor'], ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                transpose_inds: Optional[Sequence[str]]=None):

    rdms = [get_tens_rdm(tens, ket_iso, bra_iso, transpose_inds=transpose_inds) for tens in tensors]
    # print('rdms', [t.norm() for t in rdms])
    # for t in rdms:
    #     t.modify(apply=lambda x: x / t.norm())
    tot_rdm = helper_tn.sum_tens(rdms)
    return tot_rdm


def get_rdm_eig(tot_rdm: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                new_ind: str=None, max_bond=None):
    """ A = Q Lambda Q.T  --> return Q
    """

    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    # if transpose_inds is None:
    #     transpose_inds = [*ket_iso, *bra_iso]

    tot_rdm.transpose(*ket_iso, *bra_iso, inplace=True)
    nb = len(ket_iso)

    denmat_shape = tot_rdm.shape
    tens_shape = denmat_shape[:nb]
    sq_shape = np.prod(denmat_shape[:nb])

    eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

    # sort_inds = np.argsort(np.abs(eigval))[::-1]
    #
    # ## apply cut-off
    # ev_max = eigval[sort_inds[0]]
    # new_sort_inds = []
    # for si in sort_inds:
    #     if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
    #         new_sort_inds += [si]
    #     else:
    #         break
    # sort_inds = new_sort_inds

    # print('eigvals', eigval[sort_inds])

    if max_bond is not None:
        sort_inds = np.argsort(np.abs(eigval))[::-1]
        eigvec = eigvec[:, sort_inds[:max_bond]]
        print('eig trunc', np.sum(np.abs(eigval[sort_inds[max_bond:]])))
    # else:
    #     eigvec = eigvec[:, sort_inds]

    eigvec = eigvec.reshape(*tens_shape, -1)
    eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

    return eigvec


def update_1site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                 direction: 'SweepDirection', max_bond: int=None):
    """ update ket, bra with new_site
        i: int of mps site
    """
    if isinstance(site_i, qtn.Tensor):
        site_i = [site_i]

    # print('check orthog', left_site_pos)
    # helper_quimb.check_orthog(mps)
    # print('update 1 site', [t.norm() for t in site_i])


    # print('update 1 site', left_site_pos, [t.norm() for t in site_i], direction)
    if direction == SweepDirection.RIGHT:
        ind1 = left_site_pos
        ind2 = left_site_pos + 1 if left_site_pos < mps.L - 1 else None
        ## canonicalize ket to next site
        if ind1 < mps.L - 1:
            mps.cur_orthog = ind1 + direction
            # self.cur_orthog = ind1 + direction

    else:
        ind1 = left_site_pos
        ind2 = (left_site_pos - 1) if left_site_pos > 0 else None
        ## canonicalize ket to next site
        if ind1 > 0:
            # self.cur_orthog = ind1 + direction
            mps.cur_orthog = ind1 + direction

    if ind2 is None:  ## is at_end
        # if not decimate_only:
        ## update ket
        site_i = helper_tn.sum_tens(site_i)
        site_i.transpose_like(mps[ind1], inplace=True)
        mps[ind1].modify(data=site_i.data)
    else:

        x_bond = mps.bond(left_site_pos, left_site_pos + direction)    ## bond to not contract over
        ket_iso = [ind for ind in site_i[0].inds if ind != x_bond]
        bra_iso = [ind + '_' for ind in ket_iso]

        # [new_ket_site, self.ket[i]]
        tot_rdm = get_tot_rdm(site_i, ket_iso, bra_iso, )
        # print('tot rdm', tot_rdm.norm())
        # sq_shape = int(np.round(np.sqrt(tot_rdm.size)))
        # print(tot_rdm.data.reshape(sq_shape, sq_shape))
        eigvec  = get_rdm_eig(tot_rdm, ket_iso, bra_iso, new_ind=x_bond + '_',
                                         max_bond=max_bond)
        # print('eigvec', eigvec.norm())
        # print(eigvec.data)

        decimate(mps, left_site_pos, eigvec, direction)
        mps.cur_orthog = ind2

    return


def decimate(mps: 'MPS', i: int, canon_site_i: 'qtn.Tensor', direction: 'SweepDirection',):
    """ move canonical center using new canon_site_i
        assumes mps orthogonality center currently is at i --> moves to i+1, i-1
    """
    # print('decimate orthog')
    # print(helper_quimb.check_orthog(mps))
    canon_site_i = canon_site_i.copy()

    if direction == SweepDirection.RIGHT:
        ind1 = i
        ind2 = i + 1 if i < mps.L - 1 else None
        ## canonicalize ket to next site
        if i < mps.L - 1:
            mps.cur_orthog = ind1 + direction

    else:
        ind1 = i
        ind2 = (i - 1) if i > 0 else None
        ## canonicalize ket to next site
        if i > 0:
            # self.cur_orthog = ind1 + direction
            mps.cur_orthog = ind1 + direction

    # print('canon site i', canon_site_i)

    x_bond = mps.bond(i, i + direction)  ## bond to not contract over
    # x_bond = self.ket_horizontal_bond(i, direction)  ## bond to not contract over
    # ket_iso = [ind for ind in canon_site_i.inds if (ind != x_bond or ind != x_bond + '_')]
    # bra_iso = [ind + '_' for ind in ket_iso]
    if x_bond in canon_site_i.inds:
        canon_site_i = canon_site_i.reindex({x_bond: x_bond + '_'}, inplace=False)

    # mps[ind1].transpose('i(0)', x_bond, inplace=True)
    # mps[ind2].transpose('i(1)', x_bond, mps.bond(1, 2), inplace=True)
    # print('mps[ind1]', ind1, mps[ind1].inds, mps[ind1].norm())
    # print(mps[ind1].data)
    # print('mps[ind2]', ind2, mps[ind2].inds, mps[ind2].norm())
    # print(mps[ind2].data)

    # print('mps[ind1]', mps[ind1])
    # print('canon site i', canon_site_i)
    next_site = qtn.tensor_contract(canon_site_i.conj(), mps[ind1], mps[ind2])
    canon_site_i.transpose_like(mps[ind1], inplace=True)
    next_site.transpose_like(mps[ind2], inplace=True)
    mps[ind1].modify(data=canon_site_i.data)
    mps[ind2].modify(data=next_site.data)
    # mps[ind1].transpose('i(0)', x_bond, inplace=True)
    # mps[ind2].transpose('i(1)', x_bond, mps.bond(1, 2), inplace=True)
    # print('mps[ind1]', ind1, mps[ind1].inds, mps[ind1].norm())
    # print(mps[ind1].data)
    # print('mps[ind2]', ind2, mps[ind2].inds, mps[ind2].norm())
    # print(mps[ind2].data)
    return


def update_2site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                 direction: 'SweepDirection', max_bond: int=None, decimate_only=False):
    """ update ket, bra with new_site
        i: int of mps site
    """
    if isinstance(site_i, qtn.Tensor):
        site_i = [site_i]

    # print('check orthog', left_site_pos)
    # helper_quimb.check_orthog(mps)
    # print('update 1 site', [t.norm() for t in site_i])


    # print('update 2 site', left_site_pos, mps.L, direction)
    if direction == SweepDirection.RIGHT:
        ind1 = left_site_pos
        ind2 = left_site_pos + 1 if left_site_pos < mps.L - 1 else None
        at_end = ind2 == mps.L - 1
        ## canonicalize ket to next site
        if ind1 < mps.L - 1:
            mps.cur_orthog = ind1 + direction
            # self.cur_orthog = ind1 + direction

    else:
        ind2 = left_site_pos
        ind1 = left_site_pos + 1
        at_end = ind2 == 0
        ## canonicalize ket to next site
        if ind1 > 0:
            # self.cur_orthog = ind1 + direction
            mps.cur_orthog = ind1 + direction


    if at_end and not decimate_only:
        # raise NotImplementedError
        site_i = helper_tn.sum_tens(site_i)

        x_ind = mps.bond(ind1, ind1 + direction)
        left_inds = [ind for ind in mps[ind1].inds if ind != x_ind]
        q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=max_bond, bond_ind=x_ind)

        q.transpose_like(mps[ind1], inplace=True)
        mps[ind1].modify(data=q.data)

        r.transpose_like(mps[ind1 + direction], inplace=True)
        mps[ind1 + direction].modify(data=r.data)
        mps.cur_orthog = ind1 + direction

    else:

        x_bonds = [mps.bond(ind2, ind2 + direction), mps.site_ind(ind2)]    ## bonds to not contract over
        ket_iso = [ind for ind in site_i[0].inds if ind not in x_bonds]
        bra_iso = [ind + '_' for ind in ket_iso]

        # [new_ket_site, self.ket[i]]
        x_bond = mps.bond(ind1, ind2)
        tot_rdm = get_tot_rdm(site_i, ket_iso, bra_iso, )
        eigvec  = get_rdm_eig(tot_rdm, ket_iso, bra_iso, new_ind=x_bond + '_', max_bond=max_bond)
        # print('eigvec', eigvec.norm())
        # print(eigvec.data)

        decimate(mps, ind1, eigvec, direction)
        mps.cur_orthog = ind2

    # print('cur orthog', mps.cur_orthog)

    return
