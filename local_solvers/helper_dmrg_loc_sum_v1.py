from setup_.configs import *
import time
import scipy.linalg as linalg
import quimb.tensor as qtn
import helper_quimb
from local_solvers.defaults import *
import local_solvers.helper_tn as helper_tn
from local_solvers.mps_classes import MPS
from gridTN import GridTN

if TYPE_CHECKING:
    from grid import Grid


MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']

"""
like helper_dmrg_loc but retains correction terms after compression
"""


CUTOFF = 1e-24
from local_solvers.helper_dmrg_loc import get_tens_rdm, get_tot_rdm, decimate


# def is_small(gtn: 'GridTN', cur_orthog=None, ref_norm=1.0, rel_cutoff=CUTOFF, abs_cutoff=CUTOFF):
#
#     if cur_orthog is None:
#         ket_norm = gtn.frobenius_norm()
#     else:
#         ket_norm = gtn[cur_orthog].norm() * 10 ** gtn.exponent
#
#     return ket_norm > np.sqrt(abs_cutoff) or ket_norm > rel_cutoff * ref_norm

def is_small(val: Union['GridTN', 'qtn.Tensor', Numeric], exponent=0.0,
             ref_norm=None, rel_cutoff=CUTOFF, abs_cutoff=CUTOFF):

    if isinstance(val, GridTN):
        val = val.frobenius_norm()
    elif isinstance(val, qtn.Tensor):
        val = val.norm()
    elif not isinstance(val, (int, float, complex)):
        print('val' ,val, type(val))
        raise TypeError


    val *= 10**exponent
    ref_norm = 1.0 if ref_norm is None else ref_norm

    return np.abs(val) < np.sqrt(abs_cutoff) or np.abs(val) < rel_cutoff * np.abs(ref_norm)



def get_rdm_eigsys_all(tot_rdm: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                       new_ind: str=None, max_bond=None, cutoff:Numeric=CUTOFF, abs_cutoff:Numeric=CUTOFF,
                       ref_norm = None,
                       ) -> list[tuple[np.ndarray, qtn.Tensor]]:
    """ A = Q Lambda Q.T  --> return Q
    """

    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    tot_rdm.transpose(*ket_iso, *bra_iso, inplace=True)
    nb = len(ket_iso)

    denmat_shape = tot_rdm.shape
    tens_shape = denmat_shape[:nb]
    sq_shape = np.prod(denmat_shape[:nb])

    eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))
    # print('tot rdm', tot_rdm.norm(), tot_rdm.shape, len(eigval))

    sort_inds = np.argsort(np.abs(eigval))[::-1]

    ## apply basic cut-off (this can be made faster..)
    ev_max = eigval[sort_inds[0]] if ref_norm is None else ref_norm
    new_sort_inds = []
    for si in sort_inds:
        # if not is_small(eigval[si]**2, ref_norm=np.abs(ev_max)):
        #     new_sort_inds += [si]
        # print('is small', is_small(eigval[si], ref_norm=np.abs(ev_max)), eigval[si])

        # check_is_small = is_small(np.sqrt(abs(eigval[si])), ref_norm=np.abs(ev_max))
        # orig_check = ((np.abs(eigval[si]) > CUTOFF * np.abs(ev_max)) and (np.abs(eigval[si]) > abs_cutoff))
        # print('check is_small', check_is_small, 'orig_check', orig_check, check_is_small is not orig_check)
        # if not(check_is_small is not orig_check):
        #     print('is small', is_small(np.sqrt(abs(eigval[si])), ref_norm=np.abs(ev_max)), eigval[si])
        #     print('ev max', ev_max)
        #     print('abs cutoff', np.abs(eigval[si]) > CUTOFF * np.abs(ev_max), CUTOFF * np.abs(ev_max))
        #     print('abs cutoff', np.abs(eigval[si]) > abs_cutoff, abs_cutoff)
        #     raise ValueError

        # if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max) and np.abs(eigval[si]) > abs_cutoff:
        #     new_sort_inds += [si]
        if not is_small(np.sqrt(eigval[si]), ref_norm=np.abs(ev_max)):
            new_sort_inds += [si]
        else:
            break
    sort_inds = new_sort_inds
    # print('len cutoff', CUTOFF, len(sort_inds))

    # exit()

    # print('eigval', eigval, abs_cutoff)

    eigval = eigval[sort_inds]
    eigvec = eigvec[:, sort_inds]

    eigsys_all = []

    def split_eigs(eigval_, eigvec_):
        """ eigvals, eigvecs assumed to be sorted already from largest to smallest
        """
        if eigval_ is None:
            return None, None

        # if (max_bond is not None and len(eigval_) > max_bond) or cutoff > 0.0:
        #     ev_max_ = eigval_[0] if ref_norm is None else ref_norm
        #     c_ind = np.argmax((np.abs(eigval_) < cutoff * np.abs(ev_max_)) + (np.abs(eigval_) < CUTOFF))
        #     # print('ev_max_', ev_max_, cutoff)
        #     # print('eigval_', eigval_)
        #     # print('c ind', c_ind, max_bond)
        #     if c_ind == 0:
        #         c_ind = len(eigval_)
        #     max_bond_ = max(max_bond, c_ind) if max_bond is not None else c_ind
        #     ## use max_bond as a minimum bond dimension unless exact

        if (max_bond is not None and len(eigval_) > max_bond) or cutoff > CUTOFF:
            ev_max_ = eigval_[0]
            max_bond_ = max_bond
            if cutoff > 1.1*CUTOFF:     ## apply cut-off in addition to max bond
                c_ind = np.argmax(np.abs(eigval_) < cutoff * np.abs(ev_max_))
                # print('ev_max_', ev_max_, cutoff)
                # print('eigval_', eigval_)
                # print('c ind', c_ind, max_bond)
                if c_ind == 0:
                    c_ind = len(eigval_)
                max_bond_ = max(max_bond, c_ind) if max_bond is not None else c_ind
                ## use max_bond as a minimum bond dimension unless exact

            # print('rdm eigvec eff max_bond', len(eigval_), max_bond_)

            eigvec_1 = eigvec_[:, :max_bond_]
            eigvec_2 = eigvec_[:, max_bond_:]

            eigval_1 = eigval_[:max_bond_]
            eigval_2 = eigval_[max_bond_:]

            # print('eigvals', len(eigval_1), len(eigval_2))

            # if len(eigval_2) > 0 and np.linalg.norm(eigval_2) < 1.0e-30:
            #     print('norm eigval2 ', np.linalg.norm(eigval_2), ev_max_)
            #     exit()

            if len(eigval_2) == 0 or is_small(np.sqrt(np.linalg.norm(eigval_2)), ref_norm=ev_max_):
                return (eigval_1, eigvec_1), None

            # print('eigval2', np.linalg.norm(eigval_2))
            return (eigval_1, eigvec_1), (eigval_2, eigvec_2)
        else:
            return (eigval_, eigvec_), None

    if len(eigval) > 0:
        eigsys_p, eigsys_r = split_eigs(eigval, eigvec)

        eigval, eigvec = eigsys_p
        eigvec = eigvec.reshape(*tens_shape, -1)
        eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

        eigsys_all += [(eigval, eigvec)]

        while True:
            # print('eigsys r', eigsys_r)
            if eigsys_r is not None:
                eigsys_p, eigsys_r = split_eigs(*eigsys_r)

                eigval, eigvec = eigsys_p
                eigvec = eigvec.reshape(*tens_shape, -1)
                eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

                eigsys_all += [(eigval, eigvec)]
            else:
                break

    # print('eigsys all', len(eigsys_all))
    return eigsys_all


def get_rdm_eigvec_all(tot_rdm: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                    new_ind: str=None, max_bond=None, cutoff:Numeric=CUTOFF, abs_cutoff:Numeric=CUTOFF,
                    ) -> list[qtn.Tensor]:
    """ A = Q Lambda Q.T  --> return Q
    """
    eigsys_all = get_rdm_eigsys_all(tot_rdm, ket_iso, bra_iso, new_ind=new_ind,
                                    max_bond=max_bond, cutoff=cutoff, abs_cutoff=abs_cutoff)
    return [es[1] for es in eigsys_all]


def update_1site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                 direction: 'SweepDirection', max_bond: int=None, cutoff: Numeric=CUTOFF, num_tiers:int = 1,
                 ref_norm=None,
                 ) -> list['qtn.MatrixProductState']:
    """ update ket, bra with new_site
        i: int of mps site
        inplace operation for primary update, but also returns additional mps
        returns mps projected onto primary basis, mps projected onto remainder basis
            (length of list up to num_tiers + 1)
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
            mps._cur_orthog = ind1 + direction
            # self.cur_orthog = ind1 + direction

    else:
        ind1 = left_site_pos
        ind2 = (left_site_pos - 1) if left_site_pos > 0 else None
        ## canonicalize ket to next site
        if ind1 > 0:
            # self.cur_orthog = ind1 + direction
            mps._cur_orthog = ind1 + direction

    if ind2 is None:  ## is at_end
        ## update ket
        mps_ = mps
        site_i = helper_tn.sum_tens(site_i)
        if not is_small(site_i, exponent=mps_.exponent, ref_norm=ref_norm, rel_cutoff=cutoff):
            site_i.transpose_like(mps_[ind1], inplace=True)
            mps_[ind1].modify(data=site_i.data)
            return [mps_]
        else:
            return []
    else:

        x_bond = mps.bond(left_site_pos, left_site_pos + direction)    ## bond to contract over
        ket_iso = [ind for ind in site_i[0].inds if ind != x_bond]
        bra_iso = [ind + '_' for ind in ket_iso]

        tot_rdm = get_tot_rdm(site_i, ket_iso, bra_iso, )
        eigvecs_all = get_rdm_eigvec_all(tot_rdm, ket_iso, bra_iso, new_ind=x_bond + '_',
                                         max_bond=max_bond, cutoff=cutoff, abs_cutoff=CUTOFF)

        mps_all = []
        mps_copy = mps.copy()
        for ix, eigvec in enumerate(eigvecs_all):
            mps_ = mps if ix == 0 else mps_copy.copy()
            decimate(mps_, left_site_pos, eigvec, direction)
            mps_._cur_orthog = ind2
            if not is_small(mps_[ind2], exponent=mps_.exponent, ref_norm=ref_norm, rel_cutoff=CUTOFF):
                mps_all += [mps_]

        print('update 1 site', left_site_pos, mps_all)
        return mps_all      ## primary, corrections


def add_MPS_list_remainder(mps_list: Sequence[MPS_type], max_bond: int=None, cutoff: Numeric =CUTOFF,
                           cur_orthog: int=None, direction: int = 1, num_tiers:int = 1, ref_norm: Numeric=None,
                           ) -> tuple[MPS_type, list[MPS_type]]:
    """
    take the sum of mps_list, returning a primary MPS result and a list of order epsilon corrections.
    only works for num_tiers = 1 for now
    """
    assert(num_tiers==1), 'num_tiers > 1 not implemented'

    iter_mps = iter(mps_list)
    out = None
    it = 0
    while out is None or np.isneginf(out.exponent):
        out = next(iter_mps)
        it += 1

    # new_list = []
    # for mps2 in mps_list[it:]:
    #     tmp = mps2.reindex_sites(out.site_ind_id)
    #     tmp = helper_quimb.match_inner_inds(tmp, out, inplace=True)
    #
    #     tmp.exponent -= out.exponent
    #     tmp.distribute_exponent()
    #     tmp.exponent = out.exponent
    #
    #     new_list += [tmp]

    ref_exponent = out.exponent

    # print('check orthog init')
    # print(helper_quimb.check_orthog(out))
    # for tmp in mps_list:
    #     print(helper_quimb.check_orthog(tmp))
    # print('cur orthog', cur_orthog)

    if direction >= 0:

        final_i = out.L - 1
        if max_bond is not None:
            final_i = out.L - int(np.round(np.log(max_bond)/np.log(out.phys_dim())))
        print('final i ', max_bond, out.phys_dim(), final_i, out.L)

        # helper_quimb.canonize(out, i=0, scale=False)
        # for tmp in new_list:
        #     helper_quimb.canonize(tmp, i=0, scale=False)
        # cur_orthog = 0
        new_list = []
        for tmp in mps_list:
            if tmp is not None:
                tmp = tmp.reindex_sites(out.site_ind_id)
                tmp = helper_quimb.match_inner_inds(tmp, out, inplace=True)
                tmp.exponent -= ref_exponent
                tmp.distribute_exponent()
                tmp.exponent = ref_exponent

                # if tmp is not out:
                #     tmp = tmp.reindex_sites(out.site_ind_id)
                #     tmp = helper_quimb.match_inner_inds(tmp, out, inplace=True)
                #     tmp.exponent -= ref_exponent
                #     tmp.distribute_exponent()
                #     tmp.exponent = ref_exponent
                # else:
                #     tmp = tmp.copy()
                #     out = tmp
                #     print('tmp was out')

                if cur_orthog is None:
                    helper_quimb.canonize(tmp, i=0, scale=False)

                new_list += [tmp]

        cur_orthog = 0 if cur_orthog is None else cur_orthog


        norms = [tmp[cur_orthog].norm() if tmp is not None else 0.0 for tmp in new_list ]
        sort_inds = np.argsort(norms)[::-1]
        out = new_list[sort_inds[0]]
        new_list = [new_list[x] for x in sort_inds[1:]]

        # print('check orthog init 2')
        # print('out', helper_quimb.check_orthog(out))
        # print('tmp')
        # for tmp in new_list:
        #     print(helper_quimb.check_orthog(tmp))
        #
        # print('cur orthog', cur_orthog)
        # exit()


        ref_norm = (norms[sort_inds[0]] * 10 ** out.exponent) if ref_norm is None else ref_norm

        all_remainders = []
        for i in range(cur_orthog, final_i):

            out_mps_list = update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1,
                                        max_bond=max_bond, cutoff=cutoff, ref_norm=ref_norm)
            print('len out_mps_list (1)', i, len(out_mps_list), [helper_quimb.norm(oe) for oe in out_mps_list])

            if len(out_mps_list) == 0:
                return None, []

            # print('out mps list norms', [helper_quimb.norm(oe) for oe in out_mps_list],
            #       [oe[i].shape for oe in out_mps_list])
            # print('out', out.exponent)
            # helper_quimb.check_orthog(out)

            new_list_copy = [tmp.copy() for tmp in new_list]

            # print('check orthog', i)
            # print(helper_quimb.check_orthog(out))

            ### project other terms onto dominant "out" basis
            for tmp in new_list:
                decimate(tmp, i, out[i], direction=1)
                # print('tmp', tmp.exponent)
                # helper_quimb.check_orthog(tmp)

            ## continue onto the next sites???

            ## deal with the remainder terms
            for iix, out_remainder in enumerate(out_mps_list[1:]):
                print('out remainder', out_remainder[i+1].norm() * 10**ref_exponent)
                # helper_quimb.check_orthog(out_remainder)

                remainder_mps_list = []
                if not is_small(out_remainder[i + 1], exponent=out_remainder.exponent,
                                ref_norm=ref_norm, rel_cutoff=CUTOFF):
                    remainder_mps_list += [out_remainder]

                for tmp in new_list_copy:
                    tmp_remainder = tmp.copy()
                    decimate(tmp_remainder, i, out_remainder[i], direction=1)
                    ## note: norms can go to zero after decimation, yielding r_out = None
                    # if tmp[i + 1].norm() * 10**tmp.exponent > np.sqrt(CUTOFF):
                    if not is_small(tmp[i + 1], exponent=tmp.exponent, ref_norm=ref_norm):
                        remainder_mps_list += [tmp_remainder]

                # print('out mps list norms', [helper_quimb.norm(oe) for oe in remainder_mps_list])
                #
                print('\nGET remainder', iix, cur_orthog, i)
                if len(remainder_mps_list) > 0:
                    r_out, r_remainders = add_MPS_list_remainder(remainder_mps_list, cur_orthog=i + 1,
                                                                 max_bond=max_bond, cutoff=cutoff) # , ref_norm=ref_norm)

                    if r_out is None or is_small(r_out[i + 1], exponent=out.exponent, ref_norm=ref_norm):
                        all_remainders += [*r_remainders]
                    else:
                        all_remainders += [r_out, *r_remainders]

        print('final update ', final_i)
        helper_quimb.check_orthog(out)
        for tmp in new_list:
            print(helper_quimb.check_orthog(tmp))

        ### update of few sites
        submps_list = [helper_quimb.get_submpx(out, final_i, -1)]
        submps_list += [helper_quimb.get_submpx(tmp, final_i, -1) for tmp in new_list]
        out_submps = helper_quimb.add_MPS_list(submps_list, inplace=False,)
        out = helper_quimb.append_mpx(out[:final_i], out_submps)
        out.exponent = ref_exponent

        # out, = update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1, max_bond=max_bond,
        #                    cutoff=CUTOFF, ref_norm=ref_norm)

        helper_quimb.check_orthog(out)

        if out is None or is_small(out[-1], exponent=out.exponent, ref_norm=ref_norm):
            exit()
            if len(all_remainders) > 0:
                out = all_remainders[0]
                all_remainders = all_remainders[1:]
            else:
                out = None
                print('out is None')
                exit()

        print('len remainders', len(all_remainders))
        return out, all_remainders

    elif direction < 0:
        raise NotImplementedError



def add_MPS_list_remainder_v2(mps_list: Sequence[MPS_type], max_bond: int=None, cutoff: Numeric =CUTOFF,
                           cur_orthog: int=None, direction: int = 1, num_tiers:int = 1, ref_norm: Numeric=None,
                           ) -> tuple[MPS_type, list[MPS_type]]:
    """
    take the sum of mps_list, returning a primary MPS result and a list of order epsilon corrections.
    only works for num_tiers = 1 for now
    """
    assert(num_tiers==1), 'num_tiers > 1 not implemented'

    iter_mps = iter(mps_list)
    out = None
    it = 0
    while out is None or np.isneginf(out.exponent):
        out = next(iter_mps).copy()
        it += 1

    new_list = []
    for mps2 in mps_list[it:]:
        tmp = mps2.reindex_sites(out.site_ind_id)
        tmp = helper_quimb.match_inner_inds(tmp, out, inplace=True)

        tmp.exponent -= out.exponent
        tmp.distribute_exponent()
        tmp.exponent = out.exponent

        new_list += [tmp]

    if direction >= 0:

        if cur_orthog is None:
            helper_quimb.canonize(out, i=0, scale=False)
            for tmp in new_list:
                helper_quimb.canonize(tmp, i=0, scale=False)
            cur_orthog = 0

        # print('out', out.exponent)
        # helper_quimb.check_orthog(out)
        # print('cutoff', cutoff)

        # ## approx estimate of the norm
        # if ref_norm is None:
        #     # ref_norm = np.sum( [ct[cur_orthog].norm() for ct in mps_list] ) * 10**out.exponent
        #     ref_norm = out[cur_orthog].norm() * 10 ** out.exponent

        all_remainders = []
        # i = cur_orthog
        # if i < out.L - 1:
        for i in range(cur_orthog, out.L - 1):

            # print('i', i)
            ### update site i in "out"
            # print('out[i]', helper_quimb.norm(out), out.exponent, out[i].norm())
            out_mps_list = update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1,
                                        max_bond=max_bond, cutoff=cutoff, ref_norm=ref_norm)
            print('len out_mps_list (1)', i, len(out_mps_list), [helper_quimb.norm(oe) for oe in out_mps_list])

            if len(out_mps_list) == 0:
                return None, []

            # print('out mps list norms', [helper_quimb.norm(oe) for oe in out_mps_list],
            #       [oe[i].shape for oe in out_mps_list])
            # print('out', out.exponent)
            # helper_quimb.check_orthog(out)

            new_list_copy = [tmp.copy() for tmp in new_list]

            ### project other terms onto dominant "out" basis
            for tmp in new_list:
                decimate(tmp, i, out[i], direction=1)
                # print('tmp', tmp.exponent)
                # helper_quimb.check_orthog(tmp)

            ## continue onto the next sites???

            ## deal with the remainder terms
            for out_remainder in out_mps_list[1:]:

                remainder_mps_list = []
                if not is_small(out_remainder[i + 1], exponent=out_remainder.exponent,
                                ref_norm=ref_norm, rel_cutoff=CUTOFF):
                    remainder_mps_list += [out_remainder]

                for tmp in new_list_copy:
                    tmp_remainder = tmp.copy()  ## a previously updated site i shouldn't matter
                    decimate(tmp_remainder, i, out_remainder[i], direction=1)
                    ## note: norms can go to zero after decimation, yielding r_out = None
                    # if tmp[i + 1].norm() * 10**tmp.exponent > np.sqrt(CUTOFF):
                    if not is_small(tmp[i + 1], exponent=tmp.exponent, ref_norm=ref_norm, rel_cutoff=cutoff):
                        remainder_mps_list += [tmp_remainder]

                # print('out mps list norms', [helper_quimb.norm(oe) for oe in remainder_mps_list])
                #
                # print('\nGET remainder', cur_orthog, i)
                if len(remainder_mps_list) > 0:
                    r_out, r_remainders = add_MPS_list_remainder(remainder_mps_list, cur_orthog=i + 1,
                                                                 max_bond=max_bond, cutoff=cutoff, ref_norm=ref_norm)

                    if r_out is None or is_small(r_out[i + 1], exponent=out.exponent, ref_norm=ref_norm):
                        all_remainders += [*r_remainders]
                    else:
                        all_remainders += [r_out, *r_remainders]

        ### update of last site
        i = out.L - 1
        out, = update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1, max_bond=max_bond,
                           cutoff=CUTOFF, ref_norm=ref_norm)

        if out is None or is_small(out[i], exponent=out.exponent, ref_norm=ref_norm):
            exit()
            if len(all_remainders) > 0:
                out = all_remainders[0]
                all_remainders = all_remainders[1:]
            else:
                out = None
                print('out is None')
                exit()

        print('len remainders', len(all_remainders))
        return out, all_remainders

    elif direction < 0:
        raise NotImplementedError


def add_MPS_list_remainder_v1(mps_list: Sequence[MPS_type], max_bond: int=None, cutoff: Numeric =CUTOFF,
                               cur_orthog: int=None, direction: int = 1, num_tiers:int = 1, ref_norm: Numeric=None,
                               ) -> tuple[MPS_type, list[MPS_type]]:
    """
    take the sum of mps_list, returning a primary MPS result and a list of order epsilon corrections.
    only works for num_tiers = 1 for now
    """
    assert(num_tiers==1), 'num_tiers > 1 not implemented'

    iter_mps = iter(mps_list)
    out = None
    it = 0
    while out is None or np.isneginf(out.exponent):
        out = next(iter_mps).copy()
        it += 1

    new_list = []
    for mps2 in mps_list[it:]:
        tmp = mps2.reindex_sites(out.site_ind_id)
        tmp = helper_quimb.match_inner_inds(tmp, out, inplace=True)

        tmp.exponent -= out.exponent
        tmp.distribute_exponent()
        tmp.exponent = out.exponent

        new_list += [tmp]

    if direction >= 0:

        if cur_orthog is None:
            helper_quimb.canonize(out, i=0, scale=False)
            for tmp in new_list:
                helper_quimb.canonize(tmp, i=0, scale=False)
            cur_orthog = 0

        # print('out', out.exponent)
        # helper_quimb.check_orthog(out)
        # print('cutoff', cutoff)

        ## approx estimate of the norm
        if ref_norm is None:
            # ref_norm = np.sum( [ct[cur_orthog].norm() for ct in mps_list] ) * 10**out.exponent
            ref_norm = out[cur_orthog].norm() * 10 ** out.exponent

        all_remainders = []
        for i in range(cur_orthog, out.L - 1):
            # print('i', i)
            ### update site i in "out"
            # print('out[i]', helper_quimb.norm(out), out.exponent, out[i].norm())
            out_mps_list = update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1,
                                        max_bond=max_bond, cutoff=cutoff, ref_norm=ref_norm)
            print('len out_mps_list (1)', len(out_mps_list), [helper_quimb.norm(oe) for oe in out_mps_list])

            # print('out mps list norms', [helper_quimb.norm(oe) for oe in out_mps_list],
            #       [oe[i].shape for oe in out_mps_list])
            # print('out', out.exponent)
            # helper_quimb.check_orthog(out)

            new_list_copy = [tmp.copy() for tmp in new_list]

            ### project other terms onto dominant "out" basis
            for tmp in new_list:
                decimate(tmp, i, out[i], direction=1)
                # print('tmp', tmp.exponent)
                # helper_quimb.check_orthog(tmp)

            ## deal with the remainder terms
            for out_remainder in out_mps_list[1:]:

                remainder_mps_list = []
                if not is_small(out_remainder[i + 1], exponent=out_remainder.exponent, ref_norm=ref_norm, rel_cutoff=cutoff):
                    remainder_mps_list += [out_remainder]

                for tmp in new_list_copy:
                    tmp_remainder = tmp.copy()  ## a previously updated site i shouldn't matter
                    decimate(tmp_remainder, i, out_remainder[i], direction=1)
                    ## note: norms can go to zero after decimation, yielding r_out = None
                    # if tmp[i + 1].norm() * 10**tmp.exponent > np.sqrt(CUTOFF):
                    if not is_small(tmp[i + 1], exponent=tmp.exponent, ref_norm=ref_norm, rel_cutoff=cutoff):
                        remainder_mps_list += [tmp_remainder]

                # print('out mps list norms', [helper_quimb.norm(oe) for oe in remainder_mps_list])
                #
                print('\nGET remainder', cur_orthog, i)
                if len(remainder_mps_list) > 0:
                    r_out, r_remainders = add_MPS_list_remainder(remainder_mps_list, cur_orthog=i + 1,
                                                                 max_bond=max_bond, cutoff=cutoff, ref_norm=ref_norm)
                    all_remainders += [r_out, *r_remainders]

        print('len all remainders', len(all_remainders))

        ### update of last site
        i = out.L - 1
        update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]], direction=1, max_bond=max_bond,
                     cutoff=cutoff, ref_norm=ref_norm)

        return out, all_remainders

    elif direction < 0:
        raise NotImplementedError
