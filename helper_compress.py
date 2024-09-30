from setup_.configs import *
import scipy.sparse
import quimb
import quimb.tensor as qtn

import time
import helper_quimb

"""
trying to correct for local conservation. it didnt really work.
"""


# def compress_local_conservation(mps: 'qtn.MatrixProductState', direction: int, max_bond=None, cutoff=CUTOFF,
#                                 integ_mps: Optional['qtn.MatrixProductState'] = None):
#     """ direction indications coarse --> fine
#     """
#     print('mps', mps.L, helper_quimb.norm(mps), direction)
#
#     if mps.L == 1:
#         return mps
#
#     if helper_quimb.norm(mps) < CUTOFF:
#         return None
#
#     def site_norm_to_exponent(ix):
#         tag = mps.site_tag_id.format(ix)
#         tens = mps[tag]
#         if tens.norm() == 0.0:
#             mps.exponent = -np.inf
#             raise np.linalg.LinAlgError
#         mps.strip_exponent(tens)
#
#     def integrate_left(pos: int):
#         if pos <= 0:
#             return 1.
#
#         if integ_mps is None:
#             ovlp = mps[0].sum_reduce(mps.site_ind_id.format(mps.L - 1))
#             for ix in range(1, pos, 1):
#                 tmp = mps[ix].sum_reduce(mps.site_ind_id.format(ix))
#                 ovlp = qtn.tensor_contract(tmp, ovlp)
#         else:
#             raise NotImplementedError
#         return ovlp
#
#     def integrate_right(pos: int):
#         if pos >= mps.L - 1:
#             return 1.
#
#         if integ_mps is None:
#             print('integ ix', mps.L-1)
#             ovlp = mps[-1].sum_reduce(mps.site_ind_id.format(mps.L-1))
#             for ix in range(mps.L - 2, pos, -1):
#                 print('integ ix', ix, mps.L)
#                 tmp = mps[ix].sum_reduce(mps.site_ind_id.format(ix))
#                 ovlp = qtn.tensor_contract(tmp, ovlp)
#         else:
#             raise NotImplementedError
#         return ovlp
#
#
#     if direction > 0:   ## left -> right = coarse -> fine --> R2L SWEEP
#         helper_quimb.canonize(mps, i=mps.L-1)
#         for i in range(mps.L - 1, 0, -1):
#
#             if i == mps.L - 1:  ## no fine grain basis
#                 print('site i', i)
#                 helper_quimb.right_compress_site(mps, i, max_bond=max_bond, cutoff=cutoff)
#                 continue
#
#             # if i != mps.L - 4:
#             #     print('site i', i)
#             #     helper_quimb.right_compress_site(mps, i, max_bond=max_bond, cutoff=cutoff)
#             #     continue
#
#             print('site i', i)
#             print('check orthog')
#             helper_quimb.check_orthog(mps)
#
#             T1, T2 = mps[i], mps[i+1]
#             rix, lix = T1.filter_bonds(T2)
#             # print('lix', lix, 'rix', rix)
#
#             (TR, TS, TL), (TR_R, TS_R, TL_R) = decompose_tens(T1, rix, right_inds=lix, max_bond=max_bond, cutoff=cutoff)
#             print('Tl', TL, '\nTS', TS, '\nTR',TR)
#
#             if TS_R.norm() > np.sqrt(CUTOFF):
#
#                 ## get error from compression
#                 weights = integrate_right(i)
#                 print('weights', weights.data)
#                 # print('TR_R', TR_R, '\nTS_R', TS_R, '\nTL_R', TL_R)
#                 TSR_R = helper_quimb.diag_mult(TR_R, TS_R)
#                 # print('TS_R', TSR_R, 'weights', weights)
#                 remainder_weights = TSR_R @ weights if isinstance(weights, qtn.Tensor) \
#                                     else TR_R.modify(apply=lambda x: x * weights)
#                 # print('TL_R', TL_R)
#                 # print('remainder weights', remainder_weights)
#                 # cons_err = helper_quimb.diag_mult(TL_R, remainder_weights)
#                 cons_err = TL_R @ remainder_weights     ## lix inds
#                 print('cons err', cons_err.data)
#
#                 ## obtain correction vector from retained singular values
#                 print('TR', TR.data)
#                 print(TR.data @ TR.data.T.conj())
#                 print(TR.data.T.conj() @ TR.data)
#                 TSR = helper_quimb.diag_mult(TR, TS)
#                 print('TS * weights', TS.data, weights.data)
#                 print('TSR', TSR.data * weights.data)
#                 keep_weights = TSR @ weights
#                 A_eff = helper_quimb.diag_mult(TL, keep_weights)
#
#                 A_inds = lix + [ind for ind in A_eff.inds if ind not in lix]
#                 A_eff.transpose(*A_inds, inplace=True)
#                 out_shape, num_weights = A_eff.shape[:-1], A_eff.shape[-1]
#                 A_eff_sq = A_eff.data.reshape(np.prod(out_shape), num_weights)
#                 cons_err_vec = cons_err.data.reshape(np.prod(out_shape))
#                 print('A eff', A_eff_sq.shape, cons_err_vec.shape)
#                 print('A eff', np.linalg.cond(A_eff_sq))
#                 print('cons err vec', cons_err_vec)
#                 print('svals', TS.data)
#                 out = np.linalg.lstsq(A_eff_sq, cons_err_vec, rcond=-1)
#                 s_sel = out[0]
#                 s_sel = np.linalg.pinv(A_eff_sq) @ cons_err_vec
#
#                 plt.figure()
#                 plt.imshow(np.log(np.abs(np.real(A_eff_sq))))
#                 plt.colorbar()
#                 plt.figure()
#                 plt.imshow(np.log(np.abs(np.imag(A_eff_sq))))
#                 plt.colorbar()
#                 plt.show()
#
#
#                 plt.figure()
#                 plt.imshow(np.real(np.linalg.pinv(A_eff_sq)))
#                 plt.colorbar()
#                 plt.figure()
#                 plt.imshow(np.imag(np.linalg.pinv(A_eff_sq)))
#                 plt.colorbar()
#                 plt.show()
#
#                 print('A eff sq', A_eff_sq @ s_sel, np.linalg.norm(A_eff_sq @ s_sel - cons_err_vec))
#                 print('residual', out[1])
#                 print('s sel', s_sel.shape, s_sel)
#
#
#                 TS.modify(apply=lambda x: x + s_sel)
#
#             TLS = helper_quimb.diag_mult(TL, TS)
#             # print('TLS', TLS)
#             # print('T1', T1)
#             TLS.transpose_like(T1, inplace=True)
#             T1.modify(data=TLS.data)
#             print('modified T1')
#
#             # print('TR', TR)
#             # print('T2', T2)
#             TR2 = TR @ T2
#             TR2.transpose_like(T2, inplace=True)
#             T2.modify(data=TR2.data)
#             print('modified T2')
#
#             print('check orthog')
#             helper_quimb.check_orthog(mps)
#
#             helper_quimb.right_compress_site(mps, i)
#             print('mps', mps)
#
#     else:
#         raise NotImplementedError
#         for i in range(mps.L-1):
#             T1, T2 = mps[i], mps[i + 1]
#             rix, lix = T1.filter_bonds(T2)
#             (TL, TS, TR), (TL_R, TS_R, TR_R) = decompose_tens(mps[i], lix, right_inds=rix, max_bond=max_bond,
#                                                                    cutoff=cutoff)
#
#             if TS_R.norm() > np.sqrt(CUTOFF):
#                 ## get error from compression
#                 weights = integrate_left(i)
#                 TLS_R = helper_quimb.diag_mult(TL_R, TS_R)
#                 remainder_weights = TLS_R @ weights if isinstance(weights, qtn.Tensor) \
#                     else TR_R.modify(apply=lambda x: x * weights)
#                 cons_err = helper_quimb.diag_mult(TR_R, remainder_weights)
#
#                 ## obtain correction vector from retained singular values
#                 TLS = helper_quimb.diag_mult(TL, TS)
#                 keep_weights = TLS @ weights
#                 A_eff = helper_quimb.diag_mult(TR, keep_weights)
#
#                 A_inds = [ind for ind in A_eff.inds if ind not in lix] + lix
#                 A_eff.transpose(A_inds, inplace=True)
#                 out_shape, num_weights = A_eff.shape[:-1], A_eff.shape[-1]
#                 A_eff_sq = A_eff.data.reshape(np.prod(out_shape), num_weights)
#                 s_sel = np.linalg.solve(A_eff_sq, -cons_err.data)
#
#                 TS.modify(apply=lambda x: x - s_sel)
#
#             TLS = helper_quimb.diag_mult(TL, TS)
#             TLS.transpose_like(T2, inplace=True)
#             T2.modify(data=TLS.data)
#
#             TR.transpose_like(T1, inplace=True)
#             T1.modify(data=TR.data)
#
#     return mps

def check_local_cons(ref_mps: 'qtn.MatrixProductState', chk_mps: 'qtn.MatrixProductState', direction: int):

    if direction > 0:  ## coarse to fine
        for i in range(ref_mps.L):
            tmp_ref = ref_mps.copy()
            for ix in range(i, ref_mps.L):
                tmp_ref[ix].sum_reduce(ref_mps.site_ind_id.format(ix), inplace=True)
            # tmp_out = tmp_ref.contract()

            tmp_chk = chk_mps.copy()
            for ix in range(i, chk_mps.L):
                tmp_chk[ix].sum_reduce(ref_mps.site_ind_id.format(ix), inplace=True)

            # print('tmp ref', tmp_ref)
            # print('tmp chk', tmp_chk)

            if i == 0:
                print('proj val', qtn.tensor_contract(*tmp_ref.tensors))
            print(f'diff level {i}', helper_quimb.distance(tmp_ref, tmp_chk)/helper_quimb.norm(tmp_chk))
    else:
        raise NotImplementedError

    return


def target_MPS(mps: qtn.MatrixProductState, target_mps: qtn.MatrixProductState, pos: int):
    """ project mps onto space defined by target_mps
    """
    target_mps.mangle_inner_()

    ovlp = qtn.tensor_contract(*mps[:pos + 1].tensors, *target_mps.conj()[:pos + 1].tensors)
    ovlp.drop_tags()
    mod_tens = mps[pos + 1] @ ovlp
    new_tensors = target_mps[:pos + 1].tensors + (mod_tens,) + mps[pos+2:].tensors
    out = qtn.TensorNetwork(new_tensors)
    out.view_like(mps, inplace=True)
    out.exponent = mps.exponent
    return out



def coarse_grain_project(mps: 'qtn.MatrixProductState', direction: int, max_bond:int = None):

    mps_comp = mps.copy()
    helper_quimb.canonize(mps_comp, i=0, scale=False)
    helper_quimb.canonize(mps, i=0, scale=False)

    if direction > 0:    # coarse to fine

        for i in range(mps.L - 1):
            helper_quimb.left_compress_site(mps_comp, i, max_bond=max_bond)
            helper_quimb.left_compress_site(mps, i)
            print('exponents', mps_comp.exponent, mps.exponent)

            helper_quimb.check_orthog(mps_comp)

            mps_target = mps.copy()
            mps_target = target_MPS(mps_target, mps_comp, i)


            helper_quimb.check_orthog(mps)


            print('max bond', mps_target.max_bond(), mps_comp.max_bond(), mps.max_bond())
            print('compress dist', helper_quimb.distance(mps_comp, mps))
            print('target dist', helper_quimb.distance(mps_target, mps))
            check_local_cons(mps, mps_target, 1)

    return



def compress_local_conservation(mps: 'qtn.MatrixProductState', direction: int, max_bond=None, cutoff=CUTOFF,
                                integ_mps: Optional['qtn.MatrixProductState'] = None):
    """ direction indications coarse --> fine
        integration includes site of compression
    """

    coarse_grain_project(mps, direction, max_bond=max_bond)
    exit()

    print('mps', mps.L, helper_quimb.norm(mps), direction)
    ref_mps = mps.copy()

    if mps.L == 1:
        return mps

    if helper_quimb.norm(mps) < CUTOFF:
        return None

    def site_norm_to_exponent(ix):
        tag = mps.site_tag_id.format(ix)
        tens = mps[tag]
        if tens.norm() == 0.0:
            mps.exponent = -np.inf
            raise np.linalg.LinAlgError
        mps.strip_exponent(tens)

    def integrate_left(pos: int):
        if pos <= 0:
            return 1.

        if integ_mps is None:
            ovlp = mps[0].sum_reduce(mps.site_ind_id.format(mps.L - 1))
            for ix in range(1, pos, 1):
                tmp = mps[ix].sum_reduce(mps.site_ind_id.format(ix))
                ovlp = qtn.tensor_contract(tmp, ovlp)
        else:
            raise NotImplementedError
        return ovlp

    def integrate_right(pos: int):
        if pos >= mps.L - 1:
            return 1.

        # print('integ check orthog')
        # helper_quimb.check_orthog(mps)

        if integ_mps is None:
            ovlp = mps[-1].sum_reduce(mps.site_ind_id.format(mps.L-1))
            for ix in range(mps.L - 2, pos, -1):
                tmp = mps[ix].sum_reduce(mps.site_ind_id.format(ix))
                ovlp = qtn.tensor_contract(tmp, ovlp)
        else:
            raise NotImplementedError

        # tmp = mps[pos+1:]
        # tmp = qtn.tensor_contract(*(mps[pos+1:].tensors))
        # for x in range(pos+1, mps.L):
        #     tmp.sum_reduce(mps.site_ind_id.format(x), inplace=True)
        # print('tmp', tmp)
        # print('ovlp', tmp.data, ovlp.data)
        # exit()

        return ovlp


    if direction > 0:   ## left -> right = coarse -> fine --> L2R sweep
        helper_quimb.canonize(mps, i=0)
        for i in range(mps.L -1):

            # if i == mps.L - 1:  ## no fine grain basis
            #     print('site i', i)
            #     helper_quimb.right_compress_site(mps, i, max_bond=max_bond, cutoff=cutoff)
            #     continue

            if i >= 4:
                print('site i', i)
                helper_quimb.left_compress_site(mps, i) # , max_bond=max_bond, cutoff=cutoff)
                continue

            print('site i', i)
            print('check orthog')
            helper_quimb.check_orthog(mps)

            if i > 0:
                # weights = integrate_right(i)
                ref_integ = mps.copy()
                for ix in range(i, mps.L):
                    ref_integ[ix].sum_reduce(mps.site_ind_id.format(ix), inplace=True)
                tmp = qtn.tensor_contract(*(ref_integ[i:].tensors))
                ref_integ = ref_integ[:i + 1]
                ref_integ._L = i + 1
                ref_integ[i].modify(data=tmp.data, inds=tmp.inds)
                # print('ref integ', ref_integ)



            T1, T2 = mps[i], mps[i+1]
            rix, lix = T1.filter_bonds(T2)
            # print('lix', lix, 'rix', rix)

            (TL, TS, TR), (TL_R, TS_R, TR_R) = decompose_tens(T1, lix, right_inds=rix, max_bond=max_bond,
                                                              cutoff=cutoff)
            # print('Tl', TL, '\nTS', TS, '\nTR',TR)

            keep = TL @ helper_quimb.diag_mult(TR, TS)
            remainder = TL_R @ helper_quimb.diag_mult(TR_R, TS_R)
            print('ovlp keep @ remainder', keep @ remainder)

            if TS_R.norm() > np.sqrt(CUTOFF):

                ## get error from compression
                weights = integrate_right(i)

                remainder_weights = TR_R @ weights
                TSRW_R = helper_quimb.diag_mult(remainder_weights, TS_R)
                TL_R_sum = TL_R.sum_reduce(mps.site_ind_id.format(i))
                cons_err = TL_R_sum @ TSRW_R     ## lix inds
                # cons_err = helper_quimb.diag_mult(TL_R_sum, remainder_weights)
                print('cons err', cons_err.data)

                ## obtain correction vector from retained singular values
                TRW = TR @ weights
                TL_sum = TL.sum_reduce(mps.site_ind_id.format(i))
                # A_eff = TL_sum @ keep_weights #
                A_eff = helper_quimb.diag_mult(TL_sum, TRW)

                print('cons err', cons_err)
                print('Aeff', A_eff)
                # print('cons err @ A eff', cons_err @ A_eff)
                # exit()

                lix_ = [ind for ind in lix if ind != mps.site_ind_id.format(i)]
                A_inds = lix_ + [ind for ind in A_eff.inds if ind not in lix]
                A_eff.transpose(*A_inds, inplace=True)
                out_shape, num_weights = A_eff.shape[:-1], A_eff.shape[-1]
                A_eff_sq = A_eff.data.reshape(np.prod(out_shape), num_weights)
                cons_err_vec = cons_err.data.reshape(np.prod(out_shape))
                s_sel = np.linalg.pinv(A_eff_sq) @ cons_err_vec
                print('err', np.linalg.norm(A_eff_sq @ s_sel - cons_err_vec) ** 2)
                print('s sel', s_sel.shape, s_sel)
                # exit()

                TS.modify(apply=lambda x: x + s_sel)

                chk = TL_sum @ helper_quimb.diag_mult(TR @ weights , TS)
                print('chk', chk.data, chk.inds)
                ref_T1_integ = T1 @ weights
                ref_T1_integ.sum_reduce_(mps.site_ind_id.format(i))
                print('ref T1', ref_T1_integ.data, ref_T1_integ.inds)
                print('errs', ref_T1_integ.data - chk.data)
                print('orig errs', cons_err_vec)

                TS_sel = TS.copy()
                TS_sel.modify(data=s_sel)
                chk1 = TL_sum @ helper_quimb.diag_mult(TR @ weights , TS_sel)
                print('chk diff from remainder', chk1.data - cons_err_vec)

                ref_integ_2 = ref_integ.copy()
                ref_integ_2[i].modify(data=chk.data, inds=chk.inds)
                print('ref integ 2 ', i, ref_integ_2)
                print('ref integ', ref_integ)
                print('diff ref integ', helper_quimb.distance(ref_integ, ref_integ_2))
                ref_integ_2[i-1].sum_reduce(mps.site_ind_id.format(i-1), inplace=True)
                ref_integ[i-1].sum_reduce(mps.site_ind_id.format(i-1), inplace=True)
                print('DIFF ref integ', helper_quimb.distance(ref_integ, ref_integ_2))
                print()


            # TLS = helper_quimb.diag_mult(TL, TS)
            # print('TLS', TLS)
            # print('T1', T1)
            TL.transpose_like(T1, inplace=True)
            T1.modify(data=TL.data)
            print('modified T1')

            # print('TR', TR)
            # print('T2', T2)
            TSR = helper_quimb.diag_mult(TR, TS)
            TSR2 = TSR @ T2
            TSR2.transpose_like(T2, inplace=True)
            T2.modify(data=TSR2.data)
            print('modified T2')

            print('chk')
            check_local_cons(ref_mps, mps, 1)

            print('check orthog')
            helper_quimb.check_orthog(mps)
            print('--')

            # helper_quimb.left_compress_site(mps, i)
            # print('mps', mps)

    else:
        raise NotImplementedError
        for i in range(mps.L-1):
            T1, T2 = mps[i], mps[i + 1]
            rix, lix = T1.filter_bonds(T2)
            (TL, TS, TR), (TL_R, TS_R, TR_R) = decompose_tens(mps[i], lix, right_inds=rix, max_bond=max_bond,
                                                                   cutoff=cutoff)

            if TS_R.norm() > np.sqrt(CUTOFF):
                ## get error from compression
                weights = integrate_left(i)
                TLS_R = helper_quimb.diag_mult(TL_R, TS_R)
                remainder_weights = TLS_R @ weights if isinstance(weights, qtn.Tensor) \
                    else TR_R.modify(apply=lambda x: x * weights)
                cons_err = helper_quimb.diag_mult(TR_R, remainder_weights)

                ## obtain correction vector from retained singular values
                TLS = helper_quimb.diag_mult(TL, TS)
                keep_weights = TLS @ weights
                A_eff = helper_quimb.diag_mult(TR, keep_weights)

                A_inds = [ind for ind in A_eff.inds if ind not in lix] + lix
                A_eff.transpose(A_inds, inplace=True)
                out_shape, num_weights = A_eff.shape[:-1], A_eff.shape[-1]
                A_eff_sq = A_eff.data.reshape(np.prod(out_shape), num_weights)
                s_sel = np.linalg.solve(A_eff_sq, -cons_err.data)

                TS.modify(apply=lambda x: x - s_sel)

            TLS = helper_quimb.diag_mult(TL, TS)
            TLS.transpose_like(T2, inplace=True)
            T2.modify(data=TLS.data)

            TR.transpose_like(T1, inplace=True)
            T1.modify(data=TR.data)

    return mps


def decompose_tens(tens: 'qtn.Tensor', left_inds: Sequence[str], right_inds: Optional[Sequence[int]]=None,
                       max_bond=None, cutoff=CUTOFF):
    """ returns SVD decomposition and the remainder
    """

    if right_inds is None:
        right_inds = [ind for ind in tens.inds if ind not in left_inds]

    # print('decompose tens', tens)
    # print('left inds', left_inds)
    U, svals, VT = tens.split(left_inds, get='arrays', absorb=None, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)
    # print('U', U.shape, 's', svals.shape, 'VT', VT.shape)

    cut_idx = np.argmax(svals < cutoff * svals[0])
    cut_idx = len(svals) if cut_idx == 0 else cut_idx
    max_bond = len(svals) if max_bond is None else max_bond
    cut_idx = min(max_bond, cut_idx)

    U_R = U.take(np.arange(cut_idx,len(svals)), axis=-1)
    svals_R = svals[cut_idx:]
    VT_R = VT.take(np.arange(cut_idx, len(svals)), axis=0)
    # print('cut idx', cut_idx, U_R.shape, svals_R.shape, VT_R.shape)

    U =  U.take(np.arange(cut_idx), axis=-1)
    svals = svals[:cut_idx]
    VT = VT.take(np.arange(cut_idx), axis=0)

    TL = qtn.Tensor(data=U, inds=(*left_inds, f'tmp_{left_inds[0]}'))
    TS = qtn.Tensor(data=svals, inds=(f'tmp_{left_inds[0]}',))
    TR = qtn.Tensor(data=VT, inds=(f'tmp_{left_inds[0]}', *right_inds))
    # print('TL', TL, '\nTS', TS, '\nTR', TR)
    TL_R = qtn.Tensor(data=U_R, inds=(*left_inds, f'tmp_{left_inds[0]}'))
    TS_R = qtn.Tensor(data=svals_R, inds=(f'tmp_{left_inds[0]}',))
    TR_R = qtn.Tensor(data=VT_R, inds=(f'tmp_{left_inds[0]}', *right_inds))

    return (TL, TS, TR), (TL_R, TS_R, TR_R)


# def right_compress_tens(tens: 'qtn.Tensor', right_inds: Sequence[str], left_inds: Optional[Sequence[int]]=None,
#                        max_bond=None, cutoff=CUTOFF):
#     """ inplace compression of ith site of mps
#         also returns remainder
#     """
#
#     (T1, T2), (T1_R, T2_R) = left_compress_tens(tens, right_inds, right_inds=left_inds,
#                                                 max_bond=max_bond, cutoff=cutoff)
#     return (T2, T1), (T2_R, T1_R)


def check_local_cons(ref_mps: 'qtn.MatrixProductState', chk_mps: 'qtn.MatrixProductState', direction: int):

    if direction > 0:  ## coarse to fine
        for i in range(ref_mps.L):
            tmp_ref = ref_mps.copy()
            for ix in range(i, ref_mps.L):
                tmp_ref[ix].sum_reduce(ref_mps.site_ind_id.format(ix), inplace=True)
            # tmp_out = tmp_ref.contract()

            tmp_chk = chk_mps.copy()
            for ix in range(i, chk_mps.L):
                tmp_chk[ix].sum_reduce(ref_mps.site_ind_id.format(ix), inplace=True)

            # print('tmp ref', tmp_ref)
            # print('tmp chk', tmp_chk)

            if i == 0:
                print('proj val', qtn.tensor_contract(*tmp_ref.tensors))
            print(f'diff level {i}', helper_quimb.distance(tmp_ref, tmp_chk)/helper_quimb.norm(tmp_chk))
    else:
        raise NotImplementedError

    return