from setup_.configs import *
import scipy.sparse
import quimb
import quimb.tensor as qtn

import time
import helper_quimb


def coarse_grain(mps: 'qtn.MatrixProductState', i: int, direction: int, integ_mps: 'qtn.MatrixProductState'=None):

    if direction > 0:
        if i >= mps.L - 1:
            return mps

        ref_integ = mps.copy()
        for ix in range(i, mps.L):
            ref_integ[ix].sum_reduce(mps.site_ind_id.format(ix), inplace=True)
        tmp = qtn.tensor_contract(*(ref_integ[i:].tensors))
        ref_integ = ref_integ[:i + 1]
        ref_integ._L = i + 1
        ref_integ[i].modify(data=tmp.data, inds=tmp.inds)

        npts = 2**(mps.L - i)
        ref_integ.exponent = mps.exponent + np.log10(1./npts)
        return ref_integ
    else:
        raise NotImplementedError


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



def multiscale_compress(mps: 'qtn.MatrixProductState', max_bond: int):

    if max_bond is None or max_bond <= 0:
        return mps

    # mps.canonize(mps.L - 1)
    pos = int(np.round(np.log2(max_bond))) * 2 - 1
    num_integ = mps.L - pos

    integ_mps = coarse_grain(mps, pos, 1)
    print('integ mps', pos, mps.L, integ_mps)
    helper_quimb.canonize(integ_mps, i=pos, scale=False)
    sub_mps = mps[:pos + 1]
    sub_mps._L = pos + 1
    print('integ mps', integ_mps)
    print('integ mps', integ_mps.exponent, mps.exponent)

    L = integ_mps.L - 1     ## len of integrated mps (last tensor does not have physical bond)
    num_x = 1               ## number of indices/sites to use to extrapolate onto Cheb grid
    num_r = L - num_x       ## number of remaining sites
    print('L', L)

    L_tens = interpolate_fine_to_cheb(num_x, max_bond)
    L_tens = L_tens.unfuse({'dyad': [mps.site_ind_id.format(i) for i in range(num_r, L)]},
                           {'dyad': [2]*num_x})
    print('L tens', L_tens)
    print('integ mps partial', num_r, integ_mps[num_r:])
    ## projection of integ_mps and contraction with the site to the left
    proj_L = qtn.tensor_contract(*integ_mps[num_r-1:], L_tens)
    proj_L.drop_tags()
    proj_L.add_tag(integ_mps.site_ind_id.format(num_r-1))
    print('proj L', proj_L)
    ## note: this may yield MPS with rank smaller than max_bond
    ## can we estimate the rank from L_ovlp?

    ## then re-extrapolate with interpolation tensors
    TM, TR = multiscale_interpolating_tensors(max_bond)

    ## transpose inds to p l r
    if num_r > 1:
        keep_mps = integ_mps[:num_r - 1]
        keep_L = keep_mps.num_tensors
        print('keep mps', pos, num_r, keep_mps)
        for i in range(keep_L):
            inds = (integ_mps.site_ind_id.format(i),)
            if i > 0:
                left = keep_mps[i].bonds(keep_mps[i-1])
                inds = inds + tuple(left)
            if i < keep_L - 1:
                right = keep_mps[i].bonds(keep_mps[i+1])
                inds = inds + tuple(right)
            elif i == keep_L - 1:
                right = keep_mps[i].bonds(proj_L)
                inds = inds + tuple(right)
            print('transpose inds', inds)
            keep_mps[i].transpose(*inds, inplace=True)

        print('keep mps', keep_mps, num_r - 2)
        left = proj_L.bonds(keep_mps[num_r - 2])
        right = 'cheb'
        inds = (integ_mps.site_ind_id.format(num_r-1), next(iter(left)), right)
        print('proj L', proj_L, 'transpose inds', inds)
        proj_L.transpose(*inds, inplace=True)

    else:   ## num_r == 1, so keep_mps[0] is proj_L

        inds = (integ_mps.site_ind_id.format(num_r-1), 'cheb')
        proj_L.transpose(*inds, inplace=True)


    new_tensors = integ_mps.tensors[:num_r-1] + (proj_L,) + (TM,) * (mps.L - num_r-1) + (TR,)
    # new_mps = qtn.TensorNetwork(new_tensors)

    print('new tensors', new_tensors)

    new_mps = qtn.MatrixProductState([tens.data for tens in new_tensors], shape='plr',)
    new_mps.exponent = integ_mps.exponent
    print('dx', new_mps.exponent, integ_mps.exponent)
    # new_mps.view_like(mps, inplace=True)

    print('new mps', new_mps)
    helper_quimb.compress(new_mps, scale=False)
    print('compressed max bond', new_mps.max_bond())

    ## maybe measure amount to coarse grain based on |d/dx psi|
    ## remove high-frequency polynomials when mapping to Chebyshev nodes?

    return new_mps





def multiscale_interpolating_tensors(nbasis:int):

    k = np.arange(nbasis)
    cheb_grid = (np.cos( k * np.pi / (nbasis - 1) ) + 1) /  2   ##  on [0,1] instead of [-1, 1]

    ## lagrange interpolating polynomials
    ## p_j(x) = y_j prod (x - x_k) / (x_j - x_k)
    ## where y_j is the value of the function at x_j; x_k are chebyshev nodes
    ## y_j is stored in the previous tensor;
    ## x is the new (finer) chebyshev grid
    PN_ab = np.zeros((2, nbasis, nbasis))
    for j in range(nbasis):
        poly = 1
        for k in range(nbasis):
            if j != k:
                poly = poly * (0.5 * cheb_grid - cheb_grid[k]) / (cheb_grid[j] - cheb_grid[k])
        PN_ab[0,j,:] = poly

        poly = 1
        for k in range(nbasis):
            if j != k:
                poly = poly * (0.5 + 0.5 * cheb_grid - cheb_grid[k]) / (cheb_grid[j] - cheb_grid[k])
        PN_ab[1, j, :] = poly

    PN_ab = qtn.Tensor(data=PN_ab, inds=('sigma','coarse','fine'))

    ## last tensor
    ## P_j(0) and P_j(1)
    PR_b = np.zeros((2, nbasis))
    for j in range(nbasis):
        poly = 1
        for k in range(nbasis):
            if j != k:
                poly = poly * (0 - cheb_grid[k]) / (cheb_grid[j] - cheb_grid[k])
        PR_b[0,j] = poly

        poly = 1
        for k in range(nbasis):
            if j != k:
                poly = poly * (0.5 - cheb_grid[k]) / (cheb_grid[j] - cheb_grid[k])
        PR_b[1, j] = poly

    PR_b = qtn.Tensor(data=PR_b, inds=('sigma', 'coarse'))

    return PN_ab, PR_b


def interpolate_fine_to_cheb(depth:int, nbasis: int):
    """ from fine dyadic grid, get the function approximated at fine Chebyshev grids
        Lagrange interpolation tensor (from michael in section 5)
    """
    k = np.arange(nbasis)
    cheb_grid = (np.cos(k * np.pi / (nbasis - 1)) + 1) / 2  ##  on [0,1] instead of [-1, 1]
    dyadic_grid = np.linspace(0, 1, 2**depth, endpoint=False)

    lagrange_tens = np.zeros((2**depth, nbasis))
    for j in range(2**depth):
        poly = 1.
        for k in range(2**depth):
            if j != k:
                poly = poly * (cheb_grid - dyadic_grid[k]) / (dyadic_grid[j] - dyadic_grid[k])
        lagrange_tens[j,:] = poly

    L_tens = qtn.Tensor(data=lagrange_tens, inds=('dyad','cheb'))
    return L_tens








