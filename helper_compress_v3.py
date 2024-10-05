from setup_.configs import *
import scipy.sparse
import quimb
import quimb.tensor as qtn

import time
import helper_quimb
from gridTN_1D import GridTN1D

from helper_compress_v2 import coarse_grain, decompose_tens
from local_solvers.local_dmrg_compress_reference import local_reference_compress



def smooth_k_gtn(gtn: GridTN1D, filter_func: Callable =None,
                 filter_point: Numeric=0.75, filter_width: Numeric=0.01):
    """ take the FFT, filter out high frequncies, take inverse FFT
        filter_weight: parameter (0, 1) determineing shape of filter
        filter_func: callable that returns 1 - filter amount for each NORMALIZED wavevector k
    """

    def lpf(normalized_k: Numeric):
        k0, width = filter_point, filter_width
        # return np.exp((np.abs(normalized_k) - k0) / width)
        return 1. / (np.exp( (np.abs(normalized_k)-k0) / width) + 1)

    ### use a low-pass filter (unless otherwsie defined)
    qft_mpo = gtn.grid.get_qft_mpo()
    iqft_mpo = gtn.grid.get_inverse_qft_mpo()
    filter_func = lpf if filter_func is None else filter_func

    filter_vals = {}
    for ax in gtn.grid.axes:
        # ks = ax.get_qft_freqs()
        # ks_dat = ax.map_mps_to_state(ks)
        ks_dat = ax.get_qft_ks()
        kmax = np.pi / ax.dx
        filter_vals[ax] = filter_func(ks_dat/kmax)

        # plt.figure()
        # plt.plot(lpf(np.linspace(0,1)))
        #
        # plt.figure()
        # plt.plot(ks_dat/kmax, filter_func(ks_dat/kmax), 'x-')
        # plt.show()


    filter_gtn = gtn.grid.make_mps_ndim(filter_vals)

    gtn_k = gtn.apply(qft_mpo)
    gtn_k.elemental_multiply(filter_gtn, inplace=True)
    gtn_lp = gtn_k.apply(iqft_mpo)
    gtn_lp.compress()

    return gtn_lp


def smooth_nonlinear_diffusion_gtn(gtn: GridTN1D, max_bond: int, cutoff: Numeric=CUTOFF, filter_weight: Numeric=1, ) -> 'GridTN1D':
    """ apply smoothing /averaging of the form I - a D
        where D = tridiag(-1/2, 1, -1/2)  and paramater
        parameter a (filter weight)

        filter: (I - D[x] D[y] D[z]) * weight[x,y,z]
        weight: abs(\phi_j - \phi_j-1 / \phi_{j+1} - \phi_j); summed over derivatives along each direction

        this doesn't really catch discontinuities that form though...
    """
    gtn.compress(compress_opts={'max_bond': max_bond, 'cutoff': cutoff})

    ## diffusive operator
    D_mpos = {}
    for ax in gtn.grid.axes:

        # D_mat = np.diag(np.ones(ax.npts)) + np.diag(np.ons(ax.npts - 1), k=-1) + np.diag(np.ons(ax.npts - 1), k=1)
        mpo_0 = ax.get_iden_mpo()
        mpo_1 = ax.get_shift_mpo( 1, boundary_conditions=gtn.ax_deriv_configs[ax])
        mpo_2 = ax.get_shift_mpo(-1, boundary_conditions=gtn.ax_deriv_configs[ax])
        helper_quimb.scalar_multiply(mpo_1, -0.5, inplace=True)
        helper_quimb.scalar_multiply(mpo_2, -0.5, inplace=True)
        D_mpo = helper_quimb.add_MPO(mpo_0, mpo_1)
        D_mpo = helper_quimb.add_MPO(D_mpo, mpo_2, inplace=True) #, compress=True)
        D_mpos[ax] = D_mpo
    D_gtn = gtn.grid.make_mpo_ndim(D_mpos)

    weights: GridTN1D = None
    for ax in gtn.grid.axes:

        deriv_f = gtn.ax_deriv_configs[ax].copy()
        deriv_f.update(order=0, fd_type=FDType.FORWARD)
        deriv_b = gtn.ax_deriv_configs[ax].copy()
        deriv_b.update(order=0, fd_type=FDType.BACKWARD)

        slope_f = gtn.take_firstderivative(ax, ax_deriv_config={ax: deriv_f})
        slope_f.scalar_add(1.0e-8, inplace=True)
        slope_b = gtn.take_firstderivative(ax, ax_deriv_config={ax: deriv_b})
        slope_b.scalar_add(1.0e-8, inplace=True)


        #### computing phi_j - phi_{j-1} / phi_{j+1} - phi_j  ####

        # ## cross (canonicalization issues?)
        # from local_solvers.local_cross_eval import local_cross_evaluator
        # from local_solvers.terms_3 import Term_Cross
        # term = Term_Cross(slope_f.data, mps_func=lambda x: 1./x)
        # out = local_cross_evaluator([term], max_bond=max_bond * 3)
        # slope_f_inv_gtn = gtn.create_like(out)
        # slope_ratio = slope_b.elemental_multiply(slope_f_inv_gtn, compress=True, compress_opts={'max_bond': max_bond * 3})
        #
        # # ## dmrg fails!!
        # # from helper_dmrg import dmrg_solve
        # # slope_f = slope_f.apply_elemental_multiply_op()
        # # slope_f.data.mangle_inner_()
        # # slope_ratio, err, is_conv = dmrg_solve(slope_b.data, slope_f.data)
        # # print('done dmrg solve')
        #
        # # filter for values less than 0
        # # strongest filtering for slope_ratio = 1, less strong for slope_ratio -> 0 or -> infinity
        # def slope_filter_func(x):  ## max value is 1
        #     if x < 0:
        #         return min(abs(2. / (x + 1 / x)), 1e8)
        #     else:
        #         return 0
        #
        # from local_solvers.local_cross_eval import local_cross_evaluator
        # from local_solvers.terms_3 import Term_Cross
        # term = Term_Cross(slope_ratio.data, mps_func=slope_filter_func)
        # out = local_cross_evaluator( [term], max_bond=max_bond * 3 )
        # out_gtn = gtn.create_like(out)

        #### brute force
        def slope_filter_func(x):  ## max value is 1
            return np.where(x < 0, np.abs(2. / (x + 1 / x)), 0)

        slope_f_data = slope_f.get_data()
        slope_b_data = slope_b.get_data()
        slope_ratio = slope_b_data / slope_f_data
        out = slope_filter_func(slope_ratio)
        out_gtn = gtn.grid.map_state_to_mps(out)

        if weights is None:
            weights = out_gtn
        else:
            weights.add(out_gtn, inplace=True)



    ## use weights = weights**2 to ensure positive?
    weights.elemental_multiply(weights, inplace=True)
    weights.scalar_multiply(-0.5 / gtn.grid.ndim * filter_weight, inplace=True)
    ## max will be 0.5, multiply by -1 for next step (if filter_weight=1.0)

    plt.figure()
    # plt.plot(slope_ratio)
    plt.plot(out)
    plt.show()

    tmp = D_gtn.get_data()
    plt.figure()
    plt.imshow(tmp)
    plt.colorbar()
    plt.show()

    ### scale D_gtn by weights
    diag_weights = weights.apply_elemental_multiply_op()
    D_gtn = D_gtn.apply(diag_weights, inplace=True)
    final_op_gtn = gtn.grid.get_iden_mpo().add(D_gtn, compress=True)

    tmp = final_op_gtn.get_data()
    plt.figure()
    plt.imshow(tmp)
    plt.colorbar()
    plt.show()

    ## apply diffusion operator to original gtn
    out = gtn.apply(final_op_gtn)
    out.compress(compress_opts={'max_bond': max_bond})
    return out



def coarse_grain_gtn(gtn: GridTN1D, depth: int):
    """ integrates (via summation) of the finer grid (endpoint=False)
    """
    grid = gtn.grid
    Axis = grid.axes[0].__class__
    new_axes = [Axis(ax.L - depth, dx=depth*ax.dx) for ax in grid.axes]
    print('new axes', new_axes)
    print('new axes', [ax.L for ax in grid.axes])
    print('new axes', [ax.npts for ax in grid.axes])
    print('new axes', [ax.L for ax in new_axes])
    print('new axes', [ax.npts for ax in new_axes])
    new_grid = grid.create_like(new_axes)

    coarse_grain_op = grid.get_coarse_grain_mpx(depth, is_sqrt=False)
    # coarse_grain_op.data.distribute_exponent()

    L_ = gtn.L - depth
    gtn_coarse = gtn.apply(coarse_grain_op)
    coarse_mps = gtn_coarse.data
    end_tens = qtn.tensor_contract(*coarse_mps[L_ - 1:])

    coarse_mps = coarse_mps[:L_]
    coarse_mps._L = coarse_mps.num_tensors
    coarse_mps[-1].modify(data=end_tens.data, inds=end_tens.inds)
    print('exponent', gtn.data.exponent, coarse_grain_op.exponent)
    coarse_mps.exponent = gtn.data.exponent + coarse_grain_op.exponent

    coarse_gtn = gtn.__class__(new_grid, coarse_mps)
    return coarse_gtn


def coarse_grain_select_gtn(gtn: GridTN1D, depth: int):
    """ instead of integrating over the finer grid, select one of the points
    """
    raise NotImplementedError


def compress_dmrg_coarse_grain(gtn: GridTN1D, depth: int, max_bond:int):
    """ DMRG compression but comparing coarse-grained result (instead of full MPS)
    """
    grid = gtn.grid
    coarse_grain_op = grid.get_coarse_grain_mpx(depth, is_sqrt=False)
    # coarse_grain_op = grid.get_iden_mpo()  ## for testing purposes

    out = local_reference_compress(gtn.data, coarse_grain_op.data, max_bond=max_bond)
    out_gtn = gtn.create_like(out)
    return out_gtn


def compress_interpolate_1D(gtn: GridTN1D, depth: int, max_bond: int=None, cutoff: Numeric=CUTOFF):
    """ compression based on integrating out fine grid, and then interpolating
        probably doesn't work well for sequential ordering?
    """
    grid = gtn.grid
    site_ind_id = gtn.data.site_ind_id


    coarse_gtn = coarse_grain_gtn(gtn, depth)
    coarse_gtn.compress(compress_opts={'max_bond': max_bond, 'cutoff': cutoff}, inplace=True)
    coarse_mps = coarse_gtn.data
    L_ = coarse_mps.L

    def interpolate_coarse_to_fine(depth:int, influence:int, left_bc:BCType, right_bc: BCType):
        """ from fine dyadic grid, get the function approximated at fine Chebyshev grids
            Lagrange interpolation tensor (from michael in section 5)
        """

        npts = 2**L_
        num_meas = 2**influence + 1
        num_fine = 2**depth
        meas_grid = np.linspace(-0.5, 0.5, num_meas, endpoint=True) * 2 ** (depth - L_)    ## coarse grid points
        dx_coarse = meas_grid[1] - meas_grid[0]
        fine_grid = np.linspace(-0.5, 0.5, num_fine, endpoint=False) * dx_coarse

        lagrange_tens = np.zeros((npts, num_fine, npts))
        ind0 = num_meas // 2

        ## fill in values at the boundaries (left)
        if left_bc == BCType.OPEN:
            for i in range(ind0):
                diff = ind0 - i     ## number of points outside of the domain
                for j in range(num_meas):
                    j_ = j - ind0
                    if j_ < 0:  continue    ## exclude points outside of domain

                    poly = 1.
                    for k in range(diff, num_meas):
                        if j != k:
                            poly = poly * (fine_grid - meas_grid[k]) / (meas_grid[j] - meas_grid[k])

                    lagrange_tens[i, :, i + j_] += poly
        else:
            for i in range(ind0):
                for j in range(num_meas):
                    j_ = j - ind0

                    poly = 1.
                    for k in range(num_meas):
                        if j != k:
                            poly = poly * (fine_grid - meas_grid[k]) / (meas_grid[j] - meas_grid[k])

                    if left_bc == BCType.PERIODIC:
                        lagrange_tens[i, :, i + j_] += poly     # wraps around
                    elif left_bc == BCType.ANTIPERIODIC:
                        lagrange_tens[i, :, i + j_] += -poly
                    elif left_bc in [BCType.NEUMANN, BCType.ZEROGRADIENT, BCType.SYMMETRIC]:  # y_-1 = y_0
                        if i + j_ >= 0:
                            lagrange_tens[i, :, np.abs(i + j_)] += poly
                        else:
                            lagrange_tens[i, :, np.abs(i + j_) + 1] += poly
                    elif left_bc in [BCType.ZEROVALUE, BCType.DIRICHLET, BCType.ANTISYMMETRIC]:  # y_0 = 0,  y_-1 = y_1
                        lagrange_tens[i, :, np.abs(i + j_)] += -poly

        ## fill in values at the boundaries (left)
        if right_bc == BCType.OPEN:
            for i in range(npts - ind0, npts):
                diff = i + ind0 + 1 - npts
                for j in range(num_meas):
                    j_ = j - ind0
                    if j_ >= 0:  continue  ## exclude points outside of domain

                    poly = 1.
                    for k in range(num_meas - diff):    ## ignore the last few points that our outside domain
                        if j != k:
                            poly = poly * (fine_grid - meas_grid[k]) / (meas_grid[j] - meas_grid[k])

                    lagrange_tens[i, :, i + j_] += poly
        else:
            for i in range(npts - ind0, npts):
                for j in range(num_meas):
                    j_ = j - ind0

                    poly = 1.
                    for k in range(num_meas):
                        if j != k:
                            poly = poly * (fine_grid - meas_grid[k]) / (meas_grid[j] - meas_grid[k])

                    diff = npts - (i + j_)
                    if right_bc == BCType.PERIODIC:
                        lagrange_tens[i, :, -diff] += poly
                    elif right_bc == BCType.ANTIPERIODIC:
                        lagrange_tens[i, :, -diff] += -poly
                    elif right_bc in [BCType.NEUMANN, BCType.ZEROGRADIENT, BCType.SYMMETRIC]:  # y_L = y_L-1; y_L+1 = y_L-2; etc
                        if diff > 0:    # i + j_ < npts
                            lagrange_tens[i, :, -np.abs(diff)] += poly
                        else:
                            lagrange_tens[i, :, -np.abs(diff) - 1] += poly
                    elif right_bc in [BCType.ZEROVALUE, BCType.DIRICHLET, BCType.ANTISYMMETRIC]:  # y_L = 0, y_L+1 = -y_L-1
                        if diff != 0:
                            lagrange_tens[i, :, -np.abs(diff)] += -poly

        ### bulk interpolation tensor
        for j in range(num_meas):
            poly = 1.
            for k in range(num_meas):
                if j != k:
                    poly = poly * (fine_grid - meas_grid[k]) / (meas_grid[j] - meas_grid[k])

            j_ = j - ind0
            for i in range(ind0, npts - ind0):
                lagrange_tens[i, :, i + j_] += poly

        L_tens = qtn.Tensor(data=lagrange_tens, inds=('dyad_o', 'fine' ,'dyad_i'))
        return L_tens

    deriv_config: DerivativeConfiguration = gtn.ax_deriv_configs[gtn.grid.axes[0]]
    interp_tens = interpolate_coarse_to_fine(depth, 2, deriv_config.left_bc, deriv_config.right_bc)
    interp_tens.unfuse({'dyad_o': [f'u({i})' for i in range(L_)],
                        'dyad_i': [f'l({i})' for i in range(L_)]},
                       {'dyad_o': [2]*L_, 'dyad_i': [2]*L_},
                       inplace=True)
    interp_mpo = helper_quimb.mpx_from_dense(interp_tens, L_, site_inds_list=['u({})','l({})'], direction=0)

    out = helper_quimb.apply(interp_mpo, coarse_mps)
    out.mangle_inner_()

    last_tens = out[L_ - 1]
    sub_out = out[:L_ - 1]
    sub_out._L = out.L - 1
    sub_out.exponent = out.exponent

    h_bond = next(iter(last_tens.bonds(sub_out[-1])))
    last_tens.reindex({site_ind_id.format(L_ - 1): '_tmp(0)_'}, inplace=True)
    last_tens.drop_tags()
    last_tens.unfuse({'fine': [f'_tmp({i})_' for i in range(1, depth + 1)]},
                     {'fine': (2,) * depth}, inplace=True)
    last_mps = helper_quimb.mpx_from_dense(last_tens, depth + 1, ['_tmp({})_'], left_anc=[h_bond])

    out_mps = helper_quimb.append_mpx(sub_out, last_mps)
    out_gtn = gtn.create_like(out_mps)
    out_gtn.compress(inplace=True)

    return out_gtn


def compress_spline_1D(gtn: 'GridTN1D', coarsen_depth: int, max_bond: int=None, cutoff: Numeric=CUTOFF):
    """ assumes uniform grid points
        see Wikipedia page: https://en.wikipedia.org/wiki/Spline_interpolation
    """
    site_ind_id = gtn.data.site_ind_id
    site_tag_id = gtn.data.site_tag_id
    grid = gtn.grid

    coarse_gtn = coarse_grain_gtn(gtn, coarsen_depth)
    coarse_gtn.compress(inplace=True, compress_opts={'max_bond': max_bond, 'cutoff': cutoff})
    coarse_mps = coarse_gtn.data
    c_grid = coarse_gtn.grid

    assert(len(coarse_gtn.grid.axes) == 1), 'spline method only works for 1D functions right now'
    ax = gtn.grid.axes[0]
    c_ax = coarse_gtn.grid.axes[0]
    left_bc, right_bc = gtn.ax_deriv_configs[ax].bc
    # left_bc, right_bc = gtn.ax_deriv_configs[ax].left_bc, gtn.ax_deriv_configs[ax].right_bc

    shift_p_op = c_grid.get_shift_mpo({c_ax:  1}, ax_boundary_conditions={c_ax: gtn.ax_deriv_configs[ax]})
    shift_m_op = c_grid.get_shift_mpo({c_ax: -1}, ax_boundary_conditions={c_ax: gtn.ax_deriv_configs[ax]})

    #### solve for slopes
    ## (npts + 1) x (npts + 1)
    npts, dx = c_ax.npts, c_ax.dx
    Amat = np.diag([4./dx] * (npts + 1), k=0) + np.diag([1./dx] * (npts), k=1) + np.diag([1./dx] * (npts), k=-1)
    Amat[0,0] = 2./dx
    Amat[-1,-1] = 2./dx
    # A_gtn = c_grid.map_operator_to_mpo(Amat)

    ## double check boundaries
    # fd_config = DerivativeConfiguration(BCType.ZEROVALUE, fd_type=FDType.BACKWARD, order=0)  # y_{i} - y_{i-1}
    # ydiff1_gtn = coarse_gtn.take_firstderivative(c_ax, ax_deriv_config={c_ax: fd_config})
    # fd_config = DerivativeConfiguration(BCType.ZEROVALUE, fd_type=FDType.CENTER, order=1)   # y_{i+1} - y_{i-1}
    # ydiff2_gtn = coarse_gtn.take_firstderivative(c_ax, ax_deriv_config={c_ax: fd_config})
    # ydiff2_gtn.scalar_multiply(3./dx * 2, inplace=True)  ## y2-y1/(x2-x1)**2 .  ydiff_gtn already scaled by 1/2/dx

    ## approximate estimate of right bc val (more approx than in get_shift_mpo
    coarse_dat = coarse_gtn.get_data()
    right_bc_val = 0.0
    if right_bc in [BCType.OPEN, BCType.NEUMANN, BCType.SYMMETRIC, BCType.ZEROGRADIENT]:
        right_bc_val = coarse_dat[-1]
    elif right_bc in [BCType.ZEROVALUE, BCType.DIRICHLET, BCType.ANTISYMMETRIC]:
        right_bc_val = 0.0
    elif right_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
        right_bc_val = coarse_dat[0] * np.sign(right_bc)
    coarse_dat_R = np.array([*coarse_dat, right_bc_val])

    ydiff2_dat = np.zeros(c_ax.npts + 1, dtype=complex)
    ydiff2_dat[:-1] = coarse_dat_R[1:] - coarse_dat_R[:-1]
    ydiff2_dat[1:] += coarse_dat_R[1:] - coarse_dat_R[:-1]
    ydiff2_dat = ydiff2_dat * 3 / dx**2
    # ydiff2_gtn = c_grid.map_state_to_mps(ydiff2_dat)
    # ydiff1_dat = np.zeros(c_ax.npts, dtype=complex)
    ydiff1_dat = coarse_dat_R[1:] - coarse_dat_R[:-1]
    # ydiff1_dat[-1] = coarse_dat[-2] - coarse_dat[-1]    ## symmetric boundary
    # ydiff1_gtn = c_grid.map_state_to_mps(ydiff1_dat)

    # from helper_dmrg import dmrg_solve
    # A_gtn.data.mangle_inner_()
    # slopes, err, is_conv = dmrg_solve(ydiff2_gtn.data, A_gtn.data)

    # ydiff2_dat = ydiff2_gtn.get_data()
    # A_mat = A_gtn.get_data()
    slopes_dat = np.linalg.solve(Amat, ydiff2_dat)  ## npts + 1 vec
    # slopes = c_grid.map_state_to_mps(slopes_dat)

    #### compute a = k2 * (x2-x1) - (y2-y1) and b = -k1 * (x2-x1) + (y2-y1)
    # k_gtn = ydiff2_gtn.create_like(slopes)
    # km_gtn = k_gtn.apply(shift_m_op)
    # a_gtn = km_gtn.scalar_multiply(dx).add(ydiff1_gtn.scalar_multiply(-1), compress=True)
    # b_gtn = k_gtn.scalar_multiply(-dx).add(ydiff1_gtn, compress=True)
    a_dat = slopes_dat[:-1] * dx - ydiff1_dat
    b_dat = slopes_dat[1:] * -dx + ydiff1_dat
    a_gtn = c_grid.map_state_to_mps(a_dat)
    b_gtn = c_grid.map_state_to_mps(b_dat)

    #### evaluate interpolation polynomial
    tx = np.linspace(0, 1, 2**coarsen_depth, endpoint=False)        # tx = (x-x1)/(x2-x1)
    tx_ = np.linspace(1, 0, 2**coarsen_depth, endpoint=False)       # tx_ = (1- t(x)) = 1 - tx
    tx_2tx = tx_ ** 2 * tx          # (1-t)**2 * t
    tx2tx_ = tx ** 2 * tx_          # (1-t) * t**2

    Axis = c_ax.__class__
    f_ax = Axis(coarsen_depth)
    tx_mps = f_ax.map_state_to_mps(tx)
    tx_mps_ = f_ax.map_state_to_mps(tx_)
    tx_2tx_mps = f_ax.map_state_to_mps(tx_2tx)
    tx2tx__mps = f_ax.map_state_to_mps(tx2tx_)

    coarse_p_gtn = coarse_gtn.apply(shift_p_op)
    # coarse_p_gtn = c_grid.map_state_to_mps(coarse_dat_R[1:])

    qx1 = helper_quimb.append_mpx( coarse_mps, tx_mps_ )    ## outer product of coarse * fine grid interpolation
    qx2 = helper_quimb.append_mpx( coarse_p_gtn.data, tx_mps  )     # y2 * t
    qx3 = helper_quimb.append_mpx( a_gtn.data, tx_2tx_mps )
    qx4 = helper_quimb.append_mpx( b_gtn.data, tx2tx__mps )

    out_mps = helper_quimb.add_MPS_list([qx1, qx2, qx3, qx4])

    return gtn.create_like(out_mps)



def compress_cubic_interpolation(gtn: 'GridTN1D', depth: int, max_bond: int=None, cutoff: Numeric=CUTOFF):
    """ convolve a kernel along each dimension at each grid point.
        p(t) = 1/2 [1 t t^2 t^3] W [f_{-1}, f_0, f_1, f_2].T
        -- t from 0 to 1
        so we'd evaluate p at various t's, giving multiple data points per coarse data point

        for n-D
        we evaluate p(t) after convolving along one axis
        this gives us an updated function f (on a partially refined grid)
        which we then plug in when evaluating the next function.

        g(x) = sum_k c_k u(j-k) where u is the interpolation kernel
        g(x) = c_{j-1} u(t + 1) + c_{j} u(t) + c_{j+1} u(t-1) + c_{j+2} u(t-2)
        where u(t+1) = ... = a t^3 - 2 a t^2 + a t
              u(t) = ... = (a + 2) t^3 - (a + 3) t^2 + 1
              u(t-1) = ... = -(a + 2) t^3 + (2a + 3) t^2 - a t
              u(t-2) = ... = -a t^3 + a t^2
        where the value of t is from 0 to 1

        --> g(x) = - [ a (c_{j+2} - c_{j-1} + (a+2) (c_{j+1} - c_j) ] t^3
                   + [ 2a (c_{j+1} - c_{j-1} + 3 (c_{j+1} - c_j) + a (c_{j+2} - c_j) t^2
                   - a(c_{j+1} - c_{j-1}] t
                   + c_j
        where x is in the interval [x_j, x_{j+1}]
        we can write this in matrix form:
        g(x) = [1  t  t^2  t^3] [  0     1       0    0 ] [c_{j-1}]
                                [  a     0      -a    0 ] [  c_j  ]
                                [ -2a -(a+3)   2a+3   a ] [c_{j+1}]
                                [  a   (a+2)  -(a+2) -a ] [c_{j+2}]

        https://www.ncorr.com/download/publications/keysbicubic.pdf
        R. G. Keys, Cubic Convolution Interpolation for Digital Image Processing

        if a = -1/2, interpolation is third-order accurate.

    """

    mps = gtn.data
    site_ind_id = mps.site_ind_id
    site_tag_id = mps.site_tag_id

    deriv_config: DerivativeConfiguration = gtn.ax_deriv_configs[gtn.grid.axes[0]]
    left_bc, right_bc = deriv_config.left_bc, deriv_config.right_bc

    coarse_gtn = coarse_grain_gtn(gtn, depth)
    coarse_mps = coarse_gtn.data
    L_ = coarse_mps.L

    def get_kernel_mat(t: np.array, a=-0.5) -> np.ndarray:

        Amat = np.array([[0, 1., 0, 0],
                         [a, 0, -a, 0],
                         [-2*a, -(a+3), 2*a+3, a],
                         [a, a+2, -(a+2), -a]])

        t_powers = np.array([t**0, t, t**2, t**3]).T        ## powers of t along columns
        kernel = np.tensordot(t_powers, Amat, axes=(-1,0))
        return kernel

    def interpolate_coarse_to_fine(depth:int, influence:int):
        """ from fine dyadic grid, get the function approximated at fine Chebyshev grids
            Lagrange interpolation tensor (from michael in section 5)
        """

        npts = 2**L_
        num_meas = 2**influence # + 1
        num_fine = 2**depth
        meas_grid = np.linspace(0, 1, num_meas, endpoint=False)
        dx_coarse = meas_grid[1] - meas_grid[0]
        fine_grid = np.linspace(0, 1, num_fine, endpoint=False)

        L_mat = np.zeros((npts, num_fine, npts))

        kernel_mat = get_kernel_mat(fine_grid)

        ## treat the boundaries
        ## left boundary
        L_mat[0, :, :3] += kernel_mat[:, 1:]
        if left_bc == BCType.OPEN:
            ## this treatment is as if func is zero outside domain... which is not right
            L_mat[0, :, :3] += kernel_mat[:, 1:]
        else:
            if left_bc == BCType.PERIODIC:
                L_mat[0, :, -1] += kernel_mat[:, 0]  # wraps around
            elif left_bc == BCType.ANTIPERIODIC:
                L_mat[0, :, -1] += -kernel_mat[:, 0]
            elif left_bc in [BCType.NEUMANN, BCType.ZEROGRADIENT, BCType.SYMMETRIC]:  # y_-1 = y_0
                L_mat[0, :, 0] += kernel_mat[:, 0]
            elif left_bc in [BCType.ZEROVALUE, BCType.DIRICHLET, BCType.ANTISYMMETRIC]:  # y_0 = 0,  y_-1 = y_1
                L_mat[0, :, 0] += kernel_mat[:, 0]

        ## fill in values at the right boundary
        L_mat[-2, :, -3:] += kernel_mat[:, :3]
        L_mat[-1, :, -2:] += kernel_mat[:, :2]
        if right_bc == BCType.OPEN:
            pass
        else:
            if right_bc == BCType.PERIODIC:
                L_mat[-2, :, :1] += kernel_mat[:, -1:]
                L_mat[-1, :, :2] += kernel_mat[:, -2:]
            elif right_bc == BCType.ANTIPERIODIC:
                L_mat[-2, :, :1] += -kernel_mat[:, -1:]
                L_mat[-1, :, :2] += -kernel_mat[:, -2:]
            elif right_bc in [BCType.NEUMANN, BCType.ZEROGRADIENT, BCType.SYMMETRIC]:
                # y_L = y_L-1; y_L+1 = y_L-2; etc
                L_mat[-2, :, -1] += kernel_mat[:, -1]
                L_mat[-1, :, -1] += kernel_mat[:, -2]
                L_mat[-1, :, -2] += kernel_mat[:, -1]
            elif right_bc in [BCType.ZEROVALUE, BCType.DIRICHLET, BCType.ANTISYMMETRIC]:
                # y_L = 0, y_L+1 = -y_L-1
                L_mat[-1, :, -1] -= kernel_mat[:, -2]


        for i in range(1,npts-2):
            L_mat[i, :, i-1: i+3] = kernel_mat

        print('Lmat', L_mat[:5, 2, :5])

        L_tens = qtn.Tensor(data=L_mat, inds=('dyad_o', 'fine', 'dyad_i'))
        return L_tens

    interp_tens = interpolate_coarse_to_fine(depth, 2)
    interp_tens.unfuse({'dyad_o': [f'u({i})' for i in range(L_)],
                        'dyad_i': [f'l({i})' for i in range(L_)]},
                       {'dyad_o': [2] * L_, 'dyad_i': [2] * L_},
                       inplace=True)
    interp_mpo = helper_quimb.mpx_from_dense(interp_tens, L_, site_inds_list=['u({})', 'l({})'], direction=0)

    out = helper_quimb.apply(interp_mpo, coarse_mps)
    out.mangle_inner_()

    last_tens = out[L_ - 1]
    sub_out = out[:L_ - 1]
    sub_out._L = out.L - 1
    sub_out.exponent = out.exponent

    h_bond = next(iter(last_tens.bonds(sub_out[-1])))
    last_tens.reindex({site_ind_id.format(L_ - 1): '_tmp(0)_'}, inplace=True)
    last_tens.drop_tags()
    last_tens.unfuse({'fine': [f'_tmp({i})_' for i in range(1, depth + 1)]},
                     {'fine': (2,) * depth}, inplace=True)
    last_mps = helper_quimb.mpx_from_dense(last_tens, depth + 1, ['_tmp({})_'], left_anc=[h_bond])

    out_mps = helper_quimb.append_mpx(sub_out, last_mps)
    out_gtn = gtn.create_like(out_mps)
    out_gtn.compress(inplace=True)

    return out_gtn



def compress_multiscale_gtn(gtn: 'GridTN1D', coarsen_depth: int, interpolation_depth: int, max_bond: int):

    mps = gtn.data
    site_ind_id = mps.site_ind_id
    site_tag_id = mps.site_tag_id

    coarse_gtn = coarse_grain_gtn(gtn, coarsen_depth)
    coarse_mps = coarse_gtn.data
    #
    #
    # # mps.canonize(mps.L - 1)
    # # pos = int(np.round(np.log2(max_bond))) * 2 - 1
    # # num_integ = mps.L - pos
    # num_integ = coarsen_depth
    #
    #
    # integ_mps = coarse_grain(mps, pos, 1)
    # print('integ mps', pos, mps.L, integ_mps)
    # helper_quimb.canonize(integ_mps, i=pos, scale=False)
    # sub_mps = mps[:pos + 1]
    # sub_mps._L = pos + 1
    # print('integ mps', integ_mps)
    # print('integ mps', integ_mps.exponent, mps.exponent)

    L = coarse_gtn.L              ## len of integrated mps (last tensor does not have physical bond)
    num_x = interpolation_depth   ## number of indices/sites to use to extrapolate onto Cheb grid
    num_r = L - num_x       ## number of remaining sites
    print('L', L)

    L_tens = interpolate_fine_to_cheb(num_x, max_bond)
    L_tens = L_tens.unfuse({'dyad': [site_ind_id.format(i) for i in range(num_r, L)]},
                           {'dyad': [2]*num_x})
    print('L tens', L_tens)
    print('integ mps partial', num_r, coarse_mps[num_r:])
    ## projection of integ_mps and contraction with the site to the left
    proj_L = qtn.tensor_contract(*coarse_mps[num_r-1:], L_tens)
    proj_L.drop_tags()
    proj_L.add_tag(site_tag_id.format(num_r-1))
    print('proj L', proj_L)
    ## note: this may yield MPS with rank smaller than max_bond
    ## can we estimate the rank from L_ovlp?

    # check dyad to chebyshev interpolation ##
    out = qtn.tensor_contract(*coarse_mps[:num_r-1].tensors, proj_L) * 10**coarse_mps.exponent
    out.transpose(*[site_ind_id.format(i) for i in range(num_r)], 'cheb', inplace=True)
    out_data = out.data.reshape(-1)

    coarse_x = np.linspace(0, 1, 2**L, endpoint=False)
    nbasis = max_bond
    k = np.arange(nbasis)
    cheb_grid = 1 - (np.cos(k * np.pi / nbasis) + 1) / 2
    print('cheb grid', cheb_grid)
    interp_x = np.linspace(0, 1, 2**num_r, endpoint=False)
    dx_r = interp_x[1] - interp_x[0]
    xvals_cb = np.outer(np.ones(2**num_r,) * dx_r, cheb_grid) + np.outer(interp_x, np.ones(nbasis))
    xvals_cb = xvals_cb.reshape(-1)

    plt.figure()
    plt.plot(xvals_cb, 'x-')

    plt.figure()
    plt.plot(coarse_x, coarse_gtn.get_data(), 'o-', label='caorse')
    plt.plot(xvals_cb, out_data, 'x-', label='interp')
    plt.legend()
    plt.show()

    # exit()

    ## then re-extrapolate with interpolation tensors
    TM, TR = multiscale_interpolating_tensors(max_bond)

    ## transpose inds to p l r
    if num_r > 1:
        keep_mps = coarse_mps[:num_r - 1]
        keep_L = keep_mps.num_tensors
        print('keep mps', coarsen_depth, interpolation_depth, num_r, keep_mps)
        for i in range(keep_L):
            inds = (site_ind_id.format(i),)
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
        inds = (site_ind_id.format(num_r-1), next(iter(left)), right)
        print('proj L', proj_L, 'transpose inds', inds)
        proj_L.transpose(*inds, inplace=True)

    else:   ## num_r == 1, so keep_mps[0] is proj_L

        inds = (site_ind_id.format(num_r-1), 'cheb')
        proj_L.transpose(*inds, inplace=True)


    new_tensors = coarse_mps.tensors[:num_r-1] + (proj_L,) + (TM,) * (mps.L - num_r-1) + (TR,)
    # new_mps = qtn.TensorNetwork(new_tensors)

    print('new tensors', new_tensors)

    new_mps = qtn.MatrixProductState([tens.data for tens in new_tensors], shape='plr',)
    new_mps.exponent = coarse_mps.exponent
    print('dx', new_mps.exponent, coarse_mps.exponent)
    # new_mps.view_like(mps, inplace=True)

    new_gtn = gtn.create_like(new_mps)
    new_gtn.compress(inplace=True)

    # print('new mps', new_mps)
    # helper_quimb.compress(new_mps, scale=False)
    # print('compressed max bond', new_mps.max_bond())

    ## maybe measure amount to coarse grain based on |d/dx psi|
    ## remove high-frequency polynomials when mapping to Chebyshev nodes?

    return new_gtn



def multiscale_interpolating_tensors(nbasis:int):

    k = np.arange(nbasis)
    cheb_grid = 1 - (np.cos( k * np.pi / nbasis) + 1) /  2   ##  on [0,1] instead of [-1, 1]

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
    cheb_grid = 1 - (np.cos(k * np.pi / nbasis) + 1) / 2  ##  on [0,1] instead of [-1, 1]
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








