import numpy as np

import helper_dmrg
from setup.defaults import *
import time
from scipy import linalg
import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

MPO_type = Union['qtn.MatrixProductOperator', MatrixProductOperatorTN, IndexedMPO]
MPS_type = Union['qtn.MatrixProductState', MatrixProductStateTN, IndexedMPS]

from enum import IntEnum
class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    ## i'd like to switch these values
    LEFT = 1
    RIGHT = -1

class EnvironmentSide(IntEnum):
    LEFT = 1    ## int denotes shift in position when extended environment
    RIGHT = -1

class EnvironmentType(IntEnum):
    EXPEC = 1
    OVLP = 0
    NORM = 2

class SolveMethod(Enum):
    CGD = 'CGD'     # local CGD using quimb tensors
    CGDx = 'CGDx'   # local CGD using numpy
    LSQ = 'LSQ'     # least squares regression

DEFAULT_MAX_ITER = 20  # 10
DEFAULT_MAX_TOT_ITER = 50
DEFAULT_CONV_TOL = 1.0e-6
DEFAULT_SOLVE = SolveMethod.CGD
## TODO: big bug with CGDx? not in test_dmrg but with real test files (eg damp 2D).  exponent issue?


################################
### numpy optimzation method ###
################################

def conjugate_gradient_descent(A, b, x=None, max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False):
    """ solve Ax = b via conjugate gradient descent
        using normal matrices and vectors
        assumes A is psd and symmetric
    """
    norm_b = np.linalg.norm(b)
    if x is None:
        x, r = 0, b
    else:
        r = b - np.dot(A, x)
    d = r

    it = 0
    err = np.linalg.norm(r)
    min_x, min_err = x, err
    while it < max_iter and err > conv_tol * norm_b:
        # x_old = x
        alpha = np.linalg.norm(r)**2 / np.vdot(d.conj(), np.dot(A, d))

        x_ = x + alpha * d
        r_ = r - alpha * np.dot(A, d)
        beta_ = np.linalg.norm(r_)**2 / np.linalg.norm(r)**2
        d_ = r_ + beta_ * d

        x, r, d, = x_, r_, d_
        err = np.linalg.norm(r)
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

        it += 1

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')

    return x, err / norm_b


def conjugate_gradient_squared(A, b, x=None, max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, verbose=False):
    """ solve Ax = b via conjugate gradient squared algorithm
        using numpy matrices and vectors
        assumes A has positive eigenvalues but is not necessarily symmetric
    """
    norm_b = np.linalg.norm(b)
    if x is None:
        x, r = 0, b
        rT0 = b
    else:
        r = b - np.dot(A, x)
        rT0 = b - np.dot(x, A)

    it = 0
    q, p, rho = 0, 0, 1
    err = np.linalg.norm(r)
    min_x, min_err = x, err
    while it < max_iter and err > conv_tol * norm_b:

        rho_ = np.dot(rT0, r)
        beta = rho_ / rho

        u = r + beta * q
        p_ = u + beta * (q + beta * p)
        v = np.dot(A, p_)

        sigma = np.dot( rT0, v )
        alpha = rho_ / sigma

        q_ = u - alpha * v
        r_ = r - alpha * np.dot(A, u + q_)
        x_ = x + alpha * (u + q_)

        p, q, r, x = p_, q_, r_, x_
        rho = rho_
        it += 1

        err = np.linalg.norm(r)
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')

    return x, err / norm_b


def conjugate_gradient_bi(A, b, x=None, max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL):
    """ solve Ax = b via bi-conjugate gradient algorithm
        using numpy matrices and vectors
        assumes A has positive eigenvalues but is not necessarily symmetric
    """
    raise NotImplementedError


#####################################
### global mps optimzation method ###
#####################################

def mps_conjugate_gradient_descent_v2(A: qtn.MatrixProductOperator, b: qtn.MatrixProductState,
                                      x: qtn.MatrixProductState=None, max_bond=None,
                                      max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL,
                                      verbose=False):
    """ solve Ax = b via conjugate gradient descent
        using MPS, MPOs and SVD compression
    """
    compress_opts = {'max_bond': max_bond}
    norm_b = helper.norm(b)
    if x is None:
        x, r = helper.scalar_multiply(b,0), b
    else:
        # r = b - np.dot(A, x)
        r = helper.add_MPS( b, helper.scalar_multiply( helper.apply(A, x), -1), compress=True)
    d = r

    it = 0
    x_min, min_err = x, np.inf
    err = helper.norm(r)
    while it < max_iter and err > conv_tol * norm_b:
        # x_old = x.copy()
        alpha = err**2 / helper.expectation_value(d, A)
        # alpha = np.linalg.norm(r)**2 / np.vdot(d.conj(), np.dot(A, d))

        # x_ = x + alpha * d
        # r_ = r - alpha * np.dot(A, d)
        # beta_ = np.linalg.norm(r_)**2 / np.linalg.norm(r)**2
        # d_ = r_ + beta_ * d

        x_ = helper.add_MPS(x, helper.scalar_multiply(d, alpha), compress=True, compress_opts=compress_opts)
        r_ = helper.add_MPS(r, helper.scalar_multiply( helper.apply(A, d), -alpha), compress=True)
        beta_ = helper.norm(r_)**2 / err**2
        d_ = helper.add_MPS( r_, helper.scalar_multiply(d, beta_), compress=True)

        x, r, d, = x_, r_, d_
        # err = np.linalg.norm(r)
        err = helper.norm(r)
        if verbose:
            print('it', it, err / norm_b)

        if err > min_err:
            ## convergence not necessarily monotonic
            print('error went up', err / norm_b, min_err / norm_b)
            # x = x_old
            # break
        else:
            min_err = err
            x_min = x.copy()

        it += 1

    if it < max_iter:
        print(f'cgd: error {min_err / norm_b} in {it} iterations')
        conv = True
    else:
        print(f'cgd did not converge: error {min_err / norm_b} in {it} iterations')
        print('using x min')
        x = x_min
        conv = False

    return x, err / norm_b, conv


def mps_conjugate_gradient_squared_v2(A: qtn.MatrixProductOperator, b: qtn.MatrixProductState,
                                      x: qtn.MatrixProductState=None, max_bond=None,
                                      max_iter=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL):
    """ solve Ax = b via conjugate gradient descent
        using MPS, MPOs and SVD compression
    """
    compress_opts = {'max_bond': max_bond}
    norm_b = helper.norm(b)
    if x is None:
        x, r = helper.scalar_multiply(b,0), b
        rT0 = r
    else:
        # r = b - np.dot(A, x)
        r = helper.add_MPS( b, helper.scalar_multiply( helper.apply(A, x), -1), compress=True)
        rT0 = helper.add_MPS( b, helper.scalar_multiply( helper.apply( helper.mpo_transpose(A), x), -1), compress=True)

    it = 0
    zero_mps = helper.scalar_multiply(b,0)
    q, p, rho = zero_mps, zero_mps, 1
    x_min, min_err = x, np.inf
    err = helper.norm(r)
    while it < max_iter and err > conv_tol * norm_b:

        rho_ = helper.ovlp(rT0, r)
        beta = rho_ / rho

        u = helper.add_MPS(r, helper.scalar_multiply(q, beta), compress=True)
        tmp = helper.add_MPS(q, helper.scalar_multiply(p, beta))
        p_ = helper.add_MPS(u, helper.scalar_multiply(tmp, beta), compress=True)
        v = helper.apply(A, p_, compress=True)

        sigma = helper.ovlp(rT0, v)
        alpha = rho_ / sigma

        q_ = helper.add_MPS(u, helper.scalar_multiply(v, -alpha), compress=True)
        tmp_uq = helper.add_MPS(u, q_)
        tmp = helper.apply(A, tmp_uq, compress=True)
        r_ = helper.add_MPS(r, helper.scalar_multiply(tmp, -alpha), compress=True)
        x_ = helper.add_MPS(x, helper.scalar_multiply(tmp_uq, alpha),
                            compress=True, compress_opts=compress_opts)

        p, q, r, x = p_, q_, r_, x_
        rho = rho_
        it += 1

        err = helper.norm(r)
        # print('it', it, err / norm_b)

        if err < min_err:
            min_err = err
            x_min = x.copy()
        else:
            continue
            ## convergence not necessarily monotonic
            # print('error went up', err / norm_b, min_err / norm_b)

    if it < max_iter:
        print(f'cgd: error {min_err / norm_b} in {it} iterations')
        conv = True
    else:
        print(f'cgd did not converge: error {min_err / norm_b} in {it} iterations')
        print('using x min')
        x = x_min
        conv = False

    return x, err / norm_b, conv



########################
### helper functions ###
########################

def qtn_conjugate_gradient_descent_1site(As: Sequence[qtn.TensorNetwork], b: qtn.Tensor, x: qtn.Tensor,
                                         output_to_input_inds: dict[str, str],
                                         max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL,
                                         verbose=False) \
        -> tuple[qtn.Tensor, float]:
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
        A: matrix, b: target, x: solution
        r: residual = b - Ax
        d: search direction
    """
    norm_b = b.norm()
    input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.reindex(input_to_output_inds, inplace=False)  ## residual

    def contract_with_As(*x_tens, out_inds=r.inds):
        Ax_tot = None
        for A in As:
            Ax = qtn.TensorNetwork([A, *x_tens], virtual=True, check_collisions=False)
            out = Ax.contract_tags(all)

            if isinstance(out, qtn.Tensor):
                # out.transpose_like(r, inplace=True)
                out.transpose(*out_inds, inplace=True)
                out.modify(apply=lambda data: data * 10**Ax.exponent)
                if Ax_tot is None:
                    Ax_tot = out
                else:
                    Ax_tot.modify(apply = lambda data: data + out.data)
            else:
                out *= 10**Ax.exponent
                if Ax_tot is None:
                    Ax_tot = out
                else:
                    Ax_tot += out
        return Ax_tot

    if x is not None:
        # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        # Ax_tens = Ax.contract_tags(all)
        # Ax_tens.transpose_like(r, inplace=True)
        Ax_tens = contract_with_As(x)
        r.modify(apply = lambda data: data - Ax_tens.data)

    d = r.copy()       ## d: search r
    norm_r = r.norm()

    it = 0
    x_min, min_err = x, np.inf
    err = r.norm()
    while it < max_iter and err > conv_tol * norm_b:

        ## alpha = np.linalg.norm(r) ** 2 / np.vdot(d.conj(), np.dot(A, d))
        dT = d.conj()
        d.reindex(output_to_input_inds, inplace=True)
        # proj = qtn.TensorNetwork([dT, A, d], virtual=True, check_collisions=False)
        # alpha = err ** 2 / proj.contract_tags(all)
        proj = contract_with_As(dT, d)
        alpha = err ** 2 / proj

        ## update x:  x_ = x + alpha * d
        if x is None:
            x = d.copy()
            x.modify(apply = lambda data: data * alpha)
            # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        else:
            x.modify(apply = lambda data: data + alpha * d.data)

        ## update residual r:  r_ = r - alpha * np.dot(A, d)
        # r_proj = qtn.tensor_contract(*A.tensors, d)
        # r_proj.transpose_like(r, inplace=True)
        r_proj = contract_with_As(d)
        r.modify(apply = lambda data: data - alpha * r_proj.data)
        new_norm_r = r.norm()

        ## update search direction d:
        ## beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        ## d_ = r_ + beta_ * d
        beta_ = new_norm_r ** 2 / norm_r ** 2
        d.reindex(input_to_output_inds, inplace=True)
        d.transpose_like(r, inplace=True)
        d.modify(apply = lambda data: data * beta_ + r.data )

        err = new_norm_r
        norm_r = new_norm_r

        if err < min_err:
            min_err = err
            x_min = x.copy()
        # else:
        #     ## convergence not necessarily monotonic
        #     print('error went up', err, min_err)

        it += 1

        # print('it', it, err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')
            # print('using min x')
            x = x_min

    return x, err / norm_b


def qtn_conjugate_gradient_squared_1site(As: Sequence[qtn.TensorNetwork], b: qtn.Tensor, x: qtn.Tensor,
                                         output_to_input_inds: dict[str, str],
                                         max_iter=DEFAULT_MAX_ITER, conv_tol=DEFAULT_CONV_TOL,
                                         verbose=False) \
        -> tuple[qtn.Tensor, float]:
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
        A: matrix, b: target, x: solution
        r: residual = b - Ax
        d: search direction
    """
    norm_b = b.norm()
    input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.reindex(input_to_output_inds, inplace=False)  ## ket inds
    rT0 = r.copy(deep=True)

    bra_inds = r.inds   ## out inds
    ket_inds = tuple([output_to_input_inds[k] for k in r.inds])     ## in inds

    def contract_with_As(*x_tens, out_inds=bra_inds, scale=True):
        Ax_tot = None
        for A in As:
            Ax = qtn.TensorNetwork([A, *x_tens], virtual=True, check_collisions=False)
            out = Ax.contract_tags(all)

            if isinstance(out, qtn.Tensor):
                # out.transpose_like(r, inplace=True)
                out.transpose(*out_inds, inplace=True)
                if scale:
                    out.modify(apply=lambda data: data * 10 ** Ax.exponent)
                if Ax_tot is None:
                    Ax_tot = out
                else:
                    Ax_tot.modify(apply=lambda data: data + out.data)
            else:
                if scale:
                    out *= 10 ** Ax.exponent
                if Ax_tot is None:
                    Ax_tot = out
                else:
                    Ax_tot += out
        return Ax_tot

    if x is not None:
        # Ax = qtn.TensorNetwork([A, x], virtual=True, check_collisions=False)
        # Ax_tens = Ax.contract_tags(all)
        # Ax_tens.transpose_like(r, inplace=True)
        Ax_tens = contract_with_As(x)
        r.modify(apply = lambda data: data - Ax_tens.data)      ## out inds

        xT = x.reindex(input_to_output_inds, inplace=False)     ## out inds
        # AxT = qtn.TensorNetwork([As[0], xT], virtual=True, check_collisions=False)
        # AxT_tens = AxT.contract_tags(all)   ## in inds
        AxT_tens = contract_with_As(xT, out_inds=ket_inds, scale=True)
        # print('AxT inds', AxT_tens.inds, ket_inds)
        AxT_tens.reindex(input_to_output_inds, inplace=True)
        AxT_tens.transpose(*bra_inds, inplace=True)
        rT0.modify(apply=lambda data: data - AxT_tens.data)

    # rT0 = r.copy()

    zero_tens = qtn.Tensor(np.zeros(r.shape), inds=bra_inds)
    zero_tens_cc = qtn.Tensor(np.zeros(r.shape), inds=ket_inds)

    it = 0
    q, p, rho = zero_tens_cc.copy(), zero_tens_cc.copy(), 1
    # err = qtn.tensor_contract(r, rT0)
    err = r.norm()
    min_x, min_err = x, err
    while it < max_iter and np.abs(err) > conv_tol * norm_b:

        rho_ = qtn.tensor_contract(r, rT0)
        beta = rho_ / rho

        u = qtn.Tensor(data = r.data + beta * q.data, inds=ket_inds)  ## in inds
        p_ = qtn.Tensor(data = u.data + beta * (q.data + beta * p.data), inds=ket_inds)   ## in inds
        # v = qtn.tensor_contract(*A.tensors, p_)     ## out inds
        # v.transpose_like(r, inplace=True)
        v = contract_with_As(p_)

        sigma = qtn.tensor_contract(rT0, v)  ## out inds
        alpha = rho_ / sigma

        q_ = qtn.Tensor(data = u.data - alpha * v.data, inds=ket_inds)  ## in inds
        tmp_uq = qtn.Tensor(data = u.data + q_.data, inds=ket_inds)     ## in_inds
        # tmp_A = qtn.tensor_contract(*A.tensors, tmp_uq)     ## out inds
        # tmp_A.transpose_like(r, inplace=True)
        tmp_A = contract_with_As(tmp_uq)
        # r_ = r - alpha * np.dot(A, u + q_)
        # x_ = x + alpha * (u + q_)
        r.modify(data= r.data - alpha * tmp_A.data)
        x.modify(data= x.data + alpha * tmp_uq.data)

        p, q = p_, q_
        rho = rho_
        it += 1

        err = r.norm()
        # print('it', it, err)

        if err < min_err:
            min_x, min_err = x, err
        # else:
        #     print('error went up', err, min_err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err / norm_b} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err / norm_b} in {it} iterations')
            # print('using min x')
            x = min_x

    return x, min_err / norm_b


#########################
### optimization fcts ###
#########################

def dmrg_linear_combine(mps_list: Sequence[qtn.MatrixProductState],
                        operator_list: Sequence[Optional['MPO_type']],
                        weights: Sequence['float'] = None,
                        init_guess: Optional[qtn.MatrixProductState] = None,
                        # normalize=True,
                        opt_nsites=1, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None):

    """ finds approximation of x that solves: prod(A) x = sum(b * weights)
        b = mps_list, A = operator list
    """
    ### more generally, would need to update LinearSolver to add operators applied to targets.
    ### need to double check the difference in contraction computational cost?

    ### horizontal contraction: O( (Dw^2 d) d (D^2 d) + (D (dD) (Dw D) )
    ### vertical contraction: O( (Dd) D (Dw D) * 2 + Dw (d^2 Dw) D^2 )
    ###                    or O( (Dd) D (Dw D) + (d Dw) (d Dw) (D^2) + D (d D) Dw D )
    ###
    ### costs are comparable ~ O (Dw^2 D^2 + D^3 Dw)

    if weights is not None:
        mps_list = [helper.scalar_multiply(mps, weight) for mps, weight in zip(mps_list, weights)]

    init_guess = mps_list[0] if init_guess is None else init_guess

    # approx_total_norm = np.sqrt( np.sum( [np.abs(w)**2 * mps.norm()**2 for (w, mps) in zip(weights, mps_list)] ))
    # if approx_total_norm < np.sqrt(CUTOFF):
    #     return None

    solver = LinearSolver(init_guess, targets=mps_list, operators=operator_list, conv_tol=conv_tol, max_iter=max_iter,
                          max_bond=max_bond)
    return solver.solve(opt_nsites)


def dmrg_add_mps(mps_list: Sequence[qtn.MatrixProductState],
                 init_guess: Optional[qtn.MatrixProductState]=None,
                 weights: Sequence['float'] = None,
                 # normalize=True,
                 opt_nsites=1, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None, **solver_kwargs):
    """ obtain approximation of x = sum(b), weights applied to mps_list
    """
    if weights is not None:
        mps_list = [helper.scalar_multiply(mps, weight) for mps, weight in zip(mps_list, weights)]

    init_guess = mps_list[0].copy() if init_guess is None else init_guess

    # approx_total_norm = np.sqrt( np.sum( [np.abs(w)**2 * mps.norm()**2 for (w, mps) in zip(weights, mps_list)] ))
    # if approx_total_norm < np.sqrt(CUTOFF):
    #     return None

    solver = LinearSolver(init_guess, targets=mps_list, conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond,
                          **solver_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def dmrg_apply(init_mps: qtn.MatrixProductState, mpo: qtn.MatrixProductOperator,
               init_guess: qtn.MatrixProductState = None,
               opt_nsites=2, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None, **solve_kwargs):

    init_guess = init_mps if init_guess is None else init_guess
    target = MatrixProductStateTN(init_mps, [mpo])
    solver = LinearSolver(init_guess, targets=[target], conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond,
                          **solve_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def dmrg_solve(soln_mps: qtn.MatrixProductState, operator: qtn.MatrixProductOperator,
               init_guess: qtn.MatrixProductState = None, is_H = True,
               opt_nsites=2, conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None, **solve_kwargs):
    """ solve Ax = b
    """
    print('dmrg solver')
    # init_guess = helper.scalar_multiply(soln_mps, 0.) if init_guess is None else init_guess   ## should be zero, probably?
    init_guess = soln_mps if init_guess is None else init_guess  ## should be zero, probably?
    solver = LinearSolver(init_guess, targets=[soln_mps.copy()], operators=[operator], is_H = is_H,
                          conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond,
                          **solve_kwargs)
    solver.solve(opt_nsites)
    return solver.ket, solver.err, solver.is_conv


def dmrg_compress(target_mps: qtn.MatrixProductState, init_guess: Optional[qtn.MatrixProductState] = None,
                  max_iter = 10, **dmrg_opt_kwargs):
    """ compress state
        init guess should have the desired bond dimension... unless 2 site optimization is used to adapt D
    """
    dmrg_opt_kwargs.setdefault('opt_nsites',2)
    cutoff = dmrg_opt_kwargs.pop('cutoff', CUTOFF)
    cutoff_mode = dmrg_opt_kwargs.pop('cutoff_mode', CUTOFF_MODE)
    return dmrg_add_mps([target_mps], init_guess=init_guess, max_iter=max_iter, **dmrg_opt_kwargs)



def mps_conjugate_gradient_descent(A: 'qtn.MatrixProductOperator', b: Optional['qtn.MatrixProductState'] = None,
                                   x: Optional['qtn.MatrixProductState'] = None,
                                   conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_bond=None,
                                   verbose=False):
    """ perform CGD to solve Ax=b with A, b represented as quimb tensor networks / tensors objects
            A: matrix, b: target, x: solution
            r: residual = b - Ax
            d: search direction
    """
    norm_b = helper.norm(b)
    # input_to_output_inds = {item: key for key, item in output_to_input_inds.items()}

    r = b.copy()
    r.site_ind_id = A.upper_ind_id
    # r = b.reindex(input_to_output_inds, inplace=False)
    if x is not None:
        Ax = helper.apply(A, x)     ## dmrg apply? dmrg compress?
        r = dmrg_add_mps([r, Ax], weights=[1, -1], conv_tol=conv_tol, max_iter=max_iter, max_bond=max_bond)

    d = r.copy()  ## d: search r
    norm_r = helper.norm(r)

    it = 0
    err = r.norm()
    while it < max_iter and err > conv_tol * norm_b:

        ## alpha = np.linalg.norm(r) ** 2 / np.vdot(d.conj(), np.dot(A, d))
        d.site_ind_id = A.lower_ind_id
        proj = helper.expectation_value(d, A)
        alpha = err ** 2 / proj

        ## update x:  x_ = x + alpha * d
        if x is None:
            x = d.copy()
            helper.scalar_multiply(x, alpha, inplace=True)
        else:
            x = dmrg_add_mps([x, d], weights = [1, alpha], max_bond=max_bond)
            # x.modify(apply=lambda data: data + alpha * d.data)

        ## update residual r:  r_ = r - alpha * np.dot(A, d)
        r_proj = helper.apply(A, d)
        r = dmrg_add_mps([r, r_proj], weights=[1, -alpha], max_bond=max_bond)
        new_norm_r = helper.norm(r)

        ## update search direction d:
        ## beta_ = np.linalg.norm(r_) ** 2 / np.linalg.norm(r) ** 2
        ## d_ = r_ + beta_ * d
        beta_ = new_norm_r ** 2 / norm_r ** 2
        d.site_ind_id = A.upper_ind_id
        d = dmrg_add_mps([d, r], weights = [beta_, 1], max_bond=max_bond)

        err = new_norm_r
        norm_r = new_norm_r
        it += 1

        if verbose:
            print('it', it, err)

    if verbose:
        if it < max_iter:
            print(f'cgd: error {err} in {it} iterations')
        else:
            print(f'cgd did not converge: error {err} in {it} iterations')

    return x, err



""" 
Functions to perform DMRG/local optimizations for MPS, using quimb as backend
"""
class Environment:

    def __init__(self, L: int, side: 'EnvironmentSide', ket: 'qtn.MatrixProductState',
                 bra: 'MPS_type' = None,
                 # operators_list: Sequence[qtn.MatrixProductOperator] = None,
                 operator: 'MPO_type' = None,
                 init_env: 'qtn.Tensor' = None,
                 mps_inds: Sequence[int] = None):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        """
        self.L = L        ## number of sites over which we'll need to make environments for
        self.side = side
        self.ket = ket    ## do not make copy -- want to update from outside
        self.bra = bra

        # print('ENV ket', ket)
        # print('ENV bra', bra)

        # self.operators = operators_list if operators_list is not None else []
        self.operator = operator
            ## ordered from ket to bra
        # self.exponent = self.ket.exponent + self.bra.exponent
        # if operator is not None:
        #     self.exponent += self.operator.exponent

        if self.side == EnvironmentSide.RIGHT:
            self._envs = {L - 1: init_env.copy() if init_env is not None else None}
        else:
            self._envs = {0: init_env.copy() if init_env is not None else None}

        # if init_env is not None:    ## verify/update ancilla bonds
        #     end_ind = 0 if self.side == EnvironmentSide.LEFT else L - 1
        #     ket_anc, op_anc, bra_anc = None, None, None
        #     if self.operator is None and self.operator[end_ind].ndim == 4:
        #         op_anc, other = init_env.filter_bonds(self.operator[end_ind])
        #         assert (len(op_anc) == 1), 'mpo anc and environment not consistent'
        #
        #     if self.ket[end_ind].ndim == 3:
        #         ket_anc, other = init_env.filter_bonds(self.ket[end_ind])
        #         assert (len(ket_anc) == 1), 'ket anc and environment not consistent'
        #     if self.bra[end_ind].ndim == 3:
        #         bra_anc, other = init_env.filter_bonds(self.bra[end_ind])
        #         if len(bra_anc) == 0:   ## update bra to have consistent anc ind
        #             bra_anc_1, bra_anc_2 = None, None
        #             for ind in self.bra[end_ind].inds:
        #                 if ind != self.bra.site_ind(end_ind) and ind not in self.bra[end_ind + self.side].inds:
        #                     bra_anc_1 = ind
        #                     break
        #             for ind in init_env.inds:
        #                 if ind not in ket_anc and ind not in op_anc:
        #                     bra_anc_2 = ind
        #                     break
        #             self.bra[end_ind].reindex({bra_anc_1: bra_anc_2}, inplace=True)

        self.mps_inds = list(range(self.ket.L)) if mps_inds is None else mps_inds
        self.bra_horizontal_bonds = {}
        self.ket_horizontal_bonds = {}
        self.update_horizontal_bonds()


    def __getitem__(self, pos):
        return self._envs.get(pos, None)

    def __setitem__(self, pos, env:'qtn.Tensor'):
        self._envs[pos] = env

    @property
    def exponent(self):
        exp = self.ket.exponent + self.bra.exponent
        if self.operator is not None:
            exp += self.operator.exponent
        return exp

    def update_horizontal_bonds(self):
        if self.side == EnvironmentSide.RIGHT:
            init_env = self._envs[self.L - 1]

            # print('ENV update self bra', self.bra)
            # print('ENV update self ket', self.ket)

            # ## get bonds with neighbors/envs on the right
            # self.bra_horizontal_bonds, self.ket_horizontal_bonds = {}, {}
            # for i in range(self.L - 1):
            #     ind1, ind2 = self.mps_inds[i:i + 2]
            #     # print('env bra', self.bra)
            #     # print('env ket', self.ket)
            #     # print('inds', self.mps_inds, i, ind1, ind2, self.ket.site_positions)
            #     b1 = self.bra.select_tensors(self.bra.site_tag_id.format(ind1))[0]
            #     b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            #     self.bra_horizontal_bonds[i] = next(iter(b1.bonds(b2)))  # self.bra.bond(ind1, ind2)
            #
            #     k1 = self.ket.select_tensors(self.ket.site_tag_id.format(ind1))[0]
            #     k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]
            #     self.ket_horizontal_bonds[i] = next(iter(k1.bonds(k2)))  # self.ket.bond(ind1, ind2)

            self.bra_horizontal_bonds = {i: self.bra.bond(self.mps_inds[i],self.mps_inds[i + 1])
                                         for i in range(0, self.L - 1)}
            self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i],self.mps_inds[i + 1])
                                         for i in range(0, self.L - 1)}
            # print('self. bra horizontal', self.bra_horizontal_bonds)
            # print('self. ket horizontal', self.ket_horizontal_bonds)

            self.bra_horizontal_bonds[self.L - 1] = None if init_env is None \
                                                        else self.bra[self.mps_inds[-1]].bonds(init_env)
            self.ket_horizontal_bonds[self.L - 1] = None if init_env is None \
                                                        else self.ket[self.mps_inds[-1]].bonds(init_env)

        else:
            init_env = self._envs[0]

            # ## get bonds with neighbors/envs on the left
            # self.bra_horizontal_bonds, self.ket_horizontal_bonds = {}, {}
            # for i in range(1, self.L):
            #     ind1, ind2 = self.mps_inds[i-1: i+1]
            #     # print('env ket', self.ket)
            #     # print('env bra', self.bra)
            #     # print(self.mps_inds, i, ind1, ind2)
            #     # print(self.ket.site_positions)
            #     # print('tens', self.bra[i-1], self.bra[i])
            #     b1 = self.bra.select_tensors(self.bra.site_tag_id.format(ind1))[0]
            #     b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            #     self.bra_horizontal_bonds[i] = next(iter(b1.bonds(b2)))  # self.bra.bond(ind1, ind2)
            #
            #     k1 = self.ket.select_tensors(self.ket.site_tag_id.format(ind1))[0]
            #     k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]
            #     self.ket_horizontal_bonds[i] = next(iter(k1.bonds(k2)))  # self.ket.bond(ind1, ind2)

            self.bra_horizontal_bonds = {i: self.bra.bond(self.mps_inds[i], self.mps_inds[i - 1])
                                         for i in range(1, self.L)}
            self.ket_horizontal_bonds = {i: self.ket.bond(self.mps_inds[i], self.mps_inds[i - 1])
                                         for i in range(1, self.L)}

            self.bra_horizontal_bonds[0] = None if init_env is None \
                                                else self.bra[self.mps_inds[0]].bonds(init_env)
            self.ket_horizontal_bonds[0] = None if init_env is None \
                                                else self.ket[self.mps_inds[0]].bonds(init_env)


    def get(self, pos):
        return self._envs.get(pos, None)


    def modify(self, ket=None, bra=None, operator=None, init_env=None, mps_inds=None):
        if ket is not None:     self.ket = ket
        if bra is not None:     self.bra = bra
        if operator is not None:    self.operator = operator
        if mps_inds is not None:    self.mps_inds = mps_inds
        self.reinitialize(init_env=init_env)
        self.update_horizontal_bonds()


    def create_like(self, ket=None, bra=None, operator=None, init_env=None, mps_inds=None):
        new_L = self.L if ket is None else ket.L
        self_init_env = self._envs[0] if self.side is EnvironmentSide.LEFT else \
                        self._envs[self.L - 1]
        new_env = Environment(new_L, self.side,
                              ket=self.ket if ket is None else ket,
                              bra=self.bra if bra is None else bra,
                              operator=self.operator if operator is None else operator,
                              init_env=self_init_env if init_env is None else init_env,
                              mps_inds=self.mps_inds if mps_inds is None else mps_inds)
        return new_env


    def copy(self):
        # new_env = Environment(self.L, self.side, self.ket, self.bra, self.operators)
        new_env = Environment(self.L, self.side, self.ket, self.bra, self.operator, mps_inds=self.mps_inds)
        new_env._envs = {k: env.copy() if env is not None else None for k, env in self._envs.items()}
        # new_env.exponent = self.exponent
        return new_env


    def reinitialize(self, init_env=None):
        if self.side == EnvironmentSide.RIGHT:
            init_env = self._envs[self.L - 1] if init_env is None else init_env
            self._envs = {self.L - 1: init_env}
        else:
            init_env = self._envs[0] if init_env is None else init_env
            self._envs = {0: init_env}


    def reinitialize_i(self, i):
        """ reinitialize envs associate with MPS site position i
        """
        if self.side == EnvironmentSide.RIGHT:
            for ix in range(i):
                self._envs.pop(ix, None)
        else:
            for ix in range(i+1,self.L):
                self._envs.pop(ix, None)


    def extend_env(self, pos: int) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        env = self._envs[pos]
        if isinstance(self.bra, MatrixProductStateTN):
            bra_tens = self.bra[self.mps_inds[pos]]
        else:
            bra_tens = [self.bra[self.mps_inds[pos]]]

        if isinstance(self.ket, MatrixProductStateTN):
            ket_tens = self.ket[self.mps_inds[pos]]
        else:
            ket_tens = [self.ket[self.mps_inds[pos]]]

        if isinstance(self.operator, MatrixProductOperatorTN):
            op_tens = self.operator[self.mps_inds[pos]]
        elif self.operator is None:
            op_tens = []
        else:
            op_tens = [self.operator[self.mps_inds[pos]]]
            # op_tens = [mpo[pos] for mpo in self.operators]

        new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)
        if env is not None:
            new_env.add(env)
        # print('new env', new_env)
        new_env_exp = new_env.exponent
        new_env = new_env.contract_tags(all)
        new_env.modify(apply = lambda x: x * 10**new_env_exp)
        # print('new env', new_env, new_env.norm())
        # print('env exponent', new_env_exp)
        self._envs[pos + self.side.value] = new_env
        return new_env


    def build_all_envs(self, end=None):
        """ compute all right envs for <Ax|b>
        """
        start = 0 if self.side is EnvironmentSide.LEFT else self.L - 1
        end = (-1 if self.side is EnvironmentSide.RIGHT else self.L) if end is None else end

        for i in range(start, end, -1):
            self.extend_env(i)
        return


#############################
###  minimize Ax=b error  ###
#############################

class LocalSolver:
    """ primary functions:
        - solve:
            - with least squares, conjugate gradient descent (CGD)
        - apply, add + compress
        - eigsolve:  (DMRG)
            minimizes <psi|H|psi> / <psi|psi>, equivalently minimizes <psi|H|psi> - lambda <psi|psi>
            sweep through sites i:
                H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
                N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
                --> minimize <psi|H|psi> - lambda <psi|psi>
                --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
                --> H_eff * T[i] - lambda * N_eff * T[i] = 0
                lambda will be the energy of the state
        operators: list of MPOs corresponding to operators being observed
                    <x|An ... A2 A1 |x> (ordered from ket to bra)
        mps_ind_range:  in case only working on a portion of MPS (indices aren't 0 to L)
    """
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 # bra_state: Optional['qtn.MatrixProductState'] = None,
                 mps_inds: Optional[Sequence[int]] = None,
                 left_envs: Optional[Sequence['Environment']] = None,
                 right_envs: Optional[Sequence['Environment']] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL,
                 max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None):

        self.virtual = virtual
        self._left_envs = [] if left_envs is None else left_envs
        self._right_envs = [] if right_envs is None else right_envs
        self._targets = []

        self._bra = None
        self._ket = trial_state if virtual else trial_state.copy()
        self.operators = operators # if virtual else [op.copy() for op in operators]
        self.set_bra_from_ket()
        self.targets = targets

        ## left index of MPS to perform dmrg over (eg. if only acting on region of MPS)
        self.mps_inds = list(range(0, self.ket.L)) if mps_inds is None else mps_inds

        self.conv_tol = conv_tol
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_bond = max_bond

        ## results
        self.err = np.inf       # self.check_err()
        self.is_conv = False    # np.abs(self.err) < conv_tol

    @property
    def L(self):
        return len(self.mps_inds)  # self.mps_inds[1] - self.mps_inds[0]

    @property
    def num_targets(self):
        return len(self._targets)

    @property
    def num_operators(self):
        return len(self._operators)

    @property
    def left_envs(self):
        return self._left_envs

    @property
    def right_envs(self):
        return self._right_envs

    @property
    def ket(self):
        return self._ket

    @ket.setter
    def ket(self, new_ket: qtn.MatrixProductState):
        self._ket = new_ket if self.virtual else new_ket.copy()
        for target in self.targets:
            target.site_ind_id = new_ket.site_ind_id
        if self.num_operators > 0:
            for op in self.operators:
                op.lower_ind_id = new_ket.site_ind_id
            # self.operators[0].lower_ind_id = new_ket.site_ind_id
        self.reinitialize_envs()

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, new_bra: qtn.MatrixProductState):
        self._bra = new_bra if self.virtual else new_bra.copy()
        if self.num_operators > 0:
            self._bra.site_ind_id = self.operators[-1].upper_ind_id
        else:
            self._bra.site_ind_id = self.ket.site_ind_id
        self.reinitialize_envs()

    def set_bra_from_ket(self, sites=None, reinit_envs=True):

        # if self.bra is not None:
        #     print('self.bra', self.bra)
        #     print('self.ket', self.ket)
        #     print('L', self.L)
        #     for i in range(self.L):
        #         print('dict', self.get_ket_to_bra_inds(i))

        if self._bra is None:
            self._bra = self.ket.conj(mangle_inner=False)    ## mangle inner if there are operators
            self._bra.mangle_inner_(append='_')
            if self.num_operators > 0:
                self._bra.site_ind_id = self.operators[-1].upper_ind_id
            else:
                self._bra.site_ind_id = self.ket.site_ind_id

            ## reindex ancilla bonds -- this is jank/doesn't generally catch all cases but should work
            inds = self._bra[0].bonds(self.ket)
            reindex_map = {ind: ind + '_' for ind in inds if ind != self.ket.site_ind_id.format(0)}
            self._bra[0].reindex(reindex_map, inplace=True)

            inds = self._bra[self._bra.L-1].bonds(self.ket)
            reindex_map = {ind: ind + '_' for ind in inds if ind != self.ket.site_ind_id.format(self._bra.L-1)}
            self._bra[self._bra.L-1].reindex(reindex_map, inplace=True)

            if reinit_envs:
                self.reinitialize_envs()

        else:   ## inplace update
            sites = range(self.L) if sites is None else sites
            # sites = self.mps_inds if sites is None else sites
            for ix in sites:
                i = self.mps_inds[ix]
                ket_to_bra_inds = self.get_ket_to_bra_inds(ix)
                # print('site i', i)
                # print('ket', type(self.ket), self.ket[i])
                # print('bra', type(self._bra), self._bra[i])
                # print('ket to bra inds', ket_to_bra_inds)

                self._bra[i].modify(data=self.ket[i].data.conj(),
                                    inds=tuple([ket_to_bra_inds[ik] for ik in self.ket[i].inds]))
                if reinit_envs:
                    self.reinitialize_envs_i(ix)

            self._bra.exponent = self.ket.exponent

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets: Union['qtn.MatrixProductState', Sequence[qtn.MatrixProductState]]):
        if new_targets is None:
            self._targets = []
        elif isinstance(new_targets, qtn.TensorNetwork):
            self._targets = [new_targets if self.virtual else new_targets.copy()]
        else:
            self._targets = []
            for new_target in new_targets:
                new_target = new_target if self.virtual else new_target.copy()
                new_target.site_ind_id = self.bra.site_ind_id
                new_target.mangle_inner_()
                self._targets += [new_target]
        self.reinitialize_envs()    ## technically wouldn't have to update norm envs

        # self.targets_norm = np.linalg.norm( [target.norm() * 10**target.exponent for target in new_targets] )
        target_ovlps = 0.0
        for target in self._targets:
            target_ovlps += helper.ovlp(target, target.conj())
        for ti, tj in np.ndindex(len(self._targets), len(self._targets)):
            if ti > tj:
                out = helper.ovlp(self._targets[ti], self._targets[tj].conj())
                target_ovlps += 2*np.real(out)
        self.targets_norm = np.sqrt(target_ovlps)

        # print('targets norm', self.targets_norm)

    @property
    def operators(self) -> list['MPO_type']:
        return self._operators

    @property
    def operators_H(self) -> list['MPO_type']:
        return self._operators_H

    @operators.setter
    def operators(self, new_operators: Sequence['MPO_type']):
        self._operators = [] if new_operators is None else \
                            [mpo if self.virtual else mpo.copy() for mpo in new_operators]

        upper_ind_id = None if self.bra is None else self.bra.site_ind_id
        for op in self._operators:
            # op.upper_ind_id = self.bra.site_ind_id
            if op.upper_ind_id == self.ket.site_ind_id:
                op.upper_ind_id = op.upper_ind_id + '_out'
            if upper_ind_id is not None:
                op.upper_ind_id = upper_ind_id
            op.lower_ind_id = self.ket.site_ind_id
            op.mangle_inner_()
            upper_ind_id = op.upper_ind_id


        # if len(self._operators) > 0:
        #     qtn.tensor_network_align(*self.operators[::-1], inplace=True)
        #     self._operators[0].lower_ind_id = self.ket.site_ind_id
        #     # self._operators[-1].upper_ind_id = self.bra.site_ind_id

        ## Hermitian conjugate of operators
        operators_H = [] if new_operators is None else [mpo.conj() for mpo in new_operators]
        operators_H = operators_H[::-1]
        for i in range(len(operators_H)):
            mpo = operators_H[i]
            # # if i != 0:
            # #     mpo.upper_ind_id = mpo.upper_ind_id + '_H'
            # mpo.lower_ind_id = mpo.lower_ind_id + '_H'

            helper.mpo_flip_upper_lower(mpo, inplace=True, mangle_inner=True)
            # if i != 0:
            #     mpo.lower_ind_id = mpo.lower_ind_id + '_H'
            mpo.upper_ind_id = mpo.upper_ind_id + '_H'

        self._operators_H = operators_H

        self.reinitialize_envs()


    def create_like(self, copy=True, **kwargs):
        raise NotImplementedError

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)
        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        return new_solver


    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.left_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            env.reinitialize()

        for env in self.right_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            env.reinitialize()

        self.err = np.inf
        self.is_conv = False


    def reinitialize_envs_i(self, i):
        """ if boundary envs are None, uses old values
        """
        for env in self.left_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            env.reinitialize_i(i)

        for env in self.right_envs:
            # env.modify(ket=self.ket, bra=self.bra)
            env.reinitialize_i(i)


    def get_ket_to_bra_inds(self, i) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        ind1 = self.mps_inds[i]
        # print('self.mps_inds', self.mps_inds, i, ind1, self.ket.site_positions)
        # print('arg', self.ket.site_ind(i), self.bra.site_ind(i))
        # print(type(self.ket), type(self.bra))
        inds_dict = {self.ket.site_ind(ind1): self.bra.site_ind(ind1)}

        # print('self.mps inds', self.L, self.mps_inds, i)
        b1 = self.bra.select_tensors(self.bra.site_tag_id.format(ind1))[0]
        k1 = self.ket.select_tensors(self.ket.site_tag_id.format(ind1))[0]

        ## right bond
        if i < self.L-1:
            # inds_dict[self.ket.bond(i,i+1)] = self.bra.bond(i,i+1)
            ind2 = self.mps_inds[i + 1]
            b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]

            bbond = next(iter(b1.bonds(b2)))
            kbond = next(iter(k1.bonds(k2)))
            inds_dict[kbond] = bbond

        ## left bond
        if i > 0:
            # inds_dict[self.ket.bond(i, i - 1)] = self.bra.bond(i, i - 1)
            ind2 = self.mps_inds[i - 1]
            b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
            k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]

            bbond = next(iter(b1.bonds(b2)))
            kbond = next(iter(k1.bonds(k2)))
            inds_dict[kbond] = bbond

        ## ancilla bonds (L-1)
        if i == self.L-1 or i == 0:
            items = [item for k, item in inds_dict.items()]
            anc_b = [ind for ind in self.bra[ind1].inds if ind not in items]
            anc_k = [ind for ind in self.ket[ind1].inds if ind not in inds_dict.keys()]
            for (ab, ak) in zip(anc_b, anc_k):
                inds_dict[ak] = ab

        # print('inds dict', i, inds_dict, ind1)

        return inds_dict


    def get_bra_to_ket_inds(self, i) -> dict[str,str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket


    def canonize(self, i, cur_orthog=None):
        # helper.canonize(self.ket, i=i)
        ## difficulties with mps inds not continuous
        ## instead use canonize from list
        # print('canonize', i)

        # print('canonize', self.ket)

        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        i = self.get_mps_ind(i)
        updated_inds = []
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds += [lx]
                tens_left += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds:
                    updated_inds += [lx]
                tens_right += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]
        helper.canonize_tens_list(*tens_left, inplace=True)
        helper.canonize_tens_list(*(tens_right[::-1]), inplace=True)

        # tens_i = tens_left[-1]
        # tens_i_norm = tens_i.norm()
        # tens_i.modify(apply=lambda x: x / tens_i_norm)
        # self.ket.exponent = self.ket.exponent + np.log10(tens_i_norm)

        # print('updated inds', updated_inds)
        self.set_bra_from_ket(sites=updated_inds)


    def compress(self, i, cur_orthog=None, compress_opts=None):
        # helper.canonize(self.ket, i=i)
        ## difficulties with mps inds not continuous
        ## instead use canonize from list
        # print('canonize', i)

        # print('canonize', self.ket)

        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                tens_left += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                tens_right += [self.ket.select_tensors((self.ket.site_tag_id.format(lx),))[0]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]
        helper.compress_tens_list(*tens_left, inplace=True, compress_opts=compress_opts)
        helper.compress_tens_list(*(tens_right[::-1]), inplace=True, compress_opts=compress_opts)
        self.set_bra_from_ket()


    def left_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i + 1)
        helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
        self.set_bra_from_ket(sites=[i,i+1])

    def right_canonize_site(self, i):
        ind1, ind2 = self.get_mps_ind(i), self.get_mps_ind(i - 1)
        helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
        self.set_bra_from_ket(sites=[i-1,i])

    def get_mps_ind(self, i):
        return self.mps_inds[i]


    #########################

    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: int of mps site
        """
        if direction == SweepDirection.RIGHT:
            ## update ket
            ind1 = self.mps_inds[i]
            site_i.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=site_i.data)

            ## canonicalize ket to next site
            if i < self.L - 1:
                ind2 = self.mps_inds[i + 1]
                helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
                ## update bra
                self.set_bra_from_ket(sites=[i, i + 1])
            else:
                ## update bra
                self.set_bra_from_ket(sites=[i])

        else:
            ## update k
            ind1 = self.mps_inds[i]
            self.ket[ind1].modify(data=site_i.data, inds=site_i.inds)

            ## canonicalize ket to next site
            if i > 0:
                ind2 = self.mps_inds[i - 1]
                helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True)
                ## update bra
                self.set_bra_from_ket(sites=[i, i - 1])
            else:
                ## update bra
                self.set_bra_from_ket(sites=[i])

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if direction == SweepDirection.RIGHT:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i+1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])    # shared, not shared
            site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site1.data, inds=site1.inds)
            self.ket[ind2].modify(data=site2.data, inds=site2.inds)

            ## update bra
            self.set_bra_from_ket(sites=[i, i + 1])

            # ## update bra
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i)
            # bra_inds_0 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i].inds]
            # self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)
            #
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i + 1)
            # bra_inds_1 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i + 1].inds]
            # self.bra[i + 1].modify(data=self.ket[i + 1].data.conj(), inds=bra_inds_1)

        else:
            ind1, ind2 = self.mps_inds[i], self.mps_inds[i - 1]
            split_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])    # inds only on ket[i]
            site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=self.max_bond)

            ## update ket
            self.ket[ind1].modify(data=site2.data, inds=site2.inds)
            self.ket[ind2].modify(data=site1.data, inds=site1.inds)

            ## update bra
            self.set_bra_from_ket(sites=[i, i - 1])

            # ## update bra
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i)
            # bra_inds_0 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i].inds]
            # self.bra[i].modify(data=self.ket[i].data.conj(), inds=bra_inds_0)
            #
            # ket_to_bra_dict = self.get_ket_to_bra_inds(i - 1)
            # bra_inds_1 = [ket_to_bra_dict[k_ind] for k_ind in self.ket[i - 1].inds]
            # self.bra[i - 1].modify(data=self.ket[i - 1].data.conj(), inds=bra_inds_1)

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        end = nsites - 1 if end is None else end

        if canonize:
            self.canonize(end)

        for env in self.right_envs:
            for i in range(self.L - 1, end, -1):
                env.extend_env(i)
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        end = self.L - nsites if end is None else end

        if canonize:
            self.canonize(end)

        for env in self.left_envs:
            for i in range(end):
                env.extend_env(i)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos + 1)
            # helper.canonize(self.ket, i=pos + 1)
            # self.set_bra_from_ket()
        for env in self.left_envs:
            env.extend_env(pos)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos - 1)
            # helper.canonize(self.ket, i=pos - 1)
            # self.set_bra_from_ket()
        for env in self.right_envs:
            env.extend_env(pos)
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def check_err(self, *args, **kwargs) -> 'float':
        """ also updates self.err, self.is_conv
        """
        raise NotImplementedError


class DMRGSolver(LocalSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Sequence['MPO_type'],
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 normAA_env0_Ls: Sequence['qtn.Tensor'] = None,
                 normAA_env0_Rs: Sequence['qtn.Tensor'] = None,
                 mps_ind_range: Optional[tuple[int]] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None):

        self.A_envsLs, self.A_envsRs = [], []
        super().__init__(trial_state, targets=None, operators=operators, mps_inds=mps_ind_range, conv_tol=conv_tol,
                         max_iter=max_iter, max_tot_iter=max_tot_iter, max_bond=max_bond)

        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Ls[i]) for i in range(self.num_operators)]
        self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Rs[i]) for i in range(self.num_operators)]

        self.normAA_env0_Ls = [None] * (self.num_operators**2) if normAA_env0_Ls is None else normAA_env0_Ls
        self.normAA_env0_Rs = [None] * (self.num_operators**2) if normAA_env0_Rs is None else normAA_env0_Rs

        # self.A2_envsRs, self.A2_envsLs = [], []
        # for i, j in np.ndindex(self.num_operators, self.num_operators):
        #     A, AT = self.operators[i], self.operators_H[j]
        #     if isinstance(A, qtn.MatrixProductOperator):
        #         AAT = MatrixProductOperatorTN([A,AT])
        #     else:
        #         AAT = A.apply(AT)
        #
        #     self.A2_envsLs += [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Ls[i])]
        #     self.A2_envsRs += [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, AAT,
        #                                    init_env=norm_env0_Rs[i])]
        #     ## init env only works if MPOs don't have ancillas
        #
        # # self.A2_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Ls[i]) for i in range(self.num_operators)]
        # # self.A2_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, [self.A, self.AT],
        # #                               init_env=norm_env0_Rs[i]) for i in range(self.num_operators)]
        #
        # # self.left_envs = [self.A_envsL, self.A2_envsL]
        # # self.right_envs = [self.A_envsR, self.A2_envsR]

        # self.left_envs = self.A_envsLs # + self.A2_envsLs
        # self.right_envs = self.A_envsRs # + self.A2_envsRs

        ### initial results
        self.eigval = 0
        # self.update_results()


    @property
    def left_envs(self):
        return self.A_envsLs

    @property
    def right_envs(self):
        return self.A_envsRs

    def check_eigval(self):
        """ also updates self.eigval
        """
        eigval = 0
        for i in range(self.num_operators):
            eigval += helper.expectation_value(self.ket, self.operators[i], self.bra)
        self.eigval = eigval
        return eigval


    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.A_envsLs:
            env.modify(ket=self.ket, bra=self.bra)

        for env in self.A_envsRs:
            env.modify(ket=self.ket, bra=self.bra)

        self.err = np.inf
        self.is_conv = False


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L - 1] for A_envR in self.A_envsRs]

        new_solver = DMRGSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                normAA_env0_Ls=kwargs.get('normAA_env0_Ls', self.normAA_env0_Ls),
                                normAA_env0_Rs=kwargs.get('normAA_env0_Rs', self.normAA_env0_Rs),
                                mps_ind_range=kwargs.get('mps_ind_range', self.mps_inds),
                                conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                max_iter=kwargs.get('max_iter', self.max_iter),
                                max_bond=kwargs.get('max_bond', self.max_bond),
                                max_tot_iter=kwargs.get('max_tot_iter', self.max_tot_iter))
        return new_solver


    def copy(self, deep=True):
        new_solver = super().copy(deep=deep) # self.create_like()
        new_solver.A_envsLs = [env.copy() if deep else env for env in self.A_envsLs]
        new_solver.A_envsRs = [env.copy() if deep else env for env in self.A_envsRs]
        new_solver.eigval = self.eigval
        return new_solver

    #################################################
    ###    global eigenvalue/eigenvector solver   ###
    #################################################

    def eigsolve(self, nsites: int, init_direction=SweepDirection.RIGHT, target_eig=0):
        """ solve via numpy/scipy?
        """
        raise NotImplementedError

    # def eigsolve_1site(self, **eigsolve_kwargs):
    #     return self.eigsolve(1, **eigsolve_kwargs)
    #
    # def eigsolve_2site(self, **eigsolve_kwargs):
    #     return self.eigsolve(2, **eigsolve_kwargs)

    #################################################
    ### DMRG, solve for eigenvalues, eigenvectors ###
    #################################################

    def get_outer_inds(self, sites: Union[int,slice], is_bra=False):
        """ get outer inds of a section of an MPS
        """
        site_pos = list(range(self.L))[sites]

        if is_bra:
            outer_inds = [self.bra.site_ind_id.format(i) for i in site_pos]
            anc_L = self.A_envsLs[0].bra_horizontal_bonds[site_pos[0]]      ## all envs should be the same
            anc_R = self.A_envsRs[0].bra_horizontal_bonds[site_pos[-1]]
        else:
            outer_inds = [self.ket.site_ind_id.format(i) for i in site_pos]
            anc_L = self.A_envsLs[0].ket_horizontal_bonds[site_pos[0]]
            anc_R = self.A_envsRs[0].ket_horizontal_bonds[site_pos[-1]]

        if isinstance(anc_L, (list,tuple)):
            outer_inds += [*anc_L]  # a tuple / list for MPSTN
        elif isinstance(anc_L, str):
            outer_inds += [anc_L]   # norm str for qtn.MPS
        ## else is None

        if isinstance(anc_R, (list,tuple)):
            outer_inds += [*anc_R]
        elif isinstance(anc_R, str):
            outer_inds += [anc_R]
        ## else is None

        return outer_inds


    def dmrg_optimize_site(self, sites: Union[int,slice],
                           # input_inds: Sequence[str], output_inds: Sequence[str], left: 'qtn.Tensor', right: 'qtn.Tensor',
                           # H: Union[qtn.Tensor, qtn.TensorNetwork],
                           is_H: bool = True) -> \
            tuple[float, qtn.Tensor]:
        """ site optimization for eigensolver. ie: minimizes <psi|H|psi> / <psi|psi>
            H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
            N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
            --> minimize <psi|H|psi> - lambda <psi|psi>
            --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
            --> H_eff * T[i] - lambda * N_eff * T[i] = 0
            lambda will be the energy of the state

            sites: qtn.Tensor, to be optimized (of the desired shape)
            left: qtn.Tensor, left environment <psi|H|psi>
            right: qtn.Tensor, right environment <psi|H|psi>
            H: Sequence[qtn.Tensor] or qtn.TensorNetwork, corresponding MPO site of operator H
        """

        site_pos = list(range(self.L))[sites]
        site_ind_0 = self.mps_inds[0] + site_pos[0]
        site_inds = slice(site_ind_0, site_ind_0 + len(site_pos))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|A|x>
        H_eff_tot = None
        input_inds = self.get_outer_inds(sites)
        output_inds = self.get_outer_inds(sites, is_bra=True)

        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][site_pos[0]]
            A_right = self.A_envsRs[i][site_pos[-1]]

            H_eff = qtn.TensorNetwork([])
            # for A in self.operators:
            H_eff.add(self.operators[i][site_inds])
            if A_left is not None:
                H_eff.add(A_left)
            if A_right is not None:
                H_eff.add(A_right)
            # print('A eff exponent', A_eff.exponent, self.A_envsL.exponent)
            #### !!! CHANGED HERE
            H_eff_exponent = self.A_envsLs[i].exponent + H_eff.exponent 

            H_eff = H_eff.contract_tags(all)
            H_eff.transpose(*output_inds, *input_inds, inplace=True)
            H_eff.modify(apply = lambda data: data * 10**H_eff_exponent)

            if H_eff_tot is None:
                H_eff_tot = H_eff
            else:
                H_eff_tot.modify(apply = lambda data: data + H_eff.data)

        # ### L * H * R = d/dT*[i] d/dT[i] <psi|H|psi>
        # H_eff = qtn.TensorNetwork([left, right])
        # H_eff.add(H)
        # H_eff = H_eff.contract_tags(all)
        # H_eff.transpose(*output_inds, *input_inds, inplace=True)

        ### H_eff[i] * T[i] = lambda T[i]  (assumes in canonical form)
        shape_out = H_eff_tot.shape[:len(output_inds)]
        shape_in = H_eff_tot.shape[len(output_inds):]  ## shape_in = shape_out
        H_eff_data = H_eff_tot.data.reshape(np.prod(shape_out), -1)
        if is_H:
            eigvals, eigvecs = linalg.eigh(H_eff_data)
        else:
            eigvals, eigvecs = linalg.eig(H_eff_data)

        eigval = eigvals[0]
        eigvec = eigvecs[:, 0]

        eigvec = eigvec.reshape(shape_in)
        eigvec_tens = qtn.Tensor(data=eigvec, inds=tuple(input_inds))
        return eigval, eigvec_tens


    def dmrg_1site(self, **dmrg_kwargs):
        """ dmrg sweep with 1-site optimization
        """
        return self.dmrg(1, **dmrg_kwargs)

    def dmrg_2site(self, **dmrg_kwargs):
        """ dmrg sweep with 2-site optimization
        """
        return self.dmrg(1, **dmrg_kwargs)

    def dmrg(self, nsites, init_direction=SweepDirection.RIGHT):
        return self.solve(nsites, init_direction=init_direction)

    def solve(self, nsites, init_direction=SweepDirection.RIGHT, verbose=False):

        L = self.ket.L
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)   # canonicalizees ket and bra

        ## ovlp <Ax|x>, <Ax|Ax> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(nsites, canonize=False)
        else:
            self._build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        err = self.check_err(0, self.L)
        conv_it, prev_err = 0, err
        eigval = self.check_eigval()
        # print('init err', err)
        # print('init eigval', eigval)
        min_ket, min_err, min_eigval = self.ket, err, eigval    ## save minimum error solns
        min_solver = self.copy()
        direction = init_direction
        while err > self.conv_tol and conv_it < self.max_iter and it < self.max_tot_iter:

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites):
                    # ix = self.get_mps_ind(i)

                    inds = slice(i, i + nsites)
                    eigval, site_i = self.dmrg_optimize_site(inds)
                    # site_i = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)

                err = self.check_err(L-nsites, nsites)
                self.eigval = eigval

            else:

                ## left sweep
                for i in range(L - 1, nsites - 1, -1):
                    # ix = self.get_mps_ind(i)

                    inds = slice(i - nsites + 1, i + 1)
                    eigval, site_i = self.dmrg_optimize_site(inds)

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)

                err = self.check_err(0, nsites)
                self.eigval = eigval

            if verbose:
                print('err', it, err)

            if np.abs((prev_err - err)/err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            it += 1

            if err < min_err:
                min_solver = self.copy(deep=True)
                # min_ket = self.ket.copy()
                # min_eigval = eigval     ## eigval associated with minimum error
                min_err = err
            else:
                print('Warning: solve error went up', err, min_err)

            direction *= -1  ## swaps sweep direction

        ## revert to optimal results
        if err > min_err:
            self._ket = min_solver.ket
            self._bra = min_solver.bra
            self.A_envsRs = min_solver.A_envsRs
            self.A_envsLs = min_solver.A_envsLs
            self.err = min_solver.err
            self.eigval = min_solver.eigval

        return self.eigval, self.ket


    def check_err(self, left_ind, nsites):
        """ error = <x|A^T A|x> - <x|A|x>^2
        """
        old_site_ind = self.bra.site_ind_id

        if self.num_operators > 0:
            self.bra.site_ind_id = self.operators_H[-1].upper_ind_id
            # self.bra.site_ind_id = self.operators_H[-1].lower_ind_id

        ## <xA|Ax>
        exp_AA_tot = 0.0
        for i, j in np.ndindex(self.num_operators, self.num_operators):
            normAA_L = self.normAA_env0_Ls[i * self.num_operators + j]
            normAA_R = self.normAA_env0_Rs[i * self.num_operators + j]
            # norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
            #                             [mpo for mpo in self.operators_H], virtual=True)
            norm_Ax = qtn.TensorNetwork([self.ket, self.bra, self.operators[i], self.operators_H[j]], virtual=True)
            if normAA_L is not None:
                norm_Ax.add(normAA_L)
            if normAA_R is not None:
                norm_Ax.add(normAA_R)

            A_eff_exponent = norm_Ax.exponent   ## self.A_envsL equivalent exponent included in normAA_env0_Ls?
            exp_AA = norm_Ax.contract_tags(all)
            exp_AA *= 10 ** A_eff_exponent
            exp_AA_tot += exp_AA

        self.bra.site_ind_id = old_site_ind

        ## <x|A|x>
        exp_A_tot = 0.0
        for i in range(self.num_operators):
            env_tens = []
            if self.A_envsLs[i][left_ind] is not None:
                env_tens += [self.A_envsLs[i][left_ind]]
            if self.A_envsRs[i][left_ind + nsites - 1] is not None:
                env_tens += [self.A_envsRs[i][left_ind + nsites - 1]]
            # env_tens = [self.A_envsLs[i][left_ind], self.A_envsRs[i][left_ind + nsites - 1]]
            ket_tens = [t for t in self.ket[left_ind:left_ind + nsites]]
            bra_tens = [t for t in self.bra[left_ind:left_ind + nsites]]
            mpo = self.operators[i]
            if isinstance(mpo, qtn.MatrixProductOperator):
                mpo_tens = [t for t in mpo[left_ind:left_ind + nsites]]
            else:
                mpo_tens = []
                for x in range(left_ind, left_ind + nsites):
                    mpo_tens += mpo[x]

            exp_A = qtn.TensorNetwork(env_tens + ket_tens + bra_tens + mpo_tens)
            # exp_A = qtn.TensorNetwork([self.A_envsL[left_ind], self.A_envsR[left_ind + nsites - 1],
            #                            self.ket[left_ind:left_ind + nsites], self.bra[left_ind:left_ind+nsites]] +
            #                           [mpo[left_ind:left_ind + nsites] for mpo in self.operators])
            #### !!! CHANGED HERE
            A_eff_exponent = self.A_envsLs[i].exponent + exp_A.exponent
            exp_A = exp_A.contract_tags(all)
            exp_A *= 10 ** A_eff_exponent
            exp_A_tot += exp_A

        err = np.abs(exp_A_tot**2 - exp_AA_tot)
        self.err = err
        self.is_conv = err < self.conv_tol
        return err


class LinearSolver(LocalSolver):
    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 is_H = True,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 err_env0_Ls: Sequence['qtn.Tensor'] = None,
                 err_env0_Rs: Sequence['qtn.Tensor'] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_bond=None,
                 solve_type:SolveMethod=DEFAULT_SOLVE):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        init_state = trial_state
        # if max_bond is not None and len(targets) == 1:
        #     init_state = helper.compress(init_state, compress_opts={'max_bond': max_bond})
        # max_bond_tmp = min(trial_state.max_bond(), 2**(trial_state.L//2)) if max_bond is None else max_bond
        # init_state = qtn.MPS_rand_state(trial_state.L, max_bond_tmp, site_tag_id=trial_state.site_tag_id,
        #                                 site_ind_id=trial_state.site_ind_id)
        super().__init__(init_state, targets=targets, operators=operators, mps_inds=mps_inds, conv_tol=conv_tol,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter)

        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        self.A_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Ls[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        self.A_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators[i],
                                     init_env=norm_env0_Rs[i], mps_inds=self.mps_inds)
                            for i in range(self.num_operators)]
        # self.A_envsL = Environment(self.L, EnvironmentSide.LEFT, self.ket, self.bra, self.operators,
        #                            init_env = norm_env0_L)
        # self.A_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra, self.operators,
        #                            init_env = norm_env0_R)

        self.b_envsLs, self.b_envsRs = [], []
        for ib in range(self.num_targets):
            b_envL = Environment(self.L, EnvironmentSide.LEFT, self.targets[ib], self.bra, None,
                                 mps_inds = self.mps_inds,
                                 init_env = ovlp_env0_Ls[ib] if ovlp_env0_Ls is not None else None )
            b_envR = Environment(self.L, EnvironmentSide.RIGHT, self.targets[ib], self.bra, None,
                                 mps_inds = self.mps_inds,
                                 init_env=ovlp_env0_Rs[ib] if ovlp_env0_Rs is not None else None )
            self.b_envsLs += [b_envL]
            self.b_envsRs += [b_envR]

        # self.left_envs  = self.A_envsLs + self.b_envsLs
        # self.right_envs = self.A_envsRs + self.b_envsRs

        self.solve_type = solve_type
        self.is_H = is_H
        self.err_env0_Ls, self.err_env0_Rs = err_env0_Ls, err_env0_Rs
        ### each:  <xA|Ax>, <b|Ax>, <b|b> ancilla envs


    @property
    def left_envs(self):
        return self.A_envsLs + self.b_envsLs

    @property
    def right_envs(self):
        return self.A_envsRs + self.b_envsRs


    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for env in self.A_envsLs:
            env.modify(ket=self.ket, bra=self.bra, mps_inds=self.mps_inds)

        for i in range(len(self.b_envsLs)):
            env = self.b_envsLs[i]
            target = self.targets[i]
            env.modify(ket=target, bra=self.bra, mps_inds=self.mps_inds)


        for env in self.A_envsRs:
            env.modify(ket=self.ket, bra=self.bra, mps_inds=self.mps_inds)

        for i in range(len(self.b_envsRs)):
            env = self.b_envsRs[i]
            target = self.targets[i]
            env.modify(ket=target, bra=self.bra, mps_inds=self.mps_inds)

        self.err = np.inf
        self.is_conv = False


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L-1] for A_envR in self.A_envsRs]
        ovlp_env0_Ls = [b_envL[0] for b_envL in self.b_envsLs]
        ovlp_env0_Rs = [b_envR[self.L-1] for b_envR in self.b_envsRs]

        new_solver = LinearSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                  targets=kwargs.get('targets', [t.copy() if copy else t for t in self.targets]),
                                  operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                  is_H=kwargs.get('is_H', self.is_H),
                                  mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                  norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                  norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                  ovlp_env0_Ls=kwargs.get('ovlp_env0_Ls', ovlp_env0_Ls),
                                  ovlp_env0_Rs=kwargs.get('ovlp_env0_Rs', ovlp_env0_Rs),
                                  conv_tol=kwargs.get('conv_tol', self.conv_tol),
                                  max_iter=kwargs.get('max_iter', self.max_iter),
                                  max_bond=kwargs.get('max_bond', self.max_bond),
                                  max_tot_iter=kwargs.get('max_tot_iter', self.max_tot_iter),
                                  solve_type=kwargs.get('solve_type', self.solve_type))
        return new_solver

    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)
        # new_solver = LinearSolver(self.ket.copy(), targets=self.targets, operators=self.operators, is_H=self.is_H,
        #                           mps_ind_range=self.mps_ind_range, conv_tol=self.conv_tol,
        #                           max_iter=self.max_iter, max_bond=self.max_bond)

        new_solver.A_envsL = [A_envsL.copy() for A_envsL in self.A_envsLs]
        new_solver.A_envsR = [A_envsR.copy() for A_envsR in self.A_envsRs]
        new_solver.b_envsLs = [bL.copy() for bL in self.b_envsLs]
        new_solver.b_envsRs = [bR.copy() for bR in self.b_envsRs]

        # new_solver.left_envs = new_solver.A_envsLs + new_solver.b_envsLs
        # new_solver.right_envs = new_solver.A_envsRs + new_solver.b_envsRs

        new_solver.solve_type = self.solve_type
        new_solver.err = self.err
        new_solver.is_conv = self.is_conv
        return new_solver


    def _site_solve(self, sites: Union[int,slice],) -> qtn.Tensor:
        """
        sites: int or slice(start, stop, step)
        """
        # print("SITE SOLVE", self.solve_type, sites)
        site_pos = list(range(self.L))[sites]
        # site_ind_0 = self.mps_inds[0] + site_pos[0]
        # site_inds = slice( site_ind_0, site_ind_0 + len(site_pos) )
        site_inds = self.mps_inds[sites]
        # site_inds = sites if isinstance(sites, slice) else slice(sites)

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][site_pos[0]]
            A_right = self.A_envsRs[i][site_pos[-1]]

            A_eff = qtn.TensorNetwork([])
            for A in self.operators:
                if isinstance(site_inds, (list,tuple)):
                    for si in site_inds:
                        A_eff.add(A[si])
                else:
                    A_eff.add(A[site_inds])
            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)
            # print('A eff exponent', A_eff.exponent, self.A_envsL.exponent)
            #### !!! CHANGED HERE
            A_eff.exponent = self.A_envsLs[i].exponent + A_eff.exponent
            A_effs += [A_eff]


        ### BL * B * BR = d/dT*[i] <x|b>
        b_eff: Optional['qtn.Tensor'] = None
        for ti in range(self.num_targets):
            b_left = self.b_envsLs[ti][site_pos[0]]
            b_right = self.b_envsRs[ti][site_pos[-1]]
            #
            # print('b env ket', self.b_envsLs[ti].ket)
            #
            # print('b left', b_left)
            # print('b right', b_right)

            b_eff_t = qtn.TensorNetwork([])
            if b_left is not None:
                b_eff_t.add(b_left)
            if b_right is not None:
                b_eff_t.add(b_right)

            # print('site inds', site_inds, self.targets[ti])
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff_t.add(self.targets[ti][si])
            else:
                b_eff_t.add(self.targets[ti][site_inds])

            #### !!! CHANGED HERE
            b_eff_exponent = self.b_envsLs[ti].exponent + b_eff_t.exponent
            b_eff_t = b_eff_t.contract_tags(all)
            # print('b eff exponent', b_eff_exponent)
            b_eff_t.modify(apply=lambda x: 10**b_eff_exponent * x)

            if b_eff is None:
                b_eff = b_eff_t
            else:
                b_eff_t.transpose_like(b_eff, inplace=True)
                b_eff.modify(apply=lambda x: x + b_eff_t.data)

        ## change indices from bra (missing T*[i]) to ket
        bra_to_ket_inds = {} # self.get_bra_to_ket_inds(site_inds[0])
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))

        # print('self.ket', self.ket)
        # print('bra to ket inds', bra_to_ket_inds)
        bra_inds = b_eff.inds if b_eff is not None else []
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        def sum_Aeffs():
            A_eff_ = None
            for A_eff_tn in A_effs:
                A_eff_tens = A_eff_tn.contract_tags(all)
                A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
                A_eff_tens.transpose(*bra_inds, *ket_inds, inplace=True)

                if A_eff_ is None:
                    A_eff_ = A_eff_tens
                else:
                    A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
            return A_eff_

        ## solve for T[i]
        if len(A_effs) == 0:
            x_eff = b_eff.reindex(bra_to_ket_inds)
            x_eff.modify(apply= lambda x : x * 10**(-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type is SolveMethod.CGD:
                # A_eff_exponent = A_eff.exponent
                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [self.ket[si] for si in site_inds]
                else:
                    ket_tensors = self.ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*ket_inds, inplace=True)
                # x_tens.modify(apply=lambda x: x * 0.)
                if self.is_H:
                    x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
                else:
                    x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
                # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))

            elif self.solve_type is SolveMethod.CGDx:

                if isinstance(site_inds, (list, tuple)):
                    ket_tensors = [self.ket[si] for si in site_inds]
                else:
                    ket_tensors = self.ket[site_inds]
                x_tens = qtn.tensor_contract(*ket_tensors)
                x_tens.transpose(*ket_inds, inplace=True)

                # A_eff_exponent = A_eff.exponent
                A_eff = sum_Aeffs()
                out_shape = A_eff.shape[:len(bra_inds)]
                in_shape = A_eff.shape[len(bra_inds):]
                A_eff_data = A_eff.data.reshape(np.prod(out_shape), -1)

                if b_eff is not None:
                    b_eff_data = b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                if self.is_H:
                    x_eff_data, error = conjugate_gradient_descent(A_eff_data, b_eff_data, x=x_tens.data.reshape(-1))
                else:
                    x_eff_data, error = conjugate_gradient_squared(A_eff_data, b_eff_data, x=x_tens.data.reshape(-1))
                # x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
                x_eff_data = x_eff_data.reshape(in_shape)
                # x_eff_data *= 10 ** (-A_eff_exponent)
                ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
                # print('ket inds', ket_inds, A_eff)
                # exit()
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

            else:   ## ALS optimzation
                # A_eff_exponent = A_eff.exponent
                # A_eff = A_eff.contract_tags(all)
                # ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]
                # # print('bra inds', bra_inds, 'ket inds', ket_inds)
                # A_eff.transpose(*bra_inds, *ket_inds, inplace=True)
                # out_shape = A_eff.shape[:len(bra_inds)]
                # in_shape = A_eff.shape[len(bra_inds):]

                A_eff = sum_Aeffs()
                out_shape = A_eff.shape[:len(bra_inds)]
                in_shape = A_eff.shape[len(bra_inds):]
                A_eff_data = A_eff.data.reshape(np.prod(out_shape),-1)

                if b_eff is not None:
                    b_eff_data = b_eff.data.reshape(-1)
                else:
                    b_eff_data = np.zeros((A_eff_data.shape[0],))

                x_eff_data = np.linalg.solve(A_eff_data, b_eff_data)
                x_eff_data = x_eff_data.reshape(in_shape)
                # x_eff_data *= 10 ** (-A_eff_exponent)
                ## b_eff_exponent already in b_eff, A_eff already includes ket, bra exponents
                x_eff = qtn.Tensor(data=x_eff_data, inds=ket_inds)

        return x_eff


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
            err = self.check_err(dense=True)
            # check_err = self.check_err(dense=True)
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
                    site_i = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)

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
                    site_i = self._site_solve(inds)

                    ## update ket, bra
                    if nsites == 1:
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)

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
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv

        return self.ket, self.err, self.is_conv


    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)


    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)


    def dmrg_sweep(self, nsites: int, direction=SweepDirection.RIGHT, conv_tol=0,
                   skip_inds: set = None, opt_inds: set = None,
                   canonize=False, build_envs=False, verbose=False, **kwargs):
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
        if canonize:
            canon_site = 0 if direction == SweepDirection.RIGHT else L-1
            self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            if direction == SweepDirection.RIGHT:
                self._build_all_envs_right(nsites, canonize=False)
            else:
                self._build_all_envs_left(nsites, canonize=False)

        ## sweep (left to right), direction < 0
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
                site_i = self._site_solve(inds)

                ## update ket (+ canonicalization)
                if nsites == 1:
                    self._update_1site(i, site_i, direction)
                elif nsites == 2:
                    self._update_2site(i, site_i, direction)
                else:
                    raise NotImplementedError

                ## update A, b envs (left)
                self._update_envs_left(i, canonize=False)

            err = self.check_err(dense=True)

        ## left sweep (right to left), direction < 0
        else:
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
                site_i = self._site_solve(inds)

                ## update ket, bra
                if nsites == 1:
                    self._update_1site(i, site_i, direction)
                elif nsites == 2:
                    self._update_2site(i, site_i, direction)
                else:
                    raise NotImplementedError

                ## update A, b envs (right)
                self._update_envs_right(i, canonize=False)

            err = self.check_err(dense=True)

        return err


    def check_err_np(self):
        """
        """
        if self.num_operators > 0:
            A = np.zeros((2**self.L, 2**self.L))
            for op in self.operators:
                op_tens = op.contract(all) * 10**op.exponent
                op_tens.transpose(*[op.upper_ind_id.format(i) for i in self.mps_inds],
                                  *[op.lower_ind_id.format(i) for i in self.mps_inds],
                                  inplace=True)
                op_mat = op_tens.data.reshape(2**self.L, 2**self.L)
                # A = np.dot(op_mat, A)
                A += op_mat
        else:
            A = np.eye(2**self.L)

        b = np.zeros(2**self.L)
        for target in self.targets:
            target_tens = target.contract(all) * 10**target.exponent
            # print('target tens?', target_tens)
            target_tens.transpose(*[target.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
            target_vec = target_tens.data.reshape(-1)
            b = b + target_vec

        x_tens = self.ket.contract(all) * 10**self.ket.exponent
        x_tens.transpose(*[self.ket.site_ind_id.format(i) for i in self.mps_inds], inplace=True)
        x_vec = x_tens.data.reshape(-1)

        # Ax = np.dot(A, x_vec)
        # print('<xA|Ax>', np.dot(Ax, Ax), '<b|Ax>', np.dot(b, Ax), '<b|b>', self.targets_norm**2)

        err = np.linalg.norm( np.dot(A, x_vec) - b )
        return err / self.targets_norm


    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
        """

        if dense:

            ## combine err envs
            ### L, R:  <xA|Ax>, <b|Ax>, <b|b> ancilla envs; ordered by bra_ind, ket_ind
            def combine_err_envs(err_envs: Sequence['qtn.Tensor']):
                xAAx, bAx, bb = err_envs
                bb = bb.reindex({bb.inds[i]: xAAx.inds[i] for i in range(len(bb.inds))})
                tot_env = qtn.tensor_direct_product(xAAx, bb, inplace=False)

                b_size, Ax_size = bb.shape[0], xAAx.shape[0]
                tot_size = tot_env.shape[0]
                bAx = bAx.copy()
                for ind in bAx.inds:
                    bAx.expand_ind(ind, tot_size)

                    if ind == 0:    ## move b inds to correct position
                        reorder_tens = np.zeros(tot_size)
                        reorder_tens[-b_size-1:,:b_size] = np.eye(b_size)
                        bAx.modify(apply=lambda x: np.tensordot(reorder_tens, x, axes=1), inplace=True)
                tot_env.modify(apply = lambda x: x + bAx.data)
                return tot_env

            err_env_L = None if self.err_env0_Ls is None else combine_err_envs(self.err_env0_Ls)
            err_env_R = None if self.err_env0_Rs is None else combine_err_envs(self.err_env0_Rs)

            ## sum |Ax>
            if self.num_operators > 0:
                ket_copy = self.ket.copy()
                Ax = None
                for i in range(self.num_operators):
                    Ax_i = helper.apply(self.operators[i], ket_copy)
                    if Ax is None:
                        Ax = Ax_i
                    else:
                        Ax = helper.add_MPS(Ax, Ax_i, inplace=True)
            else:
                Ax = self.ket.copy()

            ## sum |b>
            total_b = self.targets[0].copy()
            # print('total b', total_b)
            for ib in range(1,self.num_targets):
                print('add b', self.targets[ib])
                total_b = helper.add_MPS(total_b, self.targets[ib], inplace=True)

            Ax.site_ind_id = total_b.site_ind_id
            # print('Ax', Ax)
            # print('total b', total_b)
            ## method=overlap method generally does not give as accurate results
            if err_env_L is None and err_env_R is None:
                err = helper.distance(Ax, total_b, method='auto') / self.targets_norm
            else:
                helper.match_inner_inds(total_b, Ax, inplace=True)
                total_b.site_ind_id = Ax.site_ind_id
                Ax.distribute_exponent()
                total_b.distribute_exponent()

                tens_list = []
                if err_env_L is not None:
                    tens_list += [err_env_L]
                    shared_, ancL_b  = total_b[0].filter_bonds(Ax[0])
                    shared_, ancL_Ax = Ax[0].filter_bonds(total_b[0])
                    total_b[0].reindex({anc1: anc2 for anc1, anc2 in zip(ancL_b, ancL_Ax)}, inplace=True)

                if err_env_R is not None:
                    tens_list += [err_env_R]
                    shared_, ancR_b  = total_b[-1].filter_bonds(Ax[-1])
                    shared_, ancR_Ax = Ax[-1].filter_bonds(total_b[-1])
                    total_b[-1].reindex({anc1: anc2 for anc1, anc2 in zip(ancR_b, ancR_Ax)}, inplace=True)

                diff = helper.add_MPS(Ax, total_b.scalar_multiply(-1, inplace=True))
                diff_conj = diff.conj()
                if err_env_L is not None:
                    diff_conj[0].reindex({anc1: anc1 + '_' for anc1 in ancL_Ax}, inplace=True)
                if err_env_L is not None:
                    diff_conj[-1].reindex({anc1: anc1 + '_' for anc1 in ancR_Ax}, inplace=True)

                tens_list += [diff, diff_conj]
                err = qtn.tensor_contract(tens_list)

        else:   ## contract tags doesn't give as small errors as helper.distance()

            old_site_ind_id = self.bra.site_ind_id

            if self.num_operators > 0:
                self.bra.site_ind_id = self.operators_H[-1].upper_ind_id

            ## <xA|Ax>
            if self.num_operators > 0:
                exp_AA_tot = 0.0
                ket_copy = self.ket.copy()
                bra_copy = self.bra.copy()
                for i, j in np.ndindex(self.num_operators, self.num_operators):
                    # norm_Ax = qtn.TensorNetwork([self.ket, self.bra] + [mpo for mpo in self.operators] +
                    #                             [mpo for mpo in self.operators_H], virtual=True)
                    norm_Ax = qtn.TensorNetwork([ket_copy, bra_copy, self.operators[i], self.operators_H[j]],
                                                virtual=False)   ## can mangle indices
                    A_eff_exponent = norm_Ax.exponent
                    # print('norm Ax', norm_Ax)
                    exp_AA = norm_Ax.contract_tags(all)
                    exp_AA *= 10 ** A_eff_exponent
                    exp_AA_tot += exp_AA
                norm_Ax = exp_AA_tot
            else:
                norm_Ax = helper.norm(self.ket) ** 2

            ## <xA|b>
            ovlp_b_tot = 0.
            if self.num_operators > 0:
                bra_copy = self.bra.copy()
                for i, ib in np.ndindex(self.num_operators, self.num_targets):
                    ovlp_b = qtn.TensorNetwork([self.targets[ib], bra_copy, self.operators_H[i].conj()], virtual=False)
                                               # [mpo for mpo in self.operators_H], virtual=True)
                                                # [helper.mpo_transpose(mpo) for mpo in self.operators_H], virtual=True)

                    # b_eff_exponent = ovlp_b.exponent
                    b_eff_exponent = self.operators_H[i].exponent + self.bra.exponent + self.targets[ib].exponent
                    ovlp_b = ovlp_b.contract_tags(all)
                    ovlp_b_tot = ovlp_b_tot + ovlp_b * 10**b_eff_exponent
            else:
                for ib in range(self.num_targets):
                    ovlp_b = qtn.TensorNetwork([self.targets[ib], self.bra], virtual=False)
                    b_eff_exponent = self.bra.exponent + self.targets[ib].exponent
                    ovlp_b = ovlp_b.contract_tags(all)
                    ovlp_b_tot = ovlp_b_tot + ovlp_b * 10 ** b_eff_exponent

            self.bra.site_ind_id = old_site_ind_id

            # if self.num_operators == 0 and self.num_targets == 1:
            #     print('err dist b', helper.distance(self.bra.conj(), self.targets[0]) / self.targets_norm)
            #     print('err dist b', helper.distance(self.bra, self.targets[0]) / self.targets_norm)
            #     print('err dist k', helper.distance(self.ket, self.targets[0]) / self.targets_norm)
            #     print('self.targets norm', self.targets_norm, helper.norm(self.targets[0].copy()))
            #     print('norm', helper.norm(self.ket)**2)
            #     print('ovlp', helper.ovlp(self.bra, self.targets[0]))
            #     print('ovlp', helper.ovlp(self.ket, self.targets[0]))
            # elif self.num_operators == 1 and self.num_targets == 1:
            #     Ax = helper.apply(self.operators[0], self.ket)
            #     print('err dist Ax', helper.distance(Ax, self.targets[0]) / self.targets_norm)
            #     print('self.targets norm', self.targets_norm, helper.norm(self.targets[0].copy()))
            #     print('norm Ax', helper.norm(Ax)**2)
            #     print('ovlp', helper.ovlp(Ax, self.targets[0]))


            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err2 = (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot))
            # print(norm_Ax, self.targets_norm**2, ovlp_b_tot)
            # if err2 < 0 and np.abs(err2) < 1.0e-12:
            #     err2 = np.abs(err2)
            err2 = np.abs(err2)
            err = np.sqrt(err2) / self.targets_norm

        self.err = err
        self.is_conv = err < self.conv_tol

        return err



