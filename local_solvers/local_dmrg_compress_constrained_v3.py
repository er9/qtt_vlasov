import numpy as np
import scipy.optimize
import cvxpy as cp
import time

import helper_quimb
from local_solvers.local_dmrg_eval import *

""" different way of including <x|A|x> = b constraints:
    --> (v1) <x|A|x> - b = 0 
    --> (v2) <x|A-bI|x> = 0
    NOTE: ASSUMES THAT FIRST EQUALITY CONSTRAINT IS A QUADRATIC NORM CONSTRAINT
    AND ASSUMES OTHER CONSTRAINTS ARE QUADRATIC CONSTRAINTS
    **** --> (v3) optimize via SDP relaxation.
"""

tol = 1e-8

### cvx routine 1 (X = xx.T --> X - xx.T > 0)
def cvx_routine_1(target_x: qtn.Tensor, terms: list[tuple['qtn.Tensor','qtn.Tensor',Numeric]], is_complex=False):
    """
    n: number of elements in the vector
    is_complex: bool describing if results are complex
    """
    target = target_x.data.reshape(-1)
    n = len(target)
    if is_complex:
        target = complex_to_real(target)
    n_ = 2 * n if is_complex else n
    r0 = np.linalg.norm(target) ** 2

    x = cp.Variable((n_, 1))
    X = cp.Variable((n_, n_), symmetric=True)

    constraints = []
    for Qeff, Aeff, b in terms:

        constraint = None
        if Qeff is not None:
            P_data = Qeff.data.reshape(n,n)

            # tmp = target_x.data.reshape(-1)
            # c_err = tmp.conj().T @ Qeff.data.reshape(n,n) @ tmp - b
            # print('init err 1', c_err)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

            constraint = cp.trace(X @ P_data)

        if Aeff is not None:
            A_data = Aeff.data.reshape(n)
            if is_complex:
                A_data = complex_to_real(A_data).reshape(1, 2*n)   # already conjugated, probably
            constraint = constraint + A_data @ x

        if b is not None:
            constraint = constraint - b

        constraints += [constraint == 0]
        # c_err = target.T @ P_data @ target - b
        # print('init err 2', c_err)


    ## Schur complement method for X - xx.T > 0
    S = cp.bmat([[X, x],[x.T, np.array([[1.0]])]])
    constraints += [S >> 0]

    prob = cp.Problem(cp.Minimize(cp.trace(X) - target.T @ x - x.T @ target + r0),
                      constraints)
    time1 = time.time()
    prob.solve(solver=cp.SCS, eps=tol/10)
    print('solver time', time.time() - time1)

    res_x = x.value

    ## initial errors

    ## check errors
    print('diff from target', np.linalg.norm(res_x - target))
    print('optimal value/cost object is', prob.value)
    Y = X.value - np.outer(res_x, res_x)
    c_err = np.linalg.norm(Y)
    if np.abs(c_err) > tol * 1.5:
        print(f'trace constraint not satisfied', c_err)
        raise RuntimeError
    # eigY = np.linalg.eigvals(Y)
    # print('min/max eig (X - xxT)', np.min(eigY), np.max(eigY))

    tot_errs = np.linalg.norm(res_x - target)
    for ix, constraint in enumerate(constraints):
        c_err = constraint.violation()
        if np.abs(c_err) > tol * 1.5:
            print(f'constraint {ix} not satisfied', c_err)
            raise RuntimeError
        tot_errs += c_err

    new_x = target_x.copy()
    if is_complex:
        res_x = real_to_complex(res_x)
    new_x.modify(data=res_x.reshape(target_x.shape))
    return new_x, tot_errs


### cvx routine 2 (Shor relaxation)
def cvx_routine_2(target_x: qtn.Tensor, terms: list[tuple['qtn.Tensor', 'qtn.Tensor', Numeric]], is_complex=False):
    """
    n: number of elements in the vector
    is_complex: bool describing if results are complex
    cost obj: 1/2 x.T P0 x + q0.T x + r0
    want: minimize x.T x - 2 * q0.T x + <q0.T|q0>
    """
    target = target_x.data.reshape(-1, 1)
    n = len(target)

    if is_complex:
        target = complex_to_real(target)
    r0 = np.real(np.linalg.norm(target)) ** 2 / 2

    num_c = len(terms)

    lambdas = cp.Variable(2*num_c, nonneg=True)
    gamma = cp.Variable((1,1))

    P0 = np.eye(2*n) if is_complex else np.eye(n)
    rs = np.zeros(2*num_c)
    constraint_evals = []

    sumP, sumQ = P0, target.reshape(-1,1) * -1
    for ix, (Qeff, Aeff, b) in enumerate(terms):

        if Qeff is not None:
            P_data = Qeff.data.reshape(n, n)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

            ## enforce equality constraint by adding the negative constraint
            sumP = sumP + cp.multiply(lambdas[ix], P_data)
            sumP = sumP + cp.multiply(lambdas[num_c + ix], -P_data)

        if Aeff is not None:
            A_data = Aeff.data.reshape(-1,1)
            if is_complex:
                A_data = complex_to_real(A_data[0])  # already conjugated, probably
            sumQ = sumQ + cp.multiply(lambdas[ix], A_data)
            sumQ = sumQ + cp.multiply(lambdas[num_c + ix], -A_data)

        if b is not None:
            rs[ix] = -np.real(b) / 2
            rs[num_c + ix] = np.real(b) / 2
        else:
            b = 0.0

        if Qeff is None:
            constraint_evals += [lambda x: A_data @ x - b]
        elif Aeff is None:
            constraint_evals += [lambda x: 0.5 * x.T @ P_data @ x - b / 2]
        else:
            constraint_evals += [lambda x: 0.5 * x.T @ P_data @ x + A_data @ x - b / 2]

        ## compute initial error
        # c_err = constraint_evals[-1](target)
        # print('init err', ix, c_err)

    def primal_cost(x_: np.ndarray):
        return 0.5 * x_.T @ P0 @ x_ - target.T @ x_ + r0

    def primal_sum_cost(x_: np.ndarray):
        sumP_ = sumP if isinstance(sumP, np.ndarray) else sumP.value
        sumQ_ = sumQ if isinstance(sumQ, np.ndarray) else sumQ.value
        return 0.5 * x_.T @ sumP_ @ x_ + sumQ_.T @ x_ + r0

    def constraint_eval(x_: np.ndarray, i: int):

        Qeff, Aeff, b = terms[i]

        if Qeff is not None:
            P_data = Qeff.data.reshape(n, n)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

        if Aeff is not None:
            A_data = Aeff.data.reshape(-1,1)
            if is_complex:
                A_data = complex_to_real(A_data[0])  # already conjugated, probably

        b = 0 if b is None else b

        if Qeff is None:
            func = lambda x: A_data @ x - b
        elif Aeff is None:
            func = lambda x: 0.5 * x.T @ P_data @ x - b / 2
        else:
            func = lambda x: 0.5 * x.T @ P_data @ x + A_data @ x - b / 2

        return func(x_)

    def opt_x() -> np.ndarray:
        """ minimum value of f(x) = 1/2 x^T P x + x^T b + r0
            occurs at x* = -P^{-1} b
            with f(x*) = 1/2 b^T P^{-1}.T P P^{-1} b - b^T P^{-1}.T b + r0
                       = -1/2 b^T P^{-1}.T b + r0
            (note: P is symmetric/Hermitian, so P^{-1} is also symmetric/Hermitian)
        """
        # print('sum P?', np.linalg.norm(sumP.value - lams.value[0] * P1 - P0))
        sumP_ = sumP if isinstance(sumP, np.ndarray) else sumP.value
        sumQ_ = sumQ if isinstance(sumQ, np.ndarray) else sumQ.value
        x_ = - np.linalg.inv(sumP_) @ sumQ_
        # print('sumP', sumP_)
        # print('sumQ', sumQ_.shape, target.shape)
        # print(np.linalg.norm(-sumQ_ - target))
        # print(np.linalg.norm(sumQ_), np.linalg.norm(target))
        return x_

    ## init errs
    print('init errs')
    for i in range(num_c):
        c_err = constraint_eval(target, i)
        print('init err', c_err)
    print('primal', primal_cost(target))

    cost = gamma + lambdas @ rs + r0
    print('type sumQ', type(sumQ))
    S = cp.bmat([[sumP, sumQ / np.sqrt(2)], [sumQ.T / np.sqrt(2), -gamma]])
    constraints = [S >> 0]
    prob = cp.Problem(cp.Maximize(cost), constraints)

    time1 = time.time()
    prob.solve(solver=cp.SCS, eps=tol/10)
    print('solve time', time.time() - time1)

    # res_lambdas = lambdas.value


    # ## check errors
    # print('optimal value/cost object is', prob.value)
    # # eigS = np.linalg.eigvals(S.value)
    # # print('max/min eig S', np.min(eigS), np.max(eigS))

    res_x = opt_x()
    print('diff from target', np.linalg.norm(res_x - target))
    print('res x primal cost', primal_cost(res_x)) #, primal_cost(res_x + 0.1), primal_cost(res_x - 0.1))
    print('res x primal cost', primal_sum_cost(res_x)) #, primal_sum_cost(res_x + 0.1), primal_sum_cost(res_x - 0.1))
    tot_errs = np.linalg.norm(res_x - target)
    for i in range(num_c):
        c_err = constraint_eval(res_x, i)
        if np.abs(c_err) > tol * 1.5:
            print(f'constraint {i} not satisfied', c_err)
        tot_errs += c_err

    new_x = target_x.copy()
    if is_complex:
        res_x = real_to_complex(res_x)
    new_x.modify(data=res_x.reshape(target_x.shape))
    return new_x, tot_errs


### cvx routine 1 (X = xx.T --> X - xx.T > 0)
def cvx_routine_1_penalty(target_x: qtn.Tensor, terms: list[tuple['qtn.Tensor','qtn.Tensor',Numeric]],
                          penalties: Optional[list[tuple['qtn.Tensor',Numeric]]]=None,
                          is_complex=False):
    """
    n: number of elements in the vector
    is_complex: bool describing if results are complex
    """
    target = target_x.data.reshape(-1)
    n = len(target)
    if is_complex:
        target = complex_to_real(target)
    n_ = 2 * n if is_complex else n
    r0 = np.linalg.norm(target) ** 2

    x = cp.Variable((n_, 1))
    X = cp.Variable((n_, n_), symmetric=True)

    constraints = []
    for Qeff, Aeff, b in terms:

        constraint = None
        if Qeff is not None:
            P_data = Qeff.data.reshape(n,n)

            # tmp = target_x.data.reshape(-1)
            # c_err = tmp.conj().T @ Qeff.data.reshape(n,n) @ tmp - b
            # print('init err 1', c_err)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

            constraint = cp.trace(X @ P_data)

        if Aeff is not None:
            A_data = Aeff.data.reshape(n)
            if is_complex:
                A_data = complex_to_real(A_data).reshape(1, 2*n)   # already conjugated, probably
            constraint = constraint + A_data @ x

        if b is not None:
            constraint = constraint - b

        constraints += [constraint == 0]
        # c_err = target.T @ P_data @ target - b
        # print('init err 2', c_err)


    ## Schur complement method for X - xx.T > 0
    S = cp.bmat([[X, x],[x.T, np.array([[1.0]])]])
    constraints += [S >> 0]

    ## cost object to minimize
    cost_obj = cp.trace(X) - target.T @ x - x.T @ target + r0
    if penalties is not None:
        for P, mu in penalties:
            P_data = P.data.reshape(n, n)
            P_data = complex_to_real_mat(P_data)
            penalty_cost = cp.trace(P_data @ X) - target.T @ P_data @ x - x.T @ P_data @ target + target.T @ P_data @ target
            # penalty_cost = cp.quad_form(x, P_data) #### doesn't work
            # penalty_cost = cp.trace(P_data @ X)
            cost_obj = cost_obj + mu * penalty_cost  #  - mu * cp.trace(P_data @ X)

    ## solve the problem
    prob = cp.Problem(cp.Minimize(cost_obj), constraints)
    time1 = time.time()
    prob.solve(solver=cp.SCS, eps=tol/10)
    print('solver time', time.time() - time1)
    res_x = x.value

    ## check errors
    print('diff from target', np.linalg.norm(res_x - target))
    print('penalty cost', penalty_cost.value)
    print('optimal value/cost object is', prob.value)
    Y = X.value - np.outer(res_x, res_x)
    c_err = np.linalg.norm(Y)
    if np.abs(c_err) > tol * 5.0: # 1.5:
        print(f'trace constraint not satisfied', c_err)
        raise RuntimeError
    # eigY = np.linalg.eigvals(Y)
    # print('min/max eig (X - xxT)', np.min(eigY), np.max(eigY))

    tot_errs = np.linalg.norm(res_x - target)
    for ix, constraint in enumerate(constraints):
        c_err = constraint.violation()
        if np.abs(c_err) > tol * 5.0:  # * 1.5:
            print(f'constraint {ix} not satisfied', c_err)
            raise RuntimeError
        tot_errs += c_err

    new_x = target_x.copy()
    if is_complex:
        res_x = real_to_complex(res_x)
    new_x.modify(data=res_x.reshape(target_x.shape))
    return new_x, tot_errs



### cvx routine 1 (X = xx.T --> X - xx.T > 0)
def cvx_routine_1_ineq(target_x: qtn.Tensor, terms: list[tuple['qtn.Tensor','qtn.Tensor',Numeric]],
                       ineq_constraints: Optional[list[tuple['qtn.Tensor','qtn.Tensor', Numeric]]]=None,
                       is_complex=False):
    """
    n: number of elements in the vector
    is_complex: bool describing if results are complex
    """
    target = target_x.data.reshape(-1)
    n = len(target)
    if is_complex:
        target = complex_to_real(target)
    n_ = 2 * n if is_complex else n
    r0 = np.linalg.norm(target) ** 2

    x = cp.Variable((n_, 1))
    X = cp.Variable((n_, n_), symmetric=True)

    constraints = []
    for Qeff, Aeff, b in terms:

        constraint = None
        if Qeff is not None:
            P_data = Qeff.data.reshape(n,n)

            # tmp = target_x.data.reshape(-1)
            # c_err = tmp.conj().T @ Qeff.data.reshape(n,n) @ tmp - b
            # print('init err 1', c_err)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

            constraint = cp.trace(X @ P_data)

        if Aeff is not None:
            A_data = Aeff.data.reshape(n)
            if is_complex:
                A_data = complex_to_real(A_data).reshape(1, 2*n)   # already conjugated, probably
            constraint = constraint + A_data @ x

        if b is not None:
            constraint = constraint - b

        constraints += [constraint == 0]
        # c_err = target.T @ P_data @ target - b
        # print('init err 2', c_err)

    for Qeff, Aeff, b in ineq_constraints:

        constraint = None
        if Qeff is not None:
            P_data = Qeff.data.reshape(n, n)
            if is_complex:
                P_data = complex_to_real_mat(P_data, is_real=False)

            constraint = cp.trace(X @ P_data)

            print('ineq constraint', target.T @ P_data @ target)

        if Aeff is not None:
            A_data = Aeff.data.reshape(n)
            if is_complex:
                A_data = complex_to_real(A_data).reshape(1, 2 * n)  # already conjugated, probably
            constraint = constraint + A_data @ x

        if b is not None:
            constraint = constraint - np.real(b)

        constraints += [constraint <= 0]
        # c_err = target.T @ P_data @ target - b
        # print('init err 2', c_err)

    ## Schur complement method for X - xx.T > 0
    S = cp.bmat([[X, x],[x.T, np.array([[1.0]])]])
    constraints += [S >> 0]

    ## cost object to minimize
    cost_obj = cp.trace(X) - target.T @ x - x.T @ target + r0

    ## solve the problem
    prob = cp.Problem(cp.Minimize(cost_obj), constraints)
    time1 = time.time()
    prob.solve(solver=cp.SCS, eps=tol/10)
    print('solver time', time.time() - time1)
    res_x = x.value

    ## check errors
    print('diff from target', np.linalg.norm(res_x - target))
    print('optimal value/cost object is', prob.value)
    Y = X.value - np.outer(res_x, res_x)
    c_err = np.linalg.norm(Y)
    if np.abs(c_err) > 1.0e-5: # tol * 5.0 # 1.5:
        print(f'trace constraint not satisfied', c_err)
        raise RuntimeError
    # eigY = np.linalg.eigvals(Y)
    # print('min/max eig (X - xxT)', np.min(eigY), np.max(eigY))

    # Q_eff = ineq_constraints[0][0]
    # P_data = Q_eff.data.reshape(n, n)
    # if is_complex:
    #     P_data = complex_to_real_mat(P_data, is_real=False)
    # print('ineq Q_eff', res_x.T @ P_data @ res_x)

    tot_errs = prob.value
    for ix, constraint in enumerate(constraints):
        c_err = constraint.violation()
        print('constraint error', ix, c_err)
        if np.abs(c_err) > 1.0e-5: # tol * 5.0:  # * 1.5:
            print(f'constraint {ix} not satisfied', c_err)
            raise RuntimeError
        tot_errs += c_err


    new_x = target_x.copy()
    if is_complex:
        res_x = real_to_complex(res_x)
    new_x.modify(data=res_x.reshape(target_x.shape))
    return new_x, tot_errs


#######################


### scipy optimizer only works for real-valued objects ###
def real_to_complex(z):  # real vector of length 2n -> complex of length n
    return z[:len(z) // 2] + 1j * z[len(z) // 2:]

def complex_to_real(z):  # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def real_to_complex_mat(A, is_real=True):  # real matrix of size 2n x 2n -> complex matrix of size n x n
    sh = len(A) // 2
    if is_real:
        return A[:sh,:sh]
    else:
        return A[:sh,:sh] + 1.j * A[:sh,sh:]

def complex_to_real_mat(A, is_real=True):  # complex matrix of size n x n
    """ (A + iB) (a + ib) = Aa + iBa + iAb - Bb
    """
    if is_real:
        return np.bmat([[np.real(A), np.zeros(A.shape)], [np.zeros(A.shape), np.real(A)]])
    else:
        return np.bmat([[np.real(A), -np.imag(A)], [np.imag(A), np.real(A)]])


def local_constrained_compress(ref_state: Union['MPS','qtn.MatrixProductState'],
                               # orthog_states: Sequence[Union['MPS', 'qtn.MatrixProductState']],
                               # lin_constraints: Sequence[tuple[Union['MPS', 'qtn.MatrixProductState'], Numeric]],
                               quad_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                                Union['MPS', 'qtn.MatrixProductState'], Numeric]],
                               geq_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                               Union['MPS', 'qtn.MatrixProductState'], Numeric]] = None,
                               penalties: Sequence[tuple['qtn.MatrixProductOperator', Numeric]] = None,
                               init_guess: Optional['qtn.MatrixProductState'] = None,
                               max_bond=None, combine_terms_func:Callable=None,
                               nsites=1, direction: SweepDirection = SweepDirection.RIGHT, grid=None,
                               ):
    """ use OrthogonalDMRGEvaluator with first projecting out orthog_states from ref_state
        quadratic constraint:
        -
    """
    init_guess = ref_state.copy() if init_guess is None else init_guess
    print('init guess', init_guess.max_bond())
    if max_bond is not None and init_guess.max_bond() > max_bond:
        helper_quimb.compress(init_guess, compress_opts={'max_bond': max_bond})
        helper_quimb.match_inner_inds(init_guess, ref_state)
        # helper_quimb.check_orthog(init_guess)

        init_guess[0].modify(apply=lambda x: x * 10**(init_guess.exponent - ref_state.exponent))
        init_guess.exponent = ref_state.exponent

    print('init guess', init_guess.max_bond())
    term = Term_DMRG(ref_state, bra=init_guess)
    solver = ConstrainedDMRGEvaluator(init_guess, [term],
                                      quad_constraints = quad_constraints,
                                      geq_constraints=geq_constraints,
                                      penalties=penalties,
                                      direction=direction, max_bond=max_bond,
                                      max_wrong_iter=2,
                                      grid=grid,
                                      )
    nsites = 1
    solver.solve(nsites)
    return solver.solution


class ConstrainedDMRGEvaluator(DMRGEvaluator):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 terms: Sequence['Term_DMRG'],
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'ConstrainedDMRGEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                 #
                 # quad_constraints: Sequence[tuple[Optional[Term], Optional[Term],'MPS', 'qtn.MatrixProductState']] = None,
                 quad_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                  Union['MPS', 'qtn.MatrixProductState'], Numeric]] = None,
                 geq_constraints: Sequence[tuple[Optional['qtn.MatrixProductOperator'],
                                                 Union['MPS', 'qtn.MatrixProductState'], Numeric]] = None,
                 penalties: Sequence[tuple['qtn.MatrixProductOperator', Numeric]] = None,
                 ):

        print('ket state max bond', ket_state.max_bond())
        super().__init__(ket_state, terms, direction=direction, max_bond=max_bond, conv_tol=conv_tol,
                         max_iter=max_iter, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         copy_obj=copy_obj, combine_terms_func=combine_terms_func,
                         grid=grid, ax_deriv_configs=ax_deriv_configs)

        print('done super init')

        if copy_obj:
            # print('self.out', self.out)
            # exit()

            self.eq_constraints = [tuple([qcterms[0].copy(ket_copy=self.ket, bra_copy=self.ket),
                                          qcterms[1].copy(bra_copy=self.ket) if qcterms[1] is not None else None,
                                          qcterms[2]])
                                   for qcterms in copy_obj.eq_constraints]
            self.geq_constraints = [tuple([qcterms[0].copy(ket_copy=self.ket, bra_copy=self.ket),
                                           qcterms[1].copy(bra_copy=self.ket) if qcterms[1] is not None else None,
                                           qcterms[2]])
                                    for qcterms in copy_obj.geq_constraints]
            self.pn_constraints = [(pnterms[0].copy(ket_copy=self.ket, bra_copy=self.ket), pnterms[1])
                                   for pnterms in copy_obj.eq_constraints]
            # self.eq_constraints = [tuple([qterm.copy(bra_copy=self.ket) for qterm in qcterms[:1]] + [qcterms[-1]])
            #                        for qcterms in copy_obj.eq_constraints]
            # self.geq_constraints = [tuple([qterm.copy(bra_copy=self.ket) for qterm in qcterms[:1]] + [qcterms[-1]])
            #                         for qcterms in copy_obj.geq_constraints]
            # self.pn_constraints = [(pnterms[0].copy(bra_copy=self.ket), pnterms[1])
            #                        for pnterms in copy_obj.eq_constraints]
        else:
            qc_terms_list = []
            for qc in quad_constraints:
                ## constraint of the form x^T Q x + A x + b = 0
                ## Q can be list of MPOs
                Qs, A, b = qc

                qc_terms = []
                ## quadratic constraint
                if (Qs is not None) and (not isinstance(Qs, (tuple, list))):
                    Qs = [Qs]
                    term_q = Term_DMRG(ket_state, operators=Qs, bra=ket_state, direction=self.direction,
                                       cur_orthog=self.cur_orthog)  ## checks self.cur_orthog here
                    term_q.initialize()
                    qc_terms += [term_q]
                else:
                    qc_terms += [None]

                ## linear constraint
                if A is not None:
                    helper_quimb.match_inner_inds(A, ket_state)
                    A.site_ind_id = ket_state.site_ind_id
                    term_a = Term_DMRG(A, bra=ket_state, direction=self.direction, cur_orthog=self.cur_orthog)
                    term_a.initialize()
                    qc_terms += [term_a]
                else:
                    qc_terms += [None]

                if len(qc_terms) > 0:
                    ## scalar constraint
                    qc_terms += [b if b is not None else 0.0]

                    ## add to list
                    qc_terms_list += [tuple(qc_terms)]

            self.eq_constraints: list[tuple[Optional[Term], Optional[Term], Numeric]] = qc_terms_list

            qc_terms_list = []
            geq_constraints = [] if geq_constraints is None else geq_constraints
            for qc in geq_constraints:
                ## constraint of the form x^T Q x + A x + b = 0
                ## Q can be list of MPOs
                Qs, A, b = qc

                qc_terms = []
                ## quadratic constraint
                if (Qs is not None) and (not isinstance(Qs, (tuple, list))):
                    Qs = [Qs]
                    term_q = Term_DMRG(ket_state, operators=Qs, bra=ket_state, direction=self.direction,
                                       cur_orthog=self.cur_orthog)  ## checks self.cur_orthog here
                    term_q.initialize()
                    qc_terms += [term_q]
                else:
                    qc_terms += [None]

                ## linear constraint
                if A is not None:
                    helper_quimb.match_inner_inds(A, ket_state)
                    A.site_ind_id = ket_state.site_ind_id
                    term_a = Term_DMRG(A, bra=ket_state, direction=self.direction, cur_orthog=self.cur_orthog)
                    term_a.initialize()
                    qc_terms += [term_a]
                else:
                    qc_terms += [None]

                if len(qc_terms) > 0:
                    ## scalar constraint
                    qc_terms += [b if b is not None else 0.0]

                    ## add to list
                    qc_terms_list += [tuple(qc_terms)]

            print('qc terms list geq', qc_terms_list)
            self.geq_constraints: list[tuple[Optional[Term], Optional[Term], Numeric]] = qc_terms_list

            ## penalties
            penalty_terms_list = []
            penalties = [] if penalties is None else penalties
            for pn in penalties:
                ## penalty of the form -mu * x^T Q x
                ## Q can be list of MPOs
                Qs, mu = pn         ## Q: operator, mu: penalty weight

                ## penalty operator
                if (Qs is not None) and (not isinstance(Qs, (tuple, list))):
                    Qs = [Qs]
                    term_q = Term_DMRG(ket_state, operators=Qs, bra=ket_state, direction=self.direction,
                                       cur_orthog=self.cur_orthog)  ## checks self.cur_orthog here
                    term_q.initialize()
                    pn_term = term_q
                else:
                    pn_term = None

                penalty_terms_list += [(pn_term, mu)]
            self.pn_constraints: list[tuple[Optional[Term], Numeric]] = penalty_terms_list


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        solve but with scipy optimizer.
        """
        print('ket max bond', self.ket.max_bond(), self.out.max_bond(), self.terms[0].ket.max_bond())

        site_list, site_err = super()._site_solve(left_site_pos, nsites, site_tens,
                                                  return_intermediates=return_intermediates)
        # for term in self.terms:
        #     print('lin terms', term.bra is self.ket, term.ket is self.ket, self.ket.max_bond(), term.ket.max_bond())
        #     should be True, False, self.max_bond(), uncompressed max_bond()

        is_complex = np.any(np.iscomplex(site_list[0].data))
        num_constraints = len(self.eq_constraints) + len(self.geq_constraints)

        ### doing the compression here is fine, but need to ensure that another 1/2 sweep is performed.
        ## if output_tensor bond dimension is larger than the desired rank, then perform low-rank approximation.
        out_tens = site_list[0]
        # print('self.out.max bond', self.out.max_bond(), self.max_bond)
        if np.any([out_tens.ind_size(ind) > self.max_bond for ind in out_tens.inds]):
            # print('compress site_list')
            left_inds = [self.ket.bond(left_site_pos, left_site_pos -1)] if left_site_pos > 0 else []
            left_inds += [self.ket.site_ind_id.format(left_site_pos)]
            ltens, rtens = qtn.tensor_split(out_tens, left_inds, cutoff=CUTOFF, max_bond=self.max_bond, renorm=False)
            comp_out_tens = qtn.tensor_contract(ltens, rtens)
            out_tens.modify(data=comp_out_tens.data, inds=comp_out_tens.inds)

        ## transpose all targets so that they are the same
        for xtens in site_list[1:]:
            xtens.transpose_like(site_list[0], inplace=True)
        site_shape = site_list[0].shape
        site_inds = site_list[0].inds

        bra_to_ket = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
        ket_to_bra = {v: k for k,v in bra_to_ket.items()}
        bra_inds = [ket_to_bra[k] for k in site_inds]

        # ## assumes the first constraint is a norm constraint
        # norm_b = self.eq_constraints[0][-1]

        ## evaulate constraints at the active site
        ## equality constraints
        eff_constraints = []
        for ix, penalty in enumerate(self.eq_constraints):
            Q, A, b = penalty
            # print('eq constraint', Q, A, b)

            Q_effs = None
            if Q is not None:
                # x_conj = site_list[0].conj()
                # x_conj.reindex({k: v for k, v in zip(site_inds, bra_inds)}, inplace=True)
                #
                # # print('Q_Effs id', id(Q_effs))
                # Q_effs = Q.get_eff_operator(left_site_pos, nsites, return_combined=False)[0]  ## 0th tier
                # c_err = qtn.tensor_contract(*Q_effs[0].tensors, site_list[0], x_conj) * 10 ** (Q_effs[0].exponent) - b
                # print('init c_err', c_err)

                Q_effs = Q.get_eff_operator(left_site_pos, nsites, return_combined=True)[0]  ## 0th tier
                Q_effs.transpose(*bra_inds, *site_inds, inplace=True)

                # c_err = qtn.tensor_contract(Q_effs, site_list[0], x_conj) - b
                # print('init c_err', c_err)

                # if True:  # ix == 2:
                #     plt.figure()
                #     sq_shape = np.prod(site_shape)
                #     plt.imshow(np.real(Q_effs.data).reshape(sq_shape, sq_shape))
                #     plt.colorbar()
                #     plt.figure()
                #     plt.imshow(np.imag(Q_effs.data).reshape(sq_shape, sq_shape))
                #     plt.colorbar()
                #     plt.show()

                ## include ket exponents
                Q_effs.modify(apply=lambda x: x * 10 ** (self.ket.exponent * 2))

            A_eff = None
            if A is not None:
                A_eff = A.get_evaluated_site(left_site_pos, nsites)
                A_eff.transpose(*bra_inds, inplace=True)

                ## include ket exponent
                A_eff.modify(lambda x: x * 10 ** self.ket.exponent)

            # print('constraints', Q_effs, A_eff, b)
            eff_constraints += [(Q_effs, A_eff, b)]

        ## inequality constraints
        geq_eff_constraints = []
        for ix, constraint in enumerate(self.geq_constraints):
            Q, A, b = constraint
            # print('eq constraint', Q, A, b)

            Q_effs = None
            if Q is not None:

                Q_effs = Q.get_eff_operator(left_site_pos, nsites, return_combined=True)[0]  ## 0th tier
                Q_effs.transpose(*bra_inds, *site_inds, inplace=True)

                ## include ket exponents
                Q_effs.modify(apply=lambda x: x * 10 ** (self.ket.exponent * 2))

            A_eff = None
            if A is not None:
                A_eff = A.get_evaluated_site(left_site_pos, nsites)
                A_eff.transpose(*bra_inds, inplace=True)

                ## include ket exponent
                A_eff.modify(lambda x: x * 10 ** self.ket.exponent)

            # n = np.product(site_shape)
            # P_data = Q_effs.data.reshape(n,n)
            # plt.figure()
            # plt.imshow(np.real(P_data))
            # plt.colorbar()
            # plt.show()

            # print('expec?', helper_quimb.expectation_value(self.ket, self.geq_constraints[0][0].operators[0]))

            # plt.figure()
            # P_data = self.grid.axes[0].map_mpo_to_operator(self.geq_constraints[0][0].operators[0])
            # plt.imshow(np.real(P_data))
            # plt.show()

            # print('constraints', Q_effs, A_eff, b)
            geq_eff_constraints += [(Q_effs, A_eff, b)]

        eff_penalties = []
        for ix, penalty in enumerate(self.pn_constraints):
            Q, mu = penalty
            # print('eq constraint', Q, A, b)

            Q_effs = None
            if Q is not None:

                Q_effs = Q.get_eff_operator(left_site_pos, nsites, return_combined=True)[0]  ## 0th tier
                Q_effs.transpose(*bra_inds, *site_inds, inplace=True)

                ## include ket exponents
                Q_effs.modify(apply=lambda x: x * 10 ** (self.ket.exponent * 2))

            # print('constraints', Q_effs, A_eff, b)
            eff_penalties += [(Q_effs, mu)]

        ## impose boundary conditions on all outputs, including intermediate states
        constrained_tens_list = []
        tot_err = site_err
        if len(self.pn_constraints) == 0:
            for ix, new_tens in enumerate(site_list):
                # x_tens, tot_err = cvx_routine_1(new_tens, eff_constraints, is_complex=is_complex)
                print('num constraints', len(eff_constraints), len(geq_eff_constraints))
                x_tens, tot_err = cvx_routine_1_ineq(new_tens, eff_constraints, geq_eff_constraints, is_complex=is_complex)
                constrained_tens_list += [x_tens]
        else:
            for ix, new_tens in enumerate(site_list):
                x_tens, tot_err = cvx_routine_1_penalty(new_tens, eff_constraints, eff_penalties, is_complex=is_complex)
                constrained_tens_list += [x_tens]

        print('self.ket is self.out', self.ket is self.out, self.ket.max_bond(), self.out.max_bond())

        return constrained_tens_list, tot_err



    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        site_list, site_err = super()._bond_solve(left_site_pos, site_tens,
                                                  return_intermediates=return_intermediates)

        raise NotImplementedError



    def update_blocks(self, i: int, direction: SweepDirection): #, new_ket_site=None):

        super().update_blocks(i, direction)

        for term in self.eq_constraints:
            Q, A, b = term
            if Q is not None:
                Q.extend_env(i, direction=direction)
            if A is not None:
                A.extend_env(i, direction=direction)

            # # print('term extend env', term)
            # term.extend_env(i, direction=direction) #, new_ket_site=new_ket_site)

        for term in self.geq_constraints:
            Q, A, b = term
            if Q is not None:
                Q.extend_env(i, direction=direction)
            if A is not None:
                A.extend_env(i, direction=direction)

        for term in self.pn_constraints:
            Q, mu = term
            if Q is not None:
                Q.extend_env(i, direction=direction)


    # ## since we are doing compression, don't update the ket here.
    # def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
    #                   grid: 'Grid'=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None):
    #     """ update ket, bra with new_site
    #         i: int of mps site
    #     """
    #
    #     helper_dmrg.update_1site(self.out, i, site_i, direction, max_bond=self.max_bond,
    #                              filter_bases=filter_bases, grid=grid, ax_deriv_configs=ax_deriv_configs)
    #     # print('out norm', helper_quimb.norm(self.out))
    #
    #     at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
    #     # print('at end?', at_end)
    #     if not at_end:
    #         # print('DMRG update KET update 1 site', i)
    #         helper_dmrg.update_1site(self.ket, i, self.ket[i], direction,
    #                                  filter_bases=filter_bases, grid=grid, ax_deriv_configs=ax_deriv_configs)
    #
    #     # print('update vecblock')
    #     for term in self.terms:
    #         term.update_intermediate_kets(i, 1, direction)
    #
    #     # print('update blocks')
    #     if not at_end:
    #         self.update_blocks(i, direction=direction)
    #
    #     return
    #
    #
    # def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: mps_site
    #     """
    #     at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)
    #     left_site_ind = i if direction == SweepDirection.RIGHT else i - 1
    #
    #     if not at_end:
    #         # raise NotImplementedError
    #         helper_dmrg.update_2site(self.out, left_site_ind, site_i, direction, max_bond=self.max_bond)
    #         helper_dmrg.update_1site(self.ket, left_site_ind, self.ket[i], direction)
    #
    #         # print('update vecblock')
    #         for term in self.terms:
    #             term.update_intermediate_kets(i, 2, direction)
    #
    #         # print('update blocks')
    #         self.update_blocks(i, direction=direction)
    #
    #
    #     else:
    #         helper_dmrg.update_2site(self.out, left_site_ind, site_i, direction, max_bond=self.max_bond)
    #         helper_dmrg.update_1site(self.ket, left_site_ind, self.ket[i], direction,
    #                                  decimate_only=True)
    #
    #         # print('update vecblock')
    #         for term in self.terms:
    #             term.update_intermediate_kets(i, 2, direction)
    #
    #     return
