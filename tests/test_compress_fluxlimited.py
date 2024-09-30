import sys

import matplotlib as mpl
import numpy as np

mpl.rcParams['lines.linewidth'] = 2.
import matplotlib.pyplot as plt

sys.path.append('../')
# sys.path.append('/mnt/c/Users/eyeri/wsl/code/tns/tns_pde_v2/')
# sys.path.append('/mnt/c/Users/eyeri/wsl/code/tns/tns_pde_v2/setup_/')
# print('sys path', sys.path)

# from setup_.configs import *
from setup_.configs import *
# from configs import *
# from gate import Gate
# from mps import MatrixProductState, MatrixProductOperator
import quimb.tensor as qtn
import helper_quimb
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
# from local_solvers.local_dmrg_compress_constrained import local_constrained_compress
# from local_solvers.local_dmrg_compress_constrained_v2 import local_constrained_compress
from local_solvers.local_dmrg_compress_constrained_v3 import local_constrained_compress
# from local_solvers.local_dmrg_compress_constrained_v4 import local_constrained_compress
# from local_solvers.terms_3 import Term_DMRG
# import local_solvers.tensor_callables as tens_fcts


L = 6
D = 16
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])

# fx = lambda x: 0.1 * np.cos(5*x) + np.heaviside(x-1, 0.5) + 2 * np.heaviside(x-1.5, 1.0) \
#                - 4 * np.heaviside(x-1.8, 1.0) + fx_imag(x) * 1.j
# fx_imag = lambda x: 1 - np.heaviside(x-1.2, 0.5)
# fx = lambda x:  x**2

x_scale = 0.1

np.random.seed(1)
data = np.random.rand(2**L) + 0.5 * np.random.rand(2**L) * 1.j
# data = data / np.linalg.norm(data)
print('data_tens', np.linalg.norm(data))

ax_x = Axis(L)
grid_X = Grid1D('GX', [ax_x])

### potential bug? doesn't work as well if exponents are not distributed

# data_tens = qtn.Tensor( data.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]) )
# init_mps = MatrixProductState.from_vec(data_tens, L, 'i{}', split_opts={'max_bond': D})
init_mps = GridTN1D.from_dense_state(data, grid_X)
init_mps.data.exponent += np.log10(x_scale)
init_mps.data.distribute_exponent()
print('init mps exp', init_mps.exponent)
print('init mps max bond', init_mps.max_bond())
data *= x_scale

plt.figure()
plt.plot(init_mps.get_data())
plt.plot(init_mps.take_firstderivative(ax_x).get_data(), '--')
# plt.plot(data)
plt.show()

### test 1:  return self in canonical form
do_test_1 = True
if do_test_1:
    D = 3

    ddx = grid_X.get_firstderivative_mpo(ax_x)

    cons_0 = grid_X.get_iden_mpo()
    # cons_0.scalar_multiply(1. / cons_0.frobenius_norm(), inplace=True)

    # cons_1 = grid_X.get_xmultiply_mpo(x_power=1)    ## this is not positive...
    cons_1 = grid_X.get_xmultiply_mpo(x_power=1)
    cons_1.data.distribute_exponent()
    print('cons 1', cons_1.exponent)

    ## maybe this it's difficult to also satisfy cons_2 because it's not orthogonal to cons_0?
    cons_2 = grid_X.get_xmultiply_mpo(x_power=2)
    cons_2.data.distribute_exponent()
    print('cons 1', cons_2.exponent)

    expec_0 = helper_quimb.expectation_value(init_mps.data, cons_0.data)
    expec_1 = helper_quimb.expectation_value(init_mps.data, cons_1.data)
    expec_2 = helper_quimb.expectation_value(init_mps.data, cons_2.data)
    print('expec 1', expec_1)

    # expec_0 = init_mps.meas_expec(cons_0, is_sqrt=True)
    # expec_1 = init_mps.meas_expec(cons_1, is_sqrt=True)

    # ########################
    # ### naive compression
    comp_gtn = init_mps.compress(compress_opts={'max_bond': D}, inplace=False) #, renorm=True)
    print('compress naive')
    print('ranks', init_mps.max_bond(), comp_gtn.max_bond())
    print('compression err', np.linalg.norm(data - comp_gtn.get_data()))
    ddx_f1 = comp_gtn.take_firstderivative(ax_x)
    print('|ddx f|^2', ddx_f1.frobenius_norm()**2)

    plt.figure()
    plt.plot(ddx_f1.get_data())
    plt.show()

    expec_0_ = helper_quimb.expectation_value(comp_gtn.data, cons_0.data)
    expec_1_ = helper_quimb.expectation_value(comp_gtn.data, cons_1.data)
    expec_2_ = helper_quimb.expectation_value(comp_gtn.data, cons_2.data)
    print('expec 0 err', expec_0_ - expec_0)
    print('expec 1 err', expec_1_ - expec_1)
    print('expec 2 err', expec_2_ - expec_2)
    print()

    ### compression with constraint ########
    print('init mps rank', init_mps.max_bond())
    constraints = [(cons_0.data, None, expec_0),
                   (cons_1.data, None, expec_1),
                   (cons_2.data, None, expec_2)
                   ]
    ddx_mpo = grid_X.get_firstderivative_mpo(ax_x)
    ddx_mpo_T = ddx_mpo.copy()
    helper_quimb.mpo_flip_upper_lower(ddx_mpo_T.data, inplace=True)
    # helper_quimb.mpo_transpose(ddx_mpo_T.data, inplace=True)
    ddx_squared = ddx_mpo.apply( ddx_mpo_T )
    print('ddx squared exponent', ddx_squared.exponent)
    # ddx_squared = grid_X.get_secondderivative_mpo(ax_x, ax_x).scalar_multiply(-1)

    # tmp_data = ddx_squared.get_data()
    # eigvals = np.linalg.eigvals(tmp_data)
    # print('eigvals', np.min(eigvals), np.max(eigvals))
    #
    # plt.figure()
    # plt.imshow(np.real(grid_X.map_mpo_to_operator(ddx_mpo_T)))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(np.real(grid_X.map_mpo_to_operator(ddx_squared)))
    # plt.colorbar()
    # plt.show()

    deriv_norm = helper_quimb.expectation_value(init_mps.data, ddx_squared.data,)
    geq_constraints = [(ddx_squared.data, None, deriv_norm)]
    ddx2_expec = helper_quimb.expectation_value(init_mps.data, ddx_squared.data,)
    print('|ddx f|^2', ddx_f1.frobenius_norm() ** 2)
    print('ddx2_expec', ddx2_expec, deriv_norm)
    iden = grid_X.get_iden_mpo()
    print('init mps', init_mps.exponent)
    comp_init = local_constrained_compress(init_mps.data, # ddx_mpo.data,
                                           constraints,
                                           geq_constraints=geq_constraints,
                                           max_bond=10, nsites=nsites, grid=grid_X)
    comp_init_gtn = grid_X.make_gridTN(comp_init)
    tot_gtn_2 = comp_init_gtn
    tot_gtn_2.compress(inplace=True)
    # tot_gtn_2 = comp_init_gtn.add(orthog_state.scalar_multiply(proj_val))

    print('compress init_MPS')
    print('ranks', tot_gtn_2.max_bond(), init_mps.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_2.get_data()))
    ddx_f2 = tot_gtn_2.take_firstderivative(ax_x)
    ddx_init = init_mps.take_firstderivative(ax_x)
    print('ddx err', np.linalg.norm(ddx_f2.get_data() - ddx_init.get_data()))
    print('|ddx f|^2', ddx_f2.frobenius_norm()**2)
    ddx2_expec = helper_quimb.expectation_value(tot_gtn_2.data, ddx_squared.data, )
    print('ddx2_expec', ddx2_expec)

    expec_0_ = helper_quimb.expectation_value(tot_gtn_2.data, cons_0.data)
    expec_1_ = helper_quimb.expectation_value(tot_gtn_2.data, cons_1.data)
    expec_2_ = helper_quimb.expectation_value(tot_gtn_2.data, cons_2.data)
    print('expec 0 err', expec_0_ - expec_0)
    print('expec 1 err', expec_1_ - expec_1)
    print('expec 2 err', expec_2_ - expec_2)
    print()

    # ########################
    # ### naive compression
    comp_gtn = init_mps.compress(compress_opts={'max_bond': D}, inplace=False) #, renorm=False)
    print('compress naive')
    print('ranks', init_mps.max_bond(), comp_gtn.max_bond())
    print('compression err', np.linalg.norm(data - comp_gtn.get_data()))
    ddx_f1 = comp_gtn.take_firstderivative(ax_x)
    ddx_init = init_mps.take_firstderivative(ax_x)
    print('ddx err', np.linalg.norm(ddx_f1.get_data() - ddx_init.get_data()))
    print('|ddx f|^2', ddx_f1.frobenius_norm()**2)
    ddx2_expec = helper_quimb.expectation_value(comp_gtn.data, ddx_squared.data, )
    print('ddx2_expec', ddx2_expec)

    expec_0_ = helper_quimb.expectation_value(comp_gtn.data, cons_0.data)
    expec_1_ = helper_quimb.expectation_value(comp_gtn.data, cons_1.data)
    expec_2_ = helper_quimb.expectation_value(comp_gtn.data, cons_2.data)
    print('expec 0 err', expec_0_ - expec_0)
    print('expec 1 err', expec_1_ - expec_1)
    print('expec 2 err', expec_2_ - expec_2)
    print()

    plt.figure()
    plt.plot(np.real(comp_gtn.get_data()), label='naive comp')
    plt.plot(np.imag(comp_gtn.get_data()), )
    plt.plot(np.real(tot_gtn_2.get_data()), '--', label='orthog init comp')
    plt.plot(np.imag(tot_gtn_2.get_data()), '--')
    plt.plot(np.real(data), 'k--', label='exact')
    plt.plot(np.imag(data), 'k:')
    plt.xlabel('v')
    plt.ylabel('x(v)')
    plt.legend()
    plt.show()

    plt.figure()
    ddx_f0 = init_mps.take_firstderivative(ax_x)
    plt.plot(np.real(ddx_f0.get_data()), 'k:', label='exact')
    plt.plot(np.imag(ddx_f0.get_data()), 'k:', label='exact')
    plt.plot(np.real(ddx_f1.get_data()), label='naive comp')
    plt.plot(np.imag(ddx_f1.get_data()), '--', label='naive comp')
    plt.plot(np.real(ddx_f2.get_data()), label='penalty comp')
    plt.plot(np.imag(ddx_f2.get_data()), '--', label='penalty comp')
    plt.xlabel('v')
    plt.ylabel("x'(v)")
    plt.title('d/dv x')
    plt.legend()
    plt.show()

    plt.figure()
    ddx_f0 = init_mps.take_firstderivative(ax_x)
    plt.plot(np.real(ddx_f1.get_data() - ddx_f0.get_data()), label='naive comp')
    plt.plot(np.imag(ddx_f1.get_data() - ddx_f0.get_data()), '--', label='naive comp')
    plt.plot(np.real(ddx_f2.get_data() - ddx_f0.get_data()), label='penalty comp')
    plt.plot(np.imag(ddx_f2.get_data() - ddx_f0.get_data()), '--', label='penalty comp')
    plt.xlabel('v')
    plt.ylabel("x'(v) - x'_0(v)")
    plt.title('difference in derivative')
    plt.legend()
    plt.show()
