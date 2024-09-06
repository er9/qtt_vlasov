import sys

import matplotlib as mpl
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
# from local_solvers.terms_3 import Term_DMRG
# import local_solvers.tensor_callables as tens_fcts


L = 6
D = 16
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])

fx = lambda x: np.cos(x / 2) / (x+0.1) + 1 + np.sin(5*x) + np.sqrt(5*x)*5.0 * np.exp(-x**2/6)
fx = lambda x: np.cos(x / 2) / (x+0.1) + 0.5 * np.cos(20*x + 1.3) + np.sin(5*x) + np.sqrt(5*x)*5.0 * np.exp(-x**2/6) + 1.j * x
# fx = lambda x:  x**2

x_scale = 0.1

data = fx(x)
data = data / np.linalg.norm(data)
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
plt.plot(data)
plt.show()

### test 1:  return self in canonical form
do_test_1 = True
if do_test_1:
    D = 3

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
    comp_init = local_constrained_compress(init_mps.data, constraints,
                                               max_bond=D, nsites=nsites)
    comp_init_gtn = grid_X.make_gridTN(comp_init)
    tot_gtn_2 = comp_init_gtn
    tot_gtn_2.compress(inplace=True)
    # tot_gtn_2 = comp_init_gtn.add(orthog_state.scalar_multiply(proj_val))

    print('compress init_MPS')
    print('ranks', tot_gtn_2.max_bond(), init_mps.max_bond())
    print('compression err', np.linalg.norm(data - tot_gtn_2.get_data()))

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


