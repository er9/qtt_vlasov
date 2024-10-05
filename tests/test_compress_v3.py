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
import helper_compress_v3


L = 6
D = 16
direction = 1
nsites = 1

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])

fx = lambda x: 0.1 * np.cos(5*x) + np.heaviside(x-1, 0.5) + 2 * np.heaviside(x-1.5, 1.0) \
               - 4 * np.heaviside(x-1.8, 1.0) + fx_imag(x) * 1.j
fx_imag = lambda x: 1 - np.heaviside(x-1.2, 0.5)
# fx = lambda x:  x**2
data = fx(x) + 1.j * fx_imag(x)

x_scale = 0.1

# np.random.seed(1)
# data = np.random.rand(2**L) + 0.5 * np.random.rand(2**L) * 1.j
# data = data / np.linalg.norm(data)
print('data_tens', np.linalg.norm(data))

ax_x = Axis(L)
grid_X = Grid1D('GX', [ax_x])

### potential bug? doesn't work as well if exponents are not distributed

# data_tens = qtn.Tensor( data.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]) )
# init_mps = MatrixProductState.from_vec(data_tens, L, 'i{}', split_opts={'max_bond': D})
init_gtn = GridTN1D.from_dense_state(data, grid_X)
init_gtn.data.exponent += np.log10(x_scale)
init_gtn.data.distribute_exponent()
print('init mps exp', init_gtn.exponent)
print('init mps max bond', init_gtn.max_bond())
data *= x_scale

plt.figure()
plt.plot(init_gtn.get_data())
plt.plot(init_gtn.take_firstderivative(ax_x).get_data(), '--')
# plt.plot(data)
plt.show()

depth = 2
coarse_data = helper_compress_v3.coarse_grain_gtn(init_gtn, depth).get_data()
x_coarse = np.linspace(0, 2*np.pi, 2**(L-depth), endpoint=False)


D = 3

ddx = grid_X.get_firstderivative_mpo(ax_x)

############################
## MPOs to measure moments #
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

expec_0 = helper_quimb.expectation_value(init_gtn.data, cons_0.data)
expec_1 = helper_quimb.expectation_value(init_gtn.data, cons_1.data)
expec_2 = helper_quimb.expectation_value(init_gtn.data, cons_2.data)
print('expec 1', expec_1)

# expec_0 = init_mps.meas_expec(cons_0, is_sqrt=True)
# expec_1 = init_mps.meas_expec(cons_1, is_sqrt=True)

#########################
### naive compression ###
comp_gtn = init_gtn.compress(compress_opts={'max_bond': D}, inplace=False) #, renorm=True)
print('compress naive')
print('ranks', init_gtn.max_bond(), comp_gtn.max_bond())
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

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn.get_data()), label='naive comp')
plt.plot(x, np.imag(comp_gtn.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compression dmrg with coarsening ########
### doesn't differ much from normal compression dmrg

comp_gtn_2 = helper_compress_v3.compress_dmrg_coarse_grain(init_gtn, 2, D)
print('compress coarse DMRG')
print('compression err', np.linalg.norm(data - comp_gtn_2.get_data()))
print('max bond', comp_gtn_2.max_bond())
ddx_f2 = comp_gtn_2.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f2.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_2.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_2.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_2.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_2.get_data()), label='coarse DMRG')
plt.plot(x, np.imag(comp_gtn_2.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with Lagrange interpolation ####
### easier to control rank, but may yield discontinuous solutions

comp_gtn_3 = helper_compress_v3.compress_interpolate_1D(init_gtn, 2, D)

print('compress Lagrangian interpolation')
print('compression err', np.linalg.norm(data - comp_gtn_3.get_data()))
print('max bond', comp_gtn_3.max_bond())
ddx_f3 = comp_gtn_3.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f3.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_3.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_3.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_3.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_3.get_data()), label='lagrange interp')
plt.plot(x, np.imag(comp_gtn_3.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
dx_c = x_coarse[1] - x_coarse[0]
plt.plot(x_coarse + dx_c/2, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse + dx_c/2, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with Michael's multiscale interpolation ####

comp_gtn_4 = helper_compress_v3.compress_multiscale_gtn(init_gtn, 2, 2, D,)

print('compress MG interpolation')
print('compression err', np.linalg.norm(data - comp_gtn_4.get_data()))
print('max bond', comp_gtn_4.max_bond())
ddx_f4 = comp_gtn_4.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f4.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_4.get_data()), label='MG interp')
plt.plot(x, np.imag(comp_gtn_4.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with Michael's multiscale interpolation ####

comp_gtn_4 = helper_compress_v3.compress_spline_1D(init_gtn, 2, D)

print('compress spline interpolation')
print('compression err', np.linalg.norm(data - comp_gtn_4.get_data()))
print('max bond', comp_gtn_4.max_bond())
ddx_f4 = comp_gtn_4.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f4.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_4.get_data()), label='spline interp')
plt.plot(x, np.imag(comp_gtn_4.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with cubic interpolation    ####

comp_gtn_4 = helper_compress_v3.compress_cubic_interpolation(init_gtn, 2, D)

print('compress cubic interpolation')
print('compression err', np.linalg.norm(data - comp_gtn_4.get_data()))
print('max bond', comp_gtn_4.max_bond())
ddx_f4 = comp_gtn_4.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f4.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_4.get_data()), label='cubic interp')
plt.plot(x, np.imag(comp_gtn_4.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with low-pass filter        ####

comp_gtn_4 = helper_compress_v3.smooth_k_gtn(init_gtn)

print('compress low-pass filter smoothing')
print('compression err', np.linalg.norm(data - comp_gtn_4.get_data()))
print('max bond', init_gtn.max_bond, comp_gtn_4.max_bond())
ddx_f4 = comp_gtn_4.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f4.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn_4.get_data()), label='cubic interp')
plt.plot(x, np.imag(comp_gtn_4.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()


#############################################
### compress with diffusion smoothing    ####

comp_gtn_4 = helper_compress_v3.smooth_nonlinear_diffusion_gtn(init_gtn, D)

print('compress weighted diffusion smoothing')
print('compression err', np.linalg.norm(data - comp_gtn_4.get_data()))
print('max bond', init_gtn.max_bond(), comp_gtn_4.max_bond())
ddx_f4 = comp_gtn_4.take_firstderivative(ax_x)
print('|ddx f|^2', ddx_f4.frobenius_norm()**2)

expec_0_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_0.data)
expec_1_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_1.data)
expec_2_ = helper_quimb.expectation_value(comp_gtn_4.data, cons_2.data)
print('expec 0 err', expec_0_ - expec_0)
print('expec 1 err', expec_1_ - expec_1)
print('expec 2 err', expec_2_ - expec_2)
print()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn.get_data()), label='naive comp')
plt.plot(x, np.imag(comp_gtn.get_data()), )
plt.plot(x, np.real(comp_gtn_4.get_data()), label='diff')
plt.plot(x, np.imag(comp_gtn_4.get_data()), )
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()



#############################################
exit()

plt.figure()
plt.plot()
plt.plot(x, np.real(comp_gtn.get_data()), label='naive comp')
plt.plot(x, np.imag(comp_gtn.get_data()), )
plt.plot(x, np.real(comp_gtn_2.get_data()), '--', label='coarse dmrg')
plt.plot(x, np.imag(comp_gtn_2.get_data()), '--')
plt.plot(x, np.real(comp_gtn_3.get_data()), '--', label='interp')
plt.plot(x, np.imag(comp_gtn_3.get_data()), '--')
plt.plot(x, np.real(comp_gtn_4.get_data()), '--', label='MG interp')
plt.plot(x, np.imag(comp_gtn_4.get_data()), '--')
plt.plot(x, np.real(data), 'k--', label='exact')
plt.plot(x, np.imag(data), 'k:')
plt.plot(x_coarse, np.real(coarse_data), 'ko--', label='exact')
plt.plot(x_coarse, np.imag(coarse_data), 'ko:')
plt.xlabel('v')
plt.ylabel('x(v)')
plt.legend()
plt.show()

# plt.figure()
# ddx_f0 = init_gtn.take_firstderivative(ax_x)
# plt.plot(np.real(ddx_f0.get_data()), 'k:', label='exact')
# plt.plot(np.imag(ddx_f0.get_data()), 'k:', label='exact')
# plt.plot(np.real(ddx_f1.get_data()), label='naive comp')
# plt.plot(np.imag(ddx_f1.get_data()), '--', label='naive comp')
# plt.plot(np.real(ddx_f2.get_data()), label='penalty comp')
# plt.plot(np.imag(ddx_f2.get_data()), '--', label='penalty comp')
# plt.xlabel('v')
# plt.ylabel("x'(v)")
# plt.title('d/dv x')
# plt.legend()
# plt.show()
#
# plt.figure()
# ddx_f0 = init_gtn.take_firstderivative(ax_x)
# plt.plot(np.real(ddx_f1.get_data() - ddx_f0.get_data()), label='naive comp')
# plt.plot(np.imag(ddx_f1.get_data() - ddx_f0.get_data()), '--', label='naive comp')
# plt.plot(np.real(ddx_f2.get_data() - ddx_f0.get_data()), label='penalty comp')
# plt.plot(np.imag(ddx_f2.get_data() - ddx_f0.get_data()), '--', label='penalty comp')
# plt.xlabel('v')
# plt.ylabel("x'(v) - x'_0(v)")
# plt.title('difference in derivative')
# plt.legend()
# plt.show()
