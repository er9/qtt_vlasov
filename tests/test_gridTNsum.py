import os
import sys
import time

sys.path.append('../')

from setup_.configs import *
# print(sys.argv)
# import pdb; pdb.set_trace()

import plot_defaults

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D_sum import GridTNSum
from field import Field
from tests.helper_EM_PML import _calculate_time_derivative_E_part
from tests.helper_EM_PML import _calculate_time_derivative_B
from tests.helper_EM_PML import *
import helper_compress_v3


#######################
# problem  parameters #
#######################

# permittivity = 1. / 2500  # 1.0    # permittivity of free space
# permeability = 1.0    # permeability of free space
permittivity = (1 / (36 * np.pi)) * 1e-9
permeability = 4 * np.pi * 1e-7

matl_config = UnitsConfiguration(
    eps0=1, mu0=1, is_cgs=False)
c = 1 / np.sqrt(permittivity * permeability)

# Spatial grid step length (spatial grid step= 1 micron and can be changed)
delta = 1e-6

# initial conditions (cavity size)
Lx = 256*delta
Ly = 256*delta

E0 = 1.             # electric field magnitude

xsource = Lx/2
ysource = Ly/2
sigma = 4e-5

### grid set-up
q = 2           # size of physical bond dimension
L = 8           # length of tensor train per dimension
K = 2           # number of dimension
npts = q ** L   # total number of grid points per dimension
TN_layout = LayoutType.SEQUENTIAL

x_vals = np.linspace(-Lx/2, Lx/2, npts, endpoint=False)
x1_vals = x_vals + (x_vals[1] - x_vals[0]) / 2
y_vals = np.linspace(-Ly/2, Ly/2, npts, endpoint=False)
y1_vals = y_vals + (y_vals[1] - y_vals[0]) / 2

x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
x1_mesh, y1_mesh = np.meshgrid(x1_vals, y1_vals)

r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
r1_mesh = np.sqrt(x1_mesh ** 2 + y1_mesh ** 2)

dx = Lx / npts
dy = Ly / npts
dr = np.sqrt(dx + dy)


fig1, ax1s = plt.subplots(1,3, sharey=True, figsize=(10,4))
ax1s[0].set_title('errs log')
ax1s[1].set_title('errs osc')
ax1s[2].set_title('errs log * osc')
ax1s[0].set_xlabel('log10(cutoff)'), ax1s[0].set_ylabel('rank')
ax1s[1].set_xlabel('log10(cutoff)')
ax1s[2].set_xlabel('log10(cutoff)')

lam = Lx/10

k = 2 * np.pi / lam
omega = c * k

#######################
# run-time parameters #
#######################

t = 0
func_sqrt = lambda r: 1. / np.sqrt((r + 1e-6 * dr) / permittivity) * 1e3 * 5
func_log = lambda r: -np.log((r + 1e-6 * dr) / permittivity) + np.log(Lx / permittivity)
func_osc = lambda r: np.cos(omega * t + k * r)
# func_osc = lambda r: np.exp(1.j * k * r)

# plt.figure()
# plt.imshow(func_sqrt(r_mesh))
#
# plt.figure()
# plt.imshow(func_log(r_mesh))
#
# plt.figure()
# plt.imshow(np.real(func_osc(r_mesh)))
#
# plt.figure()
# plt.plot(func_sqrt(r_mesh[npts//2, :]), label='sqrt')
# plt.plot(func_log(r_mesh[npts//2, :]), label='log')
# plt.plot(np.real(func_osc(r_mesh[npts//2, :])), label='osc')
# plt.legend()
#
# plt.show()

#############################
# start building GridTNs #
#############################

# define coordinates (to label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

ax_x = Axis(L, q, coordinate=X, dx=delta, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, dx=delta, xpts=y_vals)

# define a Cartesian coordinate system
coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X, Y, Z])

# define Grid object:  Grid1D means a TT geometry
grid_xy = Grid1D('XY', (ax_x, ax_y), layout_type=TN_layout)

## 2D:
log_gtn = grid_xy.map_state_to_mps(func_log(r_mesh))
sqrt_gtn = grid_xy.map_state_to_mps(func_sqrt(r_mesh))
osc_gtn = grid_xy.map_state_to_mps(func_osc(r_mesh))
np.random.seed(1)
f_gtn = grid_xy.map_state_to_mps(np.random.random(r_mesh.shape))
print('f gtn max bond', f_gtn.max_bond())



## interpolation compress
# plt.figure()
# plt.imshow(np.real(osc_gtn.get_data()))
# plt.colorbar()
# plt.show()

test_compression = False
if test_compression:

    ### measure number of summation terms depending on projected term max bond
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    errs, num_terms = [], []
    for mb in [8, 16, 32, 64]:
        gtnsum = GridTNSum(f_gtn.copy())
        gtnsum_c = gtnsum.compress(compress_opts={'max_bond': mb})
        print('gtnsum_c', len(gtnsum_c.corrections))
        norms = np.array([ct.frobenius_norm() for ct in gtnsum_c.corrections])
        num_nonzero = np.sum(norms > 1.0e-12)
        max_bonds = np.array([ct.max_bond() for ct in gtnsum_c.corrections])
        errs += [np.linalg.norm(gtnsum_c.get_data() - f_gtn.get_data())]
        num_terms += [num_nonzero]  # [len(gtnsum_c.corrections)]

        sort_inds = np.argsort(norms)[::-1]

        ax1.plot(norms[sort_inds], '--', lw=1.0, label=r'_R' + f'={mb}')
        ax1.plot(norms[sort_inds][:num_nonzero], label=r'R' + f'={mb}')
        ax2.plot(max_bonds[sort_inds], label=r'R' + f'={mb}')

    print('errs', errs)
    print('num terms', num_terms)
    print('f gtn max bond', f_gtn.max_bond())

    ax1.legend(loc='upper left', bbox_to_anchor=(1.05,0.99))
    ax1.set_ylabel('term norm')
    ax1.set_xlabel('number')

    ax2.legend(loc='upper left', bbox_to_anchor=(1.05,0.99))
    ax2.set_ylabel('term rank')
    ax2.set_xlabel('number')
    plt.show()

    exit()

    ### measure number of summation terms depending on cutoff
    errs, num_terms = [], []
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for co in [1.0e-6, 1.0e-10, 1.0e-15, 1.0e-20]:
        gtnsum = GridTNSum(f_gtn.copy())
        gtnsum_c = gtnsum.compress(compress_opts={'cutoff': co})
        print('gtnsum_c', len(gtnsum_c.corrections))
        norms = np.array([ct.frobenius_norm() for ct in gtnsum_c.corrections])
        ranks = np.array([ct.max_bond() for ct in gtnsum_c.corrections])
        sort_inds = np.argsort(norms)[::-1]

        ax1.plot(norms[sort_inds], label=r'$\epsilon$' + f'={co:1.1e}')
        ax2.plot(ranks[sort_inds], label=r'$\epsilon$' + f'={co:1.1e}')
        errs += [np.linalg.norm(gtnsum_c.get_data() - f_gtn.get_data())]
        num_terms += [len(gtnsum_c.corrections)]

    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 0.99))
    ax1.set_ylabel('term norm')
    ax1.set_xlabel('number')

    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.99))
    ax2.set_ylabel('term rank')
    ax2.set_xlabel('number')
    plt.show()

    print('errs', errs)
    print('num terms', num_terms)
    exit()

#################################

test_apply = True
if test_apply:
    errs, num_terms = [], []
    for mb in [64]:  # [8, 16, 32, 64]:
        co = 1e-30
        gtn_deriv_x = f_gtn.take_firstderivative(ax_x, inplace=False)

        gtnsum = GridTNSum(f_gtn.copy())
        gtnsum = gtnsum.compress(compress_opts={'max_bond': mb})
        print('gtnsum error', np.linalg.norm(gtnsum.get_data() - f_gtn.get_data()) / f_gtn.frobenius_norm() )
        # exit()


        print('\nTAKE FIRST DERIVATIVE')
        gtnsum_deriv_x = gtnsum.take_firstderivative(ax_x, inplace=False, compress=True,
                                                     compress_opts={'max_bond': mb, 'cutoff': co})

        diff = gtnsum_deriv_x.get_data() - gtn_deriv_x.get_data()
        err = np.linalg.norm(diff) / gtn_deriv_x.frobenius_norm()

        print('deriv err', err)
        print('gtnsum deriv x num terms', gtnsum_deriv_x.num_terms)
        norms = np.array([ct.frobenius_norm() for ct in gtnsum_deriv_x.terms])
        num_nonzero = np.sum(norms > 1.0e-12)
        print('num > 0', num_nonzero)

        ranks = np.array([ct.max_bond() for ct in gtnsum_deriv_x.terms])
        print('ranks', ranks)

        gtnsum_deriv_x.check_ovlp()
        # exit()

        print('\nv2')

        gtn_deriv_x.compress()
        gtn_deriv_x_sum = GridTNSum(gtn_deriv_x.copy())
        gtn_deriv_x_sum = gtn_deriv_x_sum.compress(compress_opts={'max_bond': mb, 'cutoff': co})
        norms_x = np.array([ct.frobenius_norm() for ct in gtn_deriv_x_sum.terms])

        diff = gtn_deriv_x_sum.get_data() - gtn_deriv_x.get_data()
        err_x = np.linalg.norm(diff) / gtn_deriv_x.frobenius_norm()

        num_terms += [(len(norms), num_nonzero, gtn_deriv_x_sum.num_terms - 1)]
        errs += [(err, err_x)]

        ranks_x = np.array([ct.max_bond() for ct in gtn_deriv_x_sum.terms])
        print('ranks x', ranks_x)

        gtn_deriv_x_sum.check_ovlp()



        plt.figure()
        plt.plot(norms, 'x-', label='sum + deriv')
        plt.plot(norms_x, '--', label=f'deriv + sum')
        plt.title(f'L={L}, ' + r'$\tilde{R}$' + f'={mb}' + r', $\tilde{\epsilon}$' + f'={co:1.0e}')
        plt.xlabel('number')
        plt.ylabel('term norm')
        plt.legend()
        plt.show()



    print('errs', errs)
    print('num terms', num_terms)
    exit()

    # errs, errs_x, num_terms = [], [], []
    # for co in [1e-1]:  # [8, 16, 32, 64]:
    #     gtnsum = GridTNSum(f_gtn.copy())
    #     gtnsum = gtnsum.compress(compress_opts={'cutoff': co})
    #     # gtnsum = gtnsum.compress(compress_opts={'max_bond': 32})
    #     gtnsum_deriv_x = gtnsum.take_firstderivative(ax_x, inplace=False, compress=True, compress_opts={'cutoff': co})
    #     gtn_deriv_x = f_gtn.take_firstderivative(ax_x, inplace=False)
    #     gtn_deriv_val = gtn_deriv_x.get_data()
    #
    #     gtn_deriv_x.compress()
    #     gtn_deriv_x_sum = GridTNSum(gtn_deriv_x.copy())
    #     gtn_deriv_x_sum = gtn_deriv_x_sum.compress(compress_opts={'cutoff': co})
    #
    #     diff = gtnsum_deriv_x.get_data() - gtn_deriv_val
    #     err = np.linalg.norm(diff) / gtn_deriv_x.frobenius_norm()
    #     errs += [err]
    #     print('deriv err', err)
    #     print('gtnsum deriv x num terms', gtnsum_deriv_x.num_terms)
    #     norms = np.array([ct.frobenius_norm() for ct in gtnsum_deriv_x.terms])
    #     norms_x = np.array([ct.frobenius_norm() for ct in gtn_deriv_x_sum.terms])
    #     num_nonzero = np.sum(norms > 1.0e-12)
    #     print('num > 0', num_nonzero)
    #     num_terms += [(len(norms), num_nonzero, gtn_deriv_x_sum.num_terms - 1)]
    #
    #     ranks = np.array([ct.max_bond() for ct in gtnsum_deriv_x.terms])
    #
    #     diff_x = gtn_deriv_x_sum.get_data() - gtn_deriv_val
    #     err_x = np.linalg.norm(diff) / gtn_deriv_x.frobenius_norm()
    #     errs_x += [err_x]
    #     ranks_x = np.array([ct.max_bond() for ct in gtn_deriv_x_sum.terms])
    #     print('ranks', ranks)
    #
    #     plt.figure()
    #     plt.plot(norms, 'x-', label='sumTT + deriv')
    #     plt.plot(norms_x, '--', label=f'deriv + sumTT')
    #     plt.title(f'L={L}, ' + r'$\tilde{\epsilon}$' + f'={co}')
    #     plt.xlabel('number')
    #     plt.ylabel('term norm')
    #     plt.legend()
    #
    #     plt.figure()
    #     plt.plot(ranks, 'x-', label='sumTT + deriv')
    #     plt.plot(ranks_x, '--', label=f'deriv + sumTT')
    #     plt.title(f'L={L}, ' + r'$\tilde{\epsilon}$' + f'={co}')
    #     plt.xlabel('number')
    #     plt.ylabel('term rank')
    #     plt.legend()
    #
    #     plt.show()
    #
    # print('errs', errs)
    # print('errs_x', errs_x)
    # print('num terms', num_terms)
    #
    # exit()


test_add = False
if test_add:
    errs, num_terms = [], []
    for mb in [64]:
        gtnsum = GridTNSum(f_gtn.copy())
        gtnsum = gtnsum.compress(compress_opts={'max_bond': mb})
        gtnsum_2 = gtnsum.copy()
        # gtnsum_2 = gtnsum_2.compress(compress_opts={'cutoff': 1.0e-10})
        gtnsum_2 = gtnsum_2.scalar_multiply(-1, inplace=False)

        out_gtnsum = gtnsum.add(gtnsum_2, compress_opts={'max_bond': mb})
        err = out_gtnsum.norm(is_sqrt=True)
        errs += [err]

        print('out gtn', out_gtnsum.num_terms)
        norms = np.array([ct.frobenius_norm() for ct in out_gtnsum.terms])
        num_nonzero = np.sum(norms > 1.0e-12)
        print('num > 0', num_nonzero)
        num_terms += [(len(norms), num_nonzero)]
        print('out norm', np.linalg.norm(out_gtnsum.get_data()), out_gtnsum.norm(is_sqrt=True))

        plt.figure()
        plt.plot(norms, 'x-', label='sum -> deriv')
        plt.title(f'L={L}, ' + r'$\tilde{R}$' + f'={mb}')
        plt.xlabel('number')
        plt.ylabel('term norm')
        plt.show()

    print('errs', errs)
    print('num terms', num_terms)

    exit()

#################################



gtnsum = GridTNSum(osc_gtn.copy())
gtnsum = gtnsum.compress(compress_opts={'max_bond': 8, 'cutoff': 1.0e-10})

print('gtnsum', gtnsum.primary.frobenius_norm(), gtnsum.primary.max_bond())
for x in gtnsum.corrections:
    print('x', x.frobenius_norm(), x.max_bond())

terms = gtnsum.terms
for i, j in np.ndindex(len(terms), len(terms)):
    if j >= i:
        print('ovlp', i,j, terms[i].ovlp(terms[j]))


# diff = gtnsum.primary.distance(osc_gtn)
diff = np.linalg.norm(gtnsum.primary.get_data() - osc_gtn.get_data())
print('diff', diff)

# diff2 = gtnsum.get_value().distance(osc_gtn)
diff2 = np.linalg.norm(gtnsum.get_value().get_data() - osc_gtn.get_data())
print('diff 2', diff2)

osc_gtn_c = osc_gtn.compress(inplace=False, compress_opts={'cutoff': 1.0e-10})
print('svd compress error', np.linalg.norm(osc_gtn_c.get_data() - osc_gtn.get_data()))

plt.figure()
plt.imshow(gtnsum.get_value().get_data())
plt.colorbar()
plt.title('gtnsum total')

plt.figure()
plt.imshow(gtnsum.get_value().get_data() - osc_gtn.get_data())
plt.colorbar()
plt.title('gtnsum total error')
plt.show()


#### take derivative ####
gtnsum_deriv_x = gtnsum.take_firstderivative(ax_x, inplace=False, compress=True)
gtn_deriv_x = osc_gtn.take_firstderivative(ax_x, inplace=False)
gtnc_deriv_x = osc_gtn_c.take_firstderivative(ax_x, inplace=False, compress=True)

print('diff deriv', np.linalg.norm(gtnsum_deriv_x.get_value().get_data() - gtn_deriv_x.get_data()))
print('diff deriv', np.linalg.norm(gtnc_deriv_x.get_data() - gtn_deriv_x.get_data()))


plt.figure()
plt.imshow(gtn_deriv_x.get_data())
plt.colorbar()
plt.title('osc gtn deriv')

plt.figure()
plt.imshow(gtnc_deriv_x.get_data())
plt.colorbar()
plt.title('osc gtn R=16 deriv')

# plt.figure()
# plt.imshow(gtnsum.primary.take_firstderivative(ax_x).get_data())
# plt.colorbar()
# plt.title('primary deriv')

plt.figure()
plt.imshow(gtnsum_deriv_x.get_value().get_data())
plt.colorbar()
plt.title('osc gtnsum deriv')
plt.show()