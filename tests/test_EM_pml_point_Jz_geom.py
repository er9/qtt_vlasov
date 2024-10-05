import os
import sys

sys.path.append('../')

from setup_.configs import *
# print(sys.argv)
# import pdb; pdb.set_trace()

from axis import Axis
from axis_map import BinaryMap, FlipBinaryMap
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D

"""
2D delta-point sinusoidal current source
Use thin-wire approximation
Numerical Electromagnetics: the FDTD Method (Iman, Marshall)
"""

plotting = True
save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  # to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/EM-fdtd-TM/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/delta_function_no_PML/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)

"""
Evaluate compressibility of
log(r/eps0) * cos(kr) where k = w/c
1/sqrt(r/eps0) exp(i k r - i omega t) 
"""

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

wavelengths = [Lx / 5, Lx / 10, Lx / 20, Lx/40, Lx/80]
rank_log_lam, rank_osc_lam, rank_log_osc_lam = [], [], []
for lam in wavelengths:

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

    map_x, map_y = BinaryMap(), FlipBinaryMap()
    ax_x = Axis(L, q, coordinate=X, dx=delta, xpts=x_vals, ax_map=map_x)
    ax_y = Axis(L, q, coordinate=Y, dx=delta, xpts=y_vals, ax_map=map_y)

    # define a Cartesian coordinate system
    coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X, Y, Z])

    # define Grid object:  Grid1D means a TT geometry
    grid_xy = Grid1D('XY', (ax_x, ax_y), layout_type=TN_layout)

    ## 1D:
    exp_1D = ax_x.map_state_to_mps(np.exp(1.j * k * x_vals))
    print('exp_1D', exp_1D.max_bond())
    helper_quimb.compress(exp_1D, compress_opts={'cutoff': 1e-6})
    print('compressed 1e-6 exp_1D', exp_1D.max_bond())

    exp_2D = grid_xy.make_mps_ndim({ax_x: exp_1D})


    ## 2D:
    log_gtn = grid_xy.map_state_to_mps(func_log(r_mesh))
    sqrt_gtn = grid_xy.map_state_to_mps(func_sqrt(r_mesh))
    osc_gtn = grid_xy.map_state_to_mps(func_osc(r_mesh))
    log_osc_gtn = grid_xy.map_state_to_mps(func_osc(r_mesh) * func_log(r_mesh))

    cutoffs = [1e-4, 1e-6, 1e-8, 1e-10]
    errs_log, errs_sqrt, errs_osc, errs_log_osc = [], [], [], []
    rank_log, rank_sqrt, rank_osc, rank_log_osc = [], [], [], []
    for cutoff in cutoffs:
        tmp = log_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_log += [ log_gtn.distance(tmp) / log_gtn.frobenius_norm() ]
        rank_log += [ tmp.max_bond() ]

        tmp = sqrt_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_sqrt += [sqrt_gtn.distance(tmp) / sqrt_gtn.frobenius_norm() ]
        rank_sqrt += [tmp.max_bond()]

        tmp = osc_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_osc += [osc_gtn.distance(tmp) / osc_gtn.frobenius_norm() ]
        rank_osc += [tmp.max_bond()]

        tmp = log_osc_gtn.copy()
        tmp.compress(compress_opts={'cutoff': cutoff})
        errs_log_osc += [log_osc_gtn.distance(tmp) / log_osc_gtn.frobenius_norm() ]
        rank_log_osc += [tmp.max_bond()]


    log_cutoffs = [np.log10(cutoff) for cutoff in cutoffs]

    # plt.figure()
    # plt.semilogy(log_cutoffs, errs_log, 'x-', label='log')
    # plt.semilogy(log_cutoffs, errs_sqrt, 'x-', label='sqrt')
    # plt.semilogy(log_cutoffs, errs_osc, 'x-', label='osc')
    # plt.semilogy(log_cutoffs, errs_log_osc, 'x-', label='log osc')
    # plt.legend()
    # plt.title('compression error')
    #
    # plt.figure()
    # plt.plot(log_cutoffs, rank_log, 'x-', label='log')
    # plt.plot(log_cutoffs, rank_sqrt, 'x-', label='sqrt')
    # plt.plot(log_cutoffs, rank_osc, 'x-', label='osc')
    # plt.plot(log_cutoffs, rank_log_osc, 'x-', label='log osc')
    # plt.legend()
    # plt.title(f'QTT rank {TN_layout}')

    ax1s[0].plot(log_cutoffs, rank_log, 'x-', label=f'wavelength={lam:1.3e}')
    ax1s[1].plot(log_cutoffs, rank_osc, 'x-', label=f'wavelength={lam:1.3e}')
    ax1s[2].plot(log_cutoffs, rank_log_osc, 'x-', label=f'wavelength={lam:1.3e}')

ax1s[2].legend(loc='upper left', bbox_to_anchor=(1.05,0.95))
# plt.title(f'QTT rank {TN_layout}')

plt.show()

