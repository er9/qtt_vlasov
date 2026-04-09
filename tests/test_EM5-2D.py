import numpy as np
import os, sys, time, glob
sys.path.append('../')
import matplotlib

from setup_.paths import save_dir, main_dir
from setup_.configs import *
import setup_test as test_setup

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
import setup_.helper as helper_test
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_EM import Maxwell


"""
dipole radiation from current source (2D)
"""

save_figs = (matplotlib.get_backend() in ['agg'])
save_data = flags.get('save_data', True)
## to load the output file and maybe restart from it (vs a saved restart file)
load_data = False  # bool(flags.get('load_data', True))
restart_from_T = flags.get('load_data', None)
restart_from_T = restart_from_T if isinstance(restart_from_T, (float, int)) else None
## to restart from a specfic restart file
restart   = bool(flags.get('restart', True))
restart_from_nt = flags.get('restart', None)
restart_from_nt = restart_from_nt if isinstance(restart_from_nt, int) else None

do_tt = flags.get('do_tt', False)    ## see test_vlasovEM_whistler-fd-v2-full
do_full = flags.get('do_full', False)
dissipation = flags.get('dissipation', 0.0)
deriv_order = flags.get('deriv_order', 2)
comp_style = flags.get('comp_style', 1)
run = flags.get('branch', 1)

clean = False  # True

if restart or save_data or save_figs:
    # if run == 1:       # Lx = 40 * lam
    #     dnum = 250423
    #     # dnum = 250517
    # elif run == 2:     # Lx = 10 * lam
    #     # dnum = 250505
    #     # dnum = 250518  # 250505
    dnum =  f'260331_run{run}'
    if dissipation == 0.:
        fnum = f'{dnum}/eta{dissipation}/'
    else:
        fnum = f'{dnum}/eta{dissipation}_c{comp_style}_m{deriv_order}/'

    if do_full:
        sdir = f'{save_dir}/data{fnum}/EM_jz2D_full/'
        fdir = f'{main_dir}/data{fnum}/EM_jz2D_full/'
    else:
        sdir = f'{save_dir}/data{fnum}/EM_jz2D/'
        fdir = f'{main_dir}/data{fnum}/EM_jz2D/'

    if not os.path.exists(sdir):
        os.makedirs(sdir, exist_ok=True)

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)
    print('sdir', sdir)
else:
    fdir = ''
    sdir = ''

plotting = True  # False

#######################
# problem parameters  #
#######################

permittivity = 1  # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor
matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)

## dipole properties
c = 1/np.sqrt(permittivity*permeability)
dp = 0.2                # dipole length
I0 = 1.0                # current strength
omega = 2 * np.pi * 2
lam = 2 * np.pi / (omega / c)

## box size
if run == 1:
    Lx = 40.0 * lam
elif run == 2:
    Lx = 10.0 * lam
Ly = Lx
Lz = Lx


#######################
# run-time parameters #
#######################

q = 2
L = flags.get('Lx', 13)
K = 2
npts = q**L
TN_layout = flags.get('layout', LayoutType.SEQUENTIAL)
x_map_key = flags.get('x_map_key', 'F')
y_map_key = flags.get('v_map_key', 'F')
if do_full or do_tt:
    TN_layout = LayoutType.PARALLEL_GROUP if do_full else LayoutType.PARALLEL
    x_map_key = y_map_key = 'F'
layout_str = f'{TN_layout.value}{x_map_key}{y_map_key}'
print('layout str', layout_str)

upwind = flags.get('do_upwind', False)

print('c', matl_config.c)       # should be 1
T = c * Lx / lam / 2
print('T', T)
dt = flags.get('dt', None)  # None  # 0.05  # 0.04  # None
print('dt', dt)
use_cfl = (dt is None)
if use_cfl:
    cfl = flags.get('cfl', 0.9)
order = 1
te_order = flags.get('te_order_EM', 1 if upwind else 31)

## restart from an saved state
load_nt = restart_from_nt
if use_cfl:
    subdir = (f'{layout_str}/L{L}_te{te_order}_up/' if upwind else f'{layout_str}/L{L}_te{te_order}/')
else:
    subdir = (f'{layout_str}/L{L}_te{te_order}_dt{dt}_up/' if upwind else f'{layout_str}/L{L}_te{te_order}_dt{dt}/')
fstr = ''
save_every_nt = flags.get('save_every_nt', 100)

#############################
## start building PDE system
#############################

##### compression options ####
## even with modest CUTOFF = 1.0e-8, div(B) is not preserved
DMAX = flags.get('DMAX', None if do_tt else None)
cutoff = flags.get('cutoff', CUTOFF)
compress_style = 3 if te_order == 4 else 1
compress_style_mod = 0

subdir = subdir + f'D{DMAX}_c{cutoff}/'

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)

te_compress_psi = te_compress_phi
te_compress_B = te_compress_E

comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

compress_opts_E = te_compress_E.get_compress_opts(1)
compress_opts_B = te_compress_B.get_compress_opts(1)


## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(-Lx,Lx,npts,endpoint=True)
x1_vals = x_vals + (x_vals[1]-x_vals[0])/2
y_vals = np.linspace(-Ly,Ly,npts,endpoint=True)
y1_vals = y_vals + (y_vals[1]-y_vals[0])/2

x_map = helper_test.get_maps(x_map_key)[0]
y_map = helper_test.get_maps(y_map_key)[0]
if do_tt or do_full:
    ax_x = Axis(1, q**L, coordinate=X, xpts=x_vals, ax_map=x_map)
    ax_y = Axis(1, q**L, coordinate=Y, xpts=y_vals, ax_map=y_map)
else:
    ax_x = Axis(L, q, coordinate=X, xpts=x_vals, ax_map=x_map)
    ax_y = Axis(L, q, coordinate=Y, xpts=y_vals, ax_map=y_map)
print('x map', x_map, y_map)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## define time step
dx, dy, = ax_x.dx, ax_y.dx,
if use_cfl:
    cfl_params = ((dx, dy), (c, c))
    dt_cfl = cfl_limit( cfl_params )
    dt = dt_cfl * cfl if dt is None else dt
    print('dt', dt, cfl * 1. / (c / ax_x.dx + c / ax_y.dx) )
num_tsteps = int(round(T/dt))
print('dt', dt, num_tsteps, )
print('1/omega', 1 / omega)
print('dx', dx, 'dy', dy)

## define QTT Grid object
grid_xyz = Grid1D('XYZ', (ax_x, ax_y), layout_type=TN_layout)

## define "delta function" in space for the current source
print('ax_x', ax_x.npts, ax_x.map, ax_y.map)
delta_fct_gtn = grid_xyz.make_mps_ndim(
    {ax_x: ax_x.get_select_elems_mps(inds=[ax_x.npts//2 + i - int(2**(L-7)) for i in range(2**(L-8))]),
     ax_y: ax_y.get_select_elems_mps(inds=[ax_y.npts//2 + i - int(2**(L-7)) for i in range(2**(L-8))]),
     })


if dissipation == 'max':
    dissipation = np.sqrt(cutoff) * np.sqrt(L * 2)
    print('(max) dissipation per time step', dissipation)
elif dissipation == 'fv':
    dissipation = ax_x.dx * dt
    print('(fv) dissipation per time step', dissipation)
else:
    dissipation = dissipation * dt

## current source (uniform current "point" source that varies with respect to time)
def Jz_gtn(t):
    jz = delta_fct_gtn.scalar_multiply(np.sin(omega * t) * I0, inplace=False)
    print('jz', t, np.sin(omega * t) * I0)
    return jz


data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
data_x1, data_y1 = np.meshgrid(x1_vals,y1_vals,indexing='ij')
# init_Ez = Ez(data_x, data_y)  # i, j, k+1/2
# init_Bx = Bx(data_x, data_y)  # i, j+1/2, k+1/2  # init at time dt/2
# init_By = By(data_x, data_y)  # i+1/2, j, k+1/2  # init at time dt/2


## configure derivatives
# deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
# deriv_zv = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
# deriv_zg1 = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER) # , offset=1)
# deriv_zv1 = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER) #, offset=1)

deriv_zg = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_zg1 = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_zv1 = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)

init_Ex_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
init_Ey_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
init_Ez_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg})
init_Bx_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
init_By_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
init_Bz_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg1})

init_E_field = Field('E', grid_xyz, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E, is_sqrt=True)
init_B_field = Field('B', grid_xyz, data={X: init_Bx_gtn, Y:init_By_gtn, Z: init_Bz_gtn},
                     compress_config=te_compress_B, is_sqrt=True)


# #### load data if restarting ####
# out = test_setup.load_data_Maxwell(fstr, init_E_field, init_B_field, None, None,
#                                    fdir=fdir + subdir, sdir=sdir + subdir, save_every_nt=save_every_nt * 10,
#                                    load_data=load_data, restart=restart, restart_from_T=restart_from_T)
#
# (init_E_field, init_B_field, _, _, ts, nrg_E_ts, nrg_B_ts, max_bond_Es,
#  max_bond_Bs, _, _,) = out
#  max_bond_Bs, _, _,) = out
#
# nt = len(ts) - 1
# restarted = True
#
# init_E_field.compress(compress_opts={'cutoff': cutoff})
# init_B_field.compress(compress_opts={'cutoff': cutoff})
#
# exit()

try:
    if not restart:
        raise IOError

    #### load data if restarting ####
    out = test_setup.load_data_Maxwell(fstr, init_E_field, init_B_field, None, None,
                                       fdir=fdir + subdir, sdir=sdir + subdir, save_every_nt=save_every_nt * 10,
                                       load_data=load_data, restart=restart, restart_from_T=restart_from_T)

    (init_E_field, init_B_field, _, _, ts, nrg_E_ts, nrg_B_ts, max_bond_Es,
     max_bond_Bs, _, _,) = out

    nt = len(ts) - 1
    restarted = True

    init_E_field.compress(compress_opts={'cutoff': cutoff})
    init_B_field.compress(compress_opts={'cutoff': cutoff})


except(IOError, OSError, NameError, ValueError, FileNotFoundError):

    ts = [0]
    nt = 0
    max_bond_Es = [[0, 0, 0]]
    max_bond_Bs = [[0, 0, 0]]
    nrg_E_ts = [0]
    nrg_B_ts = [0]



### BUILD MAXWELL EQNS SOLVER
em_sys = Maxwell(init_E_field, init_B_field, phi=None, psi=None,
                 coords_x=coord_sys, matl_params=matl_config,
                 is_electrostatic=False, clean=clean, is_yee=(te_order == 31), normalize=False,
                 upwind=upwind,
                 te_order=te_order, compress_levels=comp_levels )


while nt < num_tsteps:

    dt_ = dt if nt==0 else dt
    print('dt_', dt, 't', ts[-1])

    ## set the current density in the EM solver
    em_sys.current_density = Field('J', grid_xyz, data={Z:Jz_gtn(ts[-1] + dt_ / 2)})

    em_sys = em_sys.next_time_step(dt_, compress_level=1, is_first_time_step=(nt==0),
                                   apply_constraints=False)

    pre_dissip_E_bonds = [em_sys.E.max_bond(C) for C in [X, Y, Z]]
    pre_dissip_B_bonds = [em_sys.B.max_bond(C) for C in [X, Y, Z]]
    print('pre dissip max bond', pre_dissip_E_bonds, pre_dissip_B_bonds)

    # print('E norms', em_sys.E.norms([X,Y,Z]), np.sum(np.array(em_sys.E.norms([X,Y,Z]))**2))
    # print('B norms', em_sys.B.norms([X,Y,Z]), np.sum(np.array(em_sys.B.norms([X,Y,Z]))**2))
    if dissipation > 0.0:
        print('adding dissip', dissipation)
        for compID in em_sys.E.componentIDs:
            deriv_axes = [ax for ax in grid_xyz.axes if ax.coordinate != compID]
            em_sys.E[compID].add_dissipation(dissipation, deriv_axes=deriv_axes, deriv_order=deriv_order, inplace=True,
                                             compress_opts=None if comp_style==2 else compress_opts_E)
        for compID in em_sys.B.componentIDs:
            deriv_axes = [ax for ax in grid_xyz.axes if ax.coordinate != compID]
            em_sys.B[compID].add_dissipation(dissipation, deriv_axes=deriv_axes, deriv_order=deriv_order, inplace=True,
                                             compress_opts=None if comp_style==2 else compress_opts_B)

    print('divB', em_sys.check_divB().norm())
    print('divE', em_sys.check_poisson().norm())

    E_bonds = [em_sys.E.max_bond(C) for C in [X, Y, Z]]
    B_bonds = [em_sys.B[C].max_bond() for C in [X, Y, Z]]
    print('max bonds', E_bonds, B_bonds)

    ## obtain max QTT rank for each component of the E field
    if comp_style == 2:
        max_bond_Es += [pre_dissip_E_bonds]
        max_bond_Bs += [pre_dissip_B_bonds]
    else:
        max_bond_Es += [E_bonds]
        max_bond_Bs += [B_bonds]

    nrg_E_ts += [em_sys.E.norm()**2 / 2 * permittivity]
    nrg_B_ts += [em_sys.B.norm()**2 / 2 / permeability]

    ts += [ts[-1] + dt_]

    nt += 1
    print('result norm', nt, dt, ts[-1], em_sys.E.max_bonds(), em_sys.B.max_bonds(), DMAX, q ** L)

    if plotting and (nt == num_tsteps - 1 or nt%save_every_nt==0):

        ## Ez, Bx fields
        Ex = em_sys.E[X].get_data()
        Ey = em_sys.E[Y].get_data()
        Ez = em_sys.E[Z].get_data()
        Bx = em_sys.B[X].get_data()
        By = em_sys.B[Y].get_data()
        Bz = em_sys.B[Z].get_data()
        plt.figure()
        if Ez is not None:
            plt.figure()
            plt.imshow(Ez) #, vmin=1e-3, vmax=1e-3)
            plt.title('Ez')
            plt.colorbar()
        if Bx is not None:
            plt.figure()
            plt.imshow(Bx) #, vmin=-1e-3, vmax=1e-3)
            plt.title('Bx')
            plt.colorbar()
        plt.show()

        plt.figure()
        for i in range(3):
            plt.plot(np.array(max_bond_Bs)[:,i], label=f'B{coord_sys.coords[i]}')
        for i in range(3):
            plt.plot(np.array(max_bond_Es)[:,i], label=f'E{coord_sys.coords[i]}')
        plt.legend()
        plt.show()

        plt.close()
        plt.close()
        plt.close()

    if save_data and nt%save_every_nt==0:

        test_setup.save_data_Maxwell(fstr, em_sys, sdir + subdir + 'restart/', add_fstr=f'-nt{nt}',
                                     ts=ts, nrg_E_ts=nrg_E_ts, nrg_B_ts=nrg_B_ts,
                                     max_bond_E=max_bond_Es, max_bond_B=max_bond_Bs)


if save_data:

    test_setup.save_data_Maxwell(fstr, em_sys, fdir + subdir, # add_fstr=f'-nt{nt}',
                                 ts=ts, nrg_E_ts=nrg_E_ts, nrg_B_ts=nrg_B_ts,
                                 max_bond_E=max_bond_Es, max_bond_B=max_bond_Bs)

    for compID in em_sys.E.componentIDs:
        E_data = em_sys.E[compID].get_data(ax_select={ax_y: ax_y.npts//2})
        if E_data is not None:
            np.save(fdir + subdir + f'E{compID}_y0.npy', E_data)

    for compID in em_sys.B.componentIDs:
        B_data = em_sys.B[compID].get_data(ax_select={ax_y: ax_y.npts//2})
        if B_data is not None:
            np.save(fdir + subdir +  f'B{compID}_y0.npy', B_data)


    print('saved data')
