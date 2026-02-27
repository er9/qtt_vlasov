import numpy as np
import os, sys, time, glob
sys.path.append('../')

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
from pde_Aphi import VectorPotential


"""
dipole radiation from current source (2D)
"""

save_figs = False
save_data = flags.get('save_data', True)
## to load the output file and maybe restart from it (vs a saved restart file)
load_data = bool(flags.get('load_data', True))
restart_from_T = flags.get('load_data', None)
restart_from_T = restart_from_T if isinstance(restart_from_T, (float, int)) else None
## to restart from a specfic restart file
restart = bool(flags.get('restart', True))
restart_from_nt = flags.get('restart', None)
restart_from_nt = restart_from_nt if isinstance(restart_from_nt, int) else None

do_tt = flags.get('do_tt', False)    ## see test_vlasovEM_whistler-fd-v2-full
do_full = flags.get('do_full', False)
dissipation = flags.get('dissipation', 0.0)
deriv_order = flags.get('deriv_order', 2)
comp_style = flags.get('comp_style', 1)
run = flags.get('branch', 2)

clean = False  # True

if restart or save_data or save_figs:
    dnum = f'260225_run{run}'
    if dissipation == 0.:
        fnum = f'{dnum}/eta{dissipation}/'
    else:
        fnum = f'{dnum}/eta{dissipation}_c{comp_style}_m{deriv_order}/'

    if do_full:
        sdir = f'{save_dir}/data{fnum}/APhi_jz2D_full/'
        fdir = f'{main_dir}/data{fnum}/APhi_jz2D_full/'
    else:
        sdir = f'{save_dir}/data{fnum}/APhi_jz2D/'
        fdir = f'{main_dir}/data{fnum}/APhi_jz2D/'

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
L = flags.get('Lx', 10)
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
dt = None  # 0.05  # 0.04  # None
cfl = flags.get('cfl', 0.9)
order = 1
te_order = flags.get('te_order_EM', 31)

## restart from an saved state
load_nt = restart_from_nt
subdir = (f'{layout_str}/L{L}_te{te_order}_up/' if upwind else f'{layout_str}/L{L}_te{te_order}/')
fstr = ''
save_every_nt = flags.get('save_every_nt', 100)

#############################
## start building PDE system
#############################

##### compression options ####
## even with modest CUTOFF = 1.0e-8, div(B) is not preserved
DMAX = flags.get('DMAX', None if do_tt else None)
cutoff = flags.get('cutoff', CUTOFF)
compress_style = 1
compress_style_mod = 0

subdir = subdir + f'D{DMAX}_c{cutoff}/'
os.makedirs(fdir + subdir, exist_ok=True)
os.makedirs(sdir + subdir, exist_ok=True)

te_compress_A = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_A.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)
    te_compress_phi.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)

te_compress_dAdt = te_compress_A

comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

compress_opts = te_compress_A.get_compress_opts(1)


## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(-Lx,Lx,npts,endpoint=True)
y_vals = np.linspace(-Ly,Ly,npts,endpoint=True)

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
cfl_params = ((dx, dy), (c, c))
dt_cfl = cfl_limit( cfl_params )
dt = dt_cfl * cfl if dt is None else dt
num_tsteps = int(round(T/dt))
print('dt', dt, cfl*1./(c/ax_x.dx + c/ax_y.dx), num_tsteps, )
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


## current source (uniform current "point" source that varies with respect to time)
def Jz_gtn(t):
    jz = delta_fct_gtn.scalar_multiply(np.sin(omega * t) * I0, inplace=False)
    print('jz', t, np.sin(omega * t) * I0)
    return jz


## configure derivatives
deriv_p = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)

init_Ax_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_Ay_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_Az_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_dAx_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_dAy_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_dAz_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})
init_phi_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_p, ax_y: deriv_p})

init_A_field = Field('A', grid_xyz, data={X: init_Ax_gtn, Y: init_Ay_gtn, Z: init_Az_gtn},
                     compress_config=te_compress_A, is_sqrt=True)
init_dA_field = Field('dAdt', grid_xyz, data={X: init_dAx_gtn, Y:init_dAy_gtn, Z: init_dAz_gtn},
                     compress_config=te_compress_dAdt, is_sqrt=True)
init_phi_field = ScalarField('phi', grid_xyz, data=init_phi_gtn, compress_config=te_compress_phi, is_sqrt=True)


try:
    ## restart problematic with leapfrog bc dAdt is not saved and is not set...
    if not restart:
        raise IOError

    #### load data if restarting ####
    out = test_setup.load_data_Aphi(fstr, init_A_field, init_dA_field, init_phi_field,
                                       fdir=fdir + subdir, sdir=sdir + subdir, save_every_nt=save_every_nt,
                                       load_data=load_data, restart=restart, restart_from_T=restart_from_T)

    (init_A_field, init_dA_field, init_phi_field, ts, nrg_E_ts, nrg_B_ts, max_bond_As,
     max_bond_dAdts, max_bond_phis,) = out

    nt = len(ts) - 1
    restarted = True

    init_A_field.compress(compress_opts={'cutoff': cutoff})
    init_dA_field.compress(compress_opts={'cutoff': cutoff})
    init_phi_field.compress(compress_opts={'cutoff': cutoff})


except(IOError, OSError, NameError, ValueError, FileNotFoundError):

    ts = [0]
    nt = 0
    max_bond_As = [[0, 0, 0]]
    max_bond_dAdts = [[0, 0, 0]]
    max_bond_phis = [0]
    nrg_E_ts = [0]
    nrg_B_ts = [0]



### BUILD MAXWELL EQNS SOLVER
ap_sys = VectorPotential(init_A_field, init_dA_field, init_phi_field,
                         coords_x=coord_sys, matl_params=matl_config,
                         te_order=te_order, compress_levels=comp_levels)


while nt < num_tsteps:

    dt_ = dt if nt==0 else dt
    print('dt_', dt, 't', ts[-1])

    ## set the current density in the EM solver
    ap_sys.current_density = Field('J', grid_xyz, data={Z:Jz_gtn(ts[-1] + dt_ / 2)})
    print('Jz gtn max', np.max(np.abs(Jz_gtn(ts[-1] + dt_/2).get_data())) )

    ap_sys = ap_sys.next_time_step(dt_, compress_level=1, is_first_time_step=(nt == 0),
                                   apply_constraints=False)

    pre_dissip_A_bonds = [ap_sys.A.max_bond(C) for C in [X, Y, Z]]
    pre_dissip_phi_bonds = ap_sys.phi.max_bond()
    print('pre dissip A max bond', pre_dissip_A_bonds)
    print('pre dissip phi max bond', pre_dissip_phi_bonds)
    if ap_sys.dAdt is not None:
        pre_dissip_dAdt_bonds = [ap_sys.dAdt.max_bond(C) for C in [X, Y, Z]]
        print('pre dissip dAdt max bond', pre_dissip_dAdt_bonds)

    if dissipation > 0.0:
        raise NotImplementedError
        print('adding dissip', dissipation)
        for compID in ap_sys.A.componentIDs:
            deriv_axes = [ax for ax in grid_xyz.axes if ax.coordinate != compID]
            ap_sys.A[compID].add_dissipation(dissipation, deriv_axes=deriv_axes, deriv_order=deriv_order, inplace=True,
                                             compress_opts=None if comp_style==2 else compress_opts_E)
        # deriv_axes = grid_xyz.axes  #  [ax for ax in grid_xyz.axes if ax.coordinate != compID]
        # ap_sys.phi.component.add_dissipation(dissipation, deriv_axes=deriv_axes, deriv_order=deriv_order, inplace=True,
        #                                      compress_opts=None if comp_style==2 else compress_opts_B)

    ap_sys.A.compress()
    ap_sys.phi.compress()
    if ap_sys.dAdt is not None:
        ap_sys.dAdt.compress()

    A_bonds = [ap_sys.A.max_bond(C) for C in [X, Y, Z]]
    phi_bonds = ap_sys.phi.max_bond()
    print('max A bonds', A_bonds)
    print('max phi bonds', phi_bonds)
    if ap_sys.dAdt is not None:
        dAdt_bonds = [ap_sys.dAdt[C].max_bond() for C in [X, Y, Z]]
        print('max dAdt bonds', dAdt_bonds)

    ## obtain max QTT rank for each component of the E field
    if comp_style:
        max_bond_As += [pre_dissip_A_bonds]
        if ap_sys.dAdt is not None:
            max_bond_dAdts += [pre_dissip_dAdt_bonds]
        max_bond_phis += [pre_dissip_phi_bonds]
    else:
        max_bond_As += [A_bonds]
        if ap_sys.dAdt is not None:
            max_bond_dAdts += [dAdt_bonds]
        max_bond_phis += [phi_bonds]

    print('ap sys dt', ap_sys.dt, dt)
    # ap_sys.compute_E(compress_level=1)
    # ap_sys.compute_B(compress_level=1)
    # nrg_E_ts += [ap_sys.electric_energy_density(compress_level=1)]
    # nrg_B_ts += [ap_sys.magnetic_energy_density(compress_level=1)]

    ts += [ts[-1] + dt_]

    nt += 1
    print('result norm', nt, ts[-1], ap_sys.A.max_bonds(), ap_sys.dAdt.max_bonds(), ap_sys.phi.max_bonds(),
          DMAX, q ** L)

    if plotting and (nt == num_tsteps - 1 or nt%save_every_nt==0):
        ap_sys.compute_E(compress_level=1)
        ap_sys.compute_B(compress_level=1)

        ## Ez, Bx fields
        Ex = ap_sys.E[X].get_data()
        Ey = ap_sys.E[Y].get_data()
        Ez = ap_sys.E[Z].get_data()
        Bx = ap_sys.B[X].get_data()
        By = ap_sys.B[Y].get_data()
        Bz = ap_sys.B[Z].get_data()
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
            plt.plot(np.array(max_bond_As)[:,i], label=f'A{coord_sys.coords[i]}')
        for i in range(3):
            plt.plot(np.array(max_bond_dAdts)[:,i], label=f'dAdt {coord_sys.coords[i]}')
        plt.plot(np.array(max_bond_phis), label='phi')
        plt.legend()
        plt.show()

        plt.close()
        plt.close()
        plt.close()

    if save_data and nt%save_every_nt==0:
        ap_sys.compute_E(compress_level=1)
        ap_sys.compute_B(compress_level=1)
        ap_sys.E.save_data(sdir + subdir + f'E-nt{nt}')
        ap_sys.B.save_data(sdir + subdir + f'B-nt{nt}')

        for compID in ap_sys.E.componentIDs:
            E_data = ap_sys.E[compID].get_data(ax_select={ax_y: ax_y.npts // 2})
            if E_data is not None:
                np.save(fdir + subdir + f'E{compID}_y0.npy', E_data)

        for compID in ap_sys.B.componentIDs:
            B_data = ap_sys.B[compID].get_data(ax_select={ax_y: ax_y.npts // 2})
            if B_data is not None:
                np.save(fdir + subdir + f'B{compID}_y0.npy', B_data)

        test_setup.save_data_Aphi(fstr, ap_sys, sdir + subdir + 'restart/', add_fstr=f'-nt{nt}',
                                  ts=ts, nrg_E_ts=nrg_E_ts, nrg_B_ts=nrg_B_ts,
                                  max_bond_A=max_bond_As, max_bond_dAdt=max_bond_dAdts, max_bond_phi=max_bond_phis)


if save_data:

    test_setup.save_data_Aphi(fstr, ap_sys, sdir + subdir + 'restart/', add_fstr=f'-nt{nt}',
                              ts=ts, nrg_E_ts=nrg_E_ts, nrg_B_ts=nrg_B_ts,
                              max_bond_A=max_bond_As, max_bond_dAdt=max_bond_dAdts, max_bond_phi=max_bond_phis)

    ap_sys.compute_E(compress_level=3)
    ap_sys.compute_B(compress_level=3)

    for compID in ap_sys.E.componentIDs:
        E_data = ap_sys.E[compID].get_data(ax_select={ax_y: ax_y.npts // 2})
        if E_data is not None:
            np.save(fdir + subdir + f'E{compID}_y0.npy', E_data)

    for compID in ap_sys.B.componentIDs:
        B_data = ap_sys.B[compID].get_data(ax_select={ax_y: ax_y.npts // 2})
        if B_data is not None:
            np.save(fdir + subdir +  f'B{compID}_y0.npy', B_data)


    print('saved data')
