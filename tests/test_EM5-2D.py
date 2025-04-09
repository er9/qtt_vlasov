import numpy as np
import os, sys, time, glob
sys.path.append('../')

from setup_.paths import save_dir, main_dir
from setup_.configs import *


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

save_figs = False
save_data = flags.get('save_data', True)
## to load the output file and maybe restart from it (vs a saved restart file)
load_data = bool(flags.get('load_data', True))
restart_from_T = flags.get('load_data', None)
restart_from_T = restart_from_T if isinstance(restart_from_T, (float, int)) else None
## to restart from a specfic restart file
restart   = bool(flags.get('restart', True))
restart_from_nt = flags.get('restart', None)
restart_from_nt = restart_from_nt if isinstance(restart_from_nt, int) else None

do_tt = flags.get('do_tt', False)    ## see test_vlasovEM_whistler-fd-v2-full
do_full = flags.get('do_full', False)
dissipation = flags.get('dissipation', 0.0)

if restart or save_data or save_figs:
    fnum = f'250403/eta{dissipation}/'
    sdir = f'{save_dir}/data{fnum}/EM_jz2D/'
    fdir = f'{main_dir}/data{fnum}/EM_jz2D/'

    if not os.path.exists(sdir):
        os.makedirs(sdir, exist_ok=True)

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)
    print('sdir', sdir)


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
Lx = 40.0 * lam
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

upwind = flags.get('do_upwind', False)

print('c', matl_config.c)       # should be 1
T = c * Lx / lam / 2
print('T', T)
dt = None  # 0.05  # 0.04  # None
cfl = 0.9
order = 1
te_order = 1 if upwind else 31

## restart from an saved state
load_nt = restart_from_nt
load_fstr = fdir + (f'L{L}_te{te_order}_up/' if upwind else f'L{L}_te{te_order}/')
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


if dissipation == 'max':
    dissipation = np.sqrt(cutoff) * np.sqrt(L * 2)
    print('dissipation per time step', dissipation)


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
ax_x = Axis(L, q, coordinate=X, xpts=x_vals, ax_map=x_map)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals, ax_map=x_map)

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
print('ax-x', ax_x.npts, ax_x.map)
delta_fct_gtn = grid_xyz.make_mps_ndim({ax_x: ax_x.get_select_elems_mps(inds=[ax_x.npts//2 + i for i in range(2**(L-8))]),
                                        ax_y: ax_y.get_select_elems_mps(inds=[ax_y.npts//2 + i for i in range(2**(L-8))]),
                                        })


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
deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zg1 = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER) # , offset=1)
deriv_zv1 = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER) #, offset=1)

# deriv_zg = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
# deriv_zv = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
# deriv_zg1 = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
# deriv_zv1 = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)

if load_nt is not None:
    init_Ez_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + f'EZ2_nt{load_nt}.npy'), ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg})
    init_Bx_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + f'BX0_nt{load_nt}.npy'), ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
    init_By_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + f'BY1_nt{load_nt}.npy'), ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
    init_Ez_gtn.compress(compress_opts={'cutoff': cutoff})
    init_Bx_gtn.compress(compress_opts={'cutoff': cutoff})
    init_By_gtn.compress(compress_opts={'cutoff': cutoff})
    init_Ex_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
    init_Ey_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
    init_Bz_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg1})

    ts = [dt * load_nt]
    nt = load_nt
    max_bond_Es = [[0, 0, init_Ez_gtn.max_bond()]]
    max_bond_Bs = [[init_Bx_gtn.max_bond(), init_By_gtn.max_bond(), 0]]
    print(max_bond_Bs, max_bond_Es)

else:
    init_Ex_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
    init_Ey_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
    init_Ez_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg})
    init_Bx_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
    init_By_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
    init_Bz_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg1})

    ts = [0]
    nt = 0
    max_bond_Es = [[0, 0, 0]]
    max_bond_Bs = [[0, 0, 0]]

init_E_field = Field('E', grid_xyz, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E, is_sqrt=True)
init_B_field = Field('B', grid_xyz, data={X: init_Bx_gtn, Y:init_By_gtn, Z: init_Bz_gtn},
                     compress_config=te_compress_B, is_sqrt=True)


### BUILD MAXWELL EQNS SOLVER
em_sys = Maxwell(init_E_field, init_B_field, phi=None, psi=None,
                 coords_x=coord_sys, matl_params=matl_config,
                 is_electrostatic=False, clean=False, is_yee=(te_order == 31), normalize=False,
                 upwind=upwind,
                 te_order=te_order, compress_levels=comp_levels )


while nt < num_tsteps:

    dt_ = dt if nt==0 else dt
    print('dt_', dt, 't', ts[-1])

    ## set the current density in the EM solver
    em_sys.current_density = Field('J', grid_xyz, data={Z:Jz_gtn(ts[-1] + dt_ / 2)})

    em_sys = em_sys.next_time_step(dt_, compress_level=1, is_first_time_step=(nt==0),
                                   apply_constraints=False)
    # print('E norms', em_sys.E.norms([X,Y,Z]), np.sum(np.array(em_sys.E.norms([X,Y,Z]))**2))
    # print('B norms', em_sys.B.norms([X,Y,Z]), np.sum(np.array(em_sys.B.norms([X,Y,Z]))**2))
    if dissipation > 0.0:
        print('adding dissip', dissipation)
        for compID in em_sys.E.componentIDs:
            em_sys.E[compID].add_dissipation(dissipation, deriv_order=6, inplace=True)
        for compID in em_sys.B.componentIDs:
            em_sys.B[compID].add_dissipation(dissipation, deriv_order=6, inplace=True)

    print('divB', em_sys.check_divB().norm())
    print('divE', em_sys.check_poisson().norm())

    ## obtain max QTT rank for each component of the E field
    E_bonds = [em_sys.E.max_bond(C) for C in [X,Y,Z]]
    max_bond_Es += [E_bonds]

    ## obtain max QTT rank for each component of the B field
    B_bonds = [em_sys.B[C].max_bond() for C in [X,Y,Z]]
    max_bond_Bs += [B_bonds]

    ts += [ts[-1] + dt_]

    nt += 1
    print('result norm', nt, ts[-1], em_sys.E.max_bonds(), em_sys.B.max_bonds(), DMAX, q ** L)

    if nt == num_tsteps - 1 or nt%100==0:

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
            plt.imshow(Ez)
            plt.title('Ez')
            plt.colorbar()
        if Bx is not None:
            plt.figure()
            plt.imshow(Bx)
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

        if save_data:

            subdir = f'L{L}_te{te_order}_up/' if upwind else f'L{L}_te{te_order}/'
            if not os.path.exists(fdir + subdir):
                os.makedirs(fdir + subdir)

            np.save(fdir + subdir + f'ranks_nt{nt}.npy', np.hstack([np.array(max_bond_Es), np.array(max_bond_Bs)]))

            for compID in em_sys.E.componentIDs:
                E_data = em_sys.E[compID].get_data()
                if E_data is not None:
                    np.save(fdir + subdir + f'E{compID}_nt{nt}.npy', E_data)

            for compID in em_sys.B.componentIDs:
                E_data = em_sys.B[compID].get_data()
                if E_data is not None:
                    np.save(fdir + subdir + f'B{compID}_nt{nt}.npy', E_data)

if save_data:

    subdir = f'L{L}_te{te_order}_up/' if upwind else f'L{L}_te{te_order}/'
    if not os.path.exists(fdir + subdir):
        os.makedirs(fdir + subdir)

    np.save(fdir + subdir + f'ranks_nt{nt}.npy', np.hstack([np.array(max_bond_Es), np.array(max_bond_Bs)]))

    for compID in em_sys.E.componentIDs:
        E_data = em_sys.E[compID].get_data()
        if E_data is not None:
            np.save(fdir + subdir + f'E{compID}_nt{nt}.npy', E_data)

    for compID in em_sys.B.componentIDs:
        E_data = em_sys.B[compID].get_data()
        if E_data is not None:
            np.save(fdir + subdir + f'B{compID}_nt{nt}.npy', E_data)


    print('saved data')