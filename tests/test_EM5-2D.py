import numpy as np
import os, sys, time, glob
sys.path.append('../')

from setup_.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_EM import Maxwell

save_figs = False
save_data = True
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

"""
dipole radiation from current source
"""

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_data/EM_jz2D/'
    sdir = 'data/EM_jz2D/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/EM_jz2D/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)


#######################
# problem parameters  #
#######################

permittivity = 1  # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor
matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)

## dipole properties
c = 1/np.sqrt(permittivity*permeability)
dp = 0.2               # dipole length
I0 = 1.0                # current strength
omega = 2 * np.pi * 2
lam = 2 * np.pi / (omega / c)

## box size
Lx = 20.0 * lam
Ly = Lx
Lz = Lx


#######################
# run-time parameters #
#######################

q = 2
L = 13
K = 2
npts = q**L
TN_layout = LayoutType.SEQUENTIAL

upwind = True

print('c', matl_config.c)       # should be 1
T = 5.0
dt = None  # 0.05  # 0.04  # None
cfl = 0.9
order = 1
te_order = 1  # if upwind else 31

load_nt = 200 * 2**(L-8)
load_fstr = fdir + 'L13_te1_up/'

#############################
## start building PDE system
#############################

##### compression options ####
## even with modest CUTOFF = 1.0e-8, div(B) is not preserved
DMAX = None # 128
cutoff = 1.0e-8  # CUTOFF
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


## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(-Lx,Lx,npts,endpoint=True)
x1_vals = x_vals + (x_vals[1]-x_vals[0])/2
y_vals = np.linspace(-Ly,Ly,npts,endpoint=True)
y1_vals = y_vals + (y_vals[1]-y_vals[0])/2

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## define time step
dx, dy, = ax_x.dx, ax_y.dx,
cfl_params = ((dx, dy), (c, c))
dt_cfl = cfl_limit( cfl_params )
dt = dt_cfl * cfl if dt is None else dt
num_tsteps = 275 * 2**(L-8)  # int(T/dt)
print('dt', dt, cfl*1./(c/ax_x.dx + c/ax_y.dx), num_tsteps, )
print('1/omega', 1 / omega)
print('dx', dx, 'dy', dy)

## define QTT Grid object
grid_xyz = Grid1D('XYZ', (ax_x, ax_y), layout_type=TN_layout)

## define "delta function" in space for the current source
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

if load_fstr is not None:
    init_Ez_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + 'EZ2.npy'), ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg})
    init_Bx_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + 'BX0.npy'), ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1})
    init_By_gtn = grid_xyz.map_state_to_mps(np.load(load_fstr + 'BY1.npy'), ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg})
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

    if False: # nt == num_tsteps - 1: # or nt%100==0:

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

        # ## data along x axis
        # Ex_x = em_sys.E[X].get_data(ax_select={ax_y: ax_y.npts // 2})
        # Ey_x = em_sys.E[Y].get_data(ax_select={ax_y: ax_y.npts // 2})
        # Ez_x = em_sys.E[Z].get_data(ax_select={ax_y: ax_y.npts // 2})
        # Bx_x = em_sys.B[X].get_data(ax_select={ax_y: ax_y.npts // 2})
        # By_x = em_sys.B[Y].get_data(ax_select={ax_y: ax_y.npts // 2})
        # Bz_x = em_sys.B[Z].get_data(ax_select={ax_y: ax_y.npts // 2})
        # plt.figure()
        # if Ex_x is not None:
        #     plt.plot(ax_x.xpts, Ex_x, label='Ex')
        # if Ey_x is not None:
        #     plt.plot(ax_x.xpts, Ey_x, label='Ey')
        # if Ez_x is not None:
        #     plt.plot(ax_x.xpts, Ez_x, label='Ez')
        # if Bx_x is not None:
        #     plt.plot(ax_x.xpts, Bx_x, label='Bx')
        # if By_x is not None:
        #     plt.plot(ax_x.xpts, By_x, label='By')
        # if Bz_x is not None:
        #     plt.plot(ax_x.xpts, Bz_x, label='Bz')
        # plt.legend()
        # plt.show()
        #
        # #########################
        # ## theoretical results ##
        # #########################
        #
        # ### field magnitude ###
        # k = omega/matl_config.c
        # Z0 = np.sqrt(permeability / permittivity)
        # x_abs = np.abs(ax_x.xpts)
        # x_shift = omega * ts[-1] - k * x_abs
        # Ez_theory = -1.j * Z0 * k * I0 * ax_x.dx * ax_y.dx * dp / 4 / np.pi / x_abs * np.exp(1.j * x_shift) \
        #             * (1 + 1./1.j /k/x_abs - 1./k**2/x_abs**2)
        # Ez_theory = np.imag(Ez_theory)
        # Ez_theory = np.where(x_abs <= c * ts[-1], Ez_theory, 0)
        #
        # k = omega / matl_config.c
        # # x_abs = np.abs(ax_x.xpts)
        # # x_shift = omega * ts[-1] - k * x_abs
        # By_theory = 1.j * k * I0 * ax_x.dx * ax_y.dx * dp / 4 / np.pi / x_abs * np.exp(1.j * x_shift) \
        #             * (1 + 1./1.j/k/x_abs) * permeability
        # By_theory = np.imag(By_theory)
        # By_theory = np.where(ax_x.xpts < 0, -1 * By_theory, By_theory)
        # By_theory = np.where(x_abs <= c * ts[-1], By_theory, 0)
        # print('Z0', Z0, 'c', c, 'k', k, 'omega', omega)
        # print('c *t', c * (ts[-1] + dt_))
        # print('omega', omega * ts[-1] / k , omega / k, c)
        #
        # plt.plot(ax_x.xpts, By_theory, '--', label='By theory')
        # plt.plot(ax_x.xpts, Ez_theory, '--', label='Ez theory')
        #
        # plt.title('fields at x, y=0, z=0')
        # plt.legend()
        # plt.show()


        # ### energy density ###
        # print('B energy density')
        # nrg_B_density = em_sys.magnetic_energy_density()
        # print('get comp data')
        # nrg_B_data = nrg_B_density.get_comp_data()
        #
        # k = omega / matl_config.c
        # x_, y_ = np.meshgrid(ax_x.xpts, ax_y.xpts, indexing='ij')
        # r = np.sqrt(x_**2 + y_**2)
        # r_shift = omega * ts[-1] - k * r
        # nrg_xy_theory = permeability / 2 * (k * I0 * ax_x.dx * ax_y.dx * dp / 4 / np.pi / r)**2 \
        #                 * np.sin(r_shift)**2 * (1 + 1./r**2)
        # nrg_xy_theory = np.where(r < c * ts[-1], nrg_xy_theory, 0)
        #
        # levels = np.logspace(np.log10(1.0e-9 * ax_x.dx ** 3), np.log10(1.0e-5 * ax_x.dx ** 2), 10)
        # print('levels', levels)
        #
        # if nrg_B_density is not None:
        #     plt.figure()
        #     plt.imshow(np.real(nrg_B_data.T), vmax=0.001 * ax_x.dx ** 3)
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.title(f'|B|^2 (z=0), t{ts[-1]}')
        #     plt.colorbar()
        #     plt.contour(nrg_B_data, levels=levels)
        #     print('Bz real/imag', np.linalg.norm(np.real(nrg_B_data)), np.linalg.norm(np.imag(nrg_B_data)))
        #
        # plt.figure()
        # plt.imshow(nrg_xy_theory, vmax=0.001 * ax_x.dx ** 3)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title(f'theory |B|^2 (z=0), t{ts[-1]}')
        # plt.colorbar()
        # plt.contour(nrg_xy_theory, levels=levels)
        #
        # plt.show()


if save_data:

    fdir += f'L{L}_te{te_order}_up/' if upwind else f'L{L}_te{te_order}/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    np.save(fdir + f'ranks_nt{num_tsteps}.npy', np.hstack([np.array(max_bond_Es), np.array(max_bond_Bs)]))

    for compID in em_sys.E.componentIDs:
        E_data = em_sys.E[compID].get_data()
        if E_data is not None:
            np.save(fdir + f'E{compID}_nt{num_tsteps}.npy', E_data)

    for compID in em_sys.B.componentIDs:
        E_data = em_sys.B[compID].get_data()
        if E_data is not None:
            np.save(fdir + f'B{compID}_nt{num_tsteps}.npy', E_data)



    print('saved data')