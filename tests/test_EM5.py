import os, sys, time, glob
sys.path.append('../')

from setup.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_EM import Maxwell

save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

"""
dipole radiation from current source
"""

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/EM5/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/EM-fdtd-TM/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)


#######################
# standard parameters #
#######################

permittivity = 1.0 # 1/100  # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor
matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)


#########################
# EM pulse in 2D cavity #
#########################

q = 2
L = 7
K = 2
npts = q**L
TN_layout = LayoutType.SEQUENTIAL

## dipole properties
c = 1/np.sqrt(permittivity*permeability)
dp = 0.2               # dipole length
I0 = 1.0                # current strength
omega = 2 * np.pi
lam = 2 * np.pi / (omega / c)

## box size
Lx = 2.0 * lam
Ly = Lx
Lz = Lx

print('c', matl_config.c)       # should be 1
T = 3.0
cfl = 0.9
order = 1
te_order = 33 #223
is_yee = False


#############################
## start building PDE system
#############################

##### compression options ####
## even with modest CUTOFF = 1.0e-8, div(B) is not preserved
DMAX = 32
compress_style = 1
compress_style_mod = 0

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=1.0e-12)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

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
x_vals = np.linspace(-Lx,Lx,npts,endpoint=False)
x1_vals = x_vals + (x_vals[1]-x_vals[0])/2
y_vals = np.linspace(-Ly,Ly,npts,endpoint=False)
y1_vals = y_vals + (y_vals[1]-y_vals[0])/2
z_vals = np.linspace(-Lz,Lz,npts,endpoint=True)
z1_vals = z_vals + (z_vals[1]-z_vals[0])/2

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)
ax_z = Axis(L, q, coordinate=Z, xpts=z_vals)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, ax_z, coords=[X,Y,Z])

## define time step
dx, dy, dz = ax_x.dx, ax_y.dx, ax_z.dx
cfl_params = ((dx, dy, dz), (c, c, c))
dt = cfl * cfl_limit( cfl_params )
num_tsteps = int(T/dt)
print('dt', dt, cfl*1./(c/ax_x.dx + c/ax_y.dx + c/ax_z.dx), num_tsteps)
print('dx', dx, 'dy', dy)

## define grids
grid_xyz = Grid1D('XYZ', (ax_x, ax_y, ax_z), layout_type=TN_layout)

delta_fct_gtn = grid_xyz.make_mps_ndim({ax_x: ax_x.get_select_elems_mps(inds=[ax_x.npts//2]),
                                        ax_y: ax_y.get_select_elems_mps(inds=[ax_y.npts//2]),
                                        ax_z: ax_z.map_state_to_mps(np.abs(z1_vals) < dp/2)})


## define initial grid1DTNs
def Jz_gtn(t):
    jz = delta_fct_gtn.scalar_multiply(np.cos(omega * t) * I0, inplace=False)
    # print('jz', t, np.cos(omega * t) * I0)
    return jz


data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
data_x1, data_y1 = np.meshgrid(x1_vals,y1_vals,indexing='ij')
# init_Ez = Ez(data_x, data_y)   # i, j, k+1/2
# init_Bx = Bx(data_x, data_y)  # i, j+1/2, k+1/2  # init at time dt/2
# init_By = By(data_x, data_y)  # i+1/2, j, k+1/2  # init at time dt/2


## configure derivatives
deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zg1 = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER, offset=-1)
deriv_zv1 = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER, offset=-1)

init_Ex_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg, ax_z: deriv_zg})
init_Ey_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1, ax_z: deriv_zg})
init_Ez_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg, ax_z: deriv_zg1})
init_Bx_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zg1, ax_z: deriv_zg1})
init_By_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg, ax_z: deriv_zg1})
init_Bz_gtn = grid_xyz.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zg1, ax_z: deriv_zg})

init_E_field = Field('E', grid_xyz, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E, is_sqrt=True)
init_B_field = Field('B', grid_xyz, data={X: init_Bx_gtn, Y:init_By_gtn}, compress_config=te_compress_B,
                     is_sqrt=True)


### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, phi=None, psi=None,
                     coords_x=coord_sys, matl_params=matl_config,
                     is_electrostatic=False, clean=False, is_yee=is_yee, normalize=False,
                     te_order=te_order, compress_levels=comp_levels )

ts = [0]
nt = 0
max_bond_Es = [[0, 0, 0]]
max_bond_Bs = [[0, 0, 0]]

while nt < num_tsteps:

    em_sys.current_density = Field('J', grid_xyz, data={Z:Jz_gtn(ts[-1] + dt / 2)})

    em_sys = em_sys.next_time_step(dt, compress_level=1, is_first_time_step=(nt==0))
    em_sys.realspace_smoother(inplace=True)
    print('E norms', em_sys.E.norms([X,Y,Z]), np.sum(np.array(em_sys.E.norms([X,Y,Z]))**2))
    print('B norms', em_sys.B.norms([X,Y,Z]), np.sum(np.array(em_sys.B.norms([X,Y,Z]))**2))

    E_bonds = [em_sys.E.max_bond(C) for C in [X,Y,Z]]
    max_bond_Es += [E_bonds]

    B_bonds = [em_sys.B[C].max_bond() for C in [X,Y,Z]]
    max_bond_Bs += [B_bonds]

    ts += [ts[-1] + dt]

    if nt%50==0 or nt == num_tsteps - 1:
        nrg_E_density = em_sys.electric_energy_density()
        nrg_E_z0 = nrg_E_density.get_comp_data(ax_select={ax_z: ax_z.npts // 2})
        nrg_E_y0 = nrg_E_density.get_comp_data(ax_select={ax_y: ax_y.npts // 2})
        nrg_E_x0 = nrg_E_density.get_comp_data(ax_select={ax_x: ax_x.npts // 2})

        # levels = np.linspace(1.0e-9 * ax_x.dx**3, 1.0e-5 * ax_x.dx**2, 10)
        # print('levels', levels)
        levels = np.logspace(np.log10(1.0e-9 * ax_x.dx ** 3), np.log10(1.0e-5 * ax_x.dx ** 2), 10)
        print('levels', levels)

        plt.figure()
        plt.imshow(nrg_E_z0.T, vmax=0.001*ax_x.dx**3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'|E|^2 (z=0), t{ts[-1]}')
        plt.colorbar()
        plt.contour(nrg_E_z0, levels=levels)

        plt.figure()
        plt.imshow(nrg_E_y0.T, vmax=0.001 * ax_x.dx ** 3)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'|E|^2 (y=0), t{ts[-1]}')
        plt.colorbar()
        plt.contour(nrg_E_y0.T, levels=levels)

        print('max nrg E', np.max(nrg_E_z0))

        plt.figure()
        plt.imshow(nrg_E_x0.T, vmax=0.001*ax_x.dx**3)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(f'|E|^2 (x=0), t{ts[-1]}')
        plt.colorbar()
        plt.contour(nrg_E_x0.T, levels=levels)

        plt.show()

    nt += 1
    # norm = vp_sys.fe.integrate('all')[0] - vp_sys.fi.integrate('all')[0]
    # norm = np.sum(fe_data)*dx*dv - np.sum(init_fi)*dx*dv
    print('result norm', nt, ts[-1], em_sys.E[Z].max_bond(), DMAX, q ** L)



