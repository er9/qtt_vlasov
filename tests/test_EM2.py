import os, sys, time, glob
sys.path.append('../')

from setup.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell

save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/VM-TM/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VM-TM/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)


#######################
# standard parameters #
#######################

permittivity = 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor
mass_e = 1.0          # mass of electron
mass_i = 1836*mass_e  # mass of ion
charge = 1.0          # elementary charge

n0_e = 1.0             # electron number density
n0_i = n0_e            # ion number density
T_e  = 1.0             # electron temperature [eV]
T_i  = T_e             # ion temperature [eV]

vth_e = np.sqrt(eV*T_e/mass_e)                   # electron thermal speed
vth_i = np.sqrt(eV*T_i/mass_i)                   # ion thermal speed
wp_i  = np.sqrt(eV**2*n0_i/permittivity/mass_i)  # plasma i+ frequency
wp_e  = np.sqrt(eV**2*n0_e/permittivity/mass_e)   # plasma frequency
lamD  = vth_e/wp_e                               # Debye length


matl_config = MaterialConfiguration(eps0=permittivity, mu0=permeability)


#########################
# EM pulse in 2D cavity #
#########################

q = 2
L = 6
K = 2
npts = q**L
TN_layout = LayoutType.PARALLEL_GROUP

T = 3.0
cfl = 0.75
order = 4
te_order = 4


## initial conditions
Lx = 1.0
Ly = 1.0

E0 = 1.
beta = 25
Ez = lambda x, y: E0*np.exp(-beta*(x**2 + y**2))
c = 1/np.sqrt(permittivity*permeability)

#############################
## start building PDE system
#############################

##### compression options ####
DMAX = None
compress_style = 1
compress_style_mod = 0

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
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
x_vals = np.linspace(-Lx,Lx,npts)
y_vals = np.linspace(-Lx,Ly,npts)

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## define time step
dx, dy = ax_x.dx, ax_y.dx
cfl_params = ((dx, dy), (c, c))
dt = cfl * cfl_limit( cfl_params )
num_tsteps = int(T/dt)
print('dt',dt,  cfl*1./(c/ax_x.dx + c/ax_y.dx), num_tsteps)


## define grids
grid_xy = Grid1D('XY', (ax_x,ax_y), layout_type=TN_layout)

## define initial grid1DTNs
data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
init_Ez  = Ez(data_x,data_y)

plt.figure()
plt.imshow(init_Ez)
plt.colorbar()
plt.show()

## configure derivatives
deriv_zg = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER)

init_Ex_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x:deriv_zg, ax_y:deriv_zv})
init_Ey_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x:deriv_zv, ax_y:deriv_zg})
init_Ez_gtn = grid_xy.map_state_to_mps(init_Ez, ax_deriv_configs={ax_x:deriv_zv, ax_y:deriv_zv})
init_Bx_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x:deriv_zv, ax_y:deriv_zg})
init_By_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x:deriv_zg, ax_y:deriv_zv})
init_phi_gtn = grid_xy.make_empty_gridTN()
init_psi_gtn = grid_xy.make_empty_gridTN()

init_E_field = Field('E', grid_xy, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E)
init_B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y:init_By_gtn}, compress_config=te_compress_B)
init_phi_field = ScalarField('phi', grid_xy, data=init_phi_gtn, compress_config=te_compress_phi)
init_psi_field = ScalarField('psi', grid_xy, data=init_psi_gtn, compress_config=te_compress_psi)


### BUILD PDE
vm_sys = VlasovMaxwell(None, None, init_E_field, init_B_field, init_phi_field, init_psi_field,
                       coords_x=coord_sys, coords_ve=None, coords_vi=None,
                       matl_params=matl_config,
                       te_order=te_order, compress_levels=comp_levels,
                       )


nt = 0
Ez_t  = [init_Ez]
max_bond_Es = [[0, 0, init_E_field[Z].max_bond()]]
max_bond_Bs = [[0,0,0]] 

while nt < num_tsteps:
    vm_sys = vm_sys.next_time_step(dt, compress_level=1)

    Ez_data = vm_sys.E.get_comp_data(Z)
    Ez_t += [Ez_data]
    E_bonds = [vm_sys.E.max_bond(C) for C in [X,Y,Z]]
    print('E bonds', E_bonds)
    max_bond_Es += [E_bonds]

    B_bonds = [vm_sys.B[C].max_bond() for C in [X,Y,Z]]
    print('B bonds', B_bonds)
    max_bond_Bs += [B_bonds]

    if np.abs(nt*dt-1.5) < dt/2 or np.abs(nt*dt-3.0) < dt/2:
        plt.figure()
        plt.imshow(Ez_data)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'Ez(x, y, t={nt*dt:1.3f})')
        plt.show()

    if nt%20==0:
        plt.figure()
        plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
        plt.xlabel('x')
        plt.ylabel('Ez')
        plt.title(f'E(x, y={y_vals[npts//2]:1.2f}, t={nt*dt:1.3f})')
        plt.show()

    nt += 1
    # norm = vp_sys.fe.integrate('all')[0] - vp_sys.fi.integrate('all')[0]
    # norm = np.sum(fe_data)*dx*dv - np.sum(init_fi)*dx*dv
    print('result norm', nt, vm_sys.E[Z].max_bond(), q ** L)

Ez_data = vm_sys.E.get_comp_data(Z)

plt.figure()
plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
plt.xlabel('x')
plt.ylabel('Ez')
plt.title(f'E(x, y={y_vals[npts//2]:1.2f}, t={nt*dt:1.3f})')

plt.figure()
plt.imshow(Ez_data)
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.title(f'Ez(x, y, t={nt*dt:1.3f})')

plt.show()


