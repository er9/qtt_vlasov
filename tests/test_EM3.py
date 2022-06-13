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

"""
1D Reimann problem
"""

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

######################
# 1D Riemann problem #
######################

q = 2
L = 7
K = 1
npts = q**L
TN_layout = LayoutType.PARALLEL_GROUP

cfl = 1.0
T = 0.5
order = 4
te_order = 4

## initial conditions
Lx = 1.0
c = 1.0

def Ey(x):     return np.where(x<0, 1., -1.) 
def By(x):     return np.where(x<0, -0.75, 0.75)
Bx = lambda x: 1.0 + 0*x


#### compression options
""" comp1:   compress final state
    comp2:   + compress intermediate rk4 states
    comp3:   + compress sum of derivatives
    comp4:   + compress derivatives
    comp5:   + compress during calculation of derivative

    mod0:  no modifications 
    mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
"""
DMAX = 64
compress_style = 1
compress_style_mod = 0

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

te_compress_psi = te_compress_phi
te_compress_B = te_compress_E

print('te compress E', te_compress_E)
print('te compress phi', te_compress_phi)

comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]


#############################
## start building PDE system
#############################

## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

x_vals = np.linspace(-Lx,Lx,npts)
ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
coord_sys = CartesianCoordinateSpace('cart', ax_x, coords=[X,Y,Z])

## define time step
dt = cfl * 1./(c/ax_x.dx)
num_tsteps = int(np.ceil(T/dt))
print('max dt',  1./(c/ax_x.dx), 'use dt', dt, num_tsteps)

## configure derivatives
deriv_zg = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER)

## define grids
grid_x = Grid1D('X', (ax_x,), layout_type=TN_layout)

init_Ey  = Ey(x_vals)
init_Bx  = Bx(x_vals)
init_By  = By(x_vals)

init_Ey_gtn = grid_x.map_state_to_mps(init_Ey, ax_deriv_configs={ax_x: deriv_zg})
init_Ez_gtn = grid_x.make_empty_gridTN()
init_Bx_gtn = grid_x.map_state_to_mps(init_Bx, ax_deriv_configs={ax_x: deriv_zg})
init_By_gtn = grid_x.map_state_to_mps(init_By, ax_deriv_configs={ax_x: deriv_zg})
init_Bz_gtn = grid_x.make_empty_gridTN()


init_E_field = Field('E', grid_x, data={Y: init_Ey_gtn, Z: init_Ez_gtn}, compress_config=te_compress_E)
init_B_field = Field('B', grid_x, data={X: init_Bx_gtn, Y: init_By_gtn, Z: init_Bz_gtn}, compress_config=te_compress_B)
init_phi_field = ScalarField('phi', grid_x, data=None, compress_config=te_compress_phi)
init_psi_field = ScalarField('psi', grid_x, data=None, compress_config=te_compress_psi)

### BUILD PDE
vm_sys = VlasovMaxwell(None, None, init_E_field, init_B_field, init_phi_field, init_psi_field,
                       coords_x=coord_sys, coords_ve=None, coords_vi=None,
                       te_order=te_order, compress_levels=comp_levels,
                       )

nt = 0
Ey_t  = [init_Ey]
Ez_t  = [np.zeros_like(init_Ey)]
By_t  = [init_By]
Bz_t  = [np.zeros_like(init_By)]
max_bond_Es = [[0, init_E_field[Y].max_bond(), 0]]
max_bond_Bs = [[init_B_field[X].max_bond(), init_B_field[Y].max_bond(), 0]]

while nt < num_tsteps:
    vm_sys = vm_sys.next_time_step(dt, compress_level=1)

    # print(vp_sys.E.components.keys())
    Ey_data = vm_sys.E.get_comp_data(Y)
    Ez_data = vm_sys.E.get_comp_data(Z)
    Ey_t += [Ey_data]
    Ez_t += [Ez_data]
    E_bonds = [vm_sys.E.max_bond(C) for C in [X,Y,Z]]
    max_bond_Es += [E_bonds]

    By_data = vm_sys.B.get_comp_data(Y)
    Bz_data = vm_sys.B.get_comp_data(Z)
    By_t += [By_data]
    Bz_t += [Bz_data]
    B_bonds = [vm_sys.B.max_bond(C) for C in [X,Y,Z]]
    max_bond_Bs += [B_bonds]

    phi_data = vm_sys.phi.get_comp_data() if vm_sys.phi is not None else None
    psi_data = vm_sys.psi.get_comp_data() if vm_sys.psi is not None else None

    if False: # (nt+1)%10==0:
        fig1, ax1 = plt.subplots(2,2)
        ax1[0,0].plot( x_vals, Ey_data )
        ax1[0,1].plot( x_vals, Ez_data )
        ax1[1,0].plot( x_vals, By_data )
        ax1[1,1].plot( x_vals, Bz_data )
        ax1[0,0].set_xlabel('x'), ax1[0,0].set_ylabel('Ey')
        ax1[0,1].set_xlabel('x'), ax1[0,1].set_ylabel('Ez')
        ax1[1,0].set_xlabel('x'), ax1[1,0].set_ylabel('By')
        ax1[1,1].set_xlabel('x'), ax1[1,1].set_ylabel('Bz')
        plt.subplots_adjust(left=0.15,right=0.90, wspace=0.45)

        plt.figure()
        if phi_data is not None: plt.plot(x_vals, phi_data)
        if psi_data is not None: plt.plot(x_vals, psi_data)
        plt.legend(['phi','psi'])
        plt.show()

    nt += 1
    # norm = vp_sys.fe.integrate('all')[0] - vp_sys.fi.integrate('all')[0]
    # norm = np.sum(fe_data)*dx*dv - np.sum(init_fi)*dx*dv
    print('result norm', nt, vm_sys.E.max_bond(Z), q ** (L // 2))

ts = dt*np.arange(num_tsteps+1)
Ey_data = vm_sys.E.get_comp_data(Y)
Ez_data = vm_sys.E.get_comp_data(Z)
By_data = vm_sys.B.get_comp_data(Y)
Bz_data = vm_sys.B.get_comp_data(Z)
max_bond_Es = np.array(max_bond_Es)
max_bond_Bs = np.array(max_bond_Bs)

print(np.shape(max_bond_Bs), np.shape(max_bond_Es))

fig1, ax1 = plt.subplots(2,2)
ax1[0,0].plot( x_vals, Ey_data )
ax1[0,1].plot( x_vals, Ez_data )
ax1[1,0].plot( x_vals, By_data )
ax1[1,1].plot( x_vals, Bz_data )
ax1[0,0].set_xlabel('x'), ax1[0,0].set_ylabel(r'$E_y$')
ax1[0,1].set_xlabel('x'), ax1[0,1].set_ylabel(r'$E_z$')
ax1[1,0].set_xlabel('x'), ax1[1,0].set_ylabel(r'$B_y$')
ax1[1,1].set_xlabel('x'), ax1[1,1].set_ylabel(r'$B_z$')
plt.subplots_adjust(left=0.15,right=0.90, wspace=0.45)

fig3, ax3 = plt.subplots()
ax3.plot(ts, np.log(max_bond_Es[:,0])/np.log(q**K),label=r'$E_x$')
ax3.plot(ts, np.log(max_bond_Es[:,1])/np.log(q**K),label=r'$E_y$')
ax3.plot(ts, np.log(max_bond_Es[:,2])/np.log(q**K),label=r'$E_z$')
ax3.plot(ts, np.log(max_bond_Bs[:,0])/np.log(q**K),'--',label=r'$B_x$')
ax3.plot(ts, np.log(max_bond_Bs[:,1])/np.log(q**K),'--',label=r'$B_y$')
ax3.plot(ts, np.log(max_bond_Bs[:,2])/np.log(q**K),'--',label=r'$B_z$')
ax3.plot([ts[0],ts[-1]], [L//2]*2, 'k:')
ax3.set_xlabel('t')
ax3.set_ylabel(r'$\log_{q^K}$' + '(max bond)')
fig3.legend(bbox_to_anchor=(0.85,0.75), loc='upper right')

plt.show()

