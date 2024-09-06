import os, sys, pickle, time, glob
import scipy
sys.path.append('../')

from setup_.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

""" Particle streaming with absorbing boundary conditions
"""

save_figs = False
save_data = False
load_data = False
restart   = False

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/VP1-damp-final-CE/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/VP1-damp-final-CE/'
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
mass_i = 1.0*mass_e   # mass of ion
charge = 0.0          # elementary charge

n0_e = 1.0             # electron number density
n0_i = n0_e            # ion number density
T_e  = 1.0             # electron temperature [eV]
T_i  = T_e             # ion temperature [eV]

vth_e = np.sqrt(eV*T_e/mass_e)                   # electron thermal speed
vth_i = np.sqrt(eV*T_i/mass_i)                   # ion thermal speed
wp_e  = np.sqrt(eV**2*n0_e/permittivity/mass_e)  # plasma frequency
lamD  = vth_e/wp_e                               # Debye length

matl_config = UnitsConfiguration(e=charge, eps0=permittivity, mu0=permeability)
ion_config = IonConfiguration(mass=mass_i, T=T_i, Z=0, units_config=matl_config)
elc_config = IonConfiguration(mass=mass_e, T=T_e, Z=0, units_config=matl_config)



def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    for i in range(len(vs)):
        try:                 v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
        except(TypeError):   v2 += (vs[i]-flow)   *(vs[i]-flow)

    vth2 = eV*temp/mass
    return density*np.sqrt(1./2/np.pi/vth2) * np.exp(-v2/2/vth2)


def dv_maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0, eV=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    for i in range(len(vs)):
        try:                v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
        except TypeError:   v2 += (vs[i] - flow) * (vs[i] - flow)
    vth2  = eV*temp/mass
    grads = [ density*np.sqrt(1./2/np.pi/vth2) * np.exp(-v2/2/vth2) * (-v/vth2) for v in vs]
    return grads


def density_at_time(time_):
    """ density of particles at specific time
    """
    func = lambda v_: (xL - v_ * time_) * maxwellian(v_)
    n_tp = scipy.integrate.quad(func, 0, xL / time_)[0]

    func = lambda v_: (xL - v_ * time_) * maxwellian(-v_)
    n_tm = scipy.integrate.quad(func, 0, xL / time_)[0]

    return n_tp + n_tm




########################
# Vlasov-Poisson in 1D #
########################

q = 2
L = 6
K = 1
npts = q**L
TN_layout = LayoutType.PARALLEL_GROUP

order = 1  # lower order is better; can use larger time step
te_order = 4
cfl = 0.9


## initial conditions
x0, xL = 0, 2*np.pi

def fe(x):   return np.where(x>=0, 1., 0. ) * np.where(x<=xL, 1., 0. ) # * np.exp(-(x-np.pi)**2/2)
V_field  = lambda x:  0*x

dx = (xL-x0)/(npts-3)  # ghost cell on either end of spatial axis
x_vals = np.linspace(x0-dx,xL+dx,npts,endpoint=True)
v_vals = np.linspace(-6*vth_e,6*vth_e,npts,endpoint=False)
dx = x_vals[1]-x_vals[0]
dv = v_vals[1]-v_vals[0]
data_x, data_v = np.meshgrid(x_vals,v_vals,indexing='ij')
print(data_x.shape, data_v.shape)
print('dx',dx, 'dv e', dv)

## define time step
dt = cfl*1/(np.max(np.abs(v_vals))/dx)
num_tsteps = int(10.0/dt)
print('dt',dt,num_tsteps)


#### intiialize fields
init_fe = fe(data_x) * maxwellian(data_v,density=n0_e,flow=0.0,mass=mass_e,temp=T_e)

## define axis objects
X = Coordinate('X', CoordinateType.X)
V = Coordinate('V', CoordinateType.X)

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_v = Axis(L, q, coordinate=V, xpts=v_vals)

coords_x = CartesianCoordinateSpace('X',ax_x)
coords_v = CartesianCoordinateSpace('V',ax_v)

deriv_x = DerivativeConfiguration(left_bc=BCType.ABSORBING, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)


## define grids
grid_f = Grid1D('XV',(ax_x,ax_v), layout_type=TN_layout)

## define initial grid1DTNs
init_fe_gtn = grid_f.map_state_to_mps(init_fe, ax_deriv_configs={ax_x: deriv_x, ax_v: deriv_v})

## define Field
init_fe_field = ScalarField('fe', grid_f, data=init_fe_gtn)



### BUILD PDE
vp_sys = VlasovPoisson(init_fe_field, None, None, coords_x=coords_x, coords_ve=coords_v,
                       ion_params=ion_config, elc_params=elc_config,
                       evolve_ion=False, normalize=False, te_order=te_order)

nt = 0
maxbond_f = [init_fe_field.max_bond()]
num_e = [np.trapz(np.trapz(init_fe[1:-1,:])*dv)*dx]
dens_e = np.trapz(init_fe[1:-1,:])*dv
print('time step', nt, xL, num_e)

plt.figure()
plt.imshow(init_fe,extent=(v_vals[-1],v_vals[0],xL+dx,x0-dx))
plt.colorbar()
plt.xlabel('v')
plt.ylabel('x')
plt.show()

while nt < num_tsteps:
    vp_sys = vp_sys.next_time_step(dt, compress_level=1)
    # print(vp_sys.te_compress_opts)

    fe_field = vp_sys.fe
    fe_data = fe_field.get_comp_data()
    num_e += [np.trapz(np.trapz(fe_data[1:-1,:])*dv)*dx]
    maxbond_f += [fe_field.max_bond()]

    nt += 1

    if nt%50 == 0:
        plt.figure()
        plt.imshow(fe_data,extent=(v_vals[0],v_vals[-1],xL+dx,x0-dx))
        plt.colorbar()
        plt.xlabel('v')
        plt.ylabel('x')
        plt.show()

        plt.figure()
        plt.plot(v_vals,fe_data[npts//4,:])
        plt.show()

    print('time step', nt, num_e[-1])


ts = np.arange(num_tsteps+1)*dt
dens_t_theory = np.array([density_at_time(t_) for t_ in ts])
dens_e = np.array(dens_e)

plt.figure()
plt.plot(ts,num_e,'x',markevery=20)
plt.plot(ts,dens_t_theory)
plt.xlabel('time t')
plt.ylabel('number of particles')

plt.figure()
plt.plot(ts,np.log(maxbond_f)/np.log(4))
plt.plot([0,ts[-1]],[L/2,L/2])
plt.xlabel('t')
plt.ylabel('log_{2**K} (D)')

plt.show()

