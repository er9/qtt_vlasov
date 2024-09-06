import os, sys, pickle, time, glob

import scipy.optimize

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import plot_defaults

# from defaults import *
from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
from basis.basis_spatial import SpatialBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField


"""
for reference, see https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
du/dt + u du/dx = 0
--> flux form:  du/dt + 1/s d/dx u**2 = 0

Note: finite difference with second order centered stencil (order 1) is not the same as finite volume
finite difference: du_i = 1/dx (F_{i+1} - F_{i-1})
finite volume: 1/dx (F(u_{i+1/2} - F(u_{i+1/2}))
F(u_{i+1/2}) = F((u_{i+1} + u_{i})/2)  != F(u_{i+1}) + F(u_{i})/2
F(u_{i-1/2}) = F((u_{i} + u_{i-1})/2)  != F(u_{i+1}) + F(u_{i})/2
"""




save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)
is_sqrt = True
is_darwin = False

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_pde_v2/data15-2/VM12-test/'
    # if not os.path.exists(sdir):
    #     os.makedirs(sdir)

    fdir = 'data20/Burgers/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)


extra_str = ''

#######################
# standard parameters #
#######################

nu = 0.0 # 0.001 # 0.01  # 0.01        ## viscosity
flux_coeff = 1.0  # 1.0 # 1.0        ## nonolinear flux coeff
Lbox = 1.0

# def init_state(x):
#     return np.sin(2 * np.pi * x / Lbox)

# # shock
def init_state(x):
    return (x <= 0.5) * 1.0

### rarefaction
# def init_state(x):
#     return (x > 0.5) * 1.0


## runtime parameter
q = 2
KX = 1
Lx = 9

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'

# TN_layout = LayoutType.PARALLEL  # _GROUP
# x_map_key = 'F'  # 'F'

order = 1   ## derivative order
if order > 1:
    extra_str += f'_o{order}'
T = 1.0   # 0.01  ## for diffusion
print('T', T)

te_order = 4 # 74, 4
dt_frac = 0.9
# dt = 0.003    ## te 4:  < 0.01,  te 74:  < 0.02,  te 70 ok but ringing instability
dt = 0.001 # 25
# dt = 0.001 / 16    ## for diffusion only
save_every_nt = 50  # 50   # 50 # 25000 # 1000 // 20 # // 5

DMAX = None   # 16  # None # 32 # None
DMAX_F = DMAX
comp_style = 3


### initialize problem
x_map = helper_test.get_maps(x_map_key)[0]
ax_x = Axis(Lx, q, dx=Lbox/(q**Lx), endpoint=False, ax_map=x_map, basis=SpatialBasis())
grid_X = Grid1D('GX', [ax_x], TN_layout)
coords_x = CartesianCoordinateSpace('X', ax_x)

print('ax map', ax_x.map)
print('cfl dt', ax_x.dx / 1.0)    ### shock speed = (flux left - flux right)/ax.dx
# exit()


compress_config = CompressionConfiguration()
for i in range(3):
    compress_config.set_compress_opts(i, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)


#############################
## start building PDE system
#############################

#### define initial gridTNs
init_u_gtn = grid_X.make_gridTN(init_state(ax_x.xpts))

# deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_x = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
init_u_gtn.ax_deriv_configs.update({ax_x: deriv_x})

### time step info
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = f'burgers1_dt{dt}_T{T}_nu{nu}_nl{flux_coeff}'
print('filename', fstr)


########################
#### run simulation ####
########################

#### define fields
init_u_field = ScalarField('u', grid_X, data=init_u_gtn, is_sqrt=False, compress_config=compress_config)

##### initialize tests
ts = [0]
errs_dist = [0.0]
us = [init_state(ax_x.xpts)]
nt = 0

print('init bond dims', init_u_gtn.max_bond())
max_bond_fe = [init_u_gtn.max_bond()]

norm_u = init_u_gtn.norm(is_sqrt=is_sqrt)  ## compute ||u||^2
print('field norm', norm_u)

from pde_burgers import Burgers, Burgers_FV

### BUILD PDE
bg_sys = Burgers(init_u_field, dissip_coeff=nu, flux_coeff=flux_coeff, power=2,
                    normalize=False, zipup=True, te_order=te_order,
                    upwind=False)
print('initialized Burgers')


fig1, ax1 = plt.subplots(1, 1)
time_ = nt * dt

u_data = init_u_gtn.get_data()
u_data = ax_x.basis.get_realspace_1D(u_data, 0)
ax1.plot(ax_x.xpts, np.real(u_data), label=f't={time_:2.2f}')
ax1.plot(ax_x.xpts, u_data**2, label='u2')
ax1.plot(ax_x.xpts[:-1], np.diff(u_data**2), label='deriv')
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('x'), ax1.set_ylabel('u')

plt.show()


time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    dt_ = (dt * 0.1) if nt == 0 else dt

    print('dt/dx', dt_/ax_x.dx, dt_/(ax_x.dx**2))

    ## go backwards in time
    if nt == 100:
        bg_sys.flux_coeff *= -1

    walltime = time.time()
    bg_sys = bg_sys.next_time_step(dt_, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=(nt==0)) #, filter_bases=False) # (nt + 1) % (save_every_nt * 4) == 0)
    # bg_sys.f.component.average_fine_scale(inplace=True, spread=1)
    print('wall time', time.time() - walltime)

    u_field: 'ScalarField' = bg_sys.f
    max_bond_fe += [u_field.component.max_bond()]

    ts += [ts[-1] + dt_]
    us += [u_field.get_field_data()]
    nt += 1

    norm_u = u_field.component.norm(is_sqrt=True)
    print('result norm', nt, ts[-1], dt, norm_u, u_field.max_bond(), Lx)

    # ### compute errors
    # theory_u_gtn = grid_X.make_gtn_from_dicts([{ax_x: fe1_vxy}],data_type=DataType.MPS)
    # err = theory_u_gtn.distance(u_field.component) / np.sqrt( ax_x.npts )
    # errs_dist += [err]

    if nt % save_every_nt == 0 or ts[-1] > T:

        fig1, ax1 = plt.subplots(1,1)
        time_ = ts[-1]

        ## shock location:
        # x = x0 + (u_l + u_r)/2 * t = 0.5 + 0.5 * t

        u_data = us[-1]
        ax1.plot(ax_x.xpts, u_data)
        ax1.plot(ax_x.xpts, init_u_gtn.get_data(), 'k:', linewidth=1.0) #, label='initial state')
        # ax1.plot([0.5, 0.5] , [-1.0, 1.0], 'k--')
        # ax1.plot([0.5 + 0.5 * ts[-1]]*2,[-0.1, 1.1], 'k--')
        ax1.plot([0.5, 0.5 + ts[-1]] , [-0., 1.0], 'k--')

        # ## diffusion only with step fct init state
        ## 0.5 might not be exact position of step...
        exact_fx = (scipy.special.erf((ax_x.xpts - 0.5)/2/np.sqrt(nu*ts[-1]/2))
                            * (init_state(ax_x.xpts[-1]) - init_state(ax_x.xpts[0]))/2
                            + (init_state(ax_x.xpts[-1]) + init_state(ax_x.xpts[0]))/2)
        # ax1.plot(ax_x.xpts, exact_fx, 'k--')
        # semi-infinite: L >= 4 sqrt(nu * t) --> t <= L**2/16 / nu
        print('box length', ax_x.xpts[-1] - ax_x.xpts[0], 4 * np.sqrt(nu * ts[-1]))
        print('dt, dx, err', dt, ax_x.dx, np.linalg.norm(exact_fx - u_data)/np.sqrt(ax_x.npts))
        print('err', dt, np.linalg.norm(exact_fx - u_data) / np.sqrt(ax_x.npts))

        ax1.set_title(f'time {time_:3.2f}')
        ax1.set_xlabel('x'), ax1.set_ylabel('u')
        # ax1.legend()
        plt.show()

        # plt.figure()
        # plt.semilogy(ts, errs_dist, label='dist')
        # plt.title('error')
        # plt.xlabel('t'), plt.ylabel('error')
        # plt.grid(color='gray',axis='y')
        # plt.legend()

        plt.show()

        # helper.partition_1D_mps(u_field.component.data, Lx - 3)
        # helper.partition_1D_mps(u_field.component.data, Lx - 2)

        ######## plot EE vs bond #####
        fig7, ax7 = plt.subplots()
        ax7.semilogy(bg_sys.f.component.entanglement_entropy_all(), '-x', label='u')
        ax7.legend()
        ax7.set_xlabel('bond number')
        ax7.set_ylabel('EE')
        plt.show()


print('wall time', time.time() - time1)


if save_data:
    pickle.dump(np.array([ts, errs_dist]),
                open(fdir + 'errs_' + fstr + '.pkl', 'wb'))
    print('saved', fdir + 'errs_' + fstr + '.pkl')


##############

fig7, ax7 = plt.subplots()
ax7.semilogy(ax_x.xpts, us[-1], label='u')

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
plt.show()


######## plot EE vs bond #####
fig7, ax7 = plt.subplots()
ax7.semilogy(bg_sys.f.component.entanglement_entropy_all(), '-x', label='elec')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()



print('done')

