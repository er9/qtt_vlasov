import os, sys, pickle, time, glob
import scipy
sys.path.append('../')

from setup_.configs import *
from setup_.defaults import *
import setup_.helper as test_helper

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

""" test collision operators
"""

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV     # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 25 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0  # electron temperature [eV]
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth    # electron thermal speed
wp_e  = elc_config.wp     # plasma e' frequency
lamD  = elc_config.lamD   # Debye length
wp_i  = ion_config.wp     # plasma i+ frequency
vth_i = ion_config.vth    # ion thermal speed


######################
## useful functions ##
######################

def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    vth2 = eV*temp/mass
    return test_helper.maxwellian(*vs, vth2=vth2, density=density, flow=flow)


def dv_maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0, eV=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    for i in range(len(vs)):
        try:                            v2 += (vs[i]-flow[i])*(vs[i]-flow[i])
        except(IndexError,TypeError):   v2 += (vs[i]-flow)   *(vs[i]-flow)
    vth2  = eV*temp/mass
    grads = [ density*np.sqrt(1./2/np.pi/vth2) * np.exp(-v2/2/vth2) * (-v/vth2) for v in vs]
    return grads


def get_grid_params(idx):
    if idx == 0:
        x_bounds = -np.pi/k, np.pi/k
        ve_bounds = -6 * v0, 6 * v0
        vi_bounds = -50 * vth_i, 50 * vth_i
    else:
        raise IndexError(f'grid parameter for index {idx} not defined')
    return x_bounds, ve_bounds, vi_bounds


#####################
# define parameters #
#####################

### system parameters ###
A = 0.001
k = 0.75 / lamD
v0 = wp_e/k

### run-time parameters ###
q = 2
K = 1
Lx, Lve, Lvi = 7, 7, 7

upwind = True
TN_layout = LayoutType.SEQUENTIAL
ax_map_key = 'MF'
grid_idx = 0

T = 20     # total/final T (including load_T)
order = 1
te_order = 4
cfl = 0.9
do_adapt_dt = True

print('lambda D', lamD, 'k', k)


#################################
### define collisions ###
do_adapt_coll = True
coll_type = CollisionType.LB
coll_coeff_e = 0.5  ## to scale collision frequency if using adaptive collisions
coll_coeff_i = 0.5*wp_i
coll_rate_e = 0.5*wp_e if coll_type is not None else 0.0   ## rates if not using adaptive collisions
coll_rate_i = 0.5*wp_i if coll_type is not None else 0.0
if do_adapt_coll:
    collision_config = CollisionConfiguration(coll_type, v0_e=v0,
                                              coeff_e = coll_coeff_e, coeff_i = coll_coeff_i,
                                              rate_e=1.0, rate_i=1.0,)
else:
    collision_config = CollisionConfiguration(coll_type, v0_e=v0,
                                              coeff_e=1.0, coeff_i=1.0,
                                              rate_e=coll_rate_e, rate_i=coll_rate_i,)

print('collision config', collision_config)
if coll_type is None:
    fstr_coll = ''
else:
    if do_adapt_coll:
        fstr_coll = f'_{coll_type}{coll_coeff_e}'
    else:
        fstr_coll = f'_{coll_type}e{coll_rate_e}i{coll_rate_i}'


### compression options ###
"""
    comp1:   compress final state
    comp2:   + compress intermediate rk4 states
    comp3:   + compress sum of derivatives
    comp4:   + compress derivatives
    comp5:   + compress during calculation of derivative

    mod0:  no modifications 
    mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
"""
DMAX = None
compress_style = 1
compress_style_mod = 0

te_compress_i = CompressionConfiguration()
te_compress_e = CompressionConfiguration()
te_compress_V = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V',  te_compress_V)

comp_levels = list(range(1,6))
if compress_style_mod == 1:     # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

print('compress', DMAX, compress_style, compress_style_mod)

### compression of E before dot  ###
compress_E = True
DMAX_E = None
E_compress_opts = {'max_bond':DMAX_E,'cutoff':CUTOFF,'cutoff_mode':CUTOFF_MODE}


###############################
## start building PDE system ##
###############################

## initial conditions
fe = lambda x:  1 + A*np.cos(k*x)
fi = lambda x:  1 + 0*x
V_field  = lambda x: -A/k**2 * np.cos(k*x)
E_field  = lambda x: -A/k * np.sin(k*x)

## define grid
x_npts, ve_npts, vi_npts = q**Lx, q**Lve, q**Lvi
x_lims, ve_lims, vi_lims = get_grid_params(0)
tot_x = x_lims[1]-x_lims[0]
print('tot_x', tot_x, 2*np.pi/k)
x_vals  = np.linspace(*x_lims,x_npts,endpoint=False)
ve_vals = np.linspace(*ve_lims,ve_npts,endpoint=False)
vi_vals = np.linspace(*vi_lims,vi_npts,endpoint=False)

## define axis objects
X = Coordinate('X', CoordinateType.X)
V = Coordinate('V', CoordinateType.X)

map_x, map_v = test_helper.get_maps(ax_map_key)
ax_x  = Axis(Lx, q, coordinate=X, xpts=x_vals, ax_map=map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=vi_vals, ax_map=map_v)
ax_ve = Axis(Lve, q, coordinate=V, xpts=ve_vals, ax_map=map_v)
print('dx',ax_x.dx, 'dv e', ax_ve.dx, 'dv i', ax_vi.dx)


coords_x  = CartesianCoordinateSpace('X',ax_x)
coords_vi = CartesianCoordinateSpace('Vi',ax_vi)
coords_ve = CartesianCoordinateSpace('Ve',ax_ve)

## configure derivatives
deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)

## define grids
grid_i = Grid1D('XVi',(ax_x,ax_vi), layout_type=TN_layout)
grid_e = Grid1D('XVe',(ax_x,ax_ve), layout_type=TN_layout)
grid_V = Grid1D('X',(ax_x,), layout_type=TN_layout)


#### intiialize fields
fe_x = ax_x.map_state_to_mps(fe(x_vals))
fi_x = ax_x.map_state_to_mps(fi(x_vals))
fe_v = ax_ve.map_state_to_mps(maxwellian(ve_vals, density=n0_e, flow=v0,  mass=mass_e, temp=T_e))
fi_v = ax_vi.map_state_to_mps(maxwellian(vi_vals, density=n0_i, flow=0.0, mass=mass_i, temp=T_i))

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_V_gtn  = grid_V.map_state_to_mps(V_field(x_vals))

init_fe_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_ve: deriv_v}
init_fi_gtn.ax_deriv_configs = {ax_x: deriv_x, ax_vi: deriv_v}
init_V_gtn.ax_deriv_configs  = {ax_x: deriv_x}


f0_e = maxwellian(ve_vals,density=n0_e,flow=v0,mass=mass_e,temp=T_e)
f0_i = maxwellian(vi_vals,density=n0_i,flow=0.,mass=mass_i,temp=T_i)

vdrift_x_e = grid_e.make_mps_ndim({ax_ve: ax_ve.map_state_to_mps(ve_vals - v0)})
vdrift_x_i = grid_i.make_mps_ndim({ax_vi: ax_vi.map_state_to_mps(vi_vals)})


### define dt ###
dx, dv_e, dv_i = ax_x.dx, ax_ve.dx, ax_vi.dx
max_v_e = np.max(np.abs(ve_vals))
max_v_i = np.max(np.abs(vi_vals))
e_params = ((dx, dv_e), (max_v_e, (A/k)*eV/mass_e))
i_params = ((dx, dv_i), (max_v_i, (A/k)*eV/mass_i))

if not do_adapt_coll or coll_type is None:
    Ce_params = ((dv_e, dv_e**2/2), ((max_v_e+np.abs(v0))*coll_rate_e, (vth_e**2)*coll_rate_e))
    Ci_params = ((dv_i, dv_i**2/2), (max_v_i*coll_rate_i, (vth_i**2)*coll_rate_i))
    dt = cfl * test_helper.cfl_limit(e_params, i_params, Ce_params, Ci_params)
else:
    dt = cfl * test_helper.cfl_limit(e_params, i_params)
    coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, max_v_e + np.abs(v0), cfl)
    coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
    print('coll rate cfl', coll_rate_e, coll_rate_i)
    collision_config.base_rate_e = coll_rate_e
    collision_config.base_rate_i = coll_rate_i
    print('cfl limit scaled collision rates', collision_config.coll_rate_e, collision_config.coll_rate_i)

num_tsteps = int(T/dt)
print('dt',dt,num_tsteps)


nt = 0
ts = [0]
restarted = False

init_fe_data = init_fe_gtn.get_data()
init_fi_data = init_fi_gtn.get_data()
init_E = E_field(x_vals)

nrg_fe_ts = [np.sum(init_fe_data*(ve_vals**2))*dx*dv_e*mass_e/2 / tot_x]
nrg_fi_ts = [np.sum(init_fi_data*(vi_vals**2))*dx*dv_i*mass_i/2 / tot_x]
nrg_E_ts = [np.sum(init_E*np.conj(init_E))*dx*1./2 / tot_x]   # electric field energy

print('init nrgs', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])

norm_fe = np.sum(init_fe_data) * dx * dv_e
norm_fi = np.sum(init_fi_data) * dx * dv_i
print('norm', norm_fe, norm_fi)

print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())

init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
init_V_field = ScalarField('V', grid_V, data=init_V_gtn, compress_config=te_compress_V)

print('norm', init_fe_field.norm(), init_fi_field.norm())

## boundary conditions on V:
## V(x=edge) = 0
bc00 = np.array([0.]*(x_npts-1)+[1.]).reshape(1,x_npts) * 1./ax_x.dx**2

### BUILD PDE
vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field, coords_x=coords_x, coords_ve=coords_ve,
                       coords_vi=coords_vi, elc_params=elc_config, ion_params=ion_config, upwind=upwind,
                       potential_boundary_conditions=(bc00, None), te_order=te_order, compress_levels=comp_levels)
vp_sys.collision = collision_config
print('colls', collision_config.coll_params)

vp_sys.compress_F = compress_E
vp_sys.compress_F_opts = E_compress_opts
print('compress E?', compress_E, E_compress_opts)

vp_sys.v_from_v0_e = vdrift_x_e
vp_sys.v_from_v0_i = vdrift_x_i

dfdt_coll_e = vp_sys.get_collision_term()
dfdt_coll_i = vp_sys.get_collision_term(is_ion=True)
coll_e_data = dfdt_coll_e.get_comp_data()
coll_i_data = dfdt_coll_i.get_comp_data()

fig1,ax1 = plt.subplots(2,1)
im1 = ax1[0].imshow(coll_i_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                    aspect='auto', origin='lower')
im2 = ax1[1].imshow(coll_e_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title('collisions from maxwellian')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
plt.show()
plt.close(fig1)


#### add noise ####

fe_noise = np.random.random((x_npts,ve_npts)) * 0.3
fi_noise = np.random.random((x_npts,vi_npts)) * 0.3
fe_noise_mps = grid_e.map_state_to_mps(fe_noise)
fi_noise_mps = grid_i.map_state_to_mps(fi_noise)
vp_sys.fe.component.add(fe_noise_mps, inplace=True)
vp_sys.fi.component.add(fi_noise_mps, inplace=True)

init_fe_data = vp_sys.fe.get_comp_data()
init_fi_data = vp_sys.fi.get_comp_data()

fig1,ax1 = plt.subplots(2,1)
time_ = nt*dt
im1 = ax1[0].imshow(init_fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                    aspect='auto', origin='lower')
im2 = ax1[1].imshow(init_fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
plt.show()
plt.close(fig1)


time1 = time.time()

while ts[-1] < T:

    walltime = time.time()
    # vp_sys = vp_sys.next_time_step(dt, compress_level=1)  # , verbose_plot=(nt > 18 or nt<=5))
    dfdt_coll_e = vp_sys.get_collision_term()
    dfdt_coll_i = vp_sys.get_collision_term(is_ion=True)

    deriv0 = vp_sys.create_like(dfdt_coll_e, dfdt_coll_i)
    vp_sys = vp_sys.euler(dt, deriv0=deriv0, inplace=True)
    print('wall time', time.time() - walltime)

    pot_field = vp_sys.V
    elec_field = pot_field.gradient()
    E_data = elec_field.get_comp_data(X)

    fe_field = vp_sys.fe
    fe_data = fe_field.get_comp_data()

    fi_field = vp_sys.fi
    fi_data = fi_field.get_comp_data()

    nrg_fe_ts += [np.sum(fe_data * (ve_vals ** 2)) * dx * dv_e * mass_e / 2 / tot_x]
    nrg_fi_ts += [np.sum(fi_data * (vi_vals ** 2)) * dx * dv_i * mass_i / 2 / tot_x]
    energy_E_t = np.sum(E_data * np.conj(E_data)) * dx * 1. / 2 / tot_x
    nrg_E_ts += [energy_E_t]

    ts += [ts[-1] + dt]
    nt += 1

    if do_adapt_dt:
        e_params = ((dx, dv_e), (max_v_e, np.max(E_data)*eV/mass_e))  ## can use sqrt(sum(E**2))
        i_params = ((dx, dv_i), (max_v_i, np.max(E_data)*eV/mass_i))

        if not do_adapt_coll or coll_type is None:
            Ce_params = ((dv_e, dv_e**2/2), ((max_v_e+np.abs(v0))*coll_rate_e,
                                             (vth_e**2)*coll_rate_e))
            Ci_params = ((dv_i, dv_i**2/2), (max_v_i*coll_rate_i, (vth_i**2)*coll_rate_i))
            dt = cfl * test_helper.cfl_limit(e_params, i_params, Ce_params, Ci_params)
        else:
            dt = cfl * test_helper.cfl_limit(e_params, i_params)
            coll_rate_e = test_helper.LB_collision_freq_cfl_limit(dt, dv_e, vth_e, (max_v_e + np.abs(v0)), cfl)
            coll_rate_i = test_helper.LB_collision_freq_cfl_limit(dt, dv_i, vth_i, max_v_i, cfl)
            vp_sys.collision.base_rate_e = coll_rate_e
            vp_sys.collision.base_rate_i = coll_rate_i

    norm_e = vp_sys.fe.norm()
    norm_i = vp_sys.fi.norm()
    print('result', nt, ts[-1], dt, norm_e - norm_i, energy_E_t,
          pot_field.max_bond(), fe_field.max_bond(), 2 ** ((Lx + Lve) / 2))

    if nt%100 == 0:
        fig1,ax1 = plt.subplots(2,1)
        time_ = ts[-1]
        im1 = ax1[0].imshow(fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                            aspect='auto', origin='lower')
        im2 = ax1[1].imshow(fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')

        fig2, ax2 = plt.subplots(2, 1)
        time_ = ts[-1]
        im1 = ax2[0].plot(fi_data.T)
        im2 = ax2[1].plot(fe_data.T)

        ax2[0].set_title(f'time {time_:3.2f}')
        ax2[0].set_xlabel('x'), ax2[0].set_ylabel('fe')
        ax2[1].set_xlabel('x'), ax2[1].set_ylabel('fi')

        plt.show()



print('wall time', time.time()-time1)


fig1,ax1 = plt.subplots(2,1)
time_ = ts[-1]
vmax = max(np.max(fe_data),np.max(fi_data))
vmin = min(np.min(fe_data),np.min(fi_data))
im1 = ax1[0].imshow(fi_data.T, extent=(x_vals[0], x_vals[-1], vi_vals[0], vi_vals[-1]),
                    aspect='auto', origin='lower')
im2 = ax1[1].imshow(fe_data.T, extent=(x_vals[0], x_vals[-1], ve_vals[0], ve_vals[-1]),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:2.1f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')

fig2, ax2 = plt.subplots(2, 1)
im1 = ax2[0].plot(fi_data.T)
im2 = ax2[1].plot(fe_data.T)

ax2[0].set_title(f'time {time_:3.2f}')
ax2[0].set_xlabel('x'), ax2[0].set_ylabel('fe')
ax2[1].set_xlabel('x'), ax2[1].set_ylabel('fi')

plt.show()


