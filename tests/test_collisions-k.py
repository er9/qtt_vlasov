import os, sys, pickle, time, glob

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper

from axis import Axis
import axis_map
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

"""
Buneman instability (k-space, Hermite expansion)
"""

save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

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


def maxwellian(*vs, density=1.0, flow=0.0, mass=1.0, temp=1.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    vth2 = eV * temp / mass
    return helper_test.maxwellian(*vs, vth2=vth2, density=density, flow=flow)


def rk4_limit_FourierHermite(m_max, k_max, E_max=0):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 2.75 / (np.sqrt(2 * m_max) * (k_max + E_max))
    return dt_max


def get_maps(mkey: str):
    ax_maps = []
    for mk in mkey:
        if mk == 'F':   # forward
            ax_map = axis_map.BinaryMap()
        elif mk == 'B':     # back
            ax_map = axis_map.FlipBinaryMap()
        elif mk == 'M':
            ax_map = axis_map.MirrorMap()
        else:
            raise KeyError(f'{mk} not a valid map key')
        ax_maps += [ax_map]
    print('ax maps', ax_maps)
    return ax_maps


def get_fstr():
    """ get filename
    """
    ## parameters string
    fstr_params = f'k{k}_mr{mass_i}'

    ## collision string
    if coll_type is None:
        fstr_coll = ''
    else:
        if do_adapt_coll:
            fstr_coll = f'_{coll_type}{coll_coeff_e}'
        else:
            fstr_coll = f'_{coll_type}e{coll_rate_e}i{coll_rate_i}'

    ## compression string
    fstr_comp = f'_C{compress_style}{compress_style_mod}_mb{DMAX}'

    ## E compression string
    if compress_E and DMAX_E is not None:
        fstr_compE = f'_E{DMAX_E}'
    else:
        fstr_compE = ''

    ## grid string
    fstr_grid = f'test_TN{TN_layout.value}{ax_map_key}' + \
                f'_Ls{Lx:02d},{Lve:02d},{Lvi:02d}_'

    fstr_t = f'_dt{dt:1.3f}_T{int(T):03d}_te{te_order}'

    fstr1 = fstr_params + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t
    return fstr1


#####################
# define parameters #
#####################

### system parameters ###
A = 0.001
k = 0.75 / lamD
v0 = wp_e/k

## runtime parameter
q = 2
K = 1
Lx, Lve, Lvi = 4, 7, 7

TN_layout = LayoutType.SEQUENTIAL
ax_map_key = 'FF'

T = 20.0
# dt = 0.1  # 0.05
dt = rk4_limit_FourierHermite(q ** Lve - 1, k * (q ** Lx - 1), A / k) * 1.0
## *1.5:  the upper limit for k=0.75, T=15.0
print('dt rk4', dt)
te_order = 4

num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)


#############################
#### define collisions
do_adapt_coll = False
coll_type = CollisionType.LB
coll_coeff_e = 0.5  ## to scale collision frequency if using adaptive collisions
coll_coeff_i = 0.5 / np.sqrt(mass_i / mass_e)
coll_rate_e = 0.5 * wp_e if coll_type is not None else 0.0
coll_rate_i = 0.5 * wp_i if coll_type is not None else 0.0
collision_config = CollisionConfiguration(coll_type,
                                          coeff_e=coll_coeff_e, coeff_i=coll_coeff_i,
                                          rate_e=coll_rate_e, rate_i=coll_rate_i, v0_e=v0)

#### compression options
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

for ci in range(1, compress_style + 1):
    te_compress_i.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_e.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_V.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress V', te_compress_V)

comp_levels = list(range(1, 6))
if compress_style_mod == 1:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 4, 0]
elif compress_style_mod == 2:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 0, 0]

print('compress', DMAX, compress_style, compress_style_mod)

#### compression of E before dot
compress_E = True
DMAX_E = None
E_compress_opts = {'max_bond': DMAX_E, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}


#############################
## start building PDE system
#############################

## specify k values, hermite moments used
# dk = 0.10 -- lowest frequency mode
npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi
kmax = k * npts_x / 2
k_vals = np.linspace(-kmax, kmax, npts_x, endpoint=False)
me_vals = np.arange(0,npts_ve)
mi_vals = np.arange(0,npts_vi)

## initial conditions
## electron density:  1 + A * np.cos(k * x)
fe_x = np.zeros(npts_x, dtype=complex)
fe_x[npts_x // 2] = 1.0 * n0_e
fe_x[npts_x // 2 - 1] = A / 2
fe_x[npts_x // 2 + 1] = A / 2
fe_v = np.array([1.0/np.sqrt(2)/vth_e] + [0.] * (npts_ve - 1))

## ion density
fi_x = np.zeros(npts_x, dtype=complex)
fi_x[npts_x // 2] = 1.0 * n0_i
fi_v = np.array([1.0/np.sqrt(2)/vth_i] + [0.] * (npts_vi - 1))

## V field: -A / k ** 2 * np.cos(k * x)
V_x = np.zeros(npts_x, dtype=complex)
V_x[npts_x // 2 - 1] = -A / k ** 2 / 2
V_x[npts_x // 2 + 1] = -A / k ** 2 / 2

## E field: -A / k * np.sin(k * x)
E_x = np.zeros(npts_x, dtype=complex)
E_x[npts_x // 2 - 1] = -1.j * A / k / 2
E_x[npts_x // 2 + 1] = 1.j * A / k / 2


## define axis objects; position -> k-space,
X = Coordinate('KX', CoordinateType.X)
V = Coordinate('MV', CoordinateType.X)

map_x, map_v = get_maps(ax_map_key)
ax_x  = Axis(Lx, q, coordinate=X, dx=k, xpts=k_vals, basis=FourierBasis(), ax_map=map_x)
ax_vi = Axis(Lvi, q, coordinate=V, xpts=mi_vals, ax_map=map_v,
             # basis=AWHermiteGaussianBasis(x_scale=vth_i))
             basis=AWHermiteGaussianBasis(x_scale=np.sqrt(2) * vth_i))
ax_ve = Axis(Lve, q, coordinate=V, xpts=me_vals, ax_map=map_v,
             # basis=AWHermiteGaussianBasis(x_scale=vth_e, x_offset=v0))
             basis=AWHermiteGaussianBasis(x_scale=np.sqrt(2) * vth_e, x_offset=v0))


print('ax_x zero ind', ax_x.zero_ind)

coords_x  = CartesianCoordinateSpace('X', ax_x)
coords_vi = CartesianCoordinateSpace('Vi', ax_vi)
coords_ve = CartesianCoordinateSpace('Ve', ax_ve)

## define grids
grid_i = Grid1D('XVi', (ax_x, ax_vi), layout_type=TN_layout)
grid_e = Grid1D('XVe', (ax_x, ax_ve), layout_type=TN_layout)
grid_V = Grid1D('X', (ax_x,), layout_type=TN_layout)


## define initial grid1DTNs
fe_x = ax_x.map_state_to_mps(fe_x)
fi_x = ax_x.map_state_to_mps(fi_x)
fe_v = ax_ve.map_state_to_mps(fe_v)
fi_v = ax_vi.map_state_to_mps(fi_v)

init_fe_gtn = grid_e.make_mps_ndim({ax_x: fe_x, ax_ve: fe_v})
init_fi_gtn = grid_i.make_mps_ndim({ax_x: fi_x, ax_vi: fi_v})
init_V_gtn = grid_V.map_state_to_mps(V_x)


##############33
nt = 0
ts = [0]
restarted = False

# init_fe_data = init_fe_gtn.get_data()
# init_fi_data = init_fi_gtn.get_data()
# init_E = E_x
# print('mapped mps to vector')

v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

nrg_fe = init_fe_gtn.meas_expec(v2e_mpo) * mass_e / 2  * (k/2/np.pi)
nrg_fi = init_fi_gtn.meas_expec(v2i_mpo) * mass_i / 2  * (k/2/np.pi)
nrg_E  = (A/k/2)**2  * 2*np.pi/k  * (k/2/np.pi)

nrg_fe_ts = [nrg_fe]
nrg_fi_ts = [nrg_fi]
nrg_E_ts  = [nrg_E]
print('nrg', nrg_fe_ts[-1], nrg_fi_ts[-1], nrg_E_ts[-1])
# exit()

norm_fe = init_fe_gtn.norm()
norm_fi = init_fi_gtn.norm()
print('norm', norm_fe, norm_fi)


init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, compress_config=te_compress_i)
init_V_field  = ScalarField('V',  grid_V, data=init_V_gtn, compress_config=te_compress_V, is_sqrt=True)

print('norm', init_fe_field.norm(), init_fi_field.norm())

elec_field = init_V_field.gradient() * -1
energy_E_t = elec_field.norm() * 1. / 2   # electric field energy density
print('E field nrg before', energy_E_t )


### BUILD PDE
vp_sys = VlasovPoisson(init_fe_field, init_fi_field, init_V_field, coords_x=coords_x, coords_ve=coords_ve,
                       coords_vi=coords_vi, elc_params=elc_config, ion_params=ion_config, normalize=True,
                       te_order=te_order, compress_levels=comp_levels)
vp_sys.collision = collision_config
print('colls', collision_config.coll_params)

vp_sys.compress_F = compress_E
vp_sys.compress_F_opts = E_compress_opts
print('compress E?', compress_E, E_compress_opts)

# vp_sys.v_from_v0_e = grid_e.map_state_to_mps(ve_vals-0.0)
# vp_sys.v_from_v0_i = grid_i.map_state_to_mps(vi_vals-0.0)

dfdt_coll_e = vp_sys.get_collision_term(compress1=5, compress2=0)
dfdt_coll_i = vp_sys.get_collision_term(is_ion=True)
coll_e_data = dfdt_coll_e.get_comp_data()
coll_i_data = dfdt_coll_e.get_comp_data()

fig1, ax1 = plt.subplots(2, 1)
coll_i_real = ax_x.basis.get_realspace_1D(coll_i_data, 0, npts=512)
coll_i_real = ax_vi.basis.get_realspace_1D(coll_i_real, 1, -50 * vth_i, 50 * vth_i, npts=512)
im1 = ax1[0].imshow(np.real(coll_i_real).T, extent=(-np.pi/k, np.pi/k, -50 * vth_i, 50 * vth_i),
                    aspect='auto', origin='lower')

coll_e_real = ax_x.basis.get_realspace_1D(coll_e_data, 0, npts=512)
coll_e_real = ax_ve.basis.get_realspace_1D(coll_e_real, 1, -6 * v0, 6 * v0, npts=512)
im2 = ax1[1].imshow(np.real(coll_e_real).T, extent=(-np.pi/k, np.pi/k, -6 * v0, 6 * v0),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title('collisions from maxwellian')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
plt.show()


#################
### add noise ###
fe_noise = np.random.random((npts_x,npts_ve)) * 0.1
fi_noise = np.random.random((npts_x,npts_vi)) * 0.1
fe_noise_mps = grid_e.map_state_to_mps(fe_noise)
fi_noise_mps = grid_i.map_state_to_mps(fi_noise)
vp_sys.fe.component.add(fe_noise_mps, inplace=True)
vp_sys.fi.component.add(fi_noise_mps, inplace=True)

init_fe_data = vp_sys.fe.get_comp_data()
init_fi_data = vp_sys.fe.get_comp_data()

fig1, ax1 = plt.subplots(2, 1)
time_ = nt * dt
fi_real = ax_x.basis.get_realspace_1D(init_fi_data, 0, npts=512)
fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1, -50 * vth_i, 50 * vth_i, npts=512)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, -50 * vth_i, 50 * vth_i),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(init_fe_data, 0, npts=512)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1, -6 * v0, 6 * v0, npts=512)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, -6 * v0, 6 * v0),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')
# if save_figs:   fig1.savefig(fdir + f'plot_f_' + fstr + f'_nt{nt}.png')
# plt.close(fig1)

plt.show()



time1 = time.time()

## for measuring energy
v2e_mpo = grid_e.get_xmultiply_mpo(x_axes=[ax_ve], x_power=2)
v2i_mpo = grid_i.get_xmultiply_mpo(x_axes=[ax_vi], x_power=2)

while ts[-1] < T:

    walltime = time.time()
    # vp_sys = vp_sys.next_time_step(dt, compress_level=1) #, verbose_plot=(nt >= 40))
    dfdt_coll_e = vp_sys.get_collision_term(compress1=5, compress2=0)
    dfdt_coll_i = vp_sys.get_collision_term(is_ion=True)
    deriv0 = vp_sys.create_like(dfdt_coll_e, dfdt_coll_i)
    vp_sys = vp_sys.euler(dt, deriv0=deriv0, inplace=True)
    print('wall time', time.time() - walltime)

    pot_field = vp_sys.V
    elec_field = pot_field.gradient()
    elec_field.scalar_multiply(-1, inplace=True)
    elec_field.is_sqrt = True
    E_data = elec_field.get_comp_data(ax_x.coordinate)

    fe_field = vp_sys.fe
    fe_data = fe_field.get_comp_data()

    fi_field = vp_sys.fi
    fi_data = fi_field.get_comp_data()

    nrg_fe = fe_field.component.meas_expec(v2e_mpo) * mass_e / 2 * (k/2/np.pi)
    nrg_fi = fi_field.component.meas_expec(v2i_mpo) * mass_i / 2 * (k/2/np.pi)

    nrg_fe_ts += [nrg_fe]
    nrg_fi_ts += [nrg_fi]


    energy_E_t = elec_field.norm() * 1. / 2     # electric field energy
    nrg_E_ts += [energy_E_t]

    ts += [ts[-1] + dt]
    nt += 1

    norm = vp_sys.fe.norm() - vp_sys.fi.norm()
    print('result norm', nt, ts[-1], dt, norm, energy_E_t,
          pot_field.max_bond(), fe_field.max_bond(), Lx, Lvi, Lve)
    # if nt >= 5:   exit()

    if nt % 100 == 0:
        fig1, ax1 = plt.subplots(2, 1)
        time_ = ts[-1]
        fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
        fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1, -50 * vth_i, 50 * vth_i)
        im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi / k, np.pi / k, -50 * vth_i, 50 * vth_i),
                            aspect='auto', origin='lower')

        fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
        fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1, -6 * v0, 6 * v0)
        im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi / k, np.pi / k, -6 * v0, 6 * v0),
                            aspect='auto', origin='lower')

        fig1.colorbar(im1, ax=ax1[0])
        fig1.colorbar(im2, ax=ax1[1])
        ax1[0].set_title(f'time {time_:3.2f}')
        ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
        ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')

        fig2, ax2 = plt.subplots(2, 1)
        time_ = ts[-1]
        im1 = ax2[0].plot(fi_real.T)
        im2 = ax2[1].plot(fe_real.T)

        ax2[0].set_title(f'time {time_:3.2f}')
        ax2[0].set_xlabel('x'), ax2[0].set_ylabel('fe')
        ax2[1].set_xlabel('x'), ax2[1].set_ylabel('fi')

        plt.show()

print('wall time', time.time() - time1)

fig1, ax1 = plt.subplots(2, 1)
time_ = ts[-1]
fi_real = ax_x.basis.get_realspace_1D(fi_data, 0)
fi_real = ax_vi.basis.get_realspace_1D(fi_real, 1, -50 * vth_i, 50 * vth_i)
im1 = ax1[0].imshow(np.real(fi_real).T, extent=(-np.pi/k, np.pi/k, -50 * vth_i, 50 * vth_i),
                    aspect='auto', origin='lower')

fe_real = ax_x.basis.get_realspace_1D(fe_data, 0)
fe_real = ax_ve.basis.get_realspace_1D(fe_real, 1, -6 * v0, 6 * v0)
im2 = ax1[1].imshow(np.real(fe_real).T, extent=(-np.pi/k, np.pi/k, -6 * v0, 6 * v0),
                    aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1[0])
fig1.colorbar(im2, ax=ax1[1])
ax1[0].set_title(f'time {time_:3.2f}')
ax1[0].set_xlabel('x'), ax1[0].set_ylabel('v')
ax1[1].set_xlabel('x'), ax1[1].set_ylabel('v')

fig2, ax2 = plt.subplots(2, 1)
im1 = ax2[0].plot(fi_real.T)
im2 = ax2[1].plot(fe_real.T)

ax2[0].set_title(f'time {time_:3.2f}')
ax2[0].set_xlabel('x'), ax2[0].set_ylabel('fe')
ax2[1].set_xlabel('x'), ax2[1].set_ylabel('fi')

plt.show()