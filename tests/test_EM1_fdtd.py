import os, sys, time, glob
sys.path.append('../')


from setup.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell

"""
2D TM EM modes with staggered Yee cell
http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
"""

save_figs = False
save_data = False
load_data = False
restart   = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/EM-fdtd-TM/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    
    fdir = 'data/EM-fdtd-TM/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir',fdir)

#######################
# problem  parameters #
#######################

permittivity = 1./2500 # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor

matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability, is_cgs=True)
print('c', matl_config.c)

## initial conditions (cavity size)
Lx = 1.0
Ly = 0.5

E0 = 1.             # electric field magnitude
mx, my = 8, 5       # mode number
a = mx * np.pi/Lx
b = my * np.pi/Ly
Ez = lambda x, y: E0*np.sin(a*x)*np.sin(b*y)    # initial Ez

c = 1/np.sqrt(permittivity*permeability)    # speed of light
omega = c*np.sqrt(a**2 + b**2)              # frequency of oscillation
print('period', 2*np.pi/omega)


#######################
# run-time parameters #
#######################

q = 2           # size of physical bond dimension
L = 6           # length of tensor train per dimension
K = 2           # number of dimension
npts = q**L     # total number of grid points per dimension
TN_layout = LayoutType.PARALLEL_GROUP

cfl = 0.9       # time step size determined by CFL limit
order = 4       # order of finite difference stencil for spatial derivative (not used in FDTD)
T = 5.0 * 2 * np.pi / omega     # total simulation time
te_order = 31                   # specify time evolution scheme:  31 = FDTD, 4 = RK4.
do_adapt_dt = False             # flag to specify if time step dt is adapted according to CFL at each time-step

save_every_nt = 20      # save points (in units of time steps)

fstr1 = f'test{order}_TN{TN_layout.value}_Lx{Lx}_Ly{Ly}_mx{mx}_my{my}'  # str for naming saved files

#### define compression parameters ####
#### compress_style options
""" comp1:   compress final state
    comp2:   + compress intermediate rk4 states
    comp3:   + compress sum of derivatives
    comp4:   + compress derivatives
    comp5:   + compress during calculation of derivative

    mod0:  no modifications 
    mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
"""
DMAX = None             # maximum rank of TT (None means no maximum)
compress_style = 2      # see above
compress_style_mod = 0

## CompressionConfiguation holds all of the compression information
te_compress_E = CompressionConfiguration()
for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

te_compress_B = te_compress_E

print('te compress E', te_compress_E)


comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

fstr_comp = f'_C{compress_style}{compress_style_mod}_mb{DMAX}'

print('compress', DMAX, compress_style, compress_style_mod)

fstr1 += fstr_comp

#############################
## start building PDE system 
#############################

## define coordinates (to label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(0, Lx, npts,endpoint=False)
x1_vals = x_vals + (x_vals[1]-x_vals[0])/2
y_vals = np.linspace(0, Ly, npts,endpoint=False)
y1_vals = y_vals + (y_vals[1]-y_vals[0])/2

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)
## below: in theory one can build a not-quantized TT, though it's less developed
# ax_x = Axis(1, q**L, coordinate=X, xpts=x_vals)
# ax_y = Axis(1, q**L, coordinate=Y, xpts=y_vals)

## define a Cartesian coordinate system
coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## configure derivatives. DerivativeConfiguration contains information about derivatives
deriv_zg = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_zg1 = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER, offset=1)
deriv_zv1 = DerivativeConfiguration(left_bc=BCType.ANTISYMMETRIC, order=order, fd_type=FDType.CENTER, offset=1)

## note: endpoint is False:
# offset = 0:  0|0, 1 , 2 ..., n-1 | n , (n+1...)
# offset = 1:  (-1/2) | 1/2, 3/2, ..., n+1/2 | (n+3/2)


## define Grid object:  Grid1D means a TT geometry
grid_xy = Grid1D('XY', (ax_x, ax_y), layout_type=TN_layout)

## define initial grid1DTNs
data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
init_Ez  = Ez(data_x,data_y)    ## Ez would be staggered

init_Ez_gtn  = grid_xy.map_state_to_mps(init_Ez)    # map_state_to_mps converts np.ndarray to MPS
init_Bx_gtn  = grid_xy.make_empty_gridTN()
init_By_gtn  = grid_xy.make_empty_gridTN()

## set gridTN boundary conditions
init_Ez_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zv}     # i, j, k+1/2
init_Bx_gtn.ax_deriv_configs  = {ax_x: deriv_zv, ax_y: deriv_zg1}    # i, j+1/2, k+1/2
init_By_gtn.ax_deriv_configs  = {ax_x: deriv_zg1, ax_y: deriv_zv}    # i+1/2, j, k+1/2

print('init E', init_Ez_gtn.ax_deriv_configs[ax_x].bc)

## define Field object:  contains components for x, y, z components of the field.
init_E_field = Field('E', grid_xy, data={Z: init_Ez_gtn}, compress_config=te_compress_E)
init_B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y:init_By_gtn}, compress_config=te_compress_B)


### define time step ###
dx, dy = ax_x.dx, ax_y.dx
cfl_params = ((dx, dy), (c, c))
dt = cfl * cfl_limit( cfl_params )

num_tsteps = int(T/dt)
print('dt',dt, num_tsteps)
print('dx', dx, 'dy', dy)

if do_adapt_dt:
    fstr_t = f'cfl{cfl}_T{int(T):03d}'
else:
    fstr_t = f'dt{dt:1.3f}_T{int(T):03d}'

fstr = fstr1 + f'_L{L:02d}_{fstr_t}_te{te_order}' # '_m{mass_i}'
print(fstr)

try:
    ## load results if already finished
    if not load_data:  raise IOError
    E_t = optional_load(fdir + 'E_' + fstr + '.pkl')
    B_t = optional_load(fdir + 'B_' + fstr + '.pkl')
    max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb'))
    max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))
    max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
    max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
    nrg_fe_ts = pickle.load(open(fdir+'nrg_e_'+fstr+'.pkl','rb'))
    nrg_fi_ts = pickle.load(open(fdir+'nrg_i_'+fstr+'.pkl','rb'))
    nrg_E_ts  = pickle.load(open(fdir+'nrg_E_'+fstr+'.pkl','rb'))
    nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
    ts = pickle.load(open(fdir+'ts_'+fstr+'.pkl','rb'))

    nt = num_tsteps
    print('loaded data')

    E_data = grid_xy.map_mps_to_state(E_t) if E_t is not None else None
    B_data = grid_xy.map_mps_to_state(B_t) if B_t is not None else None

except(IOError,EOFError):

    ### initialize and start running the simulation ###

    try:
        ## try to load data from a previous save point
        if not restart:  raise ValueError
        re_fstr = fstr
        print(re_fstr)

        restart_fstrs = glob.glob(sdir+'restart/fe_'+re_fstr+'*.pkl')
        print(sdir+'restart/')
        print('restart?', len(restart_fstrs))
        nts = []
        for f in restart_fstrs:
            nt_ind = f.find('nt')
            nt_str = f[nt_ind+2:-4]
            nts += [int(nt_str)]
        nt = max(nts)
        print('nt',nt)

        E_t = optional_load(sdir + 'restart/E_' + re_fstr + f'-nt{nt}.pkl')
        B_t = optional_load(sdir + 'restart/B_' + re_fstr + f'-nt{nt}.pkl')
        max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb'))
        max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))
        print('loaded', nt, len(max_bond_E))
        nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
        nrg_B_ts = pickle.load(open(sdir + 'restart/nrg_B_' + re_fstr + f'.pkl', 'rb'))
        ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
        ts = list(ts)
        print('loaded',nt,len(nrg_fe_ts))

        max_bond_E = max_bond_E[:nt + 1]
        max_bond_B = max_bond_B[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        nrg_B_ts = nrg_B_ts[:nt + 1]
        ts = ts[:nt + 1]
        total_E0 = nrg_E_ts[0] + nrg_fi_ts[0] + nrg_fe_ts[0]

        ## technically would want to reconstruct X, Y, Z coordinates from saved data

        init_E_field.initialize_field(E_t)
        init_B_field.initialize_field(B_t)

        restarted = True
        print('loaded restart',nt)

    except(ValueError,FileNotFoundError):
        ## if no save point, use original initialization
        nt = 0
        ts = [0]
        restarted = False

        nrg_E_ts  = [np.sum(init_Ez * np.conj(init_Ez)) * dx * dy * 1. / 2]
        nrg_B_ts  = [0.0]

        max_bond_E = [init_Ez_gtn.max_bond()]
        max_bond_B = [(np.nan,np.nan)]


    ### BUILD PDE
    print('here?')
    em_sys = Maxwell(init_E_field, init_B_field,
                     coords_x=coord_sys, matl_params=matl_config, 
                     is_electrostatic=False, clean=False, is_yee=True, normalize=False,
                     te_order=te_order, compress_levels=comp_levels )

    if restarted:
        if do_adapt_dt:
            cfl_params = ((dx, dy), (c, c))
            dt = cfl * cfl_limit( cfl_params)
            print('restarted dt', dt)


    print('start')
    time1 = time.time()
    Exy_t = [init_E_field.get_comp_data(Z)]
    t0 = ts[-1]
    nt0 = nt

    # B_field = em_sys.B
    # divB = B_field.divergence(coord_sys=coord_sys)
    # divB_data = divB.get_comp_data()
    # print('divB_data', divB_data)
    # print('norm divB', np.linalg.norm(divB_data))

    while ts[-1] < T:

        walltime = time.time()
        em_sys = em_sys.next_time_step(dt, compress_level=1, is_first_time_step=(nt==0))
        print('wall time', time.time() - walltime)

        E_field = em_sys.E
        max_bond_E += [E_field[Z].max_bond()]   ## find maximum rank of QTT
        E_data = E_field.get_comp_data(Z)       ## get data (as an np.ndarray)
        Exy_t += [E_data]

        B_field = em_sys.B
        max_bond_B += [(B_field[X].max_bond(), B_field[Y].max_bond())]
        Bx_data = B_field.get_comp_data(X)
        By_data = B_field.get_comp_data(Y)

        ## measure energy of electric field
        nrg_E = np.sum(E_data * np.conj(E_data)) * dx * dy * 1. / 2
        nrg_E_ts += [nrg_E]

        ## measure energy of magnetic field
        nrg_Bx = np.sum(Bx_data * np.conj(Bx_data)) if Bx_data is not None else 0.0
        nrg_By = np.sum(By_data * np.conj(By_data)) if By_data is not None else 0.0
        nrg_B_ts += [(nrg_Bx + nrg_By) * dx * dy * 1. / 2 ]

        ts += [ts[-1]+dt]
        nt += 1

        ## update dt (not actually useful here)
        if do_adapt_dt:
            cfl_params = ((dx, dy), (c, c))
            dt = cfl * cfl_limit(cfl_params)

        ## total energy
        nrg_norm = nrg_E_ts[-1] + nrg_B_ts[-1]

        ## print out results
        print('result norm', nt, ts[-1], dt, nrg_E, nrg_norm,
              E_field[Z].max_bond(), B_field[X].max_bond(), B_field[Y].max_bond(), 2 ** L)


        divB = em_sys.get_divB()
        divB_data = divB.get_comp_data()
        print('norm divB', np.linalg.norm(divB_data))

        if nt % save_every_nt == 0:     ## routine for every few time steps (for plotting, etc)
            fig1,ax1 = plt.subplots()
            time_ = ts[-1]
            im1 = ax1.imshow(E_data.T, extent=(x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]),
                                aspect='auto', origin='lower')

            fig1.colorbar(im1, ax=ax1)
            ax1.set_title(f'time {time_:3.2f}')
            ax1.set_xlabel('x'), ax1.set_ylabel('y')
            if save_figs:   fig1.savefig(fdir+f'plot_E_'+fstr+f'_nt{nt}.png')

            plt.figure()
            plt.imshow(divB_data)
            plt.title('div(B) data')
            plt.colorbar()

            if not save_figs:
                plt.show()
            plt.close()

            if save_data:
                if not os.path.exists(sdir+'restart/'):
                    os.makedirs(sdir+'restart/')
                else:
                    try: 
                        raise NameError
                        restart_files = glob.glob(sdir+'restart/*'+fstr+f'-nt{save_nt}.pkl')
                        for f in restart_files:  os.remove(f)
                    except NameError:   pass

                save_nt = nt
                optional_dump(em_sys.E_func._components, sdir + 'restart/' + 'E_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.B._components, sdir + 'restart/' + 'B_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.phi._components, sdir + 'restart/' + 'phi_' + fstr + f'-nt{nt}.pkl')
                optional_dump(em_sys.psi._components, sdir + 'restart/' + 'psi_' + fstr + f'-nt{nt}.pkl')
                pickle.dump(max_bond_E, open(sdir + 'restart/' + 'maxD_E_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_B, open(sdir + 'restart/' + 'maxD_B_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_phi, open(sdir + 'restart/' + 'maxD_phi_' + fstr + '.pkl', 'wb'))
                pickle.dump(max_bond_psi, open(sdir + 'restart/' + 'maxD_psi_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_E_ts, open(sdir + 'restart/nrg_E_' + fstr + '.pkl', 'wb'))
                pickle.dump(nrg_B_ts, open(sdir + 'restart/nrg_B_' + fstr + '.pkl', 'wb'))
                pickle.dump(ts, open(sdir + 'restart/ts_' + fstr + '.pkl', 'wb'))


    if save_data:
        optional_dump(em_sys.E._components, fdir + 'E_' + fstr + '.pkl')
        optional_dump(em_sys.B._components, fdir + 'B_' + fstr + '.pkl')
        optional_dump(em_sys.phi._components, fdir + 'phi_' + fstr + '.pkl')
        optional_dump(em_sys.psi._components, fdir + 'psi_' + fstr + '.pkl')
        pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
        pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
        pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    print('wall time', time.time()-time1)

Exy_t = np.array(Exy_t)

fig4, ax4 = plt.subplots()
im4 = ax4.imshow(E_data)
ax4.set_xlabel('y')
ax4.set_ylabel('x')
fig4.colorbar(im4)

ts_ = ts[nt0:]

E_theory = lambda x,y,t: E0*np.sin(a*x)*np.sin(b*y)*np.cos(omega*t)
Exy_th = E_theory(data_x,data_y,ts[-1])
print('L1 diff E(x,y,t=T)',  np.sum(np.abs(Exy_th - Exy_t[-1]))/npts**2)
print('rms diff E(x,y,t=T)', np.linalg.norm(Exy_th - Exy_t[-1])/npts)

x_pt = np.argmax(np.max(init_Ez,axis=1))
y_pt = np.argmax(np.max(init_Ez,axis=0))
x_, y_ = x_vals[x_pt], y_vals[y_pt]
print('eval t at point',x_,y_)
E_th_data = E_theory(x_,y_,np.array(ts_))
print(E_th_data.shape)

fig2, ax2 = plt.subplots()
ax2.plot(ts_,Exy_t[:,x_pt,y_pt])
ax2.plot(ts_,E_th_data,'k:')
ax2.set_xlabel('t')
ax2.set_ylabel('E(x=X,y=Y)')

max_bond_B = np.array(max_bond_B)
max_bond_E = np.array(max_bond_E)

plt.show()
   
   
print('done')

