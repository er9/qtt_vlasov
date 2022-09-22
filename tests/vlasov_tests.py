import os, sys, pickle, time, glob

sys.path.append('../')

from setup.configs import *
import setup.helper as helper_test
import helper_quimb as helper
import test_config as config

from axis import Axis
import axis_map
from basis.basis import Basis
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis, RealFourierBasis
from basis.basis_hermite import HermiteBasis
from basis.basis_hermite import HermiteGaussianBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from grid import Grid
    from coord.coord_sys import CoordinateSystem




class VlasovTest:

    def __init__(self,
                 Lxs: Sequence[int] = (),
                 Lves: Sequence[int] = (),
                 Lvis: Sequence[int] = (),
                 KX: int = 0, KV: int = 0, q: int = 2,
                 x_lims: Sequence[Sequence[int]] = (),
                 ve_lims: Sequence[Sequence[int]] = (),
                 vi_lims: Sequence[Sequence[int]] = (),
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis],Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis],Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis],Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 # units_config: UnitsConfiguration = None,
                 # mass_e=1.0, mass_ratio=1836, n0_e=1.0, n0_ratio=1.0, T_e=1.0, T_ratio=1.0,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt = True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout = LayoutType.SEQUENTIAL,
                 ):

        ## plasma parameters
        # self.ion_config, self.elc_config = self.initialize_params(units_config=units_config,
        #                                                           mass_e=mass_e, mass_ratio=mass_ratio,
        #                                                           n0_e=n0_e, n0_ratio=n0_ratio,
        #                                                           T_e=T_e, T_ratio=T_ratio)

        self.ion_config = IonConfiguration() if ion_config is None else ion_config
        self.elc_config = ElcConfiguration() if elc_config is None else elc_config

        self.vth_e = self.elc_config.vth  # electron thermal speed
        self.wp_e  = self.elc_config.wp  # plasma e' frequency
        self.lamD  = self.elc_config.lamD  # Debye length
        self.wp_i  = self.ion_config.wp  # plasma i+ frequency
        self.vth_i = self.ion_config.vth  # ion thermal speed

        ##############################
        assert(len(Lxs) == KX), 'need to specify lengths of all x dimensions'
        assert(len(Lves) == KV), 'need to specify lengths of all ve dimensions'
        assert(len(Lvis) == KV), 'need to specify lengths of all ve dimensions'

        self.Lxs = Lxs
        self.Lves = Lves
        self.Lvis = Lvis
        self.KX = KX
        self.KV = KV
        self.q = q

        #############################
        ### define collisions
        self.do_adapt_coll = do_adapt_coll
        self._collision_config = None
        self.coll_type = coll_type
        self.coll_coeff_e = coll_coeff_e
        self.coll_coeff_i = coll_coeff_i
        self.coll_rate_e = coll_rate_e
        self.coll_rate_i = coll_rate_i
        # self.initialize_collision_config(coll_type=coll_type,
        #                                  coll_coeff_e=coll_coeff_e, coll_coeff_i=coll_coeff_i,
        #                                  coll_rate_e=coll_rate_e, coll_rate_i=coll_rate_i)

        #############################
        self.DMAX = DMAX
        self.compress_style = compress_style
        self.compress_style_mod = compress_style_mod

        ### compression of E before dot
        self.compress_F = compress_F
        self.compress_F_opts = {'max_bond': DMAX_F, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}

        ### time step
        self.do_adapt_dt = do_adapt_dt
        self.dt = dt
        self.dt_frac = dt_frac
        self.T = T
        self.te_order = te_order


        ### grid info
        self.grid_layout = grid_layout

        ## initialize axes, grid if available
        if len(x_lims) == 0:

            self.x_lims = x_lims
            self.ve_lims = ve_lims
            self.vi_lims = vi_lims

            self.pos_axes: Optional[list['Axis']] = None
            self.vi_axes: Optional[list['Axis']] = None
            self.ve_axes: Optional[list['Axis']] = None
            self.grid_e: Optional['Grid'] = None
            self.grid_i: Optional['Grid'] = None
            self.grid_X: Optional['Grid'] = None

            self.pos_coordsys: Optional['CoordinateSystem'] = None
            self.vi_coordsys: Optional['CoordinateSystem'] = None
            self.ve_coordsys: Optional['CoordinateSystem'] = None

            ### axis info
            self.x_map_key = None
            self.ve_map_key = None
            self.vi_map_key = None

            # if vi_map_key is None:
            #     self.ax_map_key = x_map_key + ve_map_key
            # else:
            #     self.ax_map_key = x_map_key + ve_map_key + vi_map_key

        else:
            self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                                 x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                                 basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                                 pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                                 )

            self.initialize_grid()


    @property
    def npts_x(self):
        return [self.q**Lx for Lx in self.Lxs]

    @property
    def npts_ve(self):
        return [self.q ** Lve for Lve in self.Lves]

    @property
    def npts_vi(self):
        return [self.q ** Lvi for Lvi in self.Lvis]

    @property
    def collision_config(self):
        if self._collision_config is None:
            self.initialize_collision_config(coll_type=self.coll_type,
                                             coll_coeff_e=self.coll_coeff_e, coll_coeff_i=self.coll_coeff_i,
                                             coll_rate_e=self.coll_rate_e, coll_rate_i=self.coll_rate_i)
        return self._collision_config

    def get_npts_x(self, dim):
        return self.pos_axes[dim].npts

    def get_npts_ve(self, dim):
        return self.ve_axes[dim].npts

    def get_npts_vi(self, dim):
        return self.vi_axes[dim].npts

    def get_dx(self, dim):
        return self.pos_axes[dim].dx

    def get_dve(self, dim):
        return self.ve_axes[dim].dx

    def get_dvi(self, dim):
        return self.vi_axes[dim].dx

    def get_kmax(self, dim):
        if self.is_k(dim):
            ax = self.pos_axes[dim]
            return ax.npts * ax.dx

    def get_mmax(self, dim):
        if self.is_m(dim):
            ax_e = self.ve_axes[dim]
            ax_i = self.vi_axes[dim]
            return max(ax_e.npts, ax_i.npts)

    def is_real_k(self, dim):
        return isinstance(self.pos_axes[dim].basis, RealFourierBasis)

    def is_k(self, dim):
        return isinstance(self.pos_axes[dim].basis, FourierBasis)

    def is_m(self, dim, is_ion=False):
        ax = self.vi_axes[dim] if is_ion else self.ve_axes[dim]
        return isinstance(ax.basis, HermiteBasis)

    def get_fstr_grid(self):
        """ get filename part describing grid
        """

        ## basis string
        basis_str_x, basis_str_v = '', ''
        for dim in range(self.KX):
            if self.is_k(dim):
                basis_str_x += 'k'
            elif self.is_real_k(dim):
                basis_str_x += 'r'
            else:
                basis_str_x += 'x'
        if all([str_x == basis_str_x[0] for str_x in basis_str_x]):
            basis_str_x = basis_str_x[0]

        for dim in range(self.KV):
            if self.is_m(dim):
                basis_str_v += 'm'
            else:
                basis_str_v += 'v'
        if all([str_v == basis_str_v[0] for str_v in basis_str_v]):
            basis_str_v = basis_str_v[0]

        fstr_basis = '_' + basis_str_x + basis_str_v

        ## grid string
        if self.vi_map_key is None:
            ax_map_key = self.x_map_key + self.ve_map_key
        else:
            ax_map_key = self.x_map_key + self.ve_map_key + self.vi_map_key

        fstr_grid = f'_TN{self.grid_layout.value}{ax_map_key}'
        for dim in range(self.KX):
            fstr_grid += f'_LX{self.Lxs[dim]:02d}'
        for dim in range(self.KV):
            fstr_grid += f'_LVE{self.Lves[dim]:02d}'
        for dim in range(self.KV):
            fstr_grid += f'_LVI{self.Lvis[dim]:02d}'

        fstr1 = fstr_basis + fstr_grid 
        return fstr1


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        ## parameters string
        fstr_params = f'Tr{self.ion_config.T/self.elc_config.T:1.2f}_mr{self.ion_config.mass/self.elc_config.mass:2.1f}'
        # fstr_params = f'Tr{self.ion_config.T/self.elc_config.T}_mr{self.ion_config.mass/self.elc_config.mass}'

        ## basis string
        basis_str_x, basis_str_v = '', ''
        for dim in range(self.KX):
            if self.is_k(dim):
                basis_str_x += 'k'
            elif self.is_real_k(dim):
                basis_str_x += 'r'
            else:
                basis_str_x += 'x'
        if all([str_x == basis_str_x[0] for str_x in basis_str_x]):
            basis_str_x = basis_str_x[0]

        for dim in range(self.KV):
            if self.is_m(dim):
                basis_str_v += 'm'
            else:
                basis_str_v += 'v'
        if all([str_v == basis_str_v[0] for str_v in basis_str_v]):
            basis_str_v = basis_str_v[0]

        fstr_basis = '_' + basis_str_x + basis_str_v


        ## collision string
        coll_type = self.collision_config.coll_type
        if coll_type is None:
            fstr_coll = ''
        else:
            if self.do_adapt_coll:
                # fstr_coll = f'_{coll_type}Ae{self.collision_config.coeff_e:1.3e}' + \
                fstr_coll = f'_{coll_type.value}Ae{self.collision_config.coeff_e:1.3e}' + \
                            f'i{self.collision_config.coeff_i:1.3e}'
            else:
                # fstr_coll = f'_{coll_type}Be{self.collision_config.coll_rate_e:1.3e}' + \
                fstr_coll = f'_{coll_type.value}Be{self.collision_config.coll_rate_e:1.3e}' + \
                            f'i{self.collision_config.coll_rate_i:1.3e}'

        ## compression string
        fstr_comp = f'_C{self.compress_style}{self.compress_style_mod}_mb{self.DMAX}'

        ## E compression string
        DMAX_F = self.compress_F_opts['max_bond']
        if self.compress_F and DMAX_F is not None:
            fstr_compE = f'_E{DMAX_F}'
        else:
            fstr_compE = ''

        ## dt string
        if self.do_adapt_dt:
            fstr_t = f'_dtf{self.dt_frac:1.1f}_T{self.T}_te{self.te_order}'
        else:
            if self.dt is None:
                fstr_t = f'_dtf0{self.dt_frac:1.3f}_T{self.T}_te{self.te_order}'
            else:
                fstr_t = f'_dt{self.dt:1.3f}_T{self.T}_te{self.te_order}'

        ## grid string
        if self.vi_map_key is None:
            ax_map_key = self.x_map_key + self.ve_map_key
        else:
            ax_map_key = self.x_map_key + self.ve_map_key + self.vi_map_key

        fstr_grid = f'_TN{self.grid_layout.value}{ax_map_key}'
        for dim in range(self.KX):
            fstr_grid += f'_LX{self.Lxs[dim]:02d}'
        for dim in range(self.KV):
            fstr_grid += f'_LVE{self.Lves[dim]:02d}'
        for dim in range(self.KV):
            fstr_grid += f'_LVI{self.Lvis[dim]:02d}'

        fstr1 = fstr_params + fstr_basis + fstr_comp + fstr_compE + fstr_coll + fstr_grid + fstr_t + extra_str
        return fstr1


    def initialize_params(self, units_config: UnitsConfiguration = None,
                          mass_e=1.0, mass_ratio=1836,
                          n0_e=1.0, n0_ratio=1.0,
                          T_e=1.0, T_ratio=1.0
                          ):

        plasma_config = UnitsConfiguration() if units_config is None else units_config

        mass_i = mass_ratio * mass_e  # mass of ion
        n0_i = n0_ratio * n0_e  # ion number density
        T_i = T_ratio * T_e  # ion temperature [eV]

        ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
        elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
        return ion_config, elc_config


    def get_grid_params(self, dim, grid_idx, is_k=False, is_real_k=False, is_m_e=False, is_m_i=False):
        raise NotImplementedError


    def initialize_axes(self,
                        x_lims: Union[Sequence[int], Sequence[Sequence[int]]] = (0,1),
                        ve_lims: Union[Sequence[int], Sequence[Sequence[int]]] = (-1,1),
                        vi_lims: Union[Sequence[int], Sequence[Sequence[int]]] = (-1,1),
                        x_map_key='F', ve_map_key='F', vi_map_key=None,
                        basis_xs: Union[Basis,Sequence[Basis]] = SpatialBasis(),
                        basis_ves: Union[Basis,Sequence[Basis]] = SpatialBasis(),
                        basis_vis: Optional[Union[Basis,Sequence[Basis]]] = None,
                        pos_coordsys: Optional['CoordinateSystem'] = None,
                        ve_coordsys: Optional['CoordinateSystem'] = None,
                        vi_coordsys: Optional['CoordinateSystem'] = None,
                        ):

        q = self.q
        KX = self.KX
        KV = self.KV
        Lxs, Lves, Lvis = self.Lxs, self.Lves, self.Lvis

        basis_vis = basis_ves if basis_vis is None else basis_vis

        self.x_map_key = x_map_key
        self.ve_map_key = ve_map_key
        self.vi_map_key = vi_map_key
        vi_map_key = ve_map_key if vi_map_key is None else vi_map_key

        if pos_coordsys is None:
            X = Coordinate('X', CoordinateType.X)
            Y = Coordinate('Y', CoordinateType.Y)
            Z = Coordinate('Z', CoordinateType.Z)
            pos_coordsys = CartesianCoordinateSpace('X', coords=[X, Y, Z])
        self.pos_coordsys = pos_coordsys

        if ve_coordsys is None:
            VX = Coordinate('VX', CoordinateType.X)
            VY = Coordinate('VY', CoordinateType.Y)
            VZ = Coordinate('VZ', CoordinateType.Z)
            ve_coordsys = CartesianCoordinateSpace('VE', coords=[VX, VY, VZ])
        self.ve_coordsys = ve_coordsys

        if vi_coordsys is None:
            vi_coordsys = CartesianCoordinateSpace('VI', coords=ve_coordsys.coords)
        self.vi_coordsys = vi_coordsys

        #############################
        ## start building PDE system
        #############################

        ## specify k values, hermite moments used
        x_axes, ve_axes, vi_axes = (), (), ()
        x_maps = helper_test.get_maps(x_map_key)
        ve_maps = helper_test.get_maps(ve_map_key)
        vi_maps = helper_test.get_maps(vi_map_key)

        print('initialize axes')

        for dim in range(KX):

            Lx = Lxs[dim] if isinstance(Lxs, (list,tuple)) else Lxs
            basis_x = basis_xs[dim] if isinstance(basis_xs, (list,tuple)) else basis_xs
            x_map = x_maps[dim] if len(x_maps) == KX else x_maps[0]
            x_lim = x_lims[dim] if isinstance(x_lims[dim], (list,tuple)) else x_lims
            x_coord = self.pos_coordsys.coords[dim]
            npts_x = q ** Lx

            x_vals = np.linspace(*x_lim, npts_x, endpoint=False)
            ax_x = Axis(Lx, q, coordinate=x_coord, xpts=x_vals, basis=basis_x, ax_map=x_map)
            x_axes = x_axes + (ax_x,)

        for dim in range(KV):
            Lve = Lves[dim] if isinstance(Lves,(list,tuple)) else Lves
            Lvi = Lvis[dim] if isinstance(Lvis,(list,tuple)) else Lvis

            basis_ve = basis_ves[dim] if isinstance(basis_ves, (list,tuple)) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, (list,tuple)) else basis_vis

            ve_map = ve_maps[dim] if len(ve_maps) == KV else ve_maps[0]
            vi_map = vi_maps[dim] if len(vi_maps) == KV else vi_maps[0]

            ve_lim = ve_lims[dim] if isinstance(ve_lims[dim], (list,tuple)) else ve_lims
            vi_lim = vi_lims[dim] if isinstance(vi_lims[dim], (list,tuple)) else vi_lims

            ve_coord = self.ve_coordsys.coords[dim]
            vi_coord = self.vi_coordsys.coords[dim]

            npts_ve, npts_vi =  q ** Lve, q ** Lvi
            if basis_ve.type == BasisType.SPATIAL:
                ve_vals = np.linspace(*ve_lim, npts_ve, endpoint=True)
            else:
                ve_vals = np.linspace(*ve_lim, npts_ve, endpoint=False)

            if basis_vi.type == BasisType.SPATIAL:
                vi_vals = np.linspace(*vi_lim, npts_vi, endpoint=True)
            else:
                vi_vals = np.linspace(*vi_lim, npts_vi, endpoint=False)

            ## define axis objects; position -> k-space,
            ax_ve = Axis(Lve, q, coordinate=ve_coord, xpts=ve_vals, basis=basis_ve, ax_map=ve_map)
            ax_vi = Axis(Lvi, q, coordinate=vi_coord, xpts=vi_vals, basis=basis_vi, ax_map=vi_map)

            ve_axes = ve_axes + (ax_ve,)
            vi_axes = vi_axes + (ax_vi,)

        self.x_lims = x_lims
        self.ve_lims = ve_lims
        self.vi_lims = vi_lims

        self.pos_axes = x_axes
        self.ve_axes = ve_axes
        self.vi_axes = vi_axes

        self.vi_coordsys.add_axes(self.vi_axes)
        self.ve_coordsys.add_axes(self.ve_axes)
        self.pos_coordsys.add_axes(self.pos_axes)

        return x_axes, vi_axes, ve_axes



    def initialize_collision_config(self, coll_type=None,
                                    coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                                    v0_e=0.0, v0_i=0.0):
        """
        coll_type:  None, CollisionType.LB, ...
        coll_coeff_e/i: to scale max collision frequency if using adaptive collisions
        coll_rate_e/i:  collision rate
        unless otherwise specif coll_rate_i/coll_rate_e = wp_i/wp_e
        """
        # coll_coeff_i = coll_coeff_e * self.wp_i / self.wp_e * (self.elc_config.T/self.ion_config.T)**1.5 \
        #     if coll_coeff_i is None else coll_coeff_i
        # coll_rate_i = coll_rate_e * self.wp_i / self.wp_e * (self.elc_config.T/self.ion_config.T)**1.5 \
        #     if coll_rate_i is None else coll_rate_i

        coll_coeff_i = coll_coeff_e * self.wp_i / self.wp_e * (self.vth_i / self.vth_e) ** (self.KV/2) \
            if coll_coeff_i is None else coll_coeff_i
        coll_rate_i = coll_rate_e * self.wp_i / self.wp_e * (self.vth_i / self.vth_e) ** (self.KV/2) \
            if coll_rate_i is None else coll_rate_i

        if self.do_adapt_coll and coll_type is not None:
            rates_e, rates_i = [], []
            for ind in range(self.KV):
                ax_ve = self.ve_axes[ind]
                ax_vi = self.vi_axes[ind]

                dv_e = ax_ve.dx
                dv_i = ax_vi.dx

                ve_max = np.max(np.abs(ax_ve.xpts))
                vi_max = np.max(np.abs(ax_vi.xpts))

                rate_e = LB_collision_freq_cfl_limit(self.dt, dv_e, self.vth_e, ve_max, self.dt_frac)
                rate_i = LB_collision_freq_cfl_limit(self.dt, dv_i, self.vth_i, vi_max, self.dt_frac)

                rates_e += [rate_e]
                rates_i += [rate_i]

            collision_config = CollisionConfiguration(coll_type, v0_e=v0_e, v0_i=v0_i,
                                                      coeff_e=coll_coeff_e, coeff_i=coll_coeff_i,
                                                      rate_e=np.array(rates_e), rate_i=np.array(rates_i), )
        else:
            collision_config = CollisionConfiguration(coll_type, v0_e=v0_e, v0_i=v0_i,
                                                      coeff_e=1.0, coeff_i=1.0,
                                                      rate_e=coll_rate_e, rate_i=coll_rate_i, )

        self._collision_config = collision_config
        return collision_config


    def update_collision_config(self):
        """
        coll_type:  None, CollisionType.LB, ...
        coll_coeff_e/i: to scale max collision frequency if using adaptive collisions
        coll_rate_e/i:  collision rate
        unless otherwise specif coll_rate_i/coll_rate_e = wp_i/wp_e
        """
        if self.do_adapt_coll and self._collision_config.coll_type is not None:
            rates_e, rates_i = [], []
            for ind in range(self.KV):
                ax_ve = self.ve_axes[ind]
                ax_vi = self.vi_axes[ind]

                dv_e = ax_ve.dx
                dv_i = ax_vi.dx

                ve_max = np.max(np.abs(ax_ve.xpts))
                vi_max = np.max(np.abs(ax_vi.xpts))

                rate_e = LB_collision_freq_cfl_limit(self.dt, dv_e, self.vth_e, ve_max, self.dt_frac)
                rate_i = LB_collision_freq_cfl_limit(self.dt, dv_i, self.vth_i, vi_max, self.dt_frac)

                rates_e += [rate_e]
                rates_i += [rate_i]

            self._collision_config.rate_e = np.array(rates_e)
            self._collision_config.rate_i = np.array(rates_i)

        return self._collision_config


    def get_compression_config(self, DMAX=None):
        """ Compression configuration
            comp1:   compress final state
            comp2:   + compress intermediate rk4 states
            comp3:   + compress sum of derivatives
            comp4:   + compress derivatives
            comp5:   + compress during calculation of derivative
        """
        compress_config = CompressionConfiguration()
        for ci in range(1, self.compress_style + 1):
            compress_config.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
        return compress_config


    def get_compress_levels(self):
        """ compress_levels (with modifications)
            mod0:  no modifications
            mod1:  comp5 = 0 (don't actively compress in interior of calculating derivative)
        """
        comp_levels = list(range(1, 6))
        if self.compress_style_mod == 1:  # don't compress inside the derivative
            comp_levels = [1, 2, 3, 4, 0]
        elif self.compress_style_mod == 2:  # don't compress inside the derivative
            comp_levels = [1, 2, 3, 0, 0]
        return comp_levels


    def get_dt(self, Fe_maxs: Sequence[Numeric], Fi_maxs: Sequence[Numeric]=None, kmaxs: Sequence[Numeric]=None):
        """
        """
        dt_frac = self.dt_frac
        dxs_e, coeffs_e = [], []
        dxs_i, coeffs_i = [], []

        if Fi_maxs is None:
            Fi_maxs = [0.0] * len(Fe_maxs)

        dt_spec = np.inf
        for dim in range(self.KV):
            v_max_e, v_max_i = np.max(np.abs(self.ve_lims[dim])), np.max(np.abs(self.vi_lims[dim]))

            if (dim < self.KX and self.is_k(dim)) and self.is_m(dim):
                dt_tmp = rk4_limit_FourierHermite(self.get_mmax(dim), self.get_kmax(dim), Fe_maxs[dim])
                dt_spec = min(dt_tmp, dt_spec)
            else:
                if dim < self.KX:
                    if self.is_k(dim):
                        kmax = self.get_kmax(dim) if kmaxs is None else kmaxs[dim]
                        # print('get dt kmax', kmax)
                        dt_tmp = rk4_limit_Fourier(kmax, max(v_max_e, v_max_i))
                        dt_spec = min(dt_tmp, dt_spec)
                    else:
                        dxs_e += [self.get_dx(dim)]
                        dxs_i += [self.get_dx(dim)]
                        coeffs_e += [v_max_e]
                        coeffs_i += [v_max_i]

                if dim < len(Fe_maxs):  # o.w. assume is 0
                    if self.is_m(dim):
                        dt_tmp = rk4_limit_Hermite(self.get_mmax(dim), self.get_dx(dim), Fe_maxs[dim])
                        dt_spec = min(dt_tmp, dt_spec)
                    else:
                        dxs_e += [self.get_dve(dim)]
                        dxs_i += [self.get_dvi(dim)]
                        coeffs_e += [Fe_maxs[dim]]
                        coeffs_i += [Fi_maxs[dim]]

        e_params = (tuple(dxs_e), tuple(coeffs_e))
        i_params = (tuple(dxs_i), tuple(coeffs_i))

        if not self.do_adapt_coll or self.collision_config.coll_type is None:
            dxs_e, coeffs_e = [], []
            dxs_i, coeffs_i = [], []

            for dim in range(self.KV):
                v_max_e = np.max(np.abs(self.ve_lims[dim])) + np.abs(self.collision_config.get_v0_e(dim))
                v_max_i = np.max(np.abs(self.vi_lims[dim])) + np.abs(self.collision_config.get_v0_i(dim))
                dv_e, dv_i = self.get_dve(dim), self.get_dvi(dim)

                dxs_e += [dv_e, dv_e ** 2 / 2]
                dxs_i += [dv_i, dv_i ** 2 / 2]
                coeffs_e += [v_max_e * self.collision_config.coll_rate_e,
                             (self.vth_e ** 2) * self.collision_config.coll_rate_e]
                coeffs_i += [v_max_i * self._collision_config.coll_rate_i,
                             (self.vth_i ** 2) * self.collision_config.coll_rate_i]

            Ce_params = (tuple(dxs_e), tuple(coeffs_e))
            Ci_params = (tuple(dxs_i), tuple(coeffs_i))

            dt_cfl = cfl_limit(e_params, i_params, Ce_params, Ci_params)
            dt = min(dt_spec, dt_cfl) * dt_frac

        else:
            dt_cfl = dt_frac * cfl_limit(e_params, i_params)
            dt = min(dt_spec, dt_cfl) * dt_frac

            coll_rate_e, coll_rate_i = [], []
            for dim in range(self.KV):
                v_max_e = np.max(np.abs(self.ve_lims[dim])) + np.abs(self.collision_config.get_v0_e(dim))
                v_max_i = np.max(np.abs(self.vi_lims[dim])) + np.abs(self.collision_config.get_v0_i(dim))
                dv_e, dv_i = self.get_dve(dim), self.get_dvi(dim)

                coll_rate_e += [helper_test.LB_collision_freq_cfl_limit(dt, dv_e, self.vth_e, v_max_e, dt_frac)]
                coll_rate_i += [helper_test.LB_collision_freq_cfl_limit(dt, dv_i, self.vth_i, v_max_i, dt_frac)]

            print('coll rate cfl', coll_rate_e, coll_rate_i)
            self.collision_config.base_rate_e = coll_rate_e
            self.collision_config.base_rate_i = coll_rate_i
            print('cfl limit scaled collision rates', self.collision_config.coll_rate_e, self._collision_config.coll_rate_i)

        return dt

    def initialize_grid(self):

        if self.grid_layout == LayoutType.COMB:
            grid_x_IDs = ['X', 'Y', 'Z']
            grid_vi_IDs = ['Vxi', 'Vyi', 'Vzi']
            grid_ve_IDs = ['Vxe', 'Vye', 'Vze']

            x_grids, ve_grids, vi_grids = [], [] ,[]
            for dim in range(self.KX):
                x_grids  += [Grid1D(grid_x_IDs[dim], (self.pos_axes[dim],))]

            for dim in range(self.KV):
                vi_grids += [Grid1D(grid_vi_IDs[dim], (self.vi_axes[dim],))]
                ve_grids += [Grid1D(grid_ve_IDs[dim], (self.ve_axes[dim],))]

            grid_i = GridsComb('XVi', (*x_grids, *vi_grids))
            grid_e = GridsComb('XVe', (*x_grids, *ve_grids))
            grid_X = GridsComb('X', (*x_grids,))

        else:
            grid_i = Grid1D('XVi', (*self.pos_axes, *self.vi_axes), layout_type=self.grid_layout)
            grid_e = Grid1D('XVe', (*self.pos_axes, *self.ve_axes), layout_type=self.grid_layout)
            grid_X = Grid1D('X', (*self.pos_axes,), layout_type=self.grid_layout)

        self.grid_i = grid_i
        self.grid_e = grid_e
        self.grid_X = grid_X

        return grid_i, grid_e, grid_X


class LandauDamping(VlasovTest):

    def __init__(self, ks: Union[Numeric, Sequence[Numeric]]=0.10, A=1.0e-3, grid_idx=None,
                 Lxs: Union[int, Sequence[int]] = (),
                 Lves: Union[int, Sequence[int]] = (),
                 Lvis: Union[int, Sequence[int]] = (),
                 KX: int = 0, KV: int = 0, q: int = 2,
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis], Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 # units_config: UnitsConfiguration = None,
                 # mass_e=1.0, mass_ratio=1836, n0_e=1.0, n0_ratio=1.0, T_e=1.0, T_ratio=1.0,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt=True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout=LayoutType.SEQUENTIAL,
                 ):

        self.ks = (ks,)*KX if np.isscalar(ks) else ks
        k = np.linalg.norm(ks)
        self.A = A

        if grid_idx is None:
            if k == 0.10 and A > 0.01:
                grid_idx = 1
            else:
                grid_idx = 0

        super().__init__(Lxs=Lxs, Lves=Lves, Lvis=Lvis, KX=KX, KV=KV, q=q,
                         ion_config=ion_config, elc_config=elc_config,
                         #
                         # units_config = units_config,
                         # mass_e = mass_e, mass_ratio = mass_ratio,
                         # n0_e = n0_e, n0_ratio = n0_ratio,
                         # T_e = T_e, T_ratio = T_ratio,
                         #
                         do_adapt_coll = do_adapt_coll, coll_type = coll_type,
                         coll_coeff_e = coll_coeff_e, coll_coeff_i = coll_coeff_i,
                         coll_rate_e = coll_rate_e, coll_rate_i = coll_rate_i,
                         #
                         compress_style = compress_style, compress_style_mod = compress_style_mod,
                         compress_F = compress_F, DMAX = DMAX, DMAX_F = DMAX_F,
                         #
                         do_adapt_dt = do_adapt_dt, dt = dt, dt_frac = dt_frac, T = T, te_order = te_order,
                         #
                         grid_layout = grid_layout,
                         )

        x_lims, ve_lims, vi_lims = [], [], []
        for dim in range(KX):

            basis_x = basis_xs[dim] if isinstance(basis_xs, (list,tuple)) else basis_xs
            basis_ve = basis_ves[dim] if isinstance(basis_ves, (list,tuple)) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, (list,tuple)) else basis_vis

            is_k = isinstance(basis_x, FourierBasis)
            is_real_k = isinstance(basis_x, RealFourierBasis)
            is_m_e = isinstance(basis_ve, HermiteBasis)
            is_m_i = isinstance(basis_vi, HermiteBasis)

            x_lim, ve_lim, vi_lim = self.get_grid_params(dim, grid_idx, is_k=is_k, is_real_k=is_real_k,
                                                         is_m_e=is_m_e, is_m_i=is_m_i)

            x_lims += [x_lim]
            ve_lims += [ve_lim]
            vi_lims += [vi_lim]

        self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                             x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                             basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                             pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                             )

        self.initialize_grid()


    def get_grid_params(self, dim, idx, is_k=False, is_real_k=False, is_m_e=False, is_m_i=False):

        if is_real_k:
            kmax = self.ks[dim] * (self.q ** self.Lxs[dim])
            x_bounds = 0, kmax
        elif is_k:
            kmax = self.ks[dim] * (self.q ** self.Lxs[dim]) / 2
            x_bounds = -kmax, kmax
        else:
            if idx == 0 or idx == 1 or idx == 2:
                x_bounds = -np.pi / self.ks[dim], np.pi / self.ks[dim]
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        if is_m_e:
            mmax = self.q**self.Lves[dim]
            ve_bounds = 0, mmax
        else:
            if idx == 0:
                ve_bounds = -6 * self.elc_config.vth, 6 * self.elc_config.vth
            elif idx == 1:
                ve_bounds = -30 * self.elc_config.vth, 30 * self.elc_config.vth
            elif idx == 2:
                ve_bounds = -12 * self.elc_config.vth, 12 * self.elc_config.vth
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        if is_m_i:
            mmax = self.q**self.Lvis[dim]
            vi_bounds = 0, mmax
        else:
            if idx == 0:
                vi_bounds = -6 * self.ion_config.vth, 6 * self.ion_config.vth
            elif idx == 1 or idx == 2:
                vi_bounds = -6 * self.ion_config.vth, 6 * self.ion_config.vth
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        return x_bounds, ve_bounds, vi_bounds


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        fstr1 = super().get_fstr()

        fstr_sub = ''
        for dim in range(self.KX):
            fstr_sub += f'k{self.ks[dim]:1.2f}_'
        fstr_sub += f'A{self.A:1.1e}_'
        return fstr_sub + fstr1 + extra_str



class IonAcoustic(VlasovTest):

    def __init__(self, dks: Union[Numeric, Sequence[Numeric]]=0.10, A=1.0e-3, grid_idx=0,
                 Lxs: Union[int, Sequence[int]] = 6,
                 Lves: Union[int, Sequence[int]] = 6,
                 Lvis: Union[int, Sequence[int]] = 6,
                 KX: int = 1, KV: int = 1, q: int = 2,
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis], Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 # units_config: UnitsConfiguration = None,
                 # mass_e=1.0, mass_ratio=1836, n0_e=1.0, n0_ratio=1.0, T_e=1.0, T_ratio=1.0,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt=True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout=LayoutType.SEQUENTIAL,
                 ):

        self.dks = (dks,) * KX if np.isscalar(dks) else dks
        k = np.linalg.norm(dks)
        self.A = A


        super().__init__(Lxs=Lxs, Lves=Lves, Lvis=Lvis, KX=KX, KV=KV, q=q,
                         ion_config=ion_config, elc_config=elc_config,
                         #
                         # units_config = units_config,
                         # mass_e = mass_e, mass_ratio = mass_ratio,
                         # n0_e = n0_e, n0_ratio = n0_ratio,
                         # T_e = T_e, T_ratio = T_ratio,
                         #
                         do_adapt_coll = do_adapt_coll, coll_type = coll_type,
                         coll_coeff_e = coll_coeff_e, coll_coeff_i = coll_coeff_i,
                         coll_rate_e = coll_rate_e, coll_rate_i = coll_rate_i,
                         #
                         compress_style = compress_style, compress_style_mod = compress_style_mod,
                         compress_F = compress_F, DMAX = DMAX, DMAX_F = DMAX_F,
                         #
                         do_adapt_dt = do_adapt_dt, dt = dt, dt_frac = dt_frac, T = T, te_order = te_order,
                         #
                         grid_layout = grid_layout,
                         )

        x_lims, ve_lims, vi_lims = [], [], []
        for dim in range(KX):

            basis_x = basis_xs[dim] if isinstance(basis_xs, (list,tuple)) else basis_xs
            basis_ve = basis_ves[dim] if isinstance(basis_ves, (list,tuple)) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, (list,tuple)) else basis_vis

            is_k = isinstance(basis_x, FourierBasis)
            is_real_k = isinstance(basis_x, RealFourierBasis)
            is_m_e = isinstance(basis_ve, HermiteBasis)
            is_m_i = isinstance(basis_vi, HermiteBasis)

            x_lim, ve_lim, vi_lim = self.get_grid_params(dim, grid_idx, is_k=is_k, is_real_k=is_real_k,
                                                         is_m_e=is_m_e, is_m_i=is_m_i)

            x_lims += [x_lim]
            ve_lims += [ve_lim]
            vi_lims += [vi_lim]

        self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                             x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                             basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                             pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                             )

        self.initialize_grid()


    def get_grid_params(self, dim, idx, is_k=False, is_real_k=False, is_m_e=False, is_m_i=False):

        if is_real_k:
            kmax = self.dks[dim] * (self.q ** self.Lxs[dim])
            x_bounds = 0, kmax
        elif is_k:
            kmax = self.dks[dim] * (self.q ** self.Lxs[dim]) / 2
            x_bounds = -kmax, kmax
        else:
            if idx == 0 or idx == 1 or idx==2:
                x_bounds = -np.pi / self.dks[dim], np.pi / self.dks[dim]
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        if is_m_e:
            mmax = self.q**self.Lves[dim]
            ve_bounds = 0, mmax
        else:
            if idx == 0:
                ve_bounds = -6 * self.elc_config.vth, 6 * self.elc_config.vth
            elif idx == 1:
                ve_bounds = -30 * self.elc_config.vth, 30 * self.elc_config.vth
            elif idx == 2:
                ve_bounds = -6 * self.elc_config.vth, 14. * self.elc_config.vth
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        if is_m_i:
            mmax = self.q**self.Lvis[dim]
            vi_bounds = 0, mmax
        else:
            if idx == 0:
                vi_bounds = -6 * self.ion_config.vth, 6 * self.ion_config.vth
            elif idx == 1:
                vi_bounds = -6 * self.ion_config.vth, 6 * self.ion_config.vth
            elif idx == 2:
                vi_bounds = -36. * self.ion_config.vth, 36. * self.ion_config.vth
            else:
                raise IndexError(f'grid parameter for index {idx} not defined')

        return x_bounds, ve_bounds, vi_bounds


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        fstr1 = super().get_fstr()

        fstr_sub = ''
        for dim in range(self.KX):
            fstr_sub += f'dk{self.dks[dim]:1.2f}_'
        fstr_sub += f'A{self.A:1.1e}_'
        return fstr_sub + fstr1 + extra_str



class Whistler(VlasovTest):

    def __init__(self, ks=0.10, A=1.0e-3, grid_idx=0,
                 Lxs: Union[int, Sequence[int]] = 6,
                 Lves: Union[int, Sequence[int]] = 6,
                 Lvis: Union[int, Sequence[int]] = 6,
                 KX: int = 1, KV: int = 1, q: int = 2,
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis], Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt=True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout=LayoutType.SEQUENTIAL,
                 ):

        self.ks = (ks,)*KX if np.isscalar(ks) else ks
        self.A = A

        super().__init__(Lxs=Lxs, Lves=Lves, Lvis=Lvis, KX=KX, KV=KV, q=q,
                         ion_config=ion_config, elc_config=elc_config,
                         #
                         # units_config = units_config,
                         # mass_e = mass_e, mass_ratio = mass_ratio,
                         # n0_e = n0_e, n0_ratio = n0_ratio,
                         # T_e = T_e, T_ratio = T_ratio,
                         #
                         do_adapt_coll = do_adapt_coll, coll_type = coll_type,
                         coll_coeff_e = coll_coeff_e, coll_coeff_i = coll_coeff_i,
                         coll_rate_e = coll_rate_e, coll_rate_i = coll_rate_i,
                         #
                         compress_style = compress_style, compress_style_mod = compress_style_mod,
                         compress_F = compress_F, DMAX = DMAX, DMAX_F = DMAX_F,
                         #
                         do_adapt_dt = do_adapt_dt, dt = dt, dt_frac = dt_frac, T = T, te_order = te_order,
                         #
                         grid_layout = grid_layout,
                         )

        x_lims, ve_lims, vi_lims = [], [], []
        for dim in range(max(KX,KV)):

            basis_x = basis_xs[dim] if isinstance(basis_xs, tuple) else basis_xs
            basis_ve = basis_ves[dim] if isinstance(basis_ves, tuple) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, tuple) else basis_vis

            is_k = isinstance(basis_x, FourierBasis)
            is_m_e = isinstance(basis_ve, HermiteBasis)
            is_m_i = isinstance(basis_vi, HermiteBasis)

            x_lim, ve_lim, vi_lim = self.get_grid_params(dim, grid_idx, is_k=is_k, is_m_e=is_m_e, is_m_i=is_m_i)

            x_lims += [x_lim]
            ve_lims += [ve_lim]
            vi_lims += [vi_lim]

        self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                             x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                             basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                             pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                             )

        self.initialize_grid()


    def get_grid_params(self, dim, grid_idx, is_k = False, is_real_k = False, is_m_e = False, is_m_i = False):

        if dim >= self.KX:
            x_bounds = None, None
        else:
            if is_real_k:
                kmax = self.ks[dim] * (self.q ** self.Lxs[dim])
                x_bounds = 0, kmax
            elif is_k:
                kmax = self.ks[dim] * (self.q ** self.Lxs[dim]) / 2
                x_bounds = -kmax, kmax
            else:
                if grid_idx == 0 or grid_idx == 1:
                    x_bounds = 0, 2*np.pi/self.ks[dim]
                    print(x_bounds)
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

        if dim >= self.KV:
            ve_bounds = None, None
            vi_bounds = None, None
        else:
            if is_m_e:
                mmax = self.q**self.Lves[dim]
                ve_bounds = 0, mmax
            else:
                if grid_idx == 0:
                    ve_bounds = -15 * self.elc_config.vth, 15 * self.elc_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

            if is_m_i:
                mmax = self.q**self.Lvis[dim]
                vi_bounds = 0, mmax
            else:
                if grid_idx == 0:
                    vi_bounds = -7 * self.ion_config.vth, 7 * self.ion_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

        return x_bounds, ve_bounds, vi_bounds


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        fstr1 = super().get_fstr()
        fstr_sub = ''
        for dim in range(self.KX):
            fstr_sub += f'k{self.ks[dim]:1.2f}_'
        fstr_sub += f'A{self.A:1.1e}_'

        return fstr_sub + fstr1 + extra_str


class Alfven(VlasovTest):

    def __init__(self, ks=0.10, A=1.0e-3, grid_idx=0,
                 Lxs: Union[int, Sequence[int]] = 6,
                 Lves: Union[int, Sequence[int]] = 6,
                 Lvis: Union[int, Sequence[int]] = 6,
                 KX: int = 1, KV: int = 1, q: int = 2,
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis], Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt=True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout=LayoutType.SEQUENTIAL,
                 ):

        self.ks = (ks,)*KX if np.isscalar(ks) else ks
        self.A = A

        super().__init__(Lxs=Lxs, Lves=Lves, Lvis=Lvis, KX=KX, KV=KV, q=q,
                         ion_config=ion_config, elc_config=elc_config,
                         #
                         # units_config = units_config,
                         # mass_e = mass_e, mass_ratio = mass_ratio,
                         # n0_e = n0_e, n0_ratio = n0_ratio,
                         # T_e = T_e, T_ratio = T_ratio,
                         #
                         do_adapt_coll = do_adapt_coll, coll_type = coll_type,
                         coll_coeff_e = coll_coeff_e, coll_coeff_i = coll_coeff_i,
                         coll_rate_e = coll_rate_e, coll_rate_i = coll_rate_i,
                         #
                         compress_style = compress_style, compress_style_mod = compress_style_mod,
                         compress_F = compress_F, DMAX = DMAX, DMAX_F = DMAX_F,
                         #
                         do_adapt_dt = do_adapt_dt, dt = dt, dt_frac = dt_frac, T = T, te_order = te_order,
                         #
                         grid_layout = grid_layout,
                         )

        x_lims, ve_lims, vi_lims = [], [], []
        for dim in range(max(KX,KV)):

            basis_x = basis_xs[dim] if isinstance(basis_xs, tuple) else basis_xs
            basis_ve = basis_ves[dim] if isinstance(basis_ves, tuple) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, tuple) else basis_vis

            is_k = isinstance(basis_x, FourierBasis)
            is_m_e = isinstance(basis_ve, HermiteBasis)
            is_m_i = isinstance(basis_vi, HermiteBasis)

            x_lim, ve_lim, vi_lim = self.get_grid_params(dim, grid_idx, is_k=is_k, is_m_e=is_m_e, is_m_i=is_m_i)

            x_lims += [x_lim]
            ve_lims += [ve_lim]
            vi_lims += [vi_lim]

        self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                             x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                             basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                             pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                             )

        self.initialize_grid()


    def get_grid_params(self, dim, grid_idx, is_k = False, is_real_k = False, is_m_e = False, is_m_i = False):

        if dim >= self.KX:
            x_bounds = None, None
        else:
            if is_real_k:
                kmax = self.ks[dim] * (self.q ** self.Lxs[dim])
                x_bounds = 0, kmax
            elif is_k:
                kmax = self.ks[dim] * (self.q ** self.Lxs[dim]) / 2
                x_bounds = -kmax, kmax
            else:
                if grid_idx == 0 or grid_idx == 1:
                    x_bounds = 0, 2*np.pi/self.ks[dim]
                    print(x_bounds)
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

        if dim >= self.KV:
            ve_bounds = None, None
            vi_bounds = None, None
        else:
            if is_m_e:
                mmax = self.q**self.Lves[dim]
                ve_bounds = 0, mmax
            else:
                if grid_idx == 0:
                    ve_bounds = -5 * self.elc_config.vth, 5 * self.elc_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

            if is_m_i:
                mmax = self.q**self.Lvis[dim]
                vi_bounds = 0, mmax
            else:
                if grid_idx == 0:
                    vi_bounds = -5 * self.ion_config.vth, 5 * self.ion_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

        return x_bounds, ve_bounds, vi_bounds


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        fstr1 = super().get_fstr()
        fstr_sub = ''
        for dim in range(self.KX):
            fstr_sub += f'k{self.ks[dim]:1.2f}_'
        fstr_sub += f'A{self.A:1.1e}_'

        return fstr_sub + fstr1 + extra_str


class Reconnection(VlasovTest):

    def __init__(self, lam=0.5, B0=1,  A=0.1,
                 v0es: Union[float, Sequence[float]] = 0,
                 v0is: Union[float, Sequence[float]] = 0,
                 grid_idx=0,
                 Lxs: Union[int, Sequence[int]] = 6,
                 Lves: Union[int, Sequence[int]] = 6,
                 Lvis: Union[int, Sequence[int]] = 6,
                 KX: int = 1, KV: int = 1, q: int = 2,
                 x_map_key: str = 'F', ve_map_key: str = 'F', vi_map_key: Optional[str] = None,
                 basis_xs: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_ves: Union[Sequence[Basis], Basis] = SpatialBasis(),
                 basis_vis: Optional[Union[Sequence[Basis], Basis]] = None,
                 pos_coordsys: Optional['CoordinateSystem'] = None,
                 ve_coordsys: Optional['CoordinateSystem'] = None,
                 vi_coordsys: Optional['CoordinateSystem'] = None,
                 #
                 ion_config: SpeciesConfiguration = None,
                 elc_config: SpeciesConfiguration = None,
                 #
                 do_adapt_coll=True, coll_type=None,
                 coll_coeff_e=0.005, coll_coeff_i=None, coll_rate_e=0.005, coll_rate_i=None,
                 #
                 compress_style=1, compress_style_mod=0,
                 compress_F=True, DMAX=None, DMAX_F=None,
                 #
                 do_adapt_dt=True, dt=None, dt_frac=0.9, T=100, te_order=4,
                 #
                 grid_layout=LayoutType.SEQUENTIAL,
                 ):

        units_config = ion_config.units_config
        T_i, T_e = ion_config.T, elc_config.T

        lambda_i = units_config.c / ion_config.wp      # ion inertial length

        self.lambda_i = lambda_i  # ion inertial length
        self.lam = lam
        self.B0 = B0

        self.T_i = T_i
        self.T_e = T_e

        if not isinstance(v0es, (list,tuple)):
            self.v0es = (v0es,v0es,v0es)
        else:
            self.v0es = v0es

        if not isinstance(v0is, (list,tuple)):
            self.v0is = (v0is,v0is,v0is)
        else:
            self.v0is = v0is

        self.A = A      # perturbation strength

        super().__init__(Lxs=Lxs, Lves=Lves, Lvis=Lvis, KX=KX, KV=KV, q=q,
                         ion_config=ion_config, elc_config=elc_config,
                         #
                         do_adapt_coll = do_adapt_coll, coll_type = coll_type,
                         coll_coeff_e = coll_coeff_e, coll_coeff_i = coll_coeff_i,
                         coll_rate_e = coll_rate_e, coll_rate_i = coll_rate_i,
                         #
                         compress_style = compress_style, compress_style_mod = compress_style_mod,
                         compress_F = compress_F, DMAX = DMAX, DMAX_F = DMAX_F,
                         #
                         do_adapt_dt = do_adapt_dt, dt = dt, dt_frac = dt_frac, T = T, te_order = te_order,
                         #
                         grid_layout = grid_layout,
                         )

        x_lims, ve_lims, vi_lims = [], [], []
        for dim in range(max(KX,KV)):

            basis_x = basis_xs[dim] if isinstance(basis_xs, tuple) else basis_xs
            basis_ve = basis_ves[dim] if isinstance(basis_ves, tuple) else basis_ves
            basis_vi = basis_vis[dim] if isinstance(basis_vis, tuple) else basis_vis

            is_k = isinstance(basis_x, FourierBasis)
            is_m_e = isinstance(basis_ve, HermiteBasis)
            is_m_i = isinstance(basis_vi, HermiteBasis)

            x_lim, ve_lim, vi_lim = self.get_grid_params(dim, grid_idx, is_k=is_k, is_m_e=is_m_e, is_m_i=is_m_i)

            x_lims += [x_lim]
            ve_lims += [ve_lim]
            vi_lims += [vi_lim]

        self.initialize_axes(x_lims=x_lims, ve_lims=ve_lims, vi_lims=vi_lims,
                             x_map_key=x_map_key, ve_map_key=ve_map_key, vi_map_key=vi_map_key,
                             basis_xs=basis_xs, basis_ves=basis_ves, basis_vis=basis_vis,
                             pos_coordsys=pos_coordsys, ve_coordsys=ve_coordsys, vi_coordsys=vi_coordsys,
                             )

        self.initialize_grid()


    def get_grid_params(self, dim, grid_idx, is_k = False, is_real_k = False, is_m_e = False, is_m_i = False):
        """ TODO: incorporate include endpoints vs not
        """

        if dim >= self.KX:
            x_bounds = None, None
        else:
            if is_real_k:
                if dim == 0:
                    dk = 2 * np.pi / (25.6 * self.lambda_i)
                elif dim == 1:
                    dk = 2 * np.pi / (12.8 * self.lambda_i)
                else:
                    raise ValueError(f'dim {dim} not in this system')
                kmax = dk * (self.q ** self.Lxs[dim])
                x_bounds = 0, kmax
            elif is_k:
                if dim == 0:
                    dk = 2 * np.pi / (25.6 * self.lambda_i)
                elif dim == 1:
                    dk = 2 * np.pi / (12.8 * self.lambda_i)
                else:
                    raise ValueError(f'dim {dim} not in this system')
                kmax = dk * (self.q ** self.Lxs[dim]) / 2
                x_bounds = -kmax, kmax
            else:
                if grid_idx == 0:   # four-fold symmetry (use reflecting boundaries at 0)
                    if dim == 0:
                        x_bounds = 0, 25.6 * self.lambda_i
                    elif dim == 1:
                        x_bounds = 0, 12.8 * self.lambda_i
                    else:
                        raise ValueError(f'dim {dim} not in this system')
                elif grid_idx == 1:     # full simulation space
                    if dim == 0:
                        x_bounds = -25.6 * self.lambda_i, 25.6 * self.lambda_i
                    elif dim == 1:
                        x_bounds = -12.8 * self.lambda_i, 12.8 * self.lambda_i
                    else:
                        raise ValueError(f'dim {dim} not in this system')
                else:
                    raise NotImplementedError


        if dim >= self.KV:
            ve_bounds = None, None
            vi_bounds = None, None
        else:
            if is_m_e:
                mmax = self.q**self.Lves[dim]
                ve_bounds = 0, mmax
            else:
                if grid_idx == 0 or grid_idx == 1:
                    ve_bounds = -15 * self.elc_config.vth, 15 * self.elc_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

            if is_m_i:
                mmax = self.q**self.Lvis[dim]
                vi_bounds = 0, mmax
            else:
                if grid_idx == 0 or grid_idx == 1:
                    vi_bounds = -7 * self.ion_config.vth, 7 * self.ion_config.vth
                else:
                    raise IndexError(f'grid parameter for index {grid_idx} not defined')

        return x_bounds, ve_bounds, vi_bounds


    def get_fstr(self, extra_str=''):
        """ get filename
        """
        fstr1 = super().get_fstr()
        fstr_sub = f'A{self.A:1.1e}_lam{self.lam}_'

        return fstr_sub + fstr1 + extra_str
