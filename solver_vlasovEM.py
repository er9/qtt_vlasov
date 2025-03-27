import numpy as np
import itertools
import quimb
import scipy.linalg

from setup_.defaults import *
from setup_.configs import *
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper

from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import ScalarField, Field
from pde_boltzmann import Boltzmann
from pde_EM import Maxwell

from setup_.TN_mpo import BlockedState1D, BlockedOperator1D

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridT



class SolverVlasovEM:

    def __init__(self, fe_sys: 'Boltzmann', fi_sys: 'Boltzmann', EM_sys: 'Maxwell',
                 max_bond=MAXBOND, cutoff=CUTOFF, upwind=False,
                 background_B0: dict['CoordinateType',Callable]=None,
                 background_E0: dict['CoordinateType', Callable]=None,
                 elc_params:'SpeciesConfiguration'=None,
                 ion_params: 'SpeciesConfiguration' = None,
                 time: Numeric=0.0,
                 ):

        self.sys_fe = fe_sys
        self.sys_fi = fi_sys
        self.sys_EM = EM_sys

        self.fe = fe_sys.f
        self.fi = fi_sys.f
        self.E = EM_sys.E
        self.B = EM_sys.B

        ## callable fcts that takes in time, returns Numeric or MPS
        ## to replace background E, B in EM_sys
        self.B0 = background_B0
        self.E0 = background_E0

        self.grid_e = self.fe.grid
        self.grid_i = self.fi.grid
        self.grid_X = self.E.grid

        self.max_bond=max_bond
        self.cutoff=cutoff

        self._force = self.compute_force()
        self._current = self.compute_current()
        self._envs = {}
        self.initialize_envs()

        self.upwind = upwind
        self.coords_x = fe_sys.coords_x
        self.coords_ve = fe_sys.coords_v
        self.coords_vi = fi_sys.coords_v

        self.elc_params = elc_params if elc_params is not None else ElcConfiguration()
        self.ion_params = ion_params if ion_params is not None else IonConfiguration()
        self.matl_params = elc_params.units_config

        self.time = time

    def initialize_envs(self):
        raise NotImplementedError

    @property
    def compress_opts(self):
        return {'max_bond': self.max_bond, 'cutoff': self.cutoff, 'cutoff_mode': CUTOFF_MODE}

    def compute_current(self, compress=1, compress1=3) -> 'Field':
        j = None
        if self.fe is not None:
            je = self.sys_fe.compute_flow(compress=compress1)
            je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
            j = je.copy()

        if self.fi is not None:
            ji = self.sys_fi.compute_flow(compress=compress1)
            ji.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)
            if j is not None:
                j.add(ji, compress_level=0, inplace=True)
            else:
                j = ji.copy()

        j.compress(inplace=True, compress_level=compress)
        return j

    def get_current_blocks(self, compID: 'CoordinateType', init_guess: 'qtn.MatrixProductState'=None,
                           ) -> dict['CoordinateType', 'qtn.TensorNetwork']:
        is_sqrt=True  # self.is_sqrt

        def get_current_block(is_ion):
            if is_sqrt:
                f_ket: 'qtn.MatrixProductState' = self.fi.component.data if is_ion else self.fe.component.data
                f_ket.add_tag("KET")
                f_bra: 'qtn.MatrixProductState' = f_ket.conj()
                f_bra.site_ind_id = f_bra.site_ind_id + '_'
                f_bra.add_tag("BRA")
                helper.match_inner_inds(f_bra, f_ket, append='_')

                ## velocity
                vel_mpo = self.sys_fi.velocities[compID] if is_ion else self.sys_fe.velocities[compID]
                vel_mpo.add_tag("V")
                vel_mpo.lower_ind_id = f_ket.site_ind_id
                vel_mpo.upper_ind_id = qtn.rand_uuid()

                ## element-wise mult operator
                in1_ind_id = vel_mpo.upper_ind_id
                in2_ind_id = f_bra.site_ind_id
                out_ind_id = qtn.rand_uuid()
                elem_mult_tn1d = self.grid_e.get_elemental_multiply_tn(in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id,
                                                                       out_ind_id=out_ind_id)

                ## partial integration
                integ_mpx = self.grid_e.get_integrals_mpx(axes=self.coords_ve.axes)
                integ_mpx.lower_ind_id = out_ind_id
                integ_mpx.upper_ind_id = init_guess.site_ind_id

                ## + projection onto initial guess
                j_block = qtn.TensorNetwork([f_ket, f_bra, vel_mpo, elem_mult_tn1d, integ_mpx, init_guess.conj()])

                return j_block

            else:
                f_ket: 'qtn.MatrixProductState' = self.fi.component.data if is_ion else self.fe.component.data
                f_ket.add_tag("KET")

                ## velocity
                vel_mpo = self.sys_fi.velocities[compID] if is_ion else self.sys_fe.velocities[compID]
                vel_mpo.add_tag("V")
                vel_mpo.lower_ind_id = f_ket.site_ind_id
                vel_mpo.upper_ind_id = qtn.rand_uuid()

                ## partial integration
                integ_mpx = self.grid_e.get_integrals_mpx(axes=self.coords_ve.axes)
                integ_mpx.lower_ind_id = vel_mpo.upper_ind_id
                integ_mpx.upper_ind_id = init_guess.site_ind_id

                ## + projection onto initial guess
                j_block = qtn.TensorNetwork([f_ket, vel_mpo, integ_mpx, init_guess.conj()])

                return j_block

        je_block = get_current_block(is_ion=False)
        ji_block = get_current_block(is_ion=True)
        return je_block, ji_block


    def get_force_operators(self, compID: 'CoordinateType', ):


    def compute_force(self, is_ion=False, compress=1, compress1=3, compress2=4
                          ) -> dict[CoordinateType, list[BlockedState1D]]:
        """ different from before:
            keep E contribution separate from v x B
            write v x B as a BlockedState1D
            for each comp
        """
        grid = self.grid_i if is_ion else self.grid_e
        coords_x = self.coords_x.coords
        coords_v = self.coords_vi if is_ion else self.coords_ve
        matl_params = self.ion_params if is_ion else self.elc_params
        coeff = matl_params.e * matl_params.Z / matl_params.mass

        if self.B0 is not None:
            background_B0 = {compID: B0(self.time) for compID, B0 in self.B0.items()}
            background_B0 = Field('B0', self.grid_X, background_B0, is_sqrt=True)
        else:
            background_B0 = self.sys_EM.background_B0

        if self.E0 is not None:
            background_E0 = {compID: E0(self.time) for compID, E0 in self.E0.items()}
            background_E0 = Field('E0', self.grid, background_E0, is_sqrt=True)
        else:
            background_E0 = self.sys_EM.background_E0


        ## v x B part
        B = self.B.add(background_B0, compress_level=compress2)

        out = {compID: [] for compID in self.coords_x.coords}
        for i in range(3):

            i0, i1, i2 = i, (i+1) % 3, (i+2) % 3
            c0, c1, c2 = coords_x[i0], coords_x[i1], coords_x[i2]

            try:
                v_ax: 'Axis' = coords_v.axes[i1]
                out_comp_1 = {(v_ax,): v_ax.get_xmultiply_mps(), tuple(B.grid.axes): B[c2].data}
                out_comp_1 = BlockedState1D(grid, out_comp_1)
                out_comp_1.scalar_multiply(coeff, inplace=True)
                if matl_params.is_cgs:
                    out_comp_1.scalar_multiply(1. / matl_params.c, inplace=True)
                out[c0] += [out_comp_1]
            except KeyError:
                pass

            try:
                v_ax: 'Axis' = coords_v.axes[i2]
                out_comp_2 = {(v_ax,): v_ax.get_xmultiply_mps(), tuple(B.grid.axes): B[c1].data}
                out_comp_2 = BlockedState1D(grid, out_comp_2)
                out_comp_2.scalar_multiply(-1. * coeff, inplace=True)
                if matl_params.is_cgs:
                    out_comp_2.scalar_multiply(1. / matl_params.c, inplace=True)
                out[c0] += [out_comp_2]
            except KeyError:
                pass

        ## E part
        E = self.E.add(background_E0, compress_level=compress2)
        for compID in coords_x:
            E_comp = E[compID]
            out_comp_3 = BlockedState1D(grid, {tuple(self.grid_X.axes): E_comp.data})
            out_comp_3.scalar_multiply(coeff, inplace=True)
            out[compID] += [out_comp_3]

        return out

    def update_current(self, i):
        raise NotImplementedError

    def update_force(self, i):
        raise NotImplementedError

    def solve_1site(self, dt: Numeric, time: Numeric=None):
        raise NotImplementedError

    def update_1site(self):
        raise NotImplementedError

    def solve_2site(self, dt: Numeric, time: Numeric=None):
        raise NotImplementedError

    def update_2site(self):
        raise NotImplementedError

    def solve_bond(self, dt: Numeric, time: Numeric=None):
        raise NotImplementedError

    def update_bond(self):
        raise NotImplementedError

    #########

    def get_f_derivative_dict(self, is_ion=False):

    def get_EM_derivative_dict(self,) -> dict[tuple[int, int], 'GridTN']:
        """ get operations acting on E, B to obtain dE/dt, dB/dt
            information returned as a dict with keys (output ind, input ind)
        """

        ### get curl operator
        def _get_curl_op(comp_ax_deriv_configs):

            gr = self.grid_X

            dict_mpos = {}
            for compID in self.coords_x.coords:
                i = compID.type  # self.ax_coords[axID]
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                c1_ = self.coords_x.type_coords.get(i1, None)
                c2_ = self.coords_x.type_coords.get(i2, None)

                ax1_ = self.coords_x.coord_axes[c1_]
                ax2_ = self.coords_x.coord_axes[c2_]
                # print('get curl', ax1_, ax2_)

                # if c2_ in gtn_comps and ax1_ is not None:
                if ax1_ is not None:
                    ax_deriv_configs = comp_ax_deriv_configs.get(c2_, {})  # [compID]
                    ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
                    mpo_1 = gr.get_firstderivative_mpo(ax1_, deriv_config=ax_deriv_configs.get(ax1_, None)).copy()
                else:
                    mpo_1 = None

                # if c1_ in gtn_comps and ax2_ is not None:
                if ax2_ is not None:
                    ax_deriv_configs = comp_ax_deriv_configs.get(c1_, {})  # [compID]
                    ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
                    mpo_2 = gr.get_firstderivative_mpo(ax2_, deriv_config=ax_deriv_configs.get(ax2_, None)).copy()
                    mpo_2 = mpo_2.scalar_multiply(-1, inplace=False)
                else:
                    mpo_2 = None  ## vector_field[i] = 0

                if mpo_1 is None and mpo_2 is None:
                    continue

                if mpo_2 is not None:
                    dict_mpos[(i.value, i1)] = mpo_2
                if mpo_1 is not None:
                    dict_mpos[(i.value, i2)] = mpo_1

            return dict_mpos

        ## dE/dt = curl(B) [[excludes + J]
        ## dB/dt = curl(E)
        E_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.E.components.items()}
        B_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.B.components.items()}
        dEdt_mpos = _get_curl_op(B_ax_deriv_configs)
        dBdt_mpos = _get_curl_op(E_ax_deriv_configs)

        ## adjust input, output coords accordingly
        all_mpos = {}
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        for (out_ind, in_ind), gtn in dEdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            all_mpos[(out_ind, in_ind + 3)] = gtn

        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
        for (out_ind, in_ind), gtn in dBdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(-self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(-1, inplace=True)
            all_mpos[(out_ind + 3, in_ind)] = gtn


        ## add upwinding contributions
        print('self.upwind', self.upwind, 'c', self.matl_params.c)
        if self.upwind:
            raise NotImplementedError
            uw_scale = 1.0
            for i in range(3):
                i1, i2 = (i+1) % 3, (i+2) % 3

                c1_ = self.coords_x.get_coord(i1)
                c2_ = self.coords_x.get_coord(i2)

                ax1_ = self.coords_x.get_axis(c1_) if c1_ is not None else None
                ax2_ = self.coords_x.get_axis(c2_) if c2_ is not None else None
                print('E upwind ax', c1_, c2_, ax1_, ax2_)

                compID = self.coords_x.type_coords.get(i, None)
                # print('E comp', compID, compID in self.E.components)
                if compID not in self.E.components:
                    continue

                E_ax_deriv_configs = self.E[compID].ax_deriv_configs

                sum_ddx_E = None
                if ax1_ is not None:
                    ddx1E = self.grid_X.get_secondderivative_mpo(ax1_, ax1_, deriv_config1=E_ax_deriv_configs[ax1_])
                    ddx1E = ddx1E.scalar_multiply(ax1_.dx)
                    # ddx1E = ddx1E.scalar_multiply(ax1_.dx * dt / ax1_.dx * self.matl_params.c)
                    sum_ddx_E = ddx1E
                    # print('E ax1 is not None', compID, ax1_)

                if ax2_ is not None:
                    ddx2E = self.grid_X.get_secondderivative_mpo(ax2_, ax2_, deriv_config1=E_ax_deriv_configs[ax2_])
                    ddx2E = ddx2E.scalar_multiply(ax2_.dx)
                    # ddx2E = ddx2E.scalar_multiply(ax2_.dx * dt / ax2_.dx * self.matl_params.c)
                    sum_ddx_E = ddx2E if sum_ddx_E is None else sum_ddx_E.add(ddx2E, compress=True)
                    # print('E ax2 is not None', compID, ax2_)

                if sum_ddx_E is not None:
                    if True:  # sum_ddx_E.frobenius_norm() > 1.0e-12:
                        sum_ddx_E = sum_ddx_E.scalar_multiply(uw_scale * self.matl_params.c/2, inplace=False)

                        ## electric fields
                        all_mpos[(i, i)] = sum_ddx_E  ## this term doesn't exist from original EOMs


            for i in range(3):
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                # c1_ = self.coords_x.type_coords.get(i1, None)
                # c2_ = self.coords_x.type_coords.get(i2, None)
                c1_ = self.coords_x.get_coord(i1)
                c2_ = self.coords_x.get_coord(i2)

                # ax1_ = self.coords_x.coord_axes[c1_]
                # ax2_ = self.coords_x.coord_axes[c2_]
                ax1_ = self.coords_x.get_axis(c1_) if c1_ is not None else None
                ax2_ = self.coords_x.get_axis(c2_) if c2_ is not None else None
                print('B upwind ax', c1_, c2_, ax1_, ax2_)

                compID = self.coords_x.type_coords.get(i, None)
                if compID not in self.B.components:
                    continue

                B_ax_deriv_configs = self.B[compID].ax_deriv_configs

                sum_ddx_B = None
                if ax1_ is not None:
                    ddx1B = self.grid_X.get_secondderivative_mpo(ax1_, ax1_, deriv_config1=B_ax_deriv_configs[ax1_])
                    ddx1B = ddx1B.scalar_multiply(ax1_.dx)
                    # ddx1B = ddx1B.scalar_multiply(ax1_.dx * dt / ax1_.dx * self.matl_params.c)
                    sum_ddx_B = ddx1B
                    # print('B ax1 is not None', compID, ax1_)

                if ax2_ is not None:
                    ddx2B = self.grid_X.get_secondderivative_mpo(ax2_, ax2_, deriv_config1=B_ax_deriv_configs[ax2_])
                    ddx2B = ddx2B.scalar_multiply(ax2_.dx)
                    # ddx2B = ddx2B.scalar_multiply(ax2_.dx * dt / ax2_.dx * self.matl_params.c)
                    sum_ddx_B = ddx2B if sum_ddx_B is None else sum_ddx_B.add(ddx2B, compress=True)
                    # print('B ax2 is not None', compID, ax2_)

                if sum_ddx_B is not None:
                    if True:  # sum_ddx_B.frobenius_norm() > 1.0e-12:
                        sum_ddx_B = sum_ddx_B.scalar_multiply(uw_scale * self.matl_params.c/2 , inplace=False)

                        ## magnetic fields
                        all_mpos[(i + 3, i + 3)] = sum_ddx_B  ## this term doesn't exist from original EOMs

        return all_mpos
