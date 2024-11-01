import numpy as np

import helper_quimb
from setup_.configs import *
from helper_tdvp import TDVPSolver0
import matplotlib.pyplot as plt
import helper_quimb as helper
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField
from pde_system import PDE_system
from pde_EM import Maxwell

from helper_dmrg import SolveMethod, SweepDirection
import helper_block_dmrg as helper_block
import helper_block_dmrg_2 as helper_block_2

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord.coord_sys import Coordinate, CoordinateSystem
    from grid import Grid
    from gridTN import GridTN



def evolve_E(dt: Numeric, E_comp: 'GridTN', deriv: 'GridTN', pml_correction_mps: 'GridTN'=None,
             inplace=False, compress=True, zipup=False, compress_opts: dict=None,
             compress_type=CompressType.SVD) -> 'GridTN':
    """ evolve E('zx' or 'zy') field in time via euler step
    """
    ## correct E
    if pml_correction_mps is not None:
        pml_correction_mpo = pml_correction_mps.apply_elemental_multiply_op()
        E_corrected = E_comp.apply(pml_correction_mpo, inplace=inplace, zipup=zipup)
    else:
        E_corrected = E_comp if inplace else E_comp.copy()

    deriv = deriv.scalar_multiply(dt, inplace=False)
    new_E = E_corrected.add(deriv, compress=compress, inplace=True, compress_opts=compress_opts,
                            compress_type=compress_type)
    return new_E


def evolve_B(dt: Numeric, B_comp: 'GridTN', deriv1: 'GridTN', # deriv2: 'GridTN',
             pml_correction_mps: 'GridTN'=None, compress_type: CompressType=CompressType.SVD,
             inplace=False, compress=True, zipup=False, compress_opts:dict=None) -> 'GridTN':
    """ evolve E field in time via euler step
    """
    ## correct B
    if pml_correction_mps is not None:
        pml_correction_mpo = pml_correction_mps.apply_elemental_multiply_op()
        B_corrected = B_comp.apply(pml_correction_mpo, inplace=inplace, zipup=zipup)
    else:
        B_corrected = B_comp if inplace else B_comp.copy()

    deriv1 = deriv1.scalar_multiply(dt, inplace=False)
    new_B = B_corrected.add(deriv1, compress=compress, inplace=True, compress_opts=compress_opts,
                            compress_type=compress_type)
    return new_B


# @profile
def _calculate_time_derivative_E_part(part: 'str', B: 'Field', coords_x: 'CoordinateSystem',
                                      matl_params: 'UnitsConfiguration',
                                      pml_correction_mps: 'GridTN'=None,
                                      current_density: Optional['Field']=None,
                                      zipup=False, compress=1, compress1=0,
                                      correct_flux=False,
                                      ) -> 'GridTN':
    """
    compute dE/dt = c curl(B) - eps0 J
    part: 'zx' or 'zy'
    E: [Field] electric field
    coords_x: [CoordinateSystem] for specifying x,y,z
    matl_parms: [UnitsConfigure] for specifying permittivity, permeability, speed of light...
    pml_correction_mps: [GridTN with MPS] correction term applied to curl E terms
        !!! NOTE: compared to definition in original code, this should not include the dt/dx coefficient
    zipup: parameter for how to apply the MPO. not too important
    compress: whether to compress MPS at various stages in the algorithm. default is compression at the end.
    """
    grid_X = B.grid
    X, Y, Z = coords_x.coords

    dEdt = None

    ## current
    j: Optional[Field] = current_density
    if j is not None:
        if part[0] == 'z':
            jc = current_density.components.get(Z, None)
        elif part[0] == 'y':
            jc = current_density.components.get(Y, None)
        elif part[0] == 'x':
            jc = current_density.components.get(X, None)
        else:
            raise ValueError

        if jc is not None:
            if matl_params.is_cgs:
                jc = jc.scalar_multiply(-4 * np.pi, inplace=False)
            else:
                jc = jc.scalar_multiply(-1 / matl_params.eps0, inplace=False)

        dEdt = jc

    ## add smoothing
    # dEdt_copy = dEdt.copy()
    # dEdt_copy.average_fine_scale(mu=0.50, inplace=True)
    # dEdt.average_fine_scale(mu=0.90, inplace=True)
    # print('smoothing J', part)

    ## ensure spatial derivatives are correctly taken
    for compID, comp in B.components.items():
        new_ax_deriv_configs = comp.ax_deriv_configs
        for ax in grid_X.axes:
            if comp is not None:
                deriv_config = comp.ax_deriv_configs[ax].copy()
                deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                # comp.ax_deriv_configs[ax] = deriv_config
                new_ax_deriv_configs[ax] = deriv_config
        comp.ax_deriv_configs = new_ax_deriv_configs

    # B = B.copy()
    # for compID, comp_gtn in B.components.items():
    #     comp_gtn.average_fine_scale(mu=0.90, inplace=True)
    #     print('smoothing B for dEdt')


    ## split curl B
    # curlB = B.curl(coords_x, compress_level=compress1, inner_compress_level=compress2)
    result = None
    if part == 'zx':
        Bx = B.components.get(X, None)
        if Bx is not None:
            ax_y = coords_x.get_axis(Y)
            result = Bx.take_firstderivative(ax=ax_y)

            if correct_flux:
                from helper_fluxfilter import correct_flux
                result = correct_flux(result, deriv_ax=ax_y)

            result.scalar_multiply(-1, inplace=True)
    elif part == 'zy':
        By = B.components.get(Y, None)
        result = None
        if By is not None:
            ax_x = coords_x.get_axis(X)
            result = By.take_firstderivative(ax=ax_x)

            if correct_flux:
                from helper_fluxfilter import correct_flux
                result = correct_flux(result, deriv_ax=ax_x)

    else:
        raise NotImplementedError

    if pml_correction_mps is not None and result is not None:
        pml_correction_mpo = pml_correction_mps.apply_elemental_multiply_op()
        result = result.apply(pml_correction_mpo, inplace=True, zipup=zipup)

    ## scale curl B
    if result is not None:
        if matl_params.is_cgs:
            result = result.scalar_multiply(matl_params.c, inplace=False)
        else:
            result = result.scalar_multiply(matl_params.c ** 2, inplace=False)

    ## dE/dt = -1/eps0 J + c^2 curl B
    if dEdt is not None:
        dEdt = result.add(dEdt, inplace=False, compress_level=0)
        # dEdt = dEdt.add(result, inplace=False, compress_level=0)
    else:
        dEdt = result.copy()

    # ###################
    # dEdt_data = dEdt.get_data()
    # nx, ny = dEdt_data.shape
    # plt.figure()
    # plt.imshow(dEdt_data)
    # plt.colorbar()
    # plt.title('pre smoothing')
    #
    # ### add smoothing
    # dEdt_copy = dEdt # .copy()
    # dEdt_copy.average_fine_scale(mu=0.90, inplace=True)
    # print('smoothing dEdt', part)
    #
    # dEdt_data_new = dEdt_copy.get_data()
    # plt.figure()
    # plt.imshow(dEdt_data_new)
    # plt.colorbar()
    # plt.title('post smoothing')
    #
    # plt.figure()
    # plt.plot(dEdt_data[nx // 2, :])
    # plt.plot(dEdt_data_new[nx // 2, :])
    # plt.show()
    #
    # print('compress', compress)


    if dEdt is not None:
        if compress:    dEdt.compress(inplace=True, compress_level=compress)

    return dEdt


# @profile
def _calculate_time_derivative_B(E: 'Field', coords_x: 'CoordinateSystem',
                                 matl_params: 'UnitsConfiguration',
                                 pml_correction_dict: dict['Coordinate','GridTN']=None,
                                 zipup=False, compress=1, compress1=0, compress2=0,
                                 correct_flux = False,
                                 ) -> 'Field':
    """
    dB/dt + curl(E) + gamma*grad(psi) = 0

    E: [Field] electric field
    coords_x: [CoordinateSystem] for specifying x,y,z
    matl_parms: [UnitsConfigure] for specifying permittivity, permeability, speed of light...
    pml_correction_mps: [GridTN with MPS] correction term applied to curl E terms
        !!! NOTE: compared to definition in original code, this should not include the dt/dx coefficient
    zipup: parameter for how to apply the MPO. not too important
    compress: whether to compress MPS at various stages in the algorithm. default is compression at the end.
    """
    grid_X = E.grid

    ## curl(E)
    # ensure proper derivative configuration for FDTD
    for compID, comp in E.components.items():
        new_ax_deriv_configs = comp.ax_deriv_configs
        for ax in grid_X.axes:
            deriv_config = comp.ax_deriv_configs[ax].copy()
            deriv_config.update(order=0, fd_type=FDType.FORWARD)
            new_ax_deriv_configs[ax] = deriv_config
            # comp.ax_deriv_configs[ax] = deriv_config
        comp.ax_deriv_configs = new_ax_deriv_configs


    # ## smooth E
    # E = E.copy()
    # for compID, comp_gtn in E.components.items():
    #     comp_gtn.average_fine_scale(mu=0.90, inplace=True)
    #     print('smoothing E for dBdt')

    curlE = E.curl(coords_x, compress_level=compress1, inner_compress_level=compress2)
    dBdt = curlE

    if correct_flux:
        from helper_fluxfilter import correct_flux
        for compID, comp in curlE.components.items():
            ax_x = coords_x.get_axis(compID)
            result = correct_flux(comp, deriv_ax=ax_x)
            dBdt.components[compID] = result

    if pml_correction_dict is not None:
        for compID in coords_x.coords:
            dBdt_comp = dBdt.components.get(compID, None)
            pml_correction_mps = pml_correction_dict.get(compID, None)
            if dBdt_comp is not None and pml_correction_mps is not None:
                pml_correction_mpo = pml_correction_mps.apply_elemental_multiply_op()
                dBdt_comp.apply(pml_correction_mpo, inplace=True, zipup=zipup)

    if dBdt is not None:
        if compress:
            dBdt.compress(inplace=True, compress_level=compress)

        if matl_params.is_cgs:
            dBdt = dBdt.scalar_multiply(-matl_params.c, inplace=True)
        else:
            dBdt = dBdt.scalar_multiply(-1, inplace=True)

    # ## add smoothing
    # for compID, comp_gtn in dBdt.components.items():
    #     comp_gtn.average_fine_scale(mu=0.90, inplace=True)
    #     print('smoothing dBdt')

    return dBdt



def get_divB(coords_x: 'CoordinateSystem', B: 'Field', compress=5,) -> Optional['ScalarField']:
    ax_deriv_configs = {}
    for compID in coords_x.coords:
        ax = coords_x.get_axis(compID)
        comp = B.components.get(compID, None)
        if comp is not None:
            dc = comp.ax_deriv_configs.get(ax, None)
            if dc is not None:
                dc_ = dc.copy()
                dc_.update(fd_type=FDType.FORWARD, order=0)
                ax_deriv_configs[ax] = dc_

    divB = B.divergence(coord_sys=coords_x, compress_level=compress, ax_deriv_configs=ax_deriv_configs)
    divB.is_sqrt = True
    return divB


