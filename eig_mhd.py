from setup_.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
import helper_sl as helper_sl
import helper_dlr
from helper_tdvp import TDVPSolver
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField, SCALAR_COORD
from pde_system import PDE_system

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridTN

from setup_.quimb_TN1D import MatrixProductStateTN
from eig_solver import EigSolver
import helper_dmrg as dmrg
# from helper_dmrg import SweepDirection
# from helper_dmrg import Environment, DMRGSolver


""" solves F|xi> - \lambda |xi> = 0
    
    F = 1/mu0 curl(B0) x B1 + 1/mu0 curl(B1) x B0 - grad(p1)
    where p1 = - xi \dot grad(p0) - gamma p0 div(xi)
          B1 = curl( xi x B0 )
    
    curlB0 = + [x] d/dy B0z - d/dz B0y 
             + [y] d/dz B0x - d/dx B0z 
             + [z] d/dx B0y - d/dy B0x
             
    xi x B0 = + [x] xi[y] B0[z] - xi[z] B0[y]
              + [y] xi[z] B0[x] - xi[x] B0[z]
              + [z] xi[x] B0[y] - xi[y] B0[x]
    B1 = [x] d/dy (xi[x] B0[y] - xi[y] B0[x]) - d/dz (xi[z] B0[x] - xi[x] B0[z])
         [y] d/dz (xi[y] B0[z] - xi[z] B0[y]) - d/dx (xi[x] B0[y] - xi[y] B0[x])
         [z] d/dx (xi[z] B0[x] - xi[x] B0[z]) - d/dy (xi[y] B0[z] - xi[z] B0[y]) 
    
    1. 1/mu0 (  [x] (curl(B0)[y] B1[z] - curl(B0)[z] B1[y]) 
              + [y] (curl(B0)[z] B1[x] - curl(B0)[x] B1[z])
              + [z] (curl(B0)[x] B1[y] - curl(B0)[y] B1[x]) )
    
        [x]: 1/mu0 ( (d/dz B0[x] - d/dx B0[z]) B1[z] - (d/dx B0[y] - d/dy B0[x]) B1[y] )
        [y]: 1/mu0 ( (d/dx B0[y] - d/dy B0[x]) B1[x] - (d/dy B0[z] - d/dz B0[y]) B1[z] )
        [z]: 1/mu0 ( (d/dy B0[z] - d/dz B0[y]) B1[y] - (d/dz B0[x] - d/dx B0[z]) B1[x] )
        
        dropping 1/mu0
        [x]:  (d/dz B0[x] - d/dx B0[z]) * (d/dx (xi[z] B0[x] - xi[x] B0[z]) - d/dy (xi[y] B0[z] - xi[z] B0[y]))
            - (d/dx B0[y] - d/dy B0[x]) * (d/dz (xi[y] B0[z] - xi[z] B0[y]) - d/dx (xi[x] B0[y] - xi[y] B0[x]))
        [y]:  (d/dx B0[y] - d/dy B0[x]) * (d/dy (xi[x] B0[y] - xi[y] B0[x]) - d/dz (xi[z] B0[x] - xi[x] B0[z]))
            - (d/dy B0[z] - d/dz B0[y]) * (d/dx (xi[z] B0[x] - xi[x] B0[z]) - d/dy (xi[y] B0[z] - xi[z] B0[y]))
        [x]:  (d/dy B0[z] - d/dz B0[y]) * (d/dz (xi[y] B0[z] - xi[z] B0[y]) - d/dx (xi[x] B0[y] - xi[y] B0[x]))
            - (d/dz B0[x] - d/dx B0[z]) * (d/dy (xi[x] B0[y] - xi[y] B0[x]) - d/dz (xi[z] B0[x] - xi[x] B0[z]))
        
    2. 1/mu0 (  [x] (curl(B1)[y] B0[z] - curl(B1)[z] B0[y]) 
              + [y] (curl(B1)[z] B0[x] - curl(B1)[x] B0[z])
              + [z] (curl(B1)[x] B0[y] - curl(B1)[y] B0[x]) )
    
        [x]: 1/mu0 ( (d/dz B1[x] - d/dx B1[z]) B0[z] - (d/dx B1[y] - d/dy B1[x]) B0[y] )
        [y]: 1/mu0 ( (d/dx B1[y] - d/dy B1[x]) B0[x] - (d/dy B1[z] - d/dz B1[y]) B0[z] )
        [z]: 1/mu0 ( (d/dy B1[z] - d/dz B1[y]) B0[y] - (d/dz B1[x] - d/dx B1[z]) B0[x] )
    
    3. grad( -xi \dot grad(p0) - gamma p0 div(xi) )
       grad( - (xi[x] d/dx p0 + xi[y] d/dy p0 + xi[z] d/dz p0)
             - gamma p0 (d/dx xi[x] + d/dy xi[y] + d/dz xi[z]) )
        
    
    <xi|F|xi> = <xi[x] | (F|xi>))[x] + <xi[y] | (F|xi>)[y] + <xi[z] | (F|xi>)[z]
    
    <xi|F.T F|xi> = (<xi|F.T| [x]) (F|xi>)[x] + ...  
"""


""" approach:
    build a B1 Field with TN_MPS
        dict with list of TNs that give B1 when summed together 
    build a p1 ScalarField with TN_MPS
    Field F|xi> with TN_MPS
    Envs for <xi|F|xi>
    Block DMRG with these components 
"""




class MHD(EigSolver):

    def __init__(self, displacement: Optional[Field],
                 background_B0: Field,
                 background_pressure: ScalarField,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 matl_params: Optional['SpeciesConfiguration'] = None,
                 adiabatic_index = 1.,
                 compress_levels=None,
                 ):

        self.names = {'xi': displacement.name if displacement is not None else 'xi'}
        field_names = [self.names[x] for x in ['xi']]

        super().__init__(displacement, field_names=field_names, compress_levels=compress_levels)

        self.matl_params = ElcConfiguration() if matl_params is None else matl_params

        self.coords_x = coords_x
        self.x_axes: Sequence['Axis'] = self.coords_x.axes

        if grid_X is None:
            self.grid_X = displacement.grid.get_subgrid(select_axes=self.coords_x.axes, new_gridID='GX')
        else:
            assert (grid_X.axes == self.x_axes)
            self.grid_X = grid_X

        self.B0 = background_B0
        self.p0 = background_pressure
        self.gamma = adiabatic_index


    def p1(self, displacement_vec, p0, gamma):
        """ xi = displacement vec
            p1 =  - xi \dot \grad p_0 - gamma p_0 div(xi)
        """
        return

    def B1(self, displacement_vec, B0):
        return


