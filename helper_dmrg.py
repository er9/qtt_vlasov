from setup.defaults import *
import time
import quimb.tensor as qtn

""" 
Functions to perform DMRG for MPS, using quimb as backend
"""

def optimize_site(site, left, right, H, is_H=True):
    """ site optimization for eigensolver. ie: minimizes <psi|H|psi> / <psi|psi>
        H_eff = (d/dT*[i] d/dT*[i] <psi|H|psi>)
        N_eff = d/dT*[i] d/dT[i] <psi|psi> = I if system is properly canonicalized
        --> minimize <psi|H|psi> - lambda <psi|psi>
        --> d/dT*[i] <psi|H|psi> - lambda d/dT*[i] <psi|psi> = 0
        --> H_eff * T[i] - lambda * N_eff * T[i] = 0
        lambda will be the energy of the state

        site: to be optimized (of desired shape)
    """
    ### L * H * R = d/dT*[i] d/dT[i] <psi|H|psi>
    H_eff = qtn.tensor_contract(left, H, right)

    ### H_eff[i] * T[i] = lambda T[i]  (assumes in canonical form)




