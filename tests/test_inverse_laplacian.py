"""
tests bond dimension of inverse laplacian operator
for 1-, 2-, and 3-D for different mappings

plot bond dimension vs Ls for desired cutoff accuracy
TODO: for higher dims, vary vs each dimension independently?
"""

import sys, os, pickle
sys.path.append('../')

from setup.configs import *
# import helper_quimb

from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis, RealFourierBasis
from vlasov_tests import LandauDamping

fdir = 'data/inverse_laplacian/'
if not os.path.exists(fdir):
    os.makedirs(fdir)

q = 2

cutoffs = [1.0e-10, 1.0e-18, 1.0e-20]
basis = FourierBasis()

for basis in [FourierBasis(), SpatialBasis()]:
    for K in [1,2,3]:
        for TN_layout in [LayoutType.COMB, LayoutType.SEQUENTIAL, LayoutType.PARALLEL, LayoutType.PARALLEL_GROUP]:
            for map_key in ['F','B','M','N']:

                x_ax_map_key = map_key * K

                plt.figure()
                mpo_bonds = {}
                for cutoff in cutoffs:


                    if K == 1:
                        Lxs = np.arange(5, 12)
                    elif K == 2:
                        Lxs = np.arange(5, 15)
                    elif K == 3:
                        Lxs = np.arange(5, 12)

                    max_bonds = []
                    spine_max_bonds = []


                    for L in Lxs:

                        if K == 1:
                            Ls = (L,)
                            ks = (0.75,)
                        elif K == 2:
                            Ls = (L, L,)
                            kx = 0.5
                            ky = np.sqrt(0.75 ** 2 - kx ** 2)
                            ks = (kx, ky)
                        elif K == 3:
                            Ls = (L, L, L)
                            kx, ky = 0.5, 0.2
                            kz = np.sqrt(0.75 ** 2 - kx ** 2 - ky ** 2)
                            ks = (kx, ky, kz)

                        VP_test = LandauDamping(ks=ks,
                                                Lxs=Ls, KX=K, KV=0, q=2,
                                                x_map_key = x_ax_map_key,
                                                grid_layout=TN_layout,
                                                basis_xs = basis,
                                                )

                        ### axis objects
                        x_axes = VP_test.pos_axes

                        ### ocordinates
                        X, Y, Z = VP_test.pos_coordsys.coords

                        ### define MPS grids
                        grid_i = VP_test.grid_i
                        grid_e = VP_test.grid_e
                        grid_X = VP_test.grid_X

                        ax_deriv_configs = {ax: DerivativeConfiguration(left_bc=BCType.PERIODIC) for ax in x_axes}
                        inv_laplacian_mpo = grid_X.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs,
                                                                         compress_opts={'cutoff':cutoff,
                                                                                        'cutoff_mode':'rsum2'})
                        print(inv_laplacian_mpo)

                        max_bonds += [inv_laplacian_mpo.max_bond()]

                        if TN_layout == LayoutType.COMB:
                            spine_max_bonds += [inv_laplacian_mpo.spine.max_bond()]

                        # inv_lapl_comp = inv_laplacian_mpo.compress(compress_opts={'max_bond': D})
                        # errs += [inv_lapl_comp.distance(inv_laplacian_mpo)]

                    print(TN_layout, x_ax_map_key, cutoff, ':', max_bonds)

                    plt.plot(Lxs, max_bonds, label=f'cutoff {np.sqrt(cutoff):1.1e}')
                    if TN_layout == LayoutType.COMB:
                        plt.plot(Lxs, spine_max_bonds, '--', label=f'cutoff {np.sqrt(cutoff):1.1e} (spine)')

                    mpo_bonds[cutoff] = [Lxs, max_bonds]

                plt.xlabel('L')
                plt.ylabel('max bond dimension')
                plt.ylim(0,np.max(max_bonds)*1.2)
                plt.plot(Lxs, q**(Lxs*K), 'k:', label='theory max')
                plt.legend()

                plt.show()

                fstr = f'inv_laplacian_max_bonds_K{K}_{basis.__class__}_TN{TN_layout.value}_{x_ax_map_key}'
                pickle.dump(mpo_bonds, open(fdir+fstr+'.pkl','wb'))


