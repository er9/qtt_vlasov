import numpy as np
from abc import ABC

from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
from setup_.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper

from setup_.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT

from local_solvers_old.local_solver import LocalSolver, MPS_type, MPO_type, SweepDirection
from local_solvers_old.local_solver import DEFAULT_MAX_TOT_ITER, DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DEFAULT_MAX_WRONG_ITER
from local_solvers_old.environments_v2 import TargetEnv_DMRG, OperatorEnv_DMRG
from local_solvers_old.time_integrator import TimeSolver


class TDVPSolver(TimeSolver):

    def site_time_evolution(self, dt, left_site_pos, nsites=1, sweep_direction=SweepDirection.RIGHT,
                            verbose=False, ) -> tuple[qtn.Tensor, Numeric]:
        """ solving + decimation (moving orthog center to the next site)
        """

        self._site_time_evolution(dt, left_site_pos, nsites=nsites, compress_direction=sweep_direction)

        if sweep_direction == SweepDirection.RIGHT:
            at_end = (left_site_pos == self.L - nsites)
        else:
            at_end = (left_site_pos == nsites)

        if not at_end:
            if nsites == 1:
                if verbose:  print('BACK prop (RL) bond ind', left_site_pos, sweep_direction)
                self._bond_time_evolution(-dt, left_site_pos, compress_direction=sweep_direction)

                # self.canonize(left_site_pos, cur_orthog=left_site_pos)  ## updates bra internally
                # self._update_envs_right(left_site_pos, canonize=False)

            elif nsites == 2:
                site_pos = left_site_pos if sweep_direction == SweepDirection.LEFT else left_site_pos + 1
                self._site_time_evolution(-dt, site_pos, nsites=1, compress_direction=sweep_direction)

        return self.ket


    def _site_time_evolution(self, dt, left_site_pos, nsites=1, compress_direction=SweepDirection.RIGHT,
                             ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # print('site time evolution check orthog site TE', left_site_pos, nsites, compress_direction)
        # helper.check_orthog(self.ket)
        # helper.check_orthog(self.bra)

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self.get_A_effs(left_site_pos, nsites)

        ## change indices from bra (missing T*[i]) to ket
        # bra_to_ket_inds = {}
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
            # bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))


        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == SweepDirection.RIGHT:
            self.canonize(site_inds[-1], cur_orthog=site_inds[0])     ## also sets bra from ket
            for i in range(left_site_pos, left_site_pos + nsites - 1):
                self._update_envs_left(i, canonize=False)
        else:
            self.canonize(site_inds[0], cur_orthog=site_inds[-1])  ## also sets bra from ket
            for i in range(left_site_pos + nsites - 1, left_site_pos, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _bond_time_evolution(self, dt, cur_orthog,
                             compress_direction=SweepDirection.RIGHT
                             ) -> Sequence[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        """
        # site_pos = list(range(self.L))[bond_ind]
        # bond_ind = self.mps_inds[bond_pos]

        # helper.check_orthog(self.ket)
        ket_usvt = MPS_USVT.from_MPS(self.ket, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        bra_usvt = MPS_USVT.from_MPS(self.bra, canon_site=cur_orthog, cur_orthog=cur_orthog, direction=compress_direction)
        Q_tens = ket_usvt[cur_orthog]
        S_tens = ket_usvt.get_S_tensor()
        Q_conj = bra_usvt[cur_orthog]
        S_conj = bra_usvt.get_S_tensor()

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = []
        for i in range(self.num_operators):
            A_left = self.A_envsLs[i][cur_orthog]
            A_right = self.A_envsRs[i][cur_orthog]
            # print('A left', A_left)
            # print('A right', A_right)

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            A_eff.add(A[cur_orthog])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            ## add Q(R) / (L)Q tensor
            A_eff.add([Q_tens, Q_conj])

            # A_eff.exponent = self.A_envsLs[i].exponent
            ## !!! changed = -> +=
            A_eff.exponent += A.exponent
            A_effs += [A_eff]

        ## change indices from bra (missing T*[i]) to ket
        shared, notshared = S_tens.filter_bonds(Q_tens)
        ket_inds = shared + notshared

        shared, notshared = S_conj.filter_bonds(Q_conj)
        bra_inds = shared + notshared

        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bra_inds + ket_inds)
        dist_submpx = qtn.TensorNetwork([S_tens], virtual=True)

        # for A_eff in A_effs:
        #     print('A eff', A_eff)
        # print('ket', ket_usvt)
        # print('bra', bra_usvt)
        # print('bonds_o', bra_inds, 'bonds_i', ket_inds)
        site_time_evolution(dist_submpx, dt, A_effs, [next(iter(S_tens.tags))],
                            bonds_i = ket_inds, bonds_o = bra_inds, te_order=self.te_order, inplace=True,
                            compress_direction=compress_direction, compress_opts_dict=self.compress_config
                            )

        # print('cur orthog to MPS', cur_orthog)
        new_ket = ket_usvt.to_MPS()
        ket_tens = self.ket[cur_orthog]
        new_tens = new_ket[cur_orthog]
        new_tens = new_tens.transpose_like(ket_tens, inplace=True)
        ket_tens.modify(data = new_tens.data)
        self.set_bra_from_ket(sites=[cur_orthog])

        return self.ket
